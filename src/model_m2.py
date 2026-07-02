"""M2 meta-labeling model."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import M2Config, PipelineConfig
from src.model_m1 import ASSET_CLASS_MAP, train_date_mask
from src.feature_engineering import get_feature_columns
from src.labels import get_m2_training_mask

logger = logging.getLogger(__name__)

M1_COMPONENT_COLS = ("momentum_score", "trend_score", "macro_score", "risk_penalty")
M2_META_DERIVED_COLS = (
    "m1_cs_rank",
    "m1_score_abs",
    "m1_x_vol",
    "m1_x_risk_off",
    "m1_x_macro",
)


class M2Model(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> M2Model:
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        ...

    def predict_meta_label(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """Diagnostic M3 preview: binary threshold on p_success (not an M2 model output)."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int).rename("predicted_meta_label")


def create_m2_model(cfg: M2Config) -> M2Model:
    architecture = getattr(cfg, "architecture", "global")
    if architecture == "per_asset":
        return PerAssetM2(cfg)
    return SklearnM2(cfg)


class SklearnM2(M2Model):
    def __init__(self, cfg: M2Config) -> None:
        self.cfg = cfg
        self.feature_cols: list[str] = []
        self.pipeline: Pipeline | CalibratedClassifierCV | None = None

    def _build_estimator(self) -> Pipeline:
        model_type = self.cfg.type
        if model_type == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=20,
                random_state=42,
                class_weight="balanced",
            )
        elif model_type == "gradient_boosting":
            clf = GradientBoostingClassifier(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.05,
                min_samples_leaf=20,
                random_state=42,
            )
        else:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SklearnM2:
        self.feature_cols = list(X.columns)
        base = self._build_estimator()
        if self.cfg.calibrate:
            self.pipeline = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        else:
            self.pipeline = base
        self.pipeline.fit(X[self.feature_cols].values, y.values)
        logger.info("M2 fitted on %d rows, %d features", len(X), len(self.feature_cols))
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if self.pipeline is None:
            raise RuntimeError("M2 model not fitted")
        cols = [c for c in self.feature_cols if c in X.columns]
        proba = self.pipeline.predict_proba(X[cols].values)[:, 1]
        return pd.Series(proba, index=X.index, name="p_success")


class PerAssetM2(M2Model):
    """Per-ticker M2 heads with fallback to a global model."""

    def __init__(self, cfg: M2Config) -> None:
        self.cfg = cfg
        self.global_model = SklearnM2(cfg)
        self.asset_models: dict[str, SklearnM2] = {}
        self.min_asset_samples = int(getattr(cfg, "min_asset_samples", 80))
        self.feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PerAssetM2:
        self.feature_cols = list(X.columns)
        self.global_model.fit(X, y)
        self.asset_models = {}
        if not isinstance(X.index, pd.MultiIndex):
            logger.warning("PerAssetM2: index is not MultiIndex; using global model only")
            return self
        tickers = X.index.get_level_values("ticker")
        for ticker in tickers.unique():
            mask = tickers == ticker
            n = int(mask.sum())
            if n < self.min_asset_samples:
                continue
            asset_model = SklearnM2(self.cfg)
            asset_model.fit(X.loc[mask], y.loc[mask])
            self.asset_models[str(ticker)] = asset_model
            logger.info("M2 per-asset head fitted: %s (%d rows)", ticker, n)
        logger.info(
            "PerAssetM2: %d asset heads + global fallback (%d features)",
            len(self.asset_models),
            len(self.feature_cols),
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if self.global_model.pipeline is None:
            raise RuntimeError("M2 model not fitted")
        if not isinstance(X.index, pd.MultiIndex) or not self.asset_models:
            return self.global_model.predict_proba(X)
        tickers = X.index.get_level_values("ticker")
        out = pd.Series(np.nan, index=X.index, dtype=float, name="p_success")
        for ticker in tickers.unique():
            mask = tickers == ticker
            model = self.asset_models.get(str(ticker), self.global_model)
            out.loc[mask] = model.predict_proba(X.loc[mask]).values
        return out


def _cross_sectional_m1_rank(panel: pd.DataFrame) -> pd.Series:
    if "M1_score" not in panel.columns:
        return pd.Series(np.nan, index=panel.index)
    if isinstance(panel.index, pd.MultiIndex):
        dates = panel.index.get_level_values("date")
        return panel["M1_score"].groupby(dates).rank(pct=True).rename("m1_cs_rank")
    return panel["M1_score"].rank(pct=True).rename("m1_cs_rank")


def _asset_class_dummies(panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(panel.index, pd.MultiIndex):
        return pd.DataFrame(index=panel.index)
    tickers = panel.index.get_level_values("ticker")
    classes = sorted(set(ASSET_CLASS_MAP.values()))
    out = pd.DataFrame(index=panel.index)
    for asset_class in classes:
        out[f"asset_class_{asset_class}"] = [
            1.0 if ASSET_CLASS_MAP.get(str(t), "equity") == asset_class else 0.0 for t in tickers
        ]
    return out


def build_m2_features(panel: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """M2 feature matrix: base factors + M1 context + asset encoding + interactions."""
    reset = panel.reset_index() if isinstance(panel.index, pd.MultiIndex) else panel.copy()
    base_features = get_feature_columns(reset)
    available = [c for c in base_features if c in panel.columns]

    cols: list[str] = list(dict.fromkeys(available))
    frames: list[pd.DataFrame] = [panel[available].copy()]

    m2_cfg = cfg.m2
    use_meta = getattr(m2_cfg, "use_meta_features", True)
    if use_meta:
        meta_cols = ["M1_signal", "M1_score", *M1_COMPONENT_COLS]
        for col in meta_cols:
            if col in panel.columns and col not in cols:
                cols.append(col)
                frames.append(panel[[col]])

        derived = pd.DataFrame(index=panel.index)
        derived["m1_cs_rank"] = _cross_sectional_m1_rank(panel)
        if "M1_score" in panel.columns:
            derived["m1_score_abs"] = panel["M1_score"].abs()
        if "M1_score" in panel.columns and "z_vol_12w" in panel.columns:
            derived["m1_x_vol"] = panel["M1_score"] * panel["z_vol_12w"]
        if "M1_score" in panel.columns and "risk_off" in panel.columns:
            derived["m1_x_risk_off"] = panel["M1_score"] * panel["risk_off"]
        if "M1_score" in panel.columns and "macro_score" in panel.columns:
            derived["m1_x_macro"] = panel["M1_score"] * panel["macro_score"]

        for col in M2_META_DERIVED_COLS:
            if col in derived.columns:
                cols.append(col)
        frames.append(derived[[c for c in M2_META_DERIVED_COLS if c in derived.columns]])

    if getattr(m2_cfg, "include_asset_encoding", True):
        asset_dummies = _asset_class_dummies(panel)
        if not asset_dummies.empty:
            cols.extend(list(asset_dummies.columns))
            frames.append(asset_dummies)

    out = pd.concat(frames, axis=1)
    out = out.loc[:, list(dict.fromkeys(cols))]
    return out


def resolve_m2_for_importance(m2_model: object) -> SklearnM2 | None:
    """Return the global SklearnM2 used for coefficient / importance charts."""
    if isinstance(m2_model, SklearnM2):
        return m2_model
    if isinstance(m2_model, PerAssetM2):
        return m2_model.global_model
    return None


def _m2_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    from sklearn.metrics import roc_auc_score

    mask = y_true.notna() & y_prob.notna()
    y = y_true[mask].astype(int)
    p = y_prob[mask]
    if len(y) < 2 or y.nunique() < 2 or p.nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def m2_architecture_benchmark(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Compare legacy global M2 vs the configured (enriched) model on train and test AUC."""
    from dataclasses import replace

    rows: list[dict[str, Any]] = []
    legacy_cfg = replace(
        cfg.m2,
        architecture="global",
        use_meta_features=False,
        include_asset_encoding=False,
        type="logistic_regression",
        calibrate=True,
    )
    variants: list[tuple[str, M2Config]] = [
        ("legacy_global", legacy_cfg),
        ("configured", cfg.m2),
    ]

    train_mask = get_m2_training_mask(train_panel)
    test_mask = get_m2_training_mask(test_panel)
    y_train = train_panel.loc[train_mask, "meta_label"].dropna()
    y_test = test_panel.loc[test_mask, "meta_label"].dropna()

    for name, m2_cfg in variants:
        variant_cfg = replace(cfg, models={**cfg.models, "m2": m2_cfg})
        X_train = build_m2_features(train_panel.loc[train_mask], variant_cfg).loc[y_train.index]
        X_test = build_m2_features(test_panel.loc[test_mask], variant_cfg).loc[y_test.index]
        model = create_m2_model(m2_cfg)
        model.fit(X_train, y_train)
        p_train = model.predict_proba(X_train)
        p_test = model.predict_proba(X_test)
        asset_heads = len(getattr(model, "asset_models", {})) if isinstance(model, PerAssetM2) else 0
        rows.append(
            {
                "variant": name,
                "model_type": m2_cfg.type,
                "architecture": getattr(m2_cfg, "architecture", "global"),
                "n_features": len(X_train.columns),
                "n_train": len(y_train),
                "n_test": len(y_test),
                "train_auc": _m2_auc(y_train, p_train),
                "test_auc": _m2_auc(y_test, p_test),
                "asset_heads": asset_heads,
            }
        )
    return pd.DataFrame(rows)


def fit_m2(
    panel: pd.DataFrame,
    cfg: PipelineConfig,
    train_mask: pd.Series | None = None,
) -> tuple[M2Model, pd.DataFrame]:
    if train_mask is None:
        dates = panel.index.get_level_values("date")
        train_mask = train_date_mask(dates, cfg).values
    m2_mask = get_m2_training_mask(panel) & train_mask
    train_panel = panel.loc[m2_mask]
    y = train_panel["meta_label"].dropna()
    X = build_m2_features(train_panel, cfg).loc[y.index]
    model = create_m2_model(cfg.m2)
    model.fit(X, y)
    return model, X


def predict_m2(model: M2Model, panel: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = panel.copy()
    m2_rows = get_m2_training_mask(out)
    X = build_m2_features(out, cfg)
    out["p_success"] = np.nan
    if m2_rows.any():
        out.loc[m2_rows, "p_success"] = model.predict_proba(X.loc[m2_rows]).values
        out.loc[m2_rows, "predicted_meta_label"] = model.predict_meta_label(
            X.loc[m2_rows], threshold=cfg.m2.threshold
        ).values
    return out
