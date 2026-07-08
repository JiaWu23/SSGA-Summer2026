"""Integration test requiring network access."""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_full_pipeline_smoke(tmp_path):
    from pathlib import Path
    import shutil

    import yaml

    from src.run_pipeline import run_pipeline

    root = Path(__file__).parent.parent
    cfg_path = tmp_path / "config.yaml"
    base_cfg = yaml.safe_load((root / "config" / "config.yaml").read_text())
    base_cfg["split"]["train_start"] = "2018-01-01"
    base_cfg["split"]["train_end"] = "2019-12-31"
    base_cfg["split"]["test_start"] = "2020-01-01"
    base_cfg["evaluation"] = {
        "walk_forward_enabled": False,
        "transaction_cost_bps_grid": [0, 5, 10],
    }
    base_cfg["paths"] = {
        "raw": str(tmp_path / "data/raw"),
        "processed": str(tmp_path / "data/processed"),
        "features": str(tmp_path / "data/features"),
        "predictions": str(tmp_path / "data/predictions"),
        "backtests": str(tmp_path / "data/backtests"),
        "runs": str(tmp_path / "runs"),
    }
    cfg_path.write_text(yaml.dump(base_cfg))

    # Reuse cached market/macro parquet from repo data/ if available
    src_processed = root / "data" / "processed"
    dst_processed = tmp_path / "data" / "processed"
    dst_processed.mkdir(parents=True, exist_ok=True)
    for name in ("market_weekly.parquet", "macro_weekly.parquet"):
        src = src_processed / name
        if src.exists():
            shutil.copy(src, dst_processed / name)

    summary = run_pipeline(str(cfg_path), project_root=tmp_path)
    run_dir = summary.run_dir
    assert run_dir.exists()
    assert (run_dir / "config_snapshot.yaml").exists()
    assert (tmp_path / "data" / "backtests" / "long_only" / "metrics_table.csv").exists()
    assert (tmp_path / "data" / "backtests" / "long_short" / "metrics_table.csv").exists()
    assert (tmp_path / "reports" / "final_report.md").exists()
    assert (tmp_path / "reports" / "m1_factor_analysis.md").exists()
    assert (tmp_path / "reports" / "m2_diagnostics.md").exists()
    assert (tmp_path / "reports" / "market_regime_analysis.md").exists()
    assert (tmp_path / "reports" / "m3_allocation_analysis.md").exists()
    assert (tmp_path / "data" / "backtests" / "long_only" / "m1_factor_ic.csv").exists()
    assert (tmp_path / "data" / "backtests" / "long_only" / "m1_factor_weight_tuning.csv").exists()
    assert (tmp_path / "data" / "backtests" / "long_only" / "m2_calibration_table.csv").exists()
    assert (tmp_path / "data" / "backtests" / "long_only" / "m2_architecture_benchmark.csv").exists()
    assert (tmp_path / "data" / "backtests" / "long_only" / "m3_allocation_summary.csv").exists()
    assert (tmp_path / "data" / "backtests" / "long_only" / "evaluation" / "transaction_cost_sensitivity.csv").exists()
    assert (tmp_path / "reports" / "evaluation_analysis.md").exists()
    pred = tmp_path / "data" / "predictions" / "long_only" / "panel_with_predictions.parquet"
    assert pred.exists()
    import pandas as pd

    panel = pd.read_parquet(pred)
    assert "M3_size" in panel.columns
    assert "allocation_state" in panel.columns
    assert (tmp_path / "reports" / "mode_comparison" / "m1_mode_comparison.png").exists()
    assert (tmp_path / "reports" / "final" / "long_only" / "strategy_cumulative_returns.png").exists()
    assert (tmp_path / "reports" / "assets" / "asset_component_analysis.md").exists()
