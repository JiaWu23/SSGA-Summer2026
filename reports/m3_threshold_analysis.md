# M3 Threshold Sweep Analysis

**Research use only — not investment advice.**

At the default threshold **T=0.55**, binary M3 approves ~**100%** of M1 candidates (recall ≈ 1) because calibrated `p_success` on the test set rarely falls below 0.55. This sweep finds thresholds where M3 **meaningfully rejects** candidates (`m3_zero` ≥ 5% of M1 signals) and compares test-period portfolio Sharpe vs M1-only.

## Setup

- **Test window:** `2021-01-01` to `latest`
- **Default threshold:** `0.55` (`models.m3.threshold` / `models.m2.threshold`)
- **Binary M3:** size = 1 if `p_success ≥ T`, else 0
- **Linear gated M3:** size = `max(0, 2p−1)` if `p_success ≥ T`, else 0 (research variant; production linear is ungated)
- **Meaningful rejection:** ≥5% of M1 candidates with `m3_zero`; binary also requires recall < 99%

## Recommended thresholds

### Binary M3

- **Recommended T:** `0.6`
- **Test Sharpe:** 0.9441 (vs baseline T=0.55: 0.7869)
- **Sharpe edge vs M1-only:** 0.1571
- **M3 rejection share (test candidates):** 62.8070%
- **M2 recall / precision:** 0.4286 / 0.6792
- **Apply to config?** `yes`
- **Rationale:** Best test Sharpe among thresholds with meaningful rejection (≥5% m3_zero, recall < 99% for binary): T=0.60, Sharpe 0.9441, rejection 62.8%, recall 0.429.

### Linear gated M3

- **Recommended T:** `0.6`
- **Test Sharpe:** 0.9494 (vs baseline T=0.55: 0.8597)
- **Sharpe edge vs M1-only:** 0.1624
- **M3 rejection share (test candidates):** 62.8070%
- **M2 recall / precision:** 0.4286 / 0.6792
- **Apply to config?** `yes`
- **Rationale:** Best test Sharpe among thresholds with meaningful rejection (≥5% m3_zero, recall < 99% for binary): T=0.60, Sharpe 0.9494, rejection 62.8%, recall 0.429.

## Full comparison table (test period)

| m3_mode | threshold | m1_candidates | m3_zero_count | m3_rejection_share | m3_approval_rate | mean_m3_size_on_candidates | m2_recall | m2_precision | m2_f1 | degeneracy_note | test_ann_return | test_sharpe | test_max_drawdown | sharpe_edge_vs_m1 | meaningful_rejection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary | 0.5000 | 855 | 0 | 0.0000% | 100.0000% | 1.0000 | 1.0000 | 0.5895 | 0.7417 | Binary M3 at this threshold approves all trades; strategy equals M1-only. | 8.4044% | 0.7869 | -21.0040% | 0.0000 | no |
| binary | 0.5200 | 855 | 0 | 0.0000% | 100.0000% | 1.0000 | 1.0000 | 0.5895 | 0.7417 | Binary M3 at this threshold approves all trades; strategy equals M1-only. | 8.4044% | 0.7869 | -21.0040% | 0.0000 | no |
| binary | 0.5400 | 855 | 0 | 0.0000% | 100.0000% | 1.0000 | 1.0000 | 0.5895 | 0.7417 | Binary M3 at this threshold approves all trades; strategy equals M1-only. | 8.4044% | 0.7869 | -21.0040% | 0.0000 | no |
| binary | 0.5500 | 855 | 0 | 0.0000% | 100.0000% | 1.0000 | 1.0000 | 0.5895 | 0.7417 | Binary M3 at this threshold approves all trades; strategy equals M1-only. | 8.4044% | 0.7869 | -21.0040% | -0.0000 | no |
| binary | 0.5600 | 855 | 3 | 0.3509% | 99.6491% | 0.9965 | 1.0000 | 0.5915 | 0.7434 | Binary M3 at this threshold approves all trades; strategy equals M1-only. | 8.2099% | 0.7680 | -21.8631% | -0.0189 | no |
| binary | 0.5800 | 855 | 95 | 11.1111% | 88.8889% | 0.8889 | 0.9008 | 0.5974 | 0.7184 |  | 7.6885% | 0.7450 | -20.9149% | -0.0420 | yes |
| binary | 0.6000 | 855 | 537 | 62.8070% | 37.1930% | 0.3719 | 0.4286 | 0.6792 | 0.5255 |  | 5.5759% | 0.9441 | -6.2346% | 0.1571 | yes |
| binary | 0.6200 | 855 | 840 | 98.2456% | 1.7544% | 0.0175 | 0.0159 | 0.5333 | 0.0308 |  | 0.7954% | 0.6715 | -1.5650% | -0.1155 | yes |
| binary | 0.6400 | 855 | 855 | 100.0000% | 0.0000% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000% | 0.0000 | 0.0000% | -0.7869 | yes |
| binary | 0.6600 | 855 | 855 | 100.0000% | 0.0000% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000% | 0.0000 | 0.0000% | -0.7869 | yes |
| binary | 0.6800 | 855 | 855 | 100.0000% | 0.0000% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000% | 0.0000 | 0.0000% | -0.7869 | yes |
| binary | 0.7000 | 855 | 855 | 100.0000% | 0.0000% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000% | 0.0000 | 0.0000% | -0.7869 | yes |
| linear_gated | 0.5000 | 855 | 0 | 0.0000% | 100.0000% | 0.1914 | 1.0000 | 0.5895 | 0.7417 | Binary M3 at this threshold approves all trades; strategy equals M1-only. | 1.8668% | 0.8597 | -4.4004% | 0.0728 | no |
| linear_gated | 0.5200 | 855 | 0 | 0.0000% | 100.0000% | 0.1914 | 1.0000 | 0.5895 | 0.7417 | Binary M3 at this threshold approves all trades; strategy equals M1-only. | 1.8668% | 0.8597 | -4.4004% | 0.0728 | no |
| linear_gated | 0.5400 | 855 | 0 | 0.0000% | 100.0000% | 0.1914 | 1.0000 | 0.5895 | 0.7417 | Binary M3 at this threshold approves all trades; strategy equals M1-only. | 1.8668% | 0.8597 | -4.4004% | 0.0728 | no |
| linear_gated | 0.5500 | 855 | 0 | 0.0000% | 100.0000% | 0.1914 | 1.0000 | 0.5895 | 0.7417 | Binary M3 at this threshold approves all trades; strategy equals M1-only. | 1.8668% | 0.8597 | -4.4004% | 0.0728 | no |
| linear_gated | 0.5600 | 855 | 3 | 0.3509% | 99.6491% | 0.1910 | 1.0000 | 0.5915 | 0.7434 | Binary M3 at this threshold approves all trades; strategy equals M1-only. | 1.8455% | 0.8518 | -4.4762% | 0.0648 | no |
| linear_gated | 0.5800 | 855 | 95 | 11.1111% | 88.8889% | 0.1752 | 0.9008 | 0.5974 | 0.7184 |  | 1.7611% | 0.8421 | -4.1455% | 0.0552 | yes |
| linear_gated | 0.6000 | 855 | 537 | 62.8070% | 37.1930% | 0.0805 | 0.4286 | 0.6792 | 0.5255 |  | 1.2228% | 0.9494 | -1.4415% | 0.1624 | yes |
| linear_gated | 0.6200 | 855 | 840 | 98.2456% | 1.7544% | 0.0043 | 0.0159 | 0.5333 | 0.0308 |  | 0.1947% | 0.6718 | -0.3874% | -0.1152 | yes |
| linear_gated | 0.6400 | 855 | 855 | 100.0000% | 0.0000% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000% | 0.0000 | 0.0000% | -0.7869 | yes |
| linear_gated | 0.6600 | 855 | 855 | 100.0000% | 0.0000% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000% | 0.0000 | 0.0000% | -0.7869 | yes |
| linear_gated | 0.6800 | 855 | 855 | 100.0000% | 0.0000% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000% | 0.0000 | 0.0000% | -0.7869 | yes |
| linear_gated | 0.7000 | 855 | 855 | 100.0000% | 0.0000% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000% | 0.0000 | 0.0000% | -0.7869 | yes |

![M3 threshold sweep](../data/backtests/long_only/figures/m3_threshold_sweep.png)

## Key findings

- **T=0.55 (binary):** recall ≈ 1.0000, rejection ≈ 0.0000% — effectively M1-only.
- **Binary best with rejection:** T=0.60 (Sharpe 0.9441, rejection 62.8070%).
- **ECDF sizing** (not swept here) remains the primary risk-shaping layer; threshold sweeps target interpretable binary/linear rules.

Related: [m3_allocation_analysis.md](m3_allocation_analysis.md) · [m2_diagnostics.md](m2_diagnostics.md)
