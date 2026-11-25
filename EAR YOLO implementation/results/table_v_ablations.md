# Table V. Ablation of Trigger Mask, Budget Guard and Gated Score

| Variant                               | Trigger      | Guard   | Surv. Head / Gated   | mAP@0.5 (%)   | Latency (ms)   |
|---------------------------------------|--------------|---------|----------------------|---------------|----------------|
| Full method                           | Yes          | Yes     | Yes                  | 88.9 ± 0.3    | 34.1 ± 0.2     |
| No trigger mask                       | No           | Yes     | Yes                  | 82.3 ± 0.4    | 34.5 ± 0.4     |
| No budget guard                       | Yes          | No      | Yes                  | 85.9 ± 0.3    | 56.4 ± 0.8     |
| Confidence only (no surv. head)       | Conf.        | Yes     | No                   | 81.0 ± 0.5    | 34.7 ± 0.3     |
| Gated score only on C2 (no rule)      | —            | —       | Yes                  | 85.9 ± 0.3    | 58.0 ± 0.5     |
| Rule only (raw conf., no gated score) | Yes          | Yes     | No                   | 85.4 ± 0.3    | 34.3 ± 0.3     |
| C3g hand-tuned control                | Hand         | Yes     | No                   | 84.5 ± 0.3    | 34.2 ± 0.3     |
| Scalar head (uniform map) — N1        | Yes (T1, T2) | Yes     | Uniform scalar       | 85.1 ± 0.3    | 34.3 ± 0.2     |
| Per-cell MC Dropout map + rule        | Yes (T1, T2) | Yes     | Spatial uncert.      | 86.2 ± 0.3    | 34.4 ± 0.3     |
