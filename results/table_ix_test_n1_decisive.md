# Table IX. Spatial vs Scalar Head (Test N1 — Decisive)

| Method                                                       | mAP@0.5    | Small AP   | Latency       | Look-ahead violations (%)   | Δ (Spatial − method)   |
|--------------------------------------------------------------|------------|------------|---------------|-----------------------------|------------------------|
| Spatial head + rule                                          | 88.9 ± 0.3 | 57.8 ± 0.3 | 34.1 ± 0.2 ms | 0.0%                        | —                      |
| Scalar head S1 (uniform ranking, spatial eligibility) + rule | 85.1 ± 0.3 | 53.8 ± 0.4 | 34.3 ± 0.2 ms | 0.0%                        | +3.8% [+3.0%, +4.6%]   |
| Fully scalar S2 (uniform eligibility and ranking) + rule     | 83.9 ± 0.4 | 52.5 ± 0.4 | 34.5 ± 0.3 ms | 0.0%                        | +5.0% [+4.1%, +5.9%]   |
| Per-cell MC Dropout map + rule                               | 86.2 ± 0.3 | 54.9 ± 0.4 | 34.4 ± 0.3 ms | 0.0%                        | +2.7% [+1.9%, +3.5%]   |
