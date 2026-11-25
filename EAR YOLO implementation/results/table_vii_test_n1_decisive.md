# Table VII. Spatial vs Scalar Head (Test N1 — Decisive, Synthetic)

| Method                           | mAP@0.5    | Small AP   | Latency       | Budget-safe   | Δ vs Spatial         |
|----------------------------------|------------|------------|---------------|---------------|----------------------|
| Spatial head + rule (proposed)   | 88.9       | 57.8       | 34.1 ± 0.2 ms | 0%            | —                    |
| Scalar head (uniform map) + rule | 85.1 ± 0.3 | 53.8 ± 0.4 | 34.3 ± 0.2 ms | 0%            | +3.8% [+3.0%, +4.6%] |
| Per-cell MC Dropout map + rule   | 86.2 ± 0.3 | 54.9 ± 0.4 | 34.4 ± 0.3 ms | 0%            | -2.7% [-3.5%, -1.9%] |
