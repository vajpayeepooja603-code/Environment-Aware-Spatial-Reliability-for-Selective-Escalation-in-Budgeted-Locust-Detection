# Table XI. Real Hard-Field Spatial vs Scalar Head (Test N1)

| Method                           |   mAP@0.5 (%) |   Small AP (%) | Latency (ms)   | Δ vs Spatial   |
|----------------------------------|---------------|----------------|----------------|----------------|
| Spatial head + rule (Proposed)   |          86.2 |           54.3 | 34.0 ± 0.3 ms  | —              |
| Scalar head (uniform map) + rule |          80.8 |           48.6 | 34.2 ± 0.3 ms  | +5.4           |
| Per-cell MC Dropout map + rule   |          82.1 |           50.1 | 34.3 ± 0.3 ms  | +4.1           |
