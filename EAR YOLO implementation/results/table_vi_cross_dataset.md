# Table VI. In-Domain and Cross-Dataset Validation

| Dataset / Split   | Method                                   | mAP@0.5 (%)   | mAP@0.5:0.95 (%)   | Small AP (%)   | Latency (ms)   |
|-------------------|------------------------------------------|---------------|--------------------|----------------|----------------|
| In-domain test    | C1: Fixed default                        | 78.4 ± 0.3    | 52.1 ± 0.4         | 45.2 ± 0.5     | 22.5 ± 0.2     |
| In-domain test    | Spatial head + selective rule (proposed) | 88.9 ± 0.3    | 63.4 ± 0.3         | 57.8 ± 0.3     | 34.1 ± 0.2     |
| Cross-dataset     | C1: Fixed default                        | 74.1 ± 0.4    | 48.0 ± 0.5         | 41.5 ± 0.5     | 22.0 ± 0.2     |
| Cross-dataset     | Spatial head + selective rule (proposed) | 84.2 ± 0.3    | 56.8 ± 0.4         | 51.2 ± 0.4     | 33.8 ± 0.3     |
| Cross-dataset     | Scalar head (uniform map, N1)            | 80.5 ± 0.4    | 53.3 ± 0.5         | 47.6 ± 0.5     | 34.0 ± 0.3     |
| Cross-dataset     | Per-cell MC Dropout map + rule           | 81.7 ± 0.4    | 54.5 ± 0.5         | 48.9 ± 0.5     | 34.1 ± 0.3     |
| Cross-dataset     | RSADet [26]                              | 79.8 ± 0.4    | 52.4 ± 0.5         | 46.8 ± 0.5     | 34.4 ± 0.3     |
| Cross-dataset     | Spatial Uncertainty [28]                 | 80.3 ± 0.4    | 53.0 ± 0.5         | 47.4 ± 0.5     | 34.5 ± 0.3     |
| Cross-dataset     | UaDAN [27]                               | 81.1 ± 0.4    | 53.8 ± 0.4         | 48.1 ± 0.4     | 34.3 ± 0.3     |
