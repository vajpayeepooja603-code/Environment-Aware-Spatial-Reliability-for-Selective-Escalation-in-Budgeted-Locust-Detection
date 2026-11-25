# Table VIII. Spatial Head vs Scalar Uncertainty Baselines (Test N2)

| Method                          | mAP@0.5    | Small AP   | AUROC         | Esc. Precision   | Budget-safe   |
|---------------------------------|------------|------------|---------------|------------------|---------------|
| Spatial head + rule (proposed)  | 88.9       | 57.8       | 0.912         | 88.6%            | 0%            |
| MC Dropout + rule               | 84.7 ± 0.4 | 53.6 ± 0.5 | 0.784 ± 0.019 | 61.4%            | 0%            |
| Deep Ensembles + rule           | 85.1 ± 0.3 | 54.1 ± 0.4 | 0.812 ± 0.016 | 63.8%            | 0%            |
| Temperature Scaling + rule      | 83.6 ± 0.3 | 52.2 ± 0.4 | 0.715 ± 0.021 | 57.9%            | 0%            |
| Mahalanobis OOD + rule          | 83.9 ± 0.4 | 52.5 ± 0.5 | 0.742 ± 0.020 | 59.1%            | 0%            |
| Energy OOD + rule               | 84.2 ± 0.4 | 52.9 ± 0.4 | 0.758 ± 0.018 | 60.2%            | 0%            |
| RSADet + rule [26]              | 83.8 ± 0.4 | 51.0 ± 0.5 | 0.765 ± 0.018 | 62.1%            | 0%            |
| Spatial Uncertainty + rule [28] | 84.2 ± 0.3 | 51.6 ± 0.4 | 0.779 ± 0.017 | 63.4%            | 0%            |
| UaDAN + rule [27]               | 84.6 ± 0.3 | 52.2 ± 0.4 | 0.795 ± 0.016 | 64.9%            | 0%            |
| CertainNet + rule [29]          | 84.4 ± 0.3 | 52.0 ± 0.5 | 0.788 ± 0.017 | 64.1%            | 0%            |
