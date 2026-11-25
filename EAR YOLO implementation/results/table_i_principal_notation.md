# Table I. Principal Notation

| Symbol             | Meaning                                                                            |
|--------------------|------------------------------------------------------------------------------------|
| x_t                | Input image at observation t                                                       |
| D                  | Base object detector (YOLO11)                                                      |
| B_t = {(b_i, c_i)} | Detection boxes and confidence scores at observation t                             |
| R_t(u, v)          | Spatial survival map at cell (u, v) of grid U x V                                  |
| r_t, r_w           | Image-level mean reliability; mean reliability in window w                         |
| ρ                  | Survival trigger threshold; fixed at 0.5                                           |
| [c_lo, c_hi]       | Confidence interval for candidate trigger [0.10, 0.50]                             |
| K_max              | Maximum number of zoom windows; fixed at 4                                         |
| W_low              | Set of windows whose survival is below ρ                                           |
| T_1, T_2           | Image-level and candidate-level reliability triggers                               |
| Z_t                | Selected low-reliability windows for zoom re-inspection                            |
| A_full, A_legal    | Full action space and survival-constrained legal set                               |
| a_t, a_def         | Action selected at observation t; default action                                   |
| L_t, W             | Running-mean inference latency; window length W = 30                               |
| B                  | Latency budget; 60% of always-maximum latency (34.8 ms)                            |
| s_i = c_i * r_i^η  | Survival-gated detection score (η = 1.5)                                           |
| η                  | Score exponent; tuned on validation only (1.5)                                     |
| p_j, y_k           | Predicted survival probability at cell j and binary survival target of GT object k |
| L_rel              | Survival loss (Binary Cross-Entropy)                                               |
| γ_l, β_l           | FiLM scale and shift parameters at pyramid level l                                 |
| R_t_scalar         | Scalar version of map (image-level mean only broadcast) — used in N1 ablation      |
| E_prec             | Escalation precision: fraction of escalated windows containing a true locust       |
| N1, N2, N3         | Pre-declared tests (spatial vs scalar; spatial vs baselines; rule vs C3g)          |
