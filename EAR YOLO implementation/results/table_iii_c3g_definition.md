# Table III. Exact Definition of Hand-Tuned Control C3g

| Component                     | C3g Setting                                                                                                                           |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Image-level trigger           | Mean confidence of top-10 detections < 0.45                                                                                           |
| Candidate-level trigger       | Any detection with confidence ∈ [0.15, 0.55]                                                                                          |
| Zoom selection                | Up to 4 windows with highest detection density                                                                                        |
| Resolution / enhancement      | Escalate to 640 + enhancement when either trigger fires                                                                               |
| Budget guard                  | Identical look-ahead guard of Eq. (5) with B = 34.8 ms                                                                                |
| Survival head / gated score   | None (raw confidence only)                                                                                                            |
| Difference from proposed rule | Uses confidence statistics instead of spatial survival map; no survival-gated score; zoom chosen by density rather than survival rank |
