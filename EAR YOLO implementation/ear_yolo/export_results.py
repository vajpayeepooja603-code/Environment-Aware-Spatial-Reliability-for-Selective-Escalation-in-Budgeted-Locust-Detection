"""
Results Table Exporter
Saves all verified benchmark tables from the manuscript into:
- CSV files (results/table_*.csv)
- Markdown files (results/table_*.md)
- Complete Interactive HTML Report with embedded graphs and LaTeX notation
"""

import os
import csv
from tabulate import tabulate
from .reproduce import ManuscriptResultsDatabase


def export_all_tables(output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    db = ManuscriptResultsDatabase

    print("\n" + "=" * 70)
    print(f"Exporting Verified Results Tables (I - XII) to '{output_dir}/'...")
    print("=" * 70)

    # Table I: Principal Notation
    t1_headers = ["Symbol", "Meaning"]
    t1_rows = [[r["symbol"], r["meaning"]] for r in db.TABLE_I]
    _save_csv(os.path.join(output_dir, "table_i_principal_notation.csv"), t1_headers, t1_rows)
    _save_md(os.path.join(output_dir, "table_i_principal_notation.md"), "Table I. Principal Notation", t1_headers, t1_rows)

    # Table II: Frozen Constants
    t2_headers = ["Setting", "Value"]
    t2_rows = [[r["setting"], r["value"]] for r in db.TABLE_II]
    _save_csv(os.path.join(output_dir, "table_ii_frozen_constants.csv"), t2_headers, t2_rows)
    _save_md(os.path.join(output_dir, "table_ii_frozen_constants.md"), "Table II. Frozen Constants and Implementation Settings", t2_headers, t2_rows)

    # Table III: Hand-Tuned Control C3g
    t3_headers = ["Component", "C3g Setting"]
    t3_rows = [[r["component"], r["c3g_setting"]] for r in db.TABLE_III]
    _save_csv(os.path.join(output_dir, "table_iii_c3g_definition.csv"), t3_headers, t3_rows)
    _save_md(os.path.join(output_dir, "table_iii_c3g_definition.md"), "Table III. Exact Definition of Hand-Tuned Control C3g", t3_headers, t3_rows)

    # Table IV: Main Detection Metrics (Synthetic)
    t4_headers = ["Method", "mAP@0.5 (%)", "mAP@0.5:0.95 (%)", "Small AP (%)", "Latency (ms)", "Escalation (%)"]
    t4_rows = []
    for r in db.TABLE_IV:
        t4_rows.append([
            r["method"],
            f"{r['map50']:.1f} ± {r['map50_sd']:.1f}",
            f"{r['map50_95']:.1f} ± {r['map50_95_sd']:.1f}",
            f"{r['small_ap']:.1f} ± {r['small_ap_sd']:.1f}",
            f"{r['latency']:.1f} ± {r['latency_sd']:.1f}",
            f"{r['escalation_rate']:.1f} ± {r['escalation_rate_sd']:.1f}"
        ])
    _save_csv(os.path.join(output_dir, "table_iv_main_metrics.csv"), t4_headers, t4_rows)
    _save_md(os.path.join(output_dir, "table_iv_main_metrics.md"), "Table IV. Main Detection Metrics at Matched Mean Latency (Synthetic)", t4_headers, t4_rows)

    # Table V: Ablations
    t5_headers = ["Variant", "Trigger", "Guard", "Surv. Head / Gated", "mAP@0.5 (%)", "Latency (ms)"]
    t5_rows = []
    for r in db.TABLE_V:
        t5_rows.append([
            r["variant"], r["trigger"], r["guard"], r["head"],
            f"{r['map50']:.1f} ± {r['map50_sd']:.1f}",
            f"{r['latency']:.1f} ± {r['latency_sd']:.1f}"
        ])
    _save_csv(os.path.join(output_dir, "table_v_ablations.csv"), t5_headers, t5_rows)
    _save_md(os.path.join(output_dir, "table_v_ablations.md"), "Table V. Ablation of Trigger Mask, Budget Guard and Gated Score", t5_headers, t5_rows)

    # Table VI: Cross-Dataset
    t6_headers = ["Dataset / Split", "Method", "mAP@0.5 (%)", "mAP@0.5:0.95 (%)", "Small AP (%)", "Latency (ms)"]
    t6_rows = []
    for r in db.TABLE_VI:
        t6_rows.append([
            r["dataset"], r["method"],
            f"{r['map50']:.1f} ± {r['map50_sd']:.1f}",
            f"{r['map50_95']:.1f} ± {r['map50_95_sd']:.1f}",
            f"{r['small_ap']:.1f} ± {r['small_ap_sd']:.1f}",
            f"{r['latency']:.1f} ± {r['latency_sd']:.1f}"
        ])
    _save_csv(os.path.join(output_dir, "table_vi_cross_dataset.csv"), t6_headers, t6_rows)
    _save_md(os.path.join(output_dir, "table_vi_cross_dataset.md"), "Table VI. In-Domain and Cross-Dataset Validation", t6_headers, t6_rows)

    # Table VII: Test N1 (Synthetic)
    t7_headers = ["Method", "mAP@0.5", "Small AP", "Latency", "Budget-safe", "Δ vs Spatial"]
    t7_rows = []
    for r in db.TABLE_VII:
        t7_rows.append([r["method"], r["map50"], r["small_ap"], r["latency"], r["budget_safe"], r["delta"]])
    _save_csv(os.path.join(output_dir, "table_vii_test_n1_decisive.csv"), t7_headers, t7_rows)
    _save_md(os.path.join(output_dir, "table_vii_test_n1_decisive.md"), "Table VII. Spatial vs Scalar Head (Test N1 — Decisive, Synthetic)", t7_headers, t7_rows)

    # Table VIII: Test N2
    t8_headers = ["Method", "mAP@0.5", "Small AP", "AUROC", "Esc. Precision", "Budget-safe"]
    t8_rows = []
    for r in db.TABLE_VIII:
        t8_rows.append([r["method"], r["map50"], r["small_ap"], r["auroc"], r["esc_prec"], r["budget_safe"]])
    _save_csv(os.path.join(output_dir, "table_viii_test_n2_uncertainty.csv"), t8_headers, t8_rows)
    _save_md(os.path.join(output_dir, "table_viii_test_n2_uncertainty.md"), "Table VIII. Spatial Head vs Scalar Uncertainty Baselines (Test N2)", t8_headers, t8_rows)

    # Table IX: LOPO
    t9_headers = ["Left-out family", "Survival AUROC (mean ± SD, 5 seeds)", "mAP@0.5 change vs conf-only (pts)"]
    t9_rows = []
    for r in db.TABLE_IX:
        t9_rows.append([r["family"], r["auroc"], r["delta_map"]])
    _save_csv(os.path.join(output_dir, "table_ix_leave_one_out.csv"), t9_headers, t9_rows)
    _save_md(os.path.join(output_dir, "table_ix_leave_one_out.md"), "Table IX. Leave-One-Perturbation-Out Validation of Reliability Head", t9_headers, t9_rows)

    # Table X: Real Hard-Field Main Detection Metrics
    t10_headers = ["Method", "mAP@0.5 (%)", "mAP@0.5:0.95 (%)", "Small AP (%)", "Latency (ms)", "Escalation (%)"]
    t10_rows = []
    for r in db.TABLE_X:
        t10_rows.append([
            r["method"],
            f"{r['map50']:.1f} ± {r['map50_sd']:.1f}",
            f"{r['map50_95']:.1f} ± {r['map50_95_sd']:.1f}",
            f"{r['small_ap']:.1f} ± {r['small_ap_sd']:.1f}",
            f"{r['latency']:.1f} ± {r['latency_sd']:.1f}",
            f"{r['escalation_rate']:.1f} ± {r['escalation_rate_sd']:.1f}"
        ])
    _save_csv(os.path.join(output_dir, "table_x_real_hard_field_main.csv"), t10_headers, t10_rows)
    _save_md(os.path.join(output_dir, "table_x_real_hard_field_main.md"), "Table X. Real Hard-Field Main Detection Metrics at Matched Mean Latency", t10_headers, t10_rows)

    # Table XI: Real Hard-Field Test N1
    t11_headers = ["Method", "mAP@0.5 (%)", "Small AP (%)", "Latency (ms)", "Δ vs Spatial"]
    t11_rows = []
    for r in db.TABLE_XI:
        t11_rows.append([r["method"], r["map50"], r["small_ap"], r["latency"], r["delta"]])
    _save_csv(os.path.join(output_dir, "table_xi_real_hard_field_n1.csv"), t11_headers, t11_rows)
    _save_md(os.path.join(output_dir, "table_xi_real_hard_field_n1.md"), "Table XI. Real Hard-Field Spatial vs Scalar Head (Test N1)", t11_headers, t11_rows)

    # Table XII: Per-Condition Real Hard-Field Breakdown
    t12_headers = ["Environmental Condition", "Images (n)", "C1 Default mAP@0.5 (%)", "Scalar Head + Rule (%)", "Spatial Head + Rule (%)", "Gain (+pts)"]
    t12_rows = []
    for r in db.TABLE_XII:
        t12_rows.append([r["condition"], str(r["images"]), r["c1_map50"], r["scalar_map50"], r["spatial_map50"], r["gain"]])
    _save_csv(os.path.join(output_dir, "table_xii_real_hard_field_breakdown.csv"), t12_headers, t12_rows)
    _save_md(os.path.join(output_dir, "table_xii_real_hard_field_breakdown.md"), "Table XII. Per-Condition Real Hard-Field Breakdown", t12_headers, t12_rows)

    # Master HTML Report
    _save_html_report(os.path.join(output_dir, "benchmark_report.html"),
                      t1_headers, t1_rows, t2_headers, t2_rows, t3_headers, t3_rows,
                      t4_headers, t4_rows, t5_headers, t5_rows, t6_headers, t6_rows,
                      t7_headers, t7_rows, t8_headers, t8_rows, t9_headers, t9_rows,
                      t10_headers, t10_rows, t11_headers, t11_rows, t12_headers, t12_rows)

    print(f"[*] All 12 manuscript tables exported to '{output_dir}/'.")
    print("=" * 70)


def _save_csv(path: str, headers: list, rows: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"    - Saved CSV: {path}")


def _save_md(path: str, title: str, headers: list, rows: list):
    content = f"# {title}\n\n" + tabulate(rows, headers=headers, tablefmt="github") + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"    - Saved MD:  {path}")


def _save_html_report(path: str, h1, r1, h2, r2, h3, r3, h4, r4, h5, r5, h6, r6, h7, r7, h8, r8, h9, r9, h10, r10, h11, r11, h12, r12):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EAR-YOLO: Official Benchmark & Verification Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f8fafc; color: #1e293b; margin: 40px; }}
    h1 {{ color: #0f172a; border-bottom: 3px solid #0284c7; padding-bottom: 12px; }}
    h2 {{ color: #0369a1; margin-top: 40px; border-bottom: 1px solid #cbd5e1; padding-bottom: 8px; }}
    h3 {{ color: #0f172a; margin-top: 20px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: white; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
    th {{ background-color: #f1f5f9; font-weight: 600; color: #334155; }}
    tr:hover {{ background-color: #f8fafc; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
    .badge-pass {{ background: #dcfce7; color: #15803d; }}
    .fig-container {{ text-align: center; margin: 30px 0; }}
    .fig-container img {{ max-width: 90%; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.15); }}
</style>
</head>
<body>
<h1>EAR-YOLO: Spatial Perturbation-Survival Maps for Budget-Aware Locust Detection</h1>
<p><b>Manuscript:</b> Spatial Perturbation-Survival Maps for Budget-Aware Locust Detection Under Environmental Degradation</p>
<p><b>Hardware Baseline:</b> NVIDIA Jetson Orin Nano, TensorRT 10.0, PyTorch 2.4 | <b>Budget:</b> B = 34.8 ms</p>

<h2>Part I: System Formulation & Constants</h2>
<h3>Table I. Principal Notation</h3>
{tabulate(r1, headers=h1, tablefmt="html")}

<h3>Table II. Frozen Constants and Implementation Settings</h3>
{tabulate(r2, headers=h2, tablefmt="html")}

<h3>Table III. Hand-Tuned Control C3g Definition</h3>
{tabulate(r3, headers=h3, tablefmt="html")}

<h2>Part II: Synthetic Perturbation Evaluations</h2>
<h3>Table IV. Main Detection Metrics at Matched Mean Latency</h3>
{tabulate(r4, headers=h4, tablefmt="html")}

<h3>Table V. Ablation of Trigger Mask, Budget Guard and Gated Score</h3>
{tabulate(r5, headers=h5, tablefmt="html")}

<h3>Table VI. In-Domain & Cross-Dataset Validation</h3>
{tabulate(r6, headers=h6, tablefmt="html")}

<h3>Table VII. Test N1 Decisive (Spatial vs. Scalar Head, Synthetic)</h3>
{tabulate(r7, headers=h7, tablefmt="html")}

<h3>Table VIII. Test N2 (Spatial Head vs. Uncertainty Baselines)</h3>
{tabulate(r8, headers=h8, tablefmt="html")}

<h3>Table IX. Leave-One-Perturbation-Out Validation</h3>
{tabulate(r9, headers=h9, tablefmt="html")}

<h2>Part III: Real Hard-Field Environmental Evaluations (Section 6.6)</h2>
<h3>Table X. Real Hard-Field Main Detection Metrics at Matched Latency</h3>
{tabulate(r10, headers=h10, tablefmt="html")}

<h3>Table XI. Real Hard-Field Spatial vs. Scalar Head (Test N1 Decisive)</h3>
{tabulate(r11, headers=h11, tablefmt="html")}

<h3>Table XII. Per-Condition Real Hard-Field Breakdown</h3>
{tabulate(r12, headers=h12, tablefmt="html")}

<h2>Part IV: Publication Figures</h2>
<div class="fig-container">
    <h3>Figure 4: Accuracy-Latency Frontier (Synthetic Protocol)</h3>
    <img src="../figures/fig4_accuracy_latency_frontier.png" alt="Figure 4">
</div>

<div class="fig-container">
    <h3>Figure 5: Ablation Analysis</h3>
    <img src="../figures/fig5_ablation_analysis.png" alt="Figure 5">
</div>

<div class="fig-container">
    <h3>Figure 6: Cross-Dataset Comparison</h3>
    <img src="../figures/fig6_cross_dataset_comparison.png" alt="Figure 6">
</div>

<div class="fig-container">
    <h3>Figure 7: Real Hard-Field Per-Condition Breakdown (Table XII)</h3>
    <img src="../figures/fig7_real_hard_field_breakdown.png" alt="Figure 7">
</div>

</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"    - Saved Master HTML Report: {path}")
