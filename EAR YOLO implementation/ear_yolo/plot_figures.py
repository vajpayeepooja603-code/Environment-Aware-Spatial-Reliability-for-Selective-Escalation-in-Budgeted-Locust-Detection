"""
Publication Figures Generator
Generates all figures from the manuscript in high-resolution (300 DPI):
- Fig 1: Closed-loop spatial selective escalation framework
- Fig 2: Architecture of the reliability head with FiLM & YOLO backbone
- Fig 3: Flowchart of the legal-action rule (Algorithm 1)
- Fig 4: Accuracy-latency frontier with inset zoom (Roboflow Locust test split)
- Fig 5: Ablation of trigger mask, budget guard, and gated score
- Fig 6: In-domain and cross-dataset validation bar charts
- Fig 7: Per-Condition Real Hard-Field Breakdown (Table XII)
- Demo: Live detection, spatial survival heatmap & selective zoom overlay
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def plot_fig1_closed_loop(output_path: str = "figures/fig1_closed_loop_framework.png"):
    """Generates Figure 1: Closed-loop spatial selective escalation pipeline."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis('off')

    # Styles
    box_kwargs = dict(boxstyle='round,pad=0.5', facecolor='#f0f4f8', edgecolor='#1d3557', linewidth=1.8)
    hl_kwargs = dict(boxstyle='round,pad=0.5', facecolor='#d8f3dc', edgecolor='#2d6a4f', linewidth=2.0)
    guard_kwargs = dict(boxstyle='round,pad=0.5', facecolor='#ffe5d9', edgecolor='#d00000', linewidth=2.0)

    # Nodes
    ax.text(0.12, 0.75, "Observation $x_t$\n(512×512 Image)", ha='center', va='center', bbox=box_kwargs, fontsize=10, fontweight='bold')
    ax.text(0.38, 0.75, "YOLO11 Detector\n& Spatial Survival Head", ha='center', va='center', bbox=box_kwargs, fontsize=10, fontweight='bold')
    ax.text(0.72, 0.75, "Detections $B_t$\n& Spatial Survival Map $R_t$", ha='center', va='center', bbox=hl_kwargs, fontsize=10, fontweight='bold')
    
    ax.text(0.72, 0.28, "Algorithm 1:\nSelective Escalation Rule\n(Triggers $T_1$, $T_2$, Zoom Set $Z_t$)", ha='center', va='center', bbox=hl_kwargs, fontsize=10, fontweight='bold')
    ax.text(0.38, 0.28, "Budget Guard (M1b)\nLook-ahead Mean $\\hat{L}_{t+1} \\leq B$\n(Fallback to $a_{def}$ if > $B$)", ha='center', va='center', bbox=guard_kwargs, fontsize=10, fontweight='bold')
    ax.text(0.12, 0.28, "Execution:\nAction $a_t \\in A_{legal}$\n(Zoom $Z_t$ or 640 or $a_{def}$)", ha='center', va='center', bbox=box_kwargs, fontsize=10, fontweight='bold')

    # Arrows
    arrow = dict(arrowstyle="->", lw=2, color="#1d3557")
    ax.annotate("", xy=(0.24, 0.75), xytext=(0.20, 0.75), arrowprops=arrow)
    ax.annotate("", xy=(0.58, 0.75), xytext=(0.52, 0.75), arrowprops=arrow)
    ax.annotate("", xy=(0.72, 0.40), xytext=(0.72, 0.65), arrowprops=arrow)
    ax.annotate("", xy=(0.55, 0.28), xytext=(0.60, 0.28), arrowprops=arrow)
    ax.annotate("", xy=(0.24, 0.28), xytext=(0.28, 0.28), arrowprops=arrow)
    
    # Feedback loop arrow
    ax.annotate("", xy=(0.12, 0.65), xytext=(0.12, 0.40),
                arrowprops=dict(arrowstyle="->", lw=2, color="#2b2d42", linestyle="--"))
    ax.text(0.04, 0.52, "Closed-Loop\nFeedback\n$t \\rightarrow t+1$", fontsize=9, color="#2b2d42", fontweight='bold')

    plt.title("Fig. 1. Closed-loop spatial selective escalation pipeline under running-mean budget $B$.", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved Figure 1 to: {output_path}")


def plot_fig2_architecture(output_path: str = "figures/fig2_reliability_head_architecture.png"):
    """Generates Figure 2: Reliability Head and FiLM modulation architecture."""
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.axis('off')

    desc_box = dict(boxstyle='round,pad=0.4', facecolor='#e0f2fe', edgecolor='#0369a1', lw=1.5)
    mlp_box = dict(boxstyle='round,pad=0.4', facecolor='#bae6fd', edgecolor='#0284c7', lw=1.8)
    backbone_box = dict(boxstyle='round,pad=0.4', facecolor='#fef08a', edgecolor='#ca8a04', lw=1.8)
    film_box = dict(boxstyle='round,pad=0.4', facecolor='#fed7aa', edgecolor='#ea580c', lw=1.8)
    head_box = dict(boxstyle='round,pad=0.4', facecolor='#bbf7d0', edgecolor='#16a34a', lw=1.8)

    ax.text(0.12, 0.85, "8 Descriptors $e_t$\n(Luminance, Contrast, Shadow,\nExG, Veg, Edge, Noise, Temp)", ha='center', va='center', bbox=desc_box, fontsize=9)
    ax.text(0.35, 0.85, "FiLM MLP\n$e'_t = \\text{MLP}(e_t)$", ha='center', va='center', bbox=mlp_box, fontsize=10, fontweight='bold')

    ax.text(0.12, 0.40, "YOLO11 Backbone\n(CSP Stages)", ha='center', va='center', bbox=backbone_box, fontsize=10, fontweight='bold')
    
    # Pyramid levels
    ax.text(0.35, 0.60, "P3 (stride 8)", ha='center', va='center', bbox=backbone_box, fontsize=9)
    ax.text(0.35, 0.40, "P4 (stride 16)", ha='center', va='center', bbox=backbone_box, fontsize=9)
    ax.text(0.35, 0.20, "P5 (stride 32)", ha='center', va='center', bbox=backbone_box, fontsize=9)

    # FiLM blocks
    ax.text(0.55, 0.60, "FiLM Block\n$(1+\\gamma_3)F_3 + \\beta_3$", ha='center', va='center', bbox=film_box, fontsize=9)
    ax.text(0.55, 0.40, "FiLM Block\n$(1+\\gamma_4)F_4 + \\beta_4$", ha='center', va='center', bbox=film_box, fontsize=9)
    ax.text(0.55, 0.20, "FiLM Block\n$(1+\\gamma_5)F_5 + \\beta_5$", ha='center', va='center', bbox=film_box, fontsize=9)

    # Heads
    ax.text(0.85, 0.60, "Detection Head\nBBoxes & Confidences $c_i$", ha='center', va='center', bbox=head_box, fontsize=10, fontweight='bold')
    ax.text(0.85, 0.30, "Spatial Survival Head\nPer-Cell Map $R_t(u, v) = \\sigma(z)$\nSurvival Loss $\\mathcal{L}_{rel}$ (Eq. 9)", ha='center', va='center', bbox=head_box, fontsize=10, fontweight='bold')

    # Connections
    arrow = dict(arrowstyle="->", lw=1.5, color="#334155")
    ax.annotate("", xy=(0.28, 0.85), xytext=(0.22, 0.85), arrowprops=arrow)
    ax.annotate("", xy=(0.50, 0.65), xytext=(0.42, 0.80), arrowprops=arrow)
    ax.annotate("", xy=(0.50, 0.45), xytext=(0.42, 0.80), arrowprops=arrow)
    ax.annotate("", xy=(0.50, 0.25), xytext=(0.42, 0.80), arrowprops=arrow)

    ax.annotate("", xy=(0.28, 0.60), xytext=(0.20, 0.45), arrowprops=arrow)
    ax.annotate("", xy=(0.28, 0.40), xytext=(0.20, 0.40), arrowprops=arrow)
    ax.annotate("", xy=(0.28, 0.20), xytext=(0.20, 0.35), arrowprops=arrow)

    ax.annotate("", xy=(0.47, 0.60), xytext=(0.41, 0.60), arrowprops=arrow)
    ax.annotate("", xy=(0.47, 0.40), xytext=(0.41, 0.40), arrowprops=arrow)
    ax.annotate("", xy=(0.47, 0.20), xytext=(0.41, 0.20), arrowprops=arrow)

    ax.annotate("", xy=(0.74, 0.60), xytext=(0.63, 0.50), arrowprops=arrow)
    ax.annotate("", xy=(0.74, 0.35), xytext=(0.63, 0.35), arrowprops=arrow)

    plt.title("Fig. 2. Architecture of the reliability head. Modulation by FiLM and multi-scale survival output.", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved Figure 2 to: {output_path}")


def plot_fig3_algorithm_flowchart(output_path: str = "figures/fig3_algorithm1_decision_flow.png"):
    """Generates Figure 3: Flowchart of Algorithm 1."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.axis('off')

    rect = dict(boxstyle='square,pad=0.5', facecolor='#f8fafc', edgecolor='#0f172a', lw=1.5)
    diamond = dict(boxstyle='round4,pad=0.5', facecolor='#fef3c7', edgecolor='#b45309', lw=1.8)
    action_box = dict(boxstyle='round,pad=0.5', facecolor='#dcfce7', edgecolor='#15803d', lw=2.0)
    guard_box = dict(boxstyle='round,pad=0.5', facecolor='#fee2e2', edgecolor='#b91c1c', lw=2.0)

    ax.text(0.50, 0.92, "Input Frame $x_t$ -> YOLO11 & Survival Head\nOutputs $B_t$ & Spatial Map $R_t$", ha='center', va='center', bbox=rect, fontsize=9.5)
    ax.text(0.50, 0.77, "Compute Window Survivals $r_w$ (Eq. 7)\nSet $W_{low} = \\{w : r_w < \\rho=0.5\\}$", ha='center', va='center', bbox=rect, fontsize=9.5)
    ax.text(0.50, 0.60, "Evaluate Triggers:\n$T_1: r_t < \\rho$ OR\n$T_2: \\exists c_i \\in [0.10, 0.50]$ in $W_{low}$?", ha='center', va='center', bbox=diamond, fontsize=9.5, fontweight='bold')

    ax.text(0.18, 0.42, "Neither Fired (¬T1 ∧ ¬T2)\n$A_{legal} = \\{a_{def}\\} \\cup A_{cheap}$", ha='center', va='center', bbox=rect, fontsize=9)
    ax.text(0.82, 0.42, "Trigger Fired (T1 ∨ T2)\nZoom Set $Z_t = \\text{Top-}K_{max}(W_{T2})$\n$A_{legal} = \\{a_{640}, a_{enh}\\} \\cup \\{a_{zoom, w}\\}$", ha='center', va='center', bbox=rect, fontsize=9)

    ax.text(0.50, 0.28, "Look-Ahead Budget Guard (Eq. 19):\nIs $\\hat{L}_{t+1}(a^*) = L_t + \\frac{\\hat{\\ell}(a^*) - \\ell_{t-W+1}}{W} > B$ (34.8 ms)?", ha='center', va='center', bbox=guard_box, fontsize=9.5, fontweight='bold')

    ax.text(0.20, 0.10, "Yes (Violates B)\nFallback: $a_t \\leftarrow a_{def}$", ha='center', va='center', bbox=action_box, fontsize=9.5, fontweight='bold')
    ax.text(0.80, 0.10, "No (Budget Safe)\nExecute $a_t \\leftarrow a^*$", ha='center', va='center', bbox=action_box, fontsize=9.5, fontweight='bold')

    arrow = dict(arrowstyle="->", lw=1.6, color="#0f172a")
    ax.annotate("", xy=(0.50, 0.83), xytext=(0.50, 0.87), arrowprops=arrow)
    ax.annotate("", xy=(0.50, 0.68), xytext=(0.50, 0.72), arrowprops=arrow)

    # Branches
    ax.annotate("", xy=(0.20, 0.48), xytext=(0.38, 0.58), arrowprops=arrow)
    ax.annotate("", xy=(0.80, 0.48), xytext=(0.62, 0.58), arrowprops=arrow)

    ax.annotate("", xy=(0.45, 0.34), xytext=(0.20, 0.38), arrowprops=arrow)
    ax.annotate("", xy=(0.55, 0.34), xytext=(0.80, 0.38), arrowprops=arrow)

    ax.annotate("", xy=(0.25, 0.15), xytext=(0.42, 0.22), arrowprops=arrow)
    ax.annotate("", xy=(0.75, 0.15), xytext=(0.58, 0.22), arrowprops=arrow)

    plt.title("Fig. 3. Flow of the legal-action rule (Algorithm 1) with budget guard constraint.", fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved Figure 3 to: {output_path}")


def plot_fig4_frontier(output_path: str = "figures/fig4_accuracy_latency_frontier.png"):
    """
    Generates Figure 4: Accuracy-latency frontier matching Table IV
    with the inset magnification exactly like the manuscript.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    data = [
        ("C1: Fixed default", 22.5, 78.4, "s", "black", 80),
        ("C2: Always maximum", 58.0, 83.1, "^", "tab:orange", 100),
        ("C3g: Hand-tuned control", 34.2, 84.5, "D", "orange", 80),
        ("C4: Contextual bandit", 35.0, 83.9, "v", "tab:green", 85),
        ("C5: Random matched-cost", 34.8, 76.2, "x", "deeppink", 85),
        ("LTF-YOLO [10]", 34.6, 81.2, "+", "deepskyblue", 90),
        ("AdaDet-style [15]", 34.5, 82.7, "*", "brown", 100),
        ("DynamicDet/RL-zoom [11]", 34.7, 83.4, "o", "gray", 75),
        ("UaDAN [27]", 34.5, 84.6, "o", "tab:blue", 85),
        ("RSADet [26]", 34.6, 83.8, "X", "navy", 80),
        ("Spatial Uncertainty [28]", 34.7, 84.2, "p", "purple", 80),
        ("AdaZoom [13]", 34.6, 84.1, "d", "teal", 75),
        ("DSENet++ [14]", 34.7, 85.0, "P", "maroon", 80),
        ("CertainNet [29]", 34.5, 84.4, "h", "olive", 80),
        ("Proposed (Spatial Survival Head)", 34.1, 88.9, "*", "darkviolet", 150),
        ("Masked PPO", 34.5, 89.2, "*", "limegreen", 150),
    ]

    for name, lat, map50, marker, color, s in data:
        ax.scatter(lat, map50, marker=marker, color=color, s=s, label=name, zorder=4)

    # Budget Line
    ax.axvline(x=34.8, color='black', linestyle='--', linewidth=1.5, label='Budget $B = 34.8$ ms')

    ax.set_xlabel("Mean latency (ms)", fontsize=11, fontweight='bold')
    ax.set_ylabel("mAP@0.5 (%)", fontsize=11, fontweight='bold')
    ax.set_xlim(18, 62)
    ax.set_ylim(74, 91)
    ax.grid(True, linestyle=':', alpha=0.6)

    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), ncol=2, fontsize=7.5, frameon=True, framealpha=0.9)

    axins = inset_axes(ax, width="40%", height="48%", loc='lower right', borderpad=2)
    for name, lat, map50, marker, color, s in data:
        if 33.5 <= lat <= 35.5:
            axins.scatter(lat, map50, marker=marker, color=color, s=s * 1.1, zorder=4)
            if "Proposed" in name:
                axins.annotate(f"Proposed\n({lat:.1f} ms, {map50:.1f}%)", xy=(lat, map50),
                               xytext=(lat - 0.75, map50 - 0.4), fontsize=6.8, fontweight='bold', color='darkviolet')
            elif "Masked PPO" in name:
                axins.annotate(f"Masked PPO\n({lat:.1f} ms, {map50:.1f}%)", xy=(lat, map50),
                               xytext=(lat + 0.1, map50 + 0.15), fontsize=6.8, fontweight='bold', color='green')
            elif "C3g" in name:
                axins.annotate(f"C3g ({lat:.1f} ms, {map50:.1f}%)", xy=(lat, map50),
                               xytext=(lat - 0.8, map50 - 0.5), fontsize=6.5, color='orange')
            elif "UaDAN" in name:
                axins.annotate(f"UaDAN", xy=(lat, map50), xytext=(lat - 0.5, map50 + 0.2), fontsize=6.5, color='tab:blue')
            elif "C5" in name:
                axins.annotate(f"C5 ({lat:.1f} ms, {map50:.1f}%)", xy=(lat, map50),
                               xytext=(lat + 0.05, map50 - 0.4), fontsize=6.5, color='deeppink')

    axins.axvline(x=34.8, color='black', linestyle='--', linewidth=1.2)
    axins.set_xlim(33.5, 35.5)
    axins.set_ylim(75.5, 90.0)
    axins.grid(True, linestyle=':', alpha=0.5)
    axins.tick_params(labelsize=7)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", ls="--")

    plt.title("Fig. 4. Accuracy-latency frontier on Roboflow Locust split (Budget B = 34.8 ms).", fontsize=11, fontweight='bold', pad=12)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved Figure 4 to: {output_path}")


def plot_fig5_ablation(output_path: str = "figures/fig5_ablation_analysis.png"):
    """Generates Figure 5: Ablation of components (mAP@0.5 and Latency)."""
    labels = [
        "Full\nmethod", "No trigger\nmask", "No budget\nguard", "Confidence\nonly",
        "Gated score\n(on C2)", "Rule\nonly", "C3g\ncontrol", "Scalar head\n(N1)", "Spatial\nuncert."
    ]
    map50 = [88.9, 82.3, 85.9, 81.0, 85.9, 85.4, 84.5, 85.1, 86.2]
    map50_sd = [0.3, 0.4, 0.3, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3]
    lat = [34.1, 34.5, 56.4, 34.7, 58.0, 34.3, 34.2, 34.3, 34.4]
    lat_sd = [0.2, 0.4, 0.8, 0.3, 0.5, 0.3, 0.3, 0.2, 0.3]

    colors = ['#0077b6'] + ['#6c757d'] * (len(labels) - 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot (a) mAP@0.5
    bars1 = ax1.bar(labels, map50, yerr=map50_sd, capsize=4, color=colors, edgecolor='black', linewidth=0.8)
    ax1.set_ylabel("mAP@0.5 (%)", fontsize=11, fontweight='bold')
    ax1.set_ylim(78, 91)
    ax1.set_title("(a) mAP@0.5", fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.6, f"{yval:.1f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Subplot (b) Latency
    bars2 = ax2.bar(labels, lat, yerr=lat_sd, capsize=4, color=colors, edgecolor='black', linewidth=0.8)
    ax2.axhline(y=34.8, color='#f77f00', linestyle='--', linewidth=1.5, label='Budget $B = 34.8$ ms')
    ax2.set_ylabel("Mean latency (ms)", fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 65)
    ax2.set_title("(b) Mean latency", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9.5)
    ax2.grid(axis='y', linestyle=':', alpha=0.7)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f"{yval:.1f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.suptitle("Fig. 5. Ablation of the proposed rule components (data of Table V).", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved Figure 5 to: {output_path}")


def plot_fig6_cross_dataset(output_path: str = "figures/fig6_cross_dataset_comparison.png"):
    """Generates Figure 6: In-Domain and Cross-Dataset comparison (4 subplots)."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), dpi=300)
    splits = ["In-domain", "Cross-dataset"]
    x = np.arange(len(splits))
    width = 0.25

    metric_configs = [
        ("mAP@0.5 (%)", "mAP@0.5 (%)", [78.4, 74.1], [88.9, 84.2], [85.1, 80.5], (0, 100)),
        ("mAP@0.5:0.95 (%)", "mAP@0.5:0.95 (%)", [52.1, 48.0], [63.4, 56.8], [53.8, 53.3], (0, 75)),
        ("Small-locust AP (%)", "Small AP (%)", [45.2, 41.5], [57.8, 51.2], [53.8, 47.6], (0, 70)),
        ("Latency (ms)", "Mean latency (ms)", [22.5, 22.0], [34.1, 33.8], [34.3, 34.0], (0, 42))
    ]

    for idx, (title, ylabel, c1, prop, scal, ylim) in enumerate(metric_configs):
        ax = axes[idx]
        b1 = ax.bar(x - width, c1, width, label='C1: Fixed default', color='#495057')
        b2 = ax.bar(x, prop, width, label='Proposed (Spatial)', color='#0284c7')
        b3 = ax.bar(x + width, scal, width, label='Scalar head (N1)', color='#f59e0b')

        ax.set_title(title, fontsize=10.5, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(splits, fontsize=9.5)
        ax.set_ylim(ylim)
        ax.grid(axis='y', linestyle=':', alpha=0.6)

        for bar in [b1, b2, b3]:
            for p in bar:
                h = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2.0, h + (ylim[1] * 0.015), f"{h:.1f}", ha='center', va='bottom', fontsize=7.5)

    axes[0].legend(loc='lower left', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=9.5, frameon=True)
    plt.suptitle("Fig. 6. In-domain and cross-dataset validation of fixed default, proposed spatial, and scalar head.", fontsize=11.5, fontweight='bold', y=1.12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved Figure 6 to: {output_path}")


def plot_fig7_real_hard_field_breakdown(output_path: str = "figures/fig7_real_hard_field_breakdown.png"):
    """Generates Figure 7: Per-Condition Real Hard-Field Breakdown (Table XII)."""
    conditions = ["Glare", "Shadow", "Dust / Haze", "Vegetation", "Motion Blur"]
    c1_vals = [73.4, 75.6, 72.9, 77.1, 74.2]
    scalar_vals = [79.9, 81.6, 79.2, 82.5, 80.1]
    spatial_vals = [85.7, 86.9, 84.8, 87.6, 85.8]
    gains = [5.8, 5.3, 5.6, 5.1, 5.7]

    x = np.arange(len(conditions))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    rects1 = ax.bar(x - width, c1_vals, width, label='C1 Default (512x512)', color='#64748b', edgecolor='black', lw=0.6)
    rects2 = ax.bar(x, scalar_vals, width, label='Scalar Head + Rule (Uniform N1)', color='#f59e0b', edgecolor='black', lw=0.6)
    rects3 = ax.bar(x + width, spatial_vals, width, label='Spatial Head + Rule (EAR-YOLO)', color='#0284c7', edgecolor='black', lw=0.8)

    # Annotate values and gains
    for i, (r1, r2, r3, g) in enumerate(zip(rects1, rects2, rects3, gains)):
        h3 = r3.get_height()
        ax.text(r3.get_x() + r3.get_width() / 2.0, h3 + 0.6, f"{h3:.1f}%\n(+{g} pts)", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#0284c7')

    ax.set_ylabel("mAP@0.5 (%)", fontsize=11, fontweight='bold')
    ax.set_title("Fig. 7. Per-Condition Performance on Real Hard-Field Dataset (Table XII Breakdown)", fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10.5, fontweight='bold')
    ax.set_ylim(65, 93)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.95)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved Figure 7 to: {output_path}")


def plot_live_detection_demo(sample_img: np.ndarray,
                             R_map: np.ndarray,
                             candidates: list,
                             zoom_windows: list,
                             output_path: str = "figures/detection_spatial_reliability_demo.png"):
    """
    Renders live visualization of observation with spatial reliability overlay,
    candidate detections, and targeted zoom windows.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    # 1. Raw image with degraded area
    axes[0].imshow(sample_img)
    axes[0].set_title("Input Observation $x_t$ (512×512)\n[Synthetically Degraded Field]", fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # 2. Spatial Survival Map Heatmap R_t(u, v)
    im = axes[1].imshow(R_map, cmap='RdYlGn', vmin=0.0, vmax=1.0)
    axes[1].set_title("Spatial Survival Map $R_t(u, v)$\n(16×16 Grid, Red=Degraded, Green=Clear)", fontsize=10, fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Detections & Selective Zoom Windows
    axes[2].imshow(sample_img)
    axes[2].set_title("Selective Escalation:\nBoxes + Targeted Zoom Windows $Z_t$", fontsize=10, fontweight='bold')
    axes[2].axis('off')

    for c in candidates:
        x1, y1, x2, y2 = c.box
        w_box = x2 - x1
        h_box = y2 - y1
        rect = patches.Rectangle((x1, y1), w_box, h_box, linewidth=1.8, edgecolor='cyan', facecolor='none')
        axes[2].add_patch(rect)
        axes[2].text(x1, max(10, y1 - 4), f"s={c.gated_score:.2f}", color='cyan', fontsize=7.5, fontweight='bold',
                     bbox=dict(boxstyle='square,pad=0.1', facecolor='black', alpha=0.6))

    for w in zoom_windows:
        rect = patches.Rectangle((w.x1, w.y1), w.x2 - w.x1, w.y2 - w.y1,
                                 linewidth=2.2, edgecolor='red', linestyle='--', facecolor='red', alpha=0.15)
        axes[2].add_patch(rect)
        axes[2].text(w.x1 + 6, w.y1 + 18, f"Zoom W{w.id} (r={w.mean_survival:.2f})", color='red', fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved Live Detection Demo to: {output_path}")


def generate_all_figures():
    """Generates all figures."""
    print("\n" + "=" * 70)
    print("Generating Publication Figures (300 DPI High-Resolution)...")
    print("=" * 70)
    plot_fig1_closed_loop()
    plot_fig2_architecture()
    plot_fig3_algorithm_flowchart()
    plot_fig4_frontier()
    plot_fig5_ablation()
    plot_fig6_cross_dataset()
    plot_fig7_real_hard_field_breakdown()
    print("=" * 70)
    print("All Figures generated successfully in 'figures/' directory.")
    print("=" * 70)
