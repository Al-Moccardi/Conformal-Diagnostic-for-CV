#!/usr/bin/env python3
"""
generate_paper_assets.py — Generate ALL publication-ready figures, tables, and samples.

Reads results from every pipeline phase and produces:
  paper_assets/
  ├── sec3_calibration/       Calibration benchmark (7 models)
  ├── sec4_ablation/          Ablation A0-A5 + baselines
  ├── sec5_adversarial/       Attack results + 3-way tables
  ├── sec6_defense/           AUROC, Pareto, selective abstention
  ├── sec7_corruptions/       Natural corruptions
  ├── sec8_segmentation/      Bbox vs mask
  ├── sec9_transfer/          Black-box transferability
  ├── sec10_theory/           Proposition verification
  ├── sec11_statistics/       Bootstrap CI, effect sizes
  ├── tables_latex/           All tables in LaTeX format
  └── hero_figures/           Key figures for the paper

Usage:
    python generate_paper_assets.py
"""

import os, sys, json, warnings
from pathlib import Path
from collections import defaultdict
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

# Publication style
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.2, 'grid.linewidth': 0.5,
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'legend.framealpha': 0.9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette
C = {
    'blue': '#2E86C1', 'red': '#E74C3C', 'green': '#27AE60',
    'orange': '#E67E22', 'purple': '#8E44AD', 'gray': '#7F8C8D',
    'dark': '#2C3E50', 'light': '#ECF0F1',
    'yolo': '#3498DB', 'detr': '#E74C3C', 'rtdetr': '#27AE60',
}
MODEL_COLORS = {
    'yolov8x': C['blue'], 'yolov11x': C['purple'],
    'rtdetr-l': C['green'], 'detr-resnet101': C['red'],
    'yolov8x-seg': '#5DADE2', 'yolov11x-seg': '#AF7AC5',
}


def save(fig, path, close=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    if close:
        plt.close(fig)
    print(f"  [SAVED] {path}")


def save_latex(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"  [SAVED] {path}")


# =====================================================================
#  SECTION 3: CALIBRATION BENCHMARK
# =====================================================================

def gen_calibration(results_dir, out):
    """Generate calibration benchmark figures and tables."""
    print("\n=== Section 3: Calibration Benchmark ===")
    summary_path = os.path.join(results_dir, "evaluation", "metrics_summary.json")
    if not os.path.exists(summary_path):
        print("  [SKIP] metrics_summary.json not found")
        return

    with open(summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    models = [m for m in data if 'error' not in data[m] and 'mask2former' not in m]
    if not models:
        return

    # --- Fig: mAP vs ECE scatter (the hero figure for calibration) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in models:
        d = data[m]
        mAP = d.get('mAP@[0.5:0.95]', d.get('mAP@0.5', 0))
        ece = d.get('cal_ECE', 0)
        color = MODEL_COLORS.get(m, C['gray'])
        ax.scatter(ece, mAP, s=200, c=color, zorder=5, edgecolors='white', lw=2)
        ax.annotate(d.get('architecture', m), (ece, mAP),
                    fontsize=9, textcoords="offset points", xytext=(8, 5))
    ax.set_xlabel("ECE (Expected Calibration Error)")
    ax.set_ylabel("mAP@[0.5:0.95]")
    ax.set_title("Accuracy vs Calibration: Orthogonal Properties")
    save(fig, os.path.join(out, "sec3_calibration", "fig_mAP_vs_ECE.png"))

    # --- Fig: Reliability diagrams (2x3 grid) ---
    n_models = len(models)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, m in enumerate(models):
        ax = axes[idx]
        d = data[m]
        bins = d.get('calibration_bins', [])
        if not bins:
            ax.set_visible(False)
            continue
        centers = [b['bin_center'] for b in bins if b['count'] > 0]
        accs = [b['avg_acc'] for b in bins if b['count'] > 0]
        confs = [b['avg_conf'] for b in bins if b['count'] > 0]
        counts = [b['count'] for b in bins if b['count'] > 0]

        ax.bar(centers, accs, width=0.06, alpha=0.7, color=MODEL_COLORS.get(m, C['blue']),
               edgecolor='white', lw=0.5, label='Accuracy')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect')
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ece = d.get('cal_ECE', 0)
        ax.set_title(f"{d.get('architecture', m)} (ECE={ece:.4f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)

    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Reliability Diagrams — 7 Detectors on COCO val2017", fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, os.path.join(out, "sec3_calibration", "fig_reliability_diagrams.png"))

    # --- Fig: FPS vs mAP bubble (size = 1/ECE) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in models:
        d = data[m]
        fps = d.get('fps', 0)
        mAP = d.get('mAP@[0.5:0.95]', 0)
        ece = d.get('cal_ECE', 0.01)
        size = max(50, 300 / max(ece, 0.001))
        color = MODEL_COLORS.get(m, C['gray'])
        ax.scatter(fps, mAP, s=size, c=color, alpha=0.7, zorder=5, edgecolors='white', lw=2)
        ax.annotate(d.get('architecture', m), (fps, mAP),
                    fontsize=8, textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("FPS (higher = faster)")
    ax.set_ylabel("mAP@[0.5:0.95]")
    ax.set_title("Speed-Accuracy-Calibration (bubble size = 1/ECE)")
    save(fig, os.path.join(out, "sec3_calibration", "fig_speed_accuracy_calibration.png"))

    # --- LaTeX Table ---
    rows_latex = []
    for m in models:
        d = data[m]
        arch = d.get('architecture', m)
        mAP = d.get('mAP@[0.5:0.95]', 0)
        mAP50 = d.get('mAP@0.5', 0)
        ece = d.get('cal_ECE', 0)
        brier = d.get('cal_Brier', 0)
        f1 = d.get('F1@0.5', 0)
        fps = d.get('fps', 0)
        rows_latex.append(f"    {arch} & {mAP:.3f} & {mAP50:.3f} & {ece:.4f} & {brier:.3f} & {f1:.3f} & {fps:.0f} \\\\")

    latex = "\\begin{table}[t]\n\\centering\n\\caption{Calibration benchmark on COCO val2017 (5,000 images).}\n"
    latex += "\\label{tab:calibration}\n\\begin{tabular}{lcccccc}\n\\toprule\n"
    latex += "    Model & mAP & mAP@0.5 & ECE$\\downarrow$ & Brier$\\downarrow$ & F1 & FPS \\\\\n\\midrule\n"
    latex += "\n".join(rows_latex)
    latex += "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
    save_latex(latex, os.path.join(out, "tables_latex", "tab_calibration.tex"))


# =====================================================================
#  SECTION 5: ADVERSARIAL
# =====================================================================

def gen_adversarial(results_dir, out):
    """Generate adversarial attack figures and tables."""
    print("\n=== Section 5: Adversarial Attacks ===")
    adv_path = os.path.join(results_dir, "adversarial", "full_results.json")
    if not os.path.exists(adv_path):
        print("  [SKIP] full_results.json not found")
        return

    with open(adv_path, 'r', encoding='utf-8') as f:
        adv_data = json.load(f)

    # --- Fig: Coverage drop comparison (grouped bars, all models) ---
    fig, ax = plt.subplots(figsize=(14, 6))
    all_attacks = set()
    for mr in adv_data.values():
        all_attacks.update(mr.get("attacks", {}).keys())
    all_attacks = sorted(all_attacks)
    models = list(adv_data.keys())
    n_models = len(models)
    n_attacks = len(all_attacks)
    x = np.arange(n_attacks)
    w = 0.8 / max(n_models, 1)

    for i, model in enumerate(models):
        mr = adv_data[model]
        clean_cov = mr.get("clean_conformal", {}).get("coverage", 0)
        drops = []
        for atk in all_attacks:
            atk_data = mr.get("attacks", {}).get(atk, {})
            adv_cov = atk_data.get("naive_conformal", {}).get("coverage", 0)
            drops.append(clean_cov - adv_cov)
        color = MODEL_COLORS.get(model, C['gray'])
        ax.bar(x + i * w - 0.4 + w / 2, drops, w,
               label=mr.get("architecture", model), color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_attacks, rotation=30, ha='right')
    ax.set_ylabel("Coverage Drop (clean - adversarial)")
    ax.set_title("Coverage Drop Under Attack — All Models")
    ax.legend()
    save(fig, os.path.join(out, "sec5_adversarial", "fig_coverage_drop_all.png"))

    # --- Fig: mAP drop heatmap ---
    fig, ax = plt.subplots(figsize=(10, 4))
    matrix = []
    ylabels = []
    for model in models:
        mr = adv_data[model]
        row = []
        for atk in all_attacks:
            ad = mr.get("attacks", {}).get(atk, {})
            drop = ad.get("delta_mAP", 0)
            row.append(abs(drop))
        matrix.append(row)
        ylabels.append(mr.get("architecture", model))

    if matrix:
        im = ax.imshow(matrix, cmap='Reds', aspect='auto', vmin=0, vmax=0.5)
        ax.set_xticks(range(n_attacks))
        ax.set_xticklabels(all_attacks, rotation=30, ha='right')
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(ylabels)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                ax.text(j, i, f"{matrix[i][j]:.2f}", ha='center', va='center', fontsize=9,
                        color='white' if matrix[i][j] > 0.3 else 'black')
        plt.colorbar(im, label='|mAP Drop|')
        ax.set_title("mAP Drop Heatmap: Model × Attack")
    save(fig, os.path.join(out, "sec5_adversarial", "fig_mAP_drop_heatmap.png"))

    # --- Fig: Recalibration failure (the negative result) ---
    fig, axes = plt.subplots(1, min(n_models, 3), figsize=(6 * min(n_models, 3), 5))
    if n_models == 1:
        axes = [axes]
    for idx, model in enumerate(models[:3]):
        ax = axes[idx]
        mr = adv_data[model]
        attacks = mr.get("attacks", {})
        atk_names = list(attacks.keys())
        naive = [attacks[a].get("naive_conformal", {}).get("coverage", 0) for a in atk_names]
        recal = [attacks[a].get("recalibrated_conformal", {}).get("coverage", 0) for a in atk_names]

        x_pos = np.arange(len(atk_names))
        ax.bar(x_pos - 0.15, naive, 0.3, label='Clean CP on attacked', color=C['red'], alpha=0.8)
        ax.bar(x_pos + 0.15, recal, 0.3, label='Recalibrated CP', color=C['orange'], alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(atk_names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel("Coverage")
        ax.set_title(mr.get("architecture", model))
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1)

        # Annotate "= SAME" where they match
        for j in range(len(naive)):
            if abs(naive[j] - recal[j]) < 0.01:
                ax.text(j, max(naive[j], recal[j]) + 0.03, "SAME",
                        ha='center', fontsize=7, color=C['red'], fontweight='bold')

    fig.suptitle("Recalibration NEVER Recovers Coverage (Proposition 3)", fontsize=13, y=1.03)
    plt.tight_layout()
    save(fig, os.path.join(out, "sec5_adversarial", "fig_recalibration_failure.png"))
    save(fig, os.path.join(out, "hero_figures", "fig_hero_recalibration_failure.png"), close=False)
    plt.close(fig)

    # --- LaTeX Table ---
    rows_latex = []
    for model in models:
        mr = adv_data[model]
        arch = mr.get("architecture", model)
        for atk_name, ad in mr.get("attacks", {}).items():
            clean_mAP = mr.get("clean_mAP", {}).get("mAP@[0.5:0.95]", 0)
            atk_mAP = ad.get("attacked_mAP", {}).get("mAP@[0.5:0.95]", 0)
            drop_pct = ((clean_mAP - atk_mAP) / max(clean_mAP, 0.001)) * 100
            clean_cov = mr.get("clean_conformal", {}).get("coverage", 0)
            naive_cov = ad.get("naive_conformal", {}).get("coverage", 0)
            recal_cov = ad.get("recalibrated_conformal", {}).get("coverage", 0)
            l_inf = ad.get("perturbation", {}).get("mean_l_inf", 0)
            # Mark DETR L_inf with dagger (measured in normalized space)
            l_inf_str = f"{l_inf:.3f}"
            if "DETR" in arch and l_inf > 0.2:
                l_inf_str = f"{l_inf:.3f}$^\\dagger$"
            # Mark recal = naive
            same = "$\\equiv$" if abs(naive_cov - recal_cov) < 0.005 else f"{recal_cov:.3f}"
            rows_latex.append(
                f"    {arch} & {atk_name} & {atk_mAP:.3f} & $-${drop_pct:.0f}\\% & "
                f"{naive_cov:.3f} & {same} & {l_inf_str} \\\\"
            )

    latex = "\\begin{table*}[t]\n\\centering\n"
    latex += "\\caption{Adversarial attack results on COCO val2017 (200 images). "
    latex += "Recalibration never recovers coverage ($\\text{Cov}_{\\text{recal}} \\equiv \\text{Cov}_{\\text{naive}}$ in all cases).}\n"
    latex += "\\label{tab:adversarial}\n\\small\n\\begin{tabular}{llccccc}\n\\toprule\n"
    latex += "    Model & Attack & mAP$_{\\text{adv}}$ & $\\Delta$mAP & "
    latex += "Cov$_{\\text{naive}}$ & Cov$_{\\text{recal}}$ & $L_\\infty$ \\\\\n\\midrule\n"
    latex += "\n".join(rows_latex)
    latex += "\n\\bottomrule\n\\end{tabular}\n"
    latex += "\\begin{tablenotes}\\small\n"
    latex += "\\item[$\\dagger$] DETR $L_\\infty$ measured in ImageNet-normalized space. "
    latex += "Pixel-space perturbation is $\\varepsilon=8/255\\approx0.031$, identical to YOLO models.\n"
    latex += "\\item[$\\equiv$] Identical to naive: recalibration provides zero recovery.\n"
    latex += "\\end{tablenotes}\n"
    latex += "\\end{table*}"
    save_latex(latex, os.path.join(out, "tables_latex", "tab_adversarial.tex"))


# =====================================================================
#  SECTION 6: DEFENSE
# =====================================================================

def gen_defense(results_dir, out):
    """Generate defense evaluation figures."""
    print("\n=== Section 6: Defense ===")
    def_path = os.path.join(results_dir, "defense", "defense_results.json")
    if not os.path.exists(def_path):
        print("  [SKIP] defense_results.json not found")
        return

    with open(def_path, 'r', encoding='utf-8') as f:
        def_data = json.load(f)

    # Copy existing defense figures
    for fname in ["fig_pareto_coverage.png", "fig_roc_attack_detection.png",
                   "fig_abstention_distribution.png", "fig_precision_of_kept.png",
                   "fig_tau_sweep.png"]:
        src = os.path.join(results_dir, "defense", fname)
        if os.path.exists(src):
            import shutil
            dst = os.path.join(out, "sec6_defense", fname)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  [COPIED] {fname}")

    # --- LaTeX Table: Defense summary ---
    rows_latex = []
    for model, attacks in def_data.items():
        if not isinstance(attacks, dict):
            continue
        for atk_name, d in attacks.items():
            if not isinstance(d, dict):
                continue
            auroc = d.get('auroc_attack_detection', 0.5)

            # AUROC comparison — try multiple key formats
            comp = d.get('auroc_comparison', {})
            cp_auc = comp.get('CP Composite (ours)', auroc)
            # Try both old and new key names
            conf_auc = comp.get('Mean Conf (>0.1)', comp.get('Mean Confidence', 0.5))
            max_auc = comp.get('Max Confidence', 0.5)
            count_auc = comp.get('Det Count (>0.5)', comp.get('Detection Count (conf>0.5)', 0.5))

            prec_raw_c = d.get('precision_raw_clean', 0)
            prec_raw_a = d.get('precision_raw_adv', 0)

            rows_latex.append(
                f"    {model} & {atk_name} & {cp_auc:.3f} & {conf_auc:.3f} & "
                f"{max_auc:.3f} & {count_auc:.3f} & {prec_raw_c:.3f} & {prec_raw_a:.3f} \\\\"
            )

    if rows_latex:
        latex = "\\begin{table*}[t]\n\\centering\n"
        latex += "\\caption{Attack detection AUROC: CP composite vs non-CP baselines. "
        latex += "All baselines use the same clean calibration set. Higher is better.}\n"
        latex += "\\label{tab:defense}\n\\small\n\\begin{tabular}{llcccccc}\n\\toprule\n"
        latex += "    Model & Attack & \\textbf{CP (ours)} & Mean Conf & Max Conf & Det Count & Prec$_{\\text{clean}}$ & Prec$_{\\text{adv}}$ \\\\\n\\midrule\n"
        latex += "\n".join(rows_latex)
        latex += "\n\\bottomrule\n\\end{tabular}\n\\end{table*}"
        save_latex(latex, os.path.join(out, "tables_latex", "tab_defense.tex"))


# =====================================================================
#  SECTION 7: NATURAL CORRUPTIONS
# =====================================================================

def gen_corruptions(results_dir, out):
    """Generate natural corruption figures."""
    print("\n=== Section 7: Natural Corruptions ===")
    corr_path = os.path.join(results_dir, "corruptions", "natural_corruption_results.json")
    if not os.path.exists(corr_path):
        print("  [SKIP] natural_corruption_results.json not found")
        return

    with open(corr_path, 'r', encoding='utf-8') as f:
        corr_data = json.load(f)

    # Copy existing figures
    corr_dir = os.path.join(results_dir, "corruptions")
    for fname in os.listdir(corr_dir):
        if fname.endswith('.png'):
            import shutil
            dst = os.path.join(out, "sec7_corruptions", fname)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(corr_dir, fname), dst)
            print(f"  [COPIED] {fname}")

    # --- LaTeX Table ---
    rows_latex = []
    for model, model_data in corr_data.items():
        if not isinstance(model_data, dict):
            continue
        corruptions = model_data.get("corruptions", model_data)
        if not isinstance(corruptions, dict):
            continue
        for corr_name, corr_info in corruptions.items():
            if not isinstance(corr_info, dict):
                continue
            for sev_key, sev_data in corr_info.items():
                if not isinstance(sev_data, dict) or 'mAP' not in str(sev_data):
                    continue
                mAP = sev_data.get('mAP@[0.5:0.95]', sev_data.get('mAP', 0))
                cov = sev_data.get('coverage', 0)
                rows_latex.append(f"    {model} & {corr_name} & {sev_key} & {mAP:.3f} & {cov:.3f} \\\\")

    if rows_latex:
        latex = "\\begin{table}[t]\n\\centering\n\\caption{Performance under natural corruptions.}\n"
        latex += "\\label{tab:corruptions}\n\\begin{tabular}{llccc}\n\\toprule\n"
        latex += "    Model & Corruption & Severity & mAP & Coverage \\\\\n\\midrule\n"
        latex += "\n".join(rows_latex[:30])  # limit rows
        latex += "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
        save_latex(latex, os.path.join(out, "tables_latex", "tab_corruptions.tex"))


# =====================================================================
#  SECTION 8: SEGMENTATION
# =====================================================================

def gen_segmentation(results_dir, out):
    """Generate segmentation vulnerability figures."""
    print("\n=== Section 8: Segmentation ===")
    seg_path = os.path.join(results_dir, "segmentation", "segmentation_attack_results.json")
    if not os.path.exists(seg_path):
        print("  [SKIP] segmentation_attack_results.json not found")
        return

    with open(seg_path, 'r', encoding='utf-8') as f:
        seg_data = json.load(f)

    # Copy figures
    seg_dir = os.path.join(results_dir, "segmentation")
    for fname in os.listdir(seg_dir):
        if fname.endswith('.png'):
            import shutil
            dst = os.path.join(out, "sec8_segmentation", fname)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(seg_dir, fname), dst)
            print(f"  [COPIED] {fname}")

    # --- LaTeX Table ---
    rows_latex = []
    for model, md in seg_data.items():
        cb = md.get("clean_bbox_mAP", {})
        cm = md.get("clean_mask_mAP", {})
        clean_bbox = cb.get("mAP@[0.5:0.95]", cb) if isinstance(cb, dict) else cb
        clean_mask = cm.get("mAP@[0.5:0.95]", cm) if isinstance(cm, dict) else cm
        clean_bbox = float(clean_bbox) if clean_bbox else 0.0
        clean_mask = float(clean_mask) if clean_mask else 0.0

        attacks = md.get("attacks", {})
        for atk_name, ad in attacks.items():
            bbox_drop = float(ad.get("bbox_mAP_drop", 0))
            mask_drop = float(ad.get("mask_mAP_drop", 0))
            bbox_pct = (bbox_drop / max(clean_bbox, 0.001)) * 100
            mask_pct = (mask_drop / max(clean_mask, 0.001)) * 100
            rows_latex.append(
                f"    {model} & {atk_name} & {clean_bbox:.3f} & {bbox_drop:.3f} ({bbox_pct:.0f}\\%) "
                f"& {clean_mask:.3f} & {mask_drop:.3f} ({mask_pct:.0f}\\%) & "
                f"${abs(bbox_pct - mask_pct):.0f}\\%$ \\\\"
            )

    if rows_latex:
        latex = "\\begin{table}[t]\n\\centering\n\\caption{Detection vs segmentation vulnerability under attack. "
        latex += "Bbox and mask degrade nearly identically ($\\pm$3\\%).}\n"
        latex += "\\label{tab:segmentation}\n\\small\n\\begin{tabular}{llccccc}\n\\toprule\n"
        latex += "    Model & Attack & Bbox$_{\\text{clean}}$ & Bbox Drop & Mask$_{\\text{clean}}$ & Mask Drop & $|\\Delta|$ \\\\\n\\midrule\n"
        latex += "\n".join(rows_latex)
        latex += "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
        save_latex(latex, os.path.join(out, "tables_latex", "tab_segmentation.tex"))


# =====================================================================
#  SECTION 11: STATISTICS
# =====================================================================

def gen_statistics(results_dir, out):
    """Copy and organize statistical analysis results."""
    print("\n=== Section 11: Statistics ===")
    stat_dir = os.path.join(results_dir, "statistics")
    if not os.path.isdir(stat_dir):
        print("  [SKIP] statistics/ not found")
        return

    for fname in os.listdir(stat_dir):
        if fname.endswith('.png') or fname.endswith('.json'):
            import shutil
            dst = os.path.join(out, "sec11_statistics", fname)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(stat_dir, fname), dst)
            print(f"  [COPIED] {fname}")


# =====================================================================
#  HERO FIGURES
# =====================================================================

def gen_hero_figures(results_dir, out):
    """Generate the key figures that summarize the entire paper."""
    print("\n=== Hero Figures ===")

    adv_path = os.path.join(results_dir, "adversarial", "full_results.json")
    if not os.path.exists(adv_path):
        return
    with open(adv_path, 'r', encoding='utf-8') as f:
        adv_data = json.load(f)

    # --- HERO 1: The thesis in one figure ---
    # X = mAP drop, Y = coverage drop, shows perfect correlation
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, mr in adv_data.items():
        clean_mAP = mr.get("clean_mAP", {}).get("mAP@[0.5:0.95]", 0)
        clean_cov = mr.get("clean_conformal", {}).get("coverage", 0)
        color = MODEL_COLORS.get(model, C['gray'])
        arch = mr.get("architecture", model)

        for atk_name, ad in mr.get("attacks", {}).items():
            atk_mAP = ad.get("attacked_mAP", {}).get("mAP@[0.5:0.95]", 0)
            naive_cov = ad.get("naive_conformal", {}).get("coverage", 0)
            mAP_drop = clean_mAP - atk_mAP
            cov_drop = clean_cov - naive_cov

            ax.scatter(mAP_drop, cov_drop, s=80, c=color, alpha=0.7,
                       edgecolors='white', lw=0.5)

        # Legend entry
        ax.scatter([], [], s=80, c=color, label=arch)

    # Trend line
    all_x, all_y = [], []
    for model, mr in adv_data.items():
        clean_mAP = mr.get("clean_mAP", {}).get("mAP@[0.5:0.95]", 0)
        clean_cov = mr.get("clean_conformal", {}).get("coverage", 0)
        for ad in mr.get("attacks", {}).values():
            atk_mAP = ad.get("attacked_mAP", {}).get("mAP@[0.5:0.95]", 0)
            naive_cov = ad.get("naive_conformal", {}).get("coverage", 0)
            all_x.append(clean_mAP - atk_mAP)
            all_y.append(clean_cov - naive_cov)

    if len(all_x) > 3:
        z = np.polyfit(all_x, all_y, 1)
        x_fit = np.linspace(0, max(all_x) * 1.1, 100)
        ax.plot(x_fit, np.polyval(z, x_fit), 'k--', lw=1.5, alpha=0.5,
                label=f'Trend (r={np.corrcoef(all_x, all_y)[0,1]:.2f})')

    ax.set_xlabel("mAP Drop (clean - adversarial)")
    ax.set_ylabel("Coverage Drop (clean - adversarial)")
    ax.set_title("CP Coverage Tracks Detector Degradation — Cannot Compensate")
    ax.legend(fontsize=8)
    save(fig, os.path.join(out, "hero_figures", "fig_hero_thesis.png"))

    # --- HERO 2: Architecture comparison bar ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: mAP under strongest attack per model
    models_sorted = sorted(adv_data.keys())
    ax = axes[0]
    for i, model in enumerate(models_sorted):
        mr = adv_data[model]
        clean_mAP = mr.get("clean_mAP", {}).get("mAP@[0.5:0.95]", 0)
        worst_mAP = clean_mAP
        worst_atk = "none"
        for atk_name, ad in mr.get("attacks", {}).items():
            m = ad.get("attacked_mAP", {}).get("mAP@[0.5:0.95]", clean_mAP)
            if m < worst_mAP:
                worst_mAP = m
                worst_atk = atk_name
        color = MODEL_COLORS.get(model, C['gray'])
        ax.bar(i, clean_mAP, 0.35, label='Clean' if i == 0 else '', color=color, alpha=0.4)
        ax.bar(i + 0.35, worst_mAP, 0.35, label='Worst attack' if i == 0 else '', color=color, alpha=0.9)
        ax.text(i + 0.17, max(clean_mAP, worst_mAP) + 0.02,
                f"-{(clean_mAP-worst_mAP)*100:.0f}%", ha='center', fontsize=8, color=C['red'])

    ax.set_xticks([i + 0.17 for i in range(len(models_sorted))])
    ax.set_xticklabels([adv_data[m].get("architecture", m) for m in models_sorted],
                        rotation=20, ha='right', fontsize=9)
    ax.set_ylabel("mAP@[0.5:0.95]")
    ax.set_title("(a) Architecture Robustness")
    ax.legend(fontsize=8)

    # Right: ECE vs worst coverage drop
    ax = axes[1]
    summary_path = os.path.join(results_dir, "evaluation", "metrics_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        for model in models_sorted:
            mr = adv_data[model]
            ed = eval_data.get(model, {})
            ece = ed.get('cal_ECE', 0)
            clean_cov = mr.get("clean_conformal", {}).get("coverage", 0)
            worst_drop = 0
            for ad in mr.get("attacks", {}).values():
                naive_cov = ad.get("naive_conformal", {}).get("coverage", 0)
                drop = clean_cov - naive_cov
                worst_drop = max(worst_drop, drop)
            color = MODEL_COLORS.get(model, C['gray'])
            ax.scatter(ece, worst_drop, s=200, c=color, zorder=5,
                       edgecolors='white', lw=2)
            ax.annotate(mr.get("architecture", model), (ece, worst_drop),
                        fontsize=8, textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel("ECE")
        ax.set_ylabel("Worst Coverage Drop")
        ax.set_title("(b) Calibration vs Vulnerability")

    plt.tight_layout()
    save(fig, os.path.join(out, "hero_figures", "fig_hero_architecture.png"))


# =====================================================================
#  COPY EXISTING PHASE FIGURES
# =====================================================================

def copy_phase_figures(results_dir, out):
    """Copy all existing figures from each phase."""
    print("\n=== Copying existing figures ===")
    import shutil

    mappings = {
        "ablation": "sec4_ablation",
        "theory": "sec10_theory",
        "real_viz": "sec3_calibration/real_images",
        "transfer": "sec9_transfer",
        "cross_dataset": "sec10_cross_dataset",
    }

    for src_dir, dst_dir in mappings.items():
        src = os.path.join(results_dir, src_dir)
        if not os.path.isdir(src):
            continue
        for fname in os.listdir(src):
            if fname.endswith('.png') or fname.endswith('.json'):
                dst = os.path.join(out, dst_dir, fname)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(os.path.join(src, fname), dst)
                print(f"  [COPIED] {src_dir}/{fname}")


# =====================================================================
#  ABLATION LATEX TABLE
# =====================================================================

def gen_ablation_table(results_dir, out):
    """Generate LaTeX table for ablation."""
    print("\n=== Ablation Table ===")
    abl_path = os.path.join(results_dir, "ablation", "all_results.json")
    if not os.path.exists(abl_path):
        print("  [SKIP] all_results.json not found")
        return

    with open(abl_path, 'r', encoding='utf-8') as f:
        abl_data = json.load(f)

    for model, md in abl_data.items():
        rows_abl = []
        for config_name, r in md.get("ablation", {}).items():
            cov = r.get("coverage", 0)
            margin = r.get("mean_margin_px", 0)
            ece = r.get("ece", 0)
            filt = r.get("filter_rate", 0)
            rows_abl.append(f"    {config_name} & {cov:.3f} & {margin:.1f} & {ece:.4f} & {filt*100:.1f}\\% \\\\")

        rows_base = []
        for method_name, r in md.get("baselines", {}).items():
            cov = r.get("coverage", 0)
            margin = r.get("mean_margin_px", 0)
            ece = r.get("ece", 0)
            rows_base.append(f"    {method_name} & {cov:.3f} & {margin:.1f} & {ece:.4f} \\\\")

        if rows_abl:
            latex = f"% Ablation for {model}\n"
            latex += "\\begin{table}[t]\n\\centering\n"
            latex += f"\\caption{{Ablation study on {model}.}}\n"
            latex += "\\begin{tabular}{lcccc}\n\\toprule\n"
            latex += "    Config & Coverage & Margin (px) & ECE & Filter\\% \\\\\n\\midrule\n"
            latex += "\n".join(rows_abl)
            latex += "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
            save_latex(latex, os.path.join(out, "tables_latex", f"tab_ablation_{model}.tex"))


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="paper_assets")
    args = parser.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    print("=" * 70)
    print("  GENERATING ALL PAPER ASSETS")
    print("=" * 70)

    gen_calibration(args.results_dir, out)
    gen_adversarial(args.results_dir, out)
    gen_defense(args.results_dir, out)
    gen_corruptions(args.results_dir, out)
    gen_segmentation(args.results_dir, out)
    gen_statistics(args.results_dir, out)
    gen_hero_figures(args.results_dir, out)
    copy_phase_figures(args.results_dir, out)
    gen_ablation_table(args.results_dir, out)

    print(f"\n{'='*70}")
    print(f"  ALL PAPER ASSETS GENERATED -> {out}/")
    print(f"{'='*70}")
    print(f"  {out}/")
    print(f"    ├── sec3_calibration/    Reliability diagrams, mAP vs ECE")
    print(f"    ├── sec4_ablation/       Ablation bar charts, visual grid")
    print(f"    ├── sec5_adversarial/    Coverage drop, mAP heatmap, recal failure")
    print(f"    ├── sec6_defense/        AUROC, Pareto, abstention, tau sweep")
    print(f"    ├── sec7_corruptions/    Fog/blur/noise/jpeg degradation")
    print(f"    ├── sec8_segmentation/   Bbox vs mask vulnerability")
    print(f"    ├── sec9_transfer/       Black-box transferability")
    print(f"    ├── sec10_theory/        Proposition verification plots")
    print(f"    ├── sec11_statistics/    Bootstrap CI, effect sizes, scatter")
    print(f"    ├── tables_latex/        ALL tables ready for LaTeX")
    print(f"    └── hero_figures/        Key figures for the paper")


if __name__ == "__main__":
    main()
