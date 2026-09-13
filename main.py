#!/usr/bin/env python3
"""
main.py — Unified CLI entry point for the Conformal Calibration Pipeline.

Usage:
    python main.py setup
    python main.py download
    python main.py eda
    python main.py eval --models yolov8x --max-images 100
    python main.py eval-all
    python main.py adversarial --models yolov8x --max-images 200
    python main.py defense
    python main.py statistics
    python main.py paper-assets
    python main.py full
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
#  PRE-FLIGHT: Directory creation + data checks
# =====================================================================

ALL_RESULT_DIRS = [
    "results/evaluation/predictions",
    "results/evaluation/plots",
    "results/ablation",
    "results/real_viz",
    "results/theory",
    "results/adversarial",
    "results/defense",
    "results/corruptions",
    "results/segmentation",
    "results/transfer",
    "results/cross_dataset",
    "results/statistics",
    "results/defense_baselines",
    "paper_assets/sec3_calibration",
    "paper_assets/sec4_ablation",
    "paper_assets/sec5_adversarial",
    "paper_assets/sec6_defense",
    "paper_assets/sec7_corruptions",
    "paper_assets/sec8_segmentation",
    "paper_assets/sec9_transfer",
    "paper_assets/sec10_theory",
    "paper_assets/sec11_statistics",
    "paper_assets/tables_latex",
    "paper_assets/hero_figures",
]


def ensure_directories():
    """Create ALL output directories upfront."""
    for d in ALL_RESULT_DIRS:
        os.makedirs(d, exist_ok=True)


def check_coco_ready(coco_root="data/coco"):
    """Check if COCO val2017 is downloaded and ready."""
    ann = os.path.join(coco_root, "annotations", "instances_val2017.json")
    img_dir = os.path.join(coco_root, "val2017")
    if not os.path.exists(ann):
        return False, f"Missing annotations: {ann}"
    if not os.path.isdir(img_dir):
        return False, f"Missing images dir: {img_dir}"
    n_images = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    if n_images < 100:
        return False, f"Only {n_images} images in {img_dir} (expected ~5000)"
    return True, f"COCO ready: {n_images} images"


def check_predictions_ready(pred_dir="results/evaluation/predictions"):
    """Check if model predictions exist."""
    if not os.path.isdir(pred_dir):
        return False, "No predictions directory"
    jsons = [f for f in os.listdir(pred_dir) if f.endswith('.json')]
    if not jsons:
        return False, "No prediction files"
    return True, f"{len(jsons)} prediction files found"


def check_adversarial_ready(adv_path="results/adversarial/full_results.json"):
    """Check if adversarial results exist."""
    if not os.path.exists(adv_path):
        return False, "No adversarial results"
    return True, "Adversarial results found"


def preflight(args, needs_coco=False, needs_preds=False, needs_adv=False):
    """Run pre-flight checks before any command."""
    ensure_directories()

    coco_root = getattr(args, 'coco_root', 'data/coco')

    if needs_coco:
        ok, msg = check_coco_ready(coco_root)
        if not ok:
            print(f"  [ERROR] {msg}")
            print(f"  Run: python main.py download")
            return False

    if needs_preds:
        pred_dir = getattr(args, 'pred_dir', 'results/evaluation/predictions')
        ok, msg = check_predictions_ready(pred_dir)
        if not ok:
            print(f"  [WARN] {msg}")
            print(f"  Run: python main.py eval-all")

    if needs_adv:
        adv_path = getattr(args, 'adv_results', 'results/adversarial/full_results.json')
        ok, msg = check_adversarial_ready(adv_path)
        if not ok:
            print(f"  [WARN] {msg}")
            print(f"  Run: python main.py adversarial")

    return True


def cmd_setup(args):
    """Install all dependencies."""
    print("=" * 65)
    print("  Step 0: Installing dependencies")
    print("=" * 65)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
    ])
    print("\n  [OK] Core dependencies installed.")
    print()
    print("  Optional installs (run manually if needed):")
    print("    pip install git+https://github.com/facebookresearch/detectron2.git")
    print("    pip install rfdetr")
    print()
    # Try optional packages but don't fail
    for pkg, name in [("rfdetr", "RF-DETR")]:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  [OK] {name} installed")
        except Exception:
            print(f"  [SKIP] {name} — install manually: pip install {pkg}")
    print("\n  Setup complete.\n")


def cmd_download(args):
    """Download COCO val2017 + create conformal split."""
    print("=" * 65)
    print("  Step 1: Downloading datasets")
    print("=" * 65)
    ensure_directories()

    # Check if already downloaded
    ok, msg = check_coco_ready(getattr(args, 'coco_root', 'data/coco'))
    if ok:
        print(f"  [SKIP] {msg} — already downloaded")
        return

    from download_datasets import main as dl_main
    sys.argv = ["download_datasets.py", "--dataset", args.dataset,
                "--split", "val", "--cal-ratio", str(args.cal_ratio),
                "--seed", str(args.seed)]
    dl_main()


def cmd_eda(args):
    """Run full EDA (13 plots → results/eda/)."""
    print("=" * 65)
    print("  Step 1b: Exploratory Data Analysis (13 plots)")
    print("=" * 65)
    from eda_full import main as eda_main
    eda_main()


def cmd_viz_conformal(args):
    """Generate conformal calibration impact figures (5 plots)."""
    print("=" * 65)
    print("  Conformal Calibration Visualizations (5 plots)")
    print("=" * 65)
    from visualize_conformal import main as viz_main
    viz_main()


def cmd_viz_setting2(args):
    """REMOVED — was not used in the paper."""
    print("  [REMOVED] viz-setting2 has been removed from the pipeline.")


def cmd_viz_all(args):
    """Generate all visualizations."""
    cmd_eda(args)
    cmd_viz_conformal(args)
    print("\n  Visualizations complete.")
    print("    results/eda/        (18 plots)")


def cmd_defense(args):
    """Run defense evaluation: Pareto curves, AUROC, selective abstention."""
    print("=" * 65)
    print("  Step 7: Defense Evaluation (no GPU needed)")
    print("=" * 65)
    from run_defense_eval import main as defense_main
    argv = ["run_defense_eval.py",
            "--coco-root", args.coco_root,
            "--pred-dir", args.pred_dir,
            "--adv-results", args.adv_results,
            "--output-dir", args.output_dir,
            "--alpha", str(args.alpha)]
    sys.argv = argv
    defense_main()


def cmd_transfer(args):
    """Run transfer attack evaluation (black-box)."""
    print("=" * 65)
    print("  Step 8: Transfer Attacks")
    print("=" * 65)
    from run_transfer_attacks import main as transfer_main
    argv = ["run_transfer_attacks.py",
            "--coco-root", args.coco_root,
            "--pred-dir", args.pred_dir,
            "--adv-dir", args.adv_dir,
            "--output-dir", args.output_dir,
            "--max-images", str(args.max_images),
            "--device", args.device]
    sys.argv = argv
    transfer_main()


def cmd_seg_attack(args):
    """Run adversarial attacks on segmentation models."""
    print("=" * 65)
    print("  Step 9: Segmentation Under Attack")
    print("=" * 65)
    from run_segmentation_attack import main as seg_main
    argv = ["run_segmentation_attack.py",
            "--coco-root", args.coco_root,
            "--output-dir", args.output_dir,
            "--max-images", str(args.max_images),
            "--device", args.device]
    if args.models:
        argv += ["--models"] + args.models
    sys.argv = argv
    seg_main()


def cmd_eval(args):
    """Evaluate models on COCO val2017."""
    print("=" * 65)
    print("  Step 2: Model Evaluation")
    print("=" * 65)
    if not preflight(args, needs_coco=True):
        return
    from evaluate_models import main as eval_main
    argv = ["evaluate_models.py",
            "--coco-root", args.coco_root,
            "--output-dir", args.output_dir,
            "--device", args.device]
    if args.models:
        argv += ["--models"] + args.models
    if args.max_images:
        argv += ["--max-images", str(args.max_images)]
    if args.task:
        argv += ["--task", args.task]
    sys.argv = argv
    eval_main()


def cmd_eval_all(args):
    """Evaluate ALL models (full run)."""
    args.models = None
    args.max_images = None
    args.task = "all"
    cmd_eval(args)


def cmd_eval_quick(args):
    """Quick evaluation on 100 images."""
    args.models = None
    args.max_images = 100
    args.task = "all"
    args.output_dir = "results/evaluation_quick"
    cmd_eval(args)


def cmd_conformal(args):
    """Run conformal calibration + adversarial evaluation pipeline."""
    print("=" * 65)
    print("  Step 3: Conformal Calibration + Adversarial Evaluation")
    print("=" * 65)
    from run_conformal_pipeline import main as conf_main
    argv = ["run_conformal_pipeline.py",
            "--coco-root", args.coco_root,
            "--pred-dir", args.pred_dir,
            "--output-dir", args.output_dir,
            "--alpha", str(args.alpha)]
    if args.models:
        argv += ["--models"] + args.models
    if hasattr(args, 'max_vis') and args.max_vis:
        argv += ["--max-vis", str(args.max_vis)]
    sys.argv = argv
    conf_main()


def cmd_adversarial(args):
    """Run real gradient-based adversarial attacks and evaluate impact."""
    print("=" * 65)
    print("  Step 4: Real Adversarial Attack Evaluation")
    print("=" * 65)
    from run_adversarial_eval import main as adv_main
    argv = ["run_adversarial_eval.py",
            "--coco-root", args.coco_root,
            "--output-dir", args.output_dir,
            "--device", args.device,
            "--max-images", str(args.max_images)]
    if args.models:
        argv += ["--models"] + args.models
    if args.attacks:
        argv += ["--attacks"] + args.attacks
    sys.argv = argv
    adv_main()


def cmd_real_viz(args):
    """Generate publication figures on real COCO images."""
    print("=" * 65)
    print("  Real COCO Image Visualizations")
    print("=" * 65)
    from visualize_real_images import main as viz_main
    argv = ["visualize_real_images.py",
            "--coco-root", args.coco_root,
            "--pred-dir", args.pred_dir,
            "--output-dir", args.output_dir,
            "--n-images", str(args.n_images),
            "--alpha", str(args.alpha)]
    if args.models:
        argv += ["--models"] + args.models
    sys.argv = argv
    viz_main()


def cmd_ablation(args):
    """Run ablation study + baseline comparison + cross-dataset transfer."""
    print("=" * 65)
    print("  Ablation + Baselines + Cross-Dataset")
    print("=" * 65)
    from run_ablation_and_baselines import main as abl_main
    argv = ["run_ablation_and_baselines.py",
            "--coco-root", args.coco_root,
            "--pred-dir", args.pred_dir,
            "--output-dir", args.output_dir,
            "--alpha", str(args.alpha)]
    if args.models:
        argv += ["--models"] + args.models
    sys.argv = argv
    abl_main()


def cmd_box_cal(args):
    """REMOVED — negative results on real data."""
    print("  [REMOVED] box-cal has been removed from the pipeline.")


def cmd_theory(args):
    """Verify theoretical propositions on real data."""
    print("=" * 65)
    print("  Theoretical Framework — Proposition Verification")
    print("=" * 65)
    from theoretical_framework import main as th_main
    argv = ["theoretical_framework.py",
            "--coco-root", args.coco_root,
            "--pred-dir", args.pred_dir,
            "--output-dir", args.output_dir,
            "--alpha", str(args.alpha)]
    sys.argv = argv
    th_main()


def cmd_full(args):
    """Run entire pipeline: setup -> download -> eval -> all experiments."""
    cmd_setup(args)
    cmd_download(args)
    cmd_eval_all(args)
    cmd_ablation(args)
    cmd_real_viz(args)
    cmd_theory(args)
    cmd_adversarial(args)
    cmd_defense(args)
    cmd_corruptions(args)
    cmd_defense_baselines(args)
    cmd_seg_attack(args)
    cmd_transfer(args)
    cmd_cross_voc(args)
    print("\n" + "=" * 65)
    print("  FULL PIPELINE COMPLETE")
    print("=" * 65)


def cmd_corruptions(args):
    """Run natural corruptions experiment (fog, blur, noise, JPEG)."""
    print("=" * 65)
    print("  Natural Corruptions Experiment")
    print("=" * 65)
    from run_natural_corruptions import main as corr_main
    argv = ["run_natural_corruptions.py",
            "--coco-root", args.coco_root,
            "--output-dir", getattr(args, 'output_dir', 'results/corruptions'),
            "--max-images", str(getattr(args, 'max_images', 200)),
            "--device", getattr(args, 'device', 'cuda')]
    if hasattr(args, 'models') and args.models:
        argv += ["--models"] + args.models
    sys.argv = argv
    corr_main()


def cmd_cross_voc(args):
    """Real cross-dataset: calibrate on COCO, test on VOC2012."""
    print("=" * 65)
    print("  Cross-Dataset: COCO -> VOC2012")
    print("=" * 65)
    from run_cross_dataset_voc import main as voc_main
    argv = ["run_cross_dataset_voc.py",
            "--coco-root", args.coco_root,
            "--pred-dir", getattr(args, 'pred_dir', 'results/evaluation/predictions'),
            "--output-dir", getattr(args, 'output_dir', 'results/cross_dataset'),
            "--device", getattr(args, 'device', 'cuda')]
    if hasattr(args, 'models') and args.models:
        argv += ["--models"] + args.models
    sys.argv = argv
    voc_main()


def cmd_statistics(args):
    """Statistical analysis: bootstrap CI, effect sizes, AUROC comparison."""
    print("=" * 65)
    print("  Statistical Analysis for Q1 Submission")
    print("=" * 65)
    from run_statistical_analysis import main as stat_main
    argv = ["run_statistical_analysis.py",
            "--coco-root", args.coco_root,
            "--pred-dir", getattr(args, 'pred_dir', 'results/evaluation/predictions'),
            "--adv-results", getattr(args, 'adv_results', 'results/adversarial/full_results.json'),
            "--output-dir", getattr(args, 'output_dir', 'results/statistics'),
            "--alpha", str(getattr(args, 'alpha', 0.1))]
    sys.argv = argv
    stat_main()


def cmd_paper_assets(args):
    """Generate ALL publication-ready figures, tables, and samples."""
    print("=" * 65)
    print("  Generating Paper Assets")
    print("=" * 65)
    from generate_paper_assets import main as pa_main
    argv = ["generate_paper_assets.py",
            "--results-dir", getattr(args, 'results_dir', 'results'),
            "--output-dir", getattr(args, 'output_dir', 'paper_assets')]
    sys.argv = argv
    pa_main()


def cmd_reviewer_fixes(args):
    """Address all major reviewer concerns (W1-W4, Q1-Q3, M1)."""
    print("=" * 65)
    print("  Addressing Reviewer Concerns")
    print("=" * 65)
    from reviewer_fixes import main as rev_main
    argv = ["reviewer_fixes.py",
            "--coco-root", getattr(args, 'coco_root', 'data/coco'),
            "--pred-dir", getattr(args, 'pred_dir', 'results/evaluation/predictions'),
            "--adv-results", getattr(args, 'adv_results', 'results/adversarial/full_results.json'),
            "--output-dir", getattr(args, 'output_dir', 'results/reviewer_fixes'),
            "--alpha", str(getattr(args, 'alpha', 0.1))]
    sys.argv = argv
    rev_main()


def cmd_reviewer_fixes_v2(args):
    """Address remaining concerns (W6/M1, T1, T3, M2)."""
    print("=" * 65)
    print("  Addressing Reviewer Concerns V2")
    print("=" * 65)
    from reviewer_fixes_v2 import main as rev2_main
    argv = ["reviewer_fixes_v2.py",
            "--coco-root", getattr(args, 'coco_root', 'data/coco'),
            "--pred-dir", getattr(args, 'pred_dir', 'results/evaluation/predictions'),
            "--adv-results", getattr(args, 'adv_results', 'results/adversarial/full_results.json'),
            "--output-dir", getattr(args, 'output_dir', 'results/reviewer_fixes'),
            "--alpha", str(getattr(args, 'alpha', 0.1)),
            "--fix", getattr(args, 'fix', 'all')]
    sys.argv = argv
    rev2_main()


def cmd_revision(args):
    """Run revision experiments (CP ablation, DeLong, reliability)."""
    print("=" * 65)
    print("  Running Revision Experiments")
    print("=" * 65)
    from revision_experiments import main as rev_main
    argv = ["revision_experiments.py",
            "--coco-root", getattr(args, 'coco_root', 'data/coco'),
            "--pred-dir", getattr(args, 'pred_dir', 'results/evaluation/predictions'),
            "--adv-results", getattr(args, 'adv_results', 'results/adversarial/full_results.json'),
            "--output-dir", getattr(args, 'output_dir', 'results/revision'),
            "--alpha", str(getattr(args, 'alpha', 0.1)),
            "--fix", getattr(args, 'fix', 'all')]
    sys.argv = argv
    rev_main()


def cmd_defense_baselines(args):
    """Compare CP defense vs JPEG compression, feature squeezing, naive baselines."""
    print("=" * 65)
    print("  Defense Baselines Comparison")
    print("=" * 65)
    from run_defense_baselines import main as bl_main
    argv = ["run_defense_baselines.py",
            "--coco-root", args.coco_root,
            "--output-dir", getattr(args, 'output_dir', 'results/defense_baselines'),
            "--max-images", str(getattr(args, 'max_images', 100)),
            "--device", getattr(args, 'device', 'cuda')]
    if hasattr(args, 'models') and args.models:
        argv += ["--models"] + args.models
    sys.argv = argv
    bl_main()


def main():
    parser = argparse.ArgumentParser(
        description="Conformal Calibration Pipeline — Main CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py setup
  python main.py download
  python main.py eval --models yolov8x --max-images 100
  python main.py adversarial --models yolov8x --max-images 200
  python main.py defense
  python main.py transfer
  python main.py seg-attack
  python main.py full

Available models:
  Detection only:    yolov8x, yolov11x, rtdetr-l, detr-resnet101
  Detection + Seg:   yolov8x-seg, yolov11x-seg, mask2former-swin-l
"""
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # setup
    subparsers.add_parser("setup", help="Install all dependencies")

    # download
    p_dl = subparsers.add_parser("download", help="Download datasets")
    p_dl.add_argument("--dataset", choices=["coco", "voc", "all"], default="coco")
    p_dl.add_argument("--cal-ratio", type=float, default=0.5)
    p_dl.add_argument("--seed", type=int, default=42)

    # eda
    subparsers.add_parser("eda", help="Run full EDA (13 plots)")

    # viz
    subparsers.add_parser("viz-conformal", help="Conformal impact figures (5 plots)")
    subparsers.add_parser("viz-setting2", help="[REMOVED]")
    subparsers.add_parser("viz-all", help="All visualizations")

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate specific models")
    p_eval.add_argument("--models", nargs="+", default=None)
    p_eval.add_argument("--max-images", type=int, default=None)
    p_eval.add_argument("--task", choices=["detection", "segmentation", "all"], default="all")
    p_eval.add_argument("--coco-root", default="data/coco")
    p_eval.add_argument("--output-dir", default="results/evaluation")
    p_eval.add_argument("--device", default="cuda")

    # eval-all
    p_ea = subparsers.add_parser("eval-all", help="Evaluate ALL models (full)")
    p_ea.add_argument("--coco-root", default="data/coco")
    p_ea.add_argument("--output-dir", default="results/evaluation")
    p_ea.add_argument("--device", default="cuda")

    # eval-quick
    p_eq = subparsers.add_parser("eval-quick", help="Quick eval on 100 images")
    p_eq.add_argument("--coco-root", default="data/coco")
    p_eq.add_argument("--output-dir", default="results/evaluation_quick")
    p_eq.add_argument("--device", default="cuda")

    # conformal
    p_cf = subparsers.add_parser("conformal", help="Conformal calibration + adversarial eval")
    p_cf.add_argument("--coco-root", default="data/coco")
    p_cf.add_argument("--pred-dir", default="results/evaluation/predictions")
    p_cf.add_argument("--output-dir", default="results/conformal")
    p_cf.add_argument("--alpha", type=float, default=0.1)
    p_cf.add_argument("--models", nargs="+", default=None)
    p_cf.add_argument("--max-vis", type=int, default=6)

    # adversarial
    p_adv = subparsers.add_parser("adversarial", help="Real gradient-based adversarial attacks")
    p_adv.add_argument("--coco-root", default="data/coco")
    p_adv.add_argument("--output-dir", default="results/adversarial")
    p_adv.add_argument("--models", nargs="+", default=["yolov8x"])
    p_adv.add_argument("--attacks", nargs="+",
                       default=["fgsm", "pgd-20", "pgd-50", "dag", "tog-v"])
    p_adv.add_argument("--max-images", type=int, default=200)
    p_adv.add_argument("--device", default="cuda")

    # real-viz
    p_rv = subparsers.add_parser("real-viz", help="Visualize on real COCO images")
    p_rv.add_argument("--coco-root", default="data/coco")
    p_rv.add_argument("--pred-dir", default="results/evaluation/predictions")
    p_rv.add_argument("--output-dir", default="results/real_viz")
    p_rv.add_argument("--models", nargs="+", default=None)
    p_rv.add_argument("--n-images", type=int, default=6)
    p_rv.add_argument("--alpha", type=float, default=0.1)

    # ablation
    p_abl = subparsers.add_parser("ablation", help="Ablation + baselines + cross-dataset")
    p_abl.add_argument("--coco-root", default="data/coco")
    p_abl.add_argument("--pred-dir", default="results/evaluation/predictions")
    p_abl.add_argument("--output-dir", default="results/ablation")
    p_abl.add_argument("--models", nargs="+", default=None)
    p_abl.add_argument("--alpha", type=float, default=0.1)

    # theory
    p_th = subparsers.add_parser("theory", help="Verify theoretical propositions")
    p_th.add_argument("--coco-root", default="data/coco")
    p_th.add_argument("--pred-dir", default="results/evaluation/predictions")
    p_th.add_argument("--output-dir", default="results/theory")
    p_th.add_argument("--alpha", type=float, default=0.1)

    # defense (NEW)
    p_def = subparsers.add_parser("defense", help="Defense evaluation: Pareto, AUROC, selective abstention")
    p_def.add_argument("--coco-root", default="data/coco")
    p_def.add_argument("--pred-dir", default="results/evaluation/predictions")
    p_def.add_argument("--adv-results", default="results/adversarial/full_results.json")
    p_def.add_argument("--output-dir", default="results/defense")
    p_def.add_argument("--alpha", type=float, default=0.1)

    # transfer (NEW)
    p_tr = subparsers.add_parser("transfer", help="Transfer attack evaluation (black-box)")
    p_tr.add_argument("--coco-root", default="data/coco")
    p_tr.add_argument("--pred-dir", default="results/evaluation/predictions")
    p_tr.add_argument("--adv-dir", default="results/adversarial")
    p_tr.add_argument("--output-dir", default="results/transfer")
    p_tr.add_argument("--max-images", type=int, default=200)
    p_tr.add_argument("--device", default="cuda")

    # seg-attack
    p_sa = subparsers.add_parser("seg-attack", help="Adversarial attacks on segmentation models")
    p_sa.add_argument("--coco-root", default="data/coco")
    p_sa.add_argument("--output-dir", default="results/segmentation")
    p_sa.add_argument("--models", nargs="+", default=None)
    p_sa.add_argument("--max-images", type=int, default=100)
    p_sa.add_argument("--device", default="cuda")

    # corruptions (GAP 1)
    p_corr = subparsers.add_parser("corruptions", help="Natural corruptions: fog, blur, noise, JPEG")
    p_corr.add_argument("--coco-root", default="data/coco")
    p_corr.add_argument("--output-dir", default="results/corruptions")
    p_corr.add_argument("--models", nargs="+", default=None)
    p_corr.add_argument("--max-images", type=int, default=200)
    p_corr.add_argument("--device", default="cuda")

    # defense-baselines (GAP 2 + GAP 4)
    p_db = subparsers.add_parser("defense-baselines", help="CP vs JPEG/FeatureSqueeze/naive AUROC")
    p_db.add_argument("--coco-root", default="data/coco")
    p_db.add_argument("--output-dir", default="results/defense_baselines")
    p_db.add_argument("--models", nargs="+", default=None)
    p_db.add_argument("--max-images", type=int, default=100)
    p_db.add_argument("--device", default="cuda")

    # cross-voc (GAP 5)
    p_cv = subparsers.add_parser("cross-voc", help="Real cross-dataset: COCO -> VOC2012")
    p_cv.add_argument("--coco-root", default="data/coco")
    p_cv.add_argument("--pred-dir", default="results/evaluation/predictions")
    p_cv.add_argument("--output-dir", default="results/cross_dataset")
    p_cv.add_argument("--models", nargs="+", default=None)
    p_cv.add_argument("--device", default="cuda")

    # statistics (Q1 rigor)
    p_st = subparsers.add_parser("statistics", help="Bootstrap CI, effect sizes, AUROC comparison")
    p_st.add_argument("--coco-root", default="data/coco")
    p_st.add_argument("--pred-dir", default="results/evaluation/predictions")
    p_st.add_argument("--adv-results", default="results/adversarial/full_results.json")
    p_st.add_argument("--output-dir", default="results/statistics")
    p_st.add_argument("--alpha", type=float, default=0.1)

    # paper-assets
    p_pa = subparsers.add_parser("paper-assets", help="Generate ALL figures, tables, samples for paper")
    p_pa.add_argument("--results-dir", default="results")
    p_pa.add_argument("--output-dir", default="paper_assets")

    # reviewer-fixes
    p_rf = subparsers.add_parser("reviewer-fixes", help="Address reviewer concerns W1-W4, Q1-Q3, M1")
    p_rf.add_argument("--coco-root", default="data/coco")
    p_rf.add_argument("--pred-dir", default="results/evaluation/predictions")
    p_rf.add_argument("--adv-results", default="results/adversarial/full_results.json")
    p_rf.add_argument("--output-dir", default="results/reviewer_fixes")
    p_rf.add_argument("--alpha", type=float, default=0.1)

    # reviewer-fixes-v2
    p_rf2 = subparsers.add_parser("reviewer-fixes-v2", help="Address W6/M1/T1/T3/M2")
    p_rf2.add_argument("--coco-root", default="data/coco")
    p_rf2.add_argument("--pred-dir", default="results/evaluation/predictions")
    p_rf2.add_argument("--adv-results", default="results/adversarial/full_results.json")
    p_rf2.add_argument("--output-dir", default="results/reviewer_fixes")
    p_rf2.add_argument("--alpha", type=float, default=0.1)
    p_rf2.add_argument("--fix", default="all", choices=["all","squeezing","coverage","linf","adaptive"])

    # revision
    p_rev = subparsers.add_parser("revision", help="Revision experiments (CP ablation, DeLong, reliability)")
    p_rev.add_argument("--coco-root", default="data/coco")
    p_rev.add_argument("--pred-dir", default="results/evaluation/predictions")
    p_rev.add_argument("--adv-results", default="results/adversarial/full_results.json")
    p_rev.add_argument("--output-dir", default="results/revision")
    p_rev.add_argument("--alpha", type=float, default=0.1)
    p_rev.add_argument("--fix", default="all", choices=["all","ablation","delong","reliability","coverage_gap","runtime","class_coverage","multi_iou","pvalues"])

    # full
    p_full = subparsers.add_parser("full", help="Run entire pipeline")
    p_full.add_argument("--dataset", default="coco")
    p_full.add_argument("--cal-ratio", type=float, default=0.5)
    p_full.add_argument("--seed", type=int, default=42)
    p_full.add_argument("--coco-root", default="data/coco")
    p_full.add_argument("--output-dir", default="results/evaluation")
    p_full.add_argument("--pred-dir", default="results/evaluation/predictions")
    p_full.add_argument("--adv-results", default="results/adversarial/full_results.json")
    p_full.add_argument("--device", default="cuda")
    p_full.add_argument("--alpha", type=float, default=0.1)
    p_full.add_argument("--models", nargs="+", default=None)
    p_full.add_argument("--attacks", nargs="+", default=None)
    p_full.add_argument("--max-images", type=int, default=200)
    p_full.add_argument("--n-images", type=int, default=6)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "setup": cmd_setup,
        "download": cmd_download,
        "eda": cmd_eda,
        "viz-conformal": cmd_viz_conformal,
        "viz-setting2": cmd_viz_setting2,
        "viz-all": cmd_viz_all,
        "eval": cmd_eval,
        "eval-all": cmd_eval_all,
        "eval-quick": cmd_eval_quick,
        "conformal": cmd_conformal,
        "adversarial": cmd_adversarial,
        "real-viz": cmd_real_viz,
        "ablation": cmd_ablation,
        "theory": cmd_theory,
        "defense": cmd_defense,
        "transfer": cmd_transfer,
        "seg-attack": cmd_seg_attack,
        "corruptions": cmd_corruptions,
        "defense-baselines": cmd_defense_baselines,
        "cross-voc": cmd_cross_voc,
        "statistics": cmd_statistics,
        "paper-assets": cmd_paper_assets,
        "reviewer-fixes": cmd_reviewer_fixes,
        "reviewer-fixes-v2": cmd_reviewer_fixes_v2,
        "revision": cmd_revision,
        "full": cmd_full,
    }

    # Pre-flight: create all directories
    ensure_directories()

    # Pre-flight: check data for commands that need it
    needs_coco = {"eval", "eval-all", "eval-quick", "conformal", "adversarial",
                  "real-viz", "ablation", "theory", "corruptions", "seg-attack",
                  "transfer", "cross-voc", "defense", "statistics", "full",
                  "defense-baselines"}
    needs_preds = {"ablation", "theory", "defense", "transfer", "statistics",
                   "paper-assets"}
    needs_adv = {"defense", "statistics"}

    if args.command in needs_coco:
        coco_root = getattr(args, 'coco_root', 'data/coco')
        ok, msg = check_coco_ready(coco_root)
        if not ok and args.command != "full":
            print(f"\n  [ERROR] COCO data not ready: {msg}")
            print(f"  Run first: python main.py download\n")
            return

    if args.command in needs_preds:
        pred_dir = getattr(args, 'pred_dir', 'results/evaluation/predictions')
        ok, msg = check_predictions_ready(pred_dir)
        if not ok:
            print(f"\n  [WARN] {msg}")
            print(f"  Run first: python main.py eval-all\n")

    if args.command in needs_adv:
        adv_path = getattr(args, 'adv_results', 'results/adversarial/full_results.json')
        ok, msg = check_adversarial_ready(adv_path)
        if not ok:
            print(f"\n  [WARN] {msg}")
            print(f"  Run first: python main.py adversarial\n")

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
