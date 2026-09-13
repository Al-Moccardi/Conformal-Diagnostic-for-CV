#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Conformal Calibration Pipeline — Setup & Run Commands
# ═══════════════════════════════════════════════════════════════
#
#  Tested on: Ubuntu 22.04 / CUDA 12.x / Python 3.10+
#  GPU: NVIDIA A100 / V100 / RTX 3090+ recommended (≥16GB VRAM)
#
# ═══════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════════════════"
echo "  CONFORMAL CALIBRATION PIPELINE"
echo "═══════════════════════════════════════════════════════════"

# ── STEP 0: Environment Setup ────────────────────────────────

setup_env() {
    echo ""
    echo "▶ Step 0: Setting up Python environment..."

    # Create conda env (optional, recommended)
    # conda create -n confcal python=3.10 -y
    # conda activate confcal

    # Core dependencies
    pip install -r requirements.txt

    # PyTorch with CUDA (adjust for your CUDA version)
    # pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    # Detectron2 (Faster R-CNN, Mask R-CNN)
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

    # RF-DETR
    pip install rfdetr

    # SAM2 (optional)
    # pip install 'git+https://github.com/facebookresearch/sam2.git'

    echo "  [OK] Environment ready"
}

# ── STEP 1: Download Data ────────────────────────────────────

download_data() {
    echo ""
    echo "▶ Step 1: Downloading COCO val2017 + annotations..."

    python download_datasets.py --dataset coco --split val --cal-ratio 0.5 --seed 42

    echo ""
    echo "  [OK] Data downloaded and conformal split created"
    echo "  Files:"
    echo "    data/coco/val2017/              (5K images)"
    echo "    data/coco/annotations/          (instances_val2017.json)"
    echo "    data/coco/annotations/          (instances_val2017_calibration.json)"
    echo "    data/coco/annotations/          (instances_val2017_test.json)"
}

# ── STEP 1b: EDA ─────────────────────────────────────────────

run_eda() {
    echo ""
    echo "▶ Step 1b: Running full EDA (13 plots)..."

    python eda_full.py

    echo "  [OK] EDA plots → results/eda/"
}

# ── STEP 2: Model Evaluation ─────────────────────────────────

evaluate_all() {
    echo ""
    echo "▶ Step 2: Evaluating ALL models on COCO val2017..."
    echo "  This will download model weights automatically on first run."
    echo "  Estimated time: ~2-4 hours on A100 (all 10 models × 5K images)"
    echo ""

    python evaluate_models.py \
        --coco-root data/coco \
        --output-dir results/evaluation \
        --task all \
        --device cuda

    echo "  [OK] Results → results/evaluation/"
}

evaluate_quick() {
    echo ""
    echo "▶ Step 2 (QUICK): Evaluating on 100 images only..."

    python evaluate_models.py \
        --coco-root data/coco \
        --output-dir results/evaluation_quick \
        --task all \
        --max-images 100 \
        --device cuda

    echo "  [OK] Quick results → results/evaluation_quick/"
}

evaluate_detection_only() {
    echo ""
    echo "▶ Step 2 (Detection only): YOLOv8, YOLOv11, RT-DETR, Faster R-CNN, DETR..."

    python evaluate_models.py \
        --coco-root data/coco \
        --output-dir results/evaluation \
        --task detection \
        --device cuda
}

evaluate_segmentation_only() {
    echo ""
    echo "▶ Step 2 (Segmentation only): YOLOv8-seg, YOLOv11-seg, Mask R-CNN, Mask2Former..."

    python evaluate_models.py \
        --coco-root data/coco \
        --output-dir results/evaluation \
        --task segmentation \
        --device cuda
}

evaluate_single_model() {
    MODEL=$1
    echo ""
    echo "▶ Step 2 (Single model): Evaluating ${MODEL}..."

    python evaluate_models.py \
        --coco-root data/coco \
        --output-dir results/evaluation \
        --models ${MODEL} \
        --device cuda
}

# ── USAGE ─────────────────────────────────────────────────────

run_conformal_viz() {
    echo ""
    echo "▶ Conformal calibration visualizations (5 impact figures)..."
    python visualize_conformal.py
    echo "  [OK] → results/eda/conformal_viz_*.png"
}

run_setting2_viz() {
    echo ""
    echo "▶ Setting 2 visualizations (attacks + conformal + cross-dataset)..."
    python visualize_setting2.py
    echo "  [OK] → results/attacks/setting2_*.png"
}

run_all_viz() {
    run_eda
    run_conformal_viz
    run_setting2_viz
    echo ""
    echo "  ALL visualizations complete."
    echo "    results/eda/             (13 EDA + 5 conformal = 18 plots)"
    echo "    results/attacks/         (5 Setting 2 plots)"
}

usage() {
    echo ""
    echo "Usage: bash run.sh <command>"
    echo ""
    echo "Commands:"
    echo "  setup           Install all dependencies"
    echo "  download        Download COCO val2017 + create conformal split"
    echo "  eda             Run full EDA (13 plots → results/eda/)"
    echo "  viz-conformal   Conformal calibration impact figures (5 plots)"
    echo "  viz-setting2    Setting 2: attacks + calibration + cross-dataset (5 plots)"
    echo "  viz-all         All visualizations (23 plots total)"
    echo "  eval-all        Evaluate ALL models (full, ~2-4h on A100)"
    echo "  eval-quick      Evaluate ALL models on 100 images (quick test)"
    echo "  eval-det        Evaluate detection models only"
    echo "  eval-seg        Evaluate segmentation models only"
    echo "  eval <model>    Evaluate a single model"
    echo "  full            Run entire pipeline (download → eda → eval)"
    echo ""
    echo "Available models:"
    echo "  Detection:    yolov8x, yolov11x, rtdetr-l, faster-rcnn-r101, detr-resnet101"
    echo "  Det + Seg:    yolov8x-seg, yolov11x-seg, mask-rcnn-r101, mask2former-swin-l"
    echo ""
    echo "Examples:"
    echo "  bash run.sh setup"
    echo "  bash run.sh download"
    echo "  bash run.sh eda"
    echo "  bash run.sh eval-quick"
    echo "  bash run.sh eval yolov8x"
    echo "  bash run.sh eval mask2former-swin-l"
    echo "  bash run.sh full"
}

# ── FULL PIPELINE ─────────────────────────────────────────────

full_pipeline() {
    setup_env
    download_data
    run_eda
    run_conformal_viz
    run_setting2_viz
    evaluate_all
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  FULL PIPELINE COMPLETE"
    echo "═══════════════════════════════════════════════════════════"
    echo "  results/"
    echo "    ├── eda/                  (18 EDA + conformal plots)"
    echo "    ├── attacks/              (5 Setting 2 plots)"
    echo "    └── evaluation/"
    echo "        ├── metrics_summary.json"
    echo "        ├── metrics_table.txt"
    echo "        ├── predictions/      (per-model COCO preds)"
    echo "        └── plots/            (6 comparison plots)"
}

# ── DISPATCHER ────────────────────────────────────────────────

case "${1}" in
    setup)          setup_env ;;
    download)       download_data ;;
    eda)            run_eda ;;
    viz-conformal)  run_conformal_viz ;;
    viz-setting2)   run_setting2_viz ;;
    viz-all)        run_all_viz ;;
    eval-all)       evaluate_all ;;
    eval-quick)     evaluate_quick ;;
    eval-det)       evaluate_detection_only ;;
    eval-seg)       evaluate_segmentation_only ;;
    eval)           evaluate_single_model "${2}" ;;
    full)           full_pipeline ;;
    *)              usage ;;
esac
