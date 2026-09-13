#!/usr/bin/env pwsh
# ================================================================
#  COMPLETE PIPELINE — From zero to Q1 paper
#  All strings use SINGLE QUOTES only
#  Estimated total: ~9 hours on RTX 4070 Laptop GPU (8GB VRAM)
# ================================================================

$ErrorActionPreference = 'Continue'

Write-Host ''
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host '  PHASE 0: VERIFY ENVIRONMENT' -ForegroundColor Cyan
Write-Host '================================================================' -ForegroundColor Cyan
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
if ($LASTEXITCODE -ne 0) {
    Write-Host '  [ERROR] PyTorch not working. Run:' -ForegroundColor Red
    Write-Host '  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124'
    exit 1
}

Write-Host ''
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host '  PHASE 1: TESTS' -ForegroundColor Cyan
Write-Host '================================================================' -ForegroundColor Cyan
python test_pipeline.py

Write-Host ''
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host '  PHASE 2: DOWNLOAD COCO' -ForegroundColor Cyan
Write-Host '================================================================' -ForegroundColor Cyan
python main.py download

Write-Host ''
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host '  PHASE 3: MODEL EVALUATION - 7 models x 5000 images ~90min' -ForegroundColor Cyan
Write-Host '================================================================' -ForegroundColor Cyan
python main.py eval-all

Write-Host ''
Write-Host '================================================================' -ForegroundColor Green
Write-Host '  PHASE 4: ABLATION + BASELINES ~10min' -ForegroundColor Green
Write-Host '================================================================' -ForegroundColor Green
python main.py ablation --models yolov8x yolov11x rtdetr-l detr-resnet101

Write-Host ''
Write-Host '================================================================' -ForegroundColor Green
Write-Host '  PHASE 5: REAL COCO VISUALIZATIONS ~5min' -ForegroundColor Green
Write-Host '================================================================' -ForegroundColor Green
python main.py real-viz --models yolov8x rtdetr-l --n-images 8

Write-Host ''
Write-Host '================================================================' -ForegroundColor Green
Write-Host '  PHASE 6: THEORETICAL PROPOSITIONS ~5min no GPU' -ForegroundColor Green
Write-Host '================================================================' -ForegroundColor Green
python main.py theory

Write-Host ''
Write-Host '================================================================' -ForegroundColor Yellow
Write-Host '  PHASE 7: ADVERSARIAL ATTACKS - 3 models x 5 attacks x 200 images ~4hrs' -ForegroundColor Yellow
Write-Host '================================================================' -ForegroundColor Yellow
python main.py adversarial --models yolov8x yolov11x --attacks fgsm pgd-20 pgd-50 dag tog-v --max-images 200
python main.py adversarial --models detr-resnet101 --attacks fgsm pgd-20 --max-images 100

Write-Host ''
Write-Host '================================================================' -ForegroundColor Yellow
Write-Host '  PHASE 8: DEFENSE EVALUATION + BASELINE AUROC ~5min no GPU' -ForegroundColor Yellow
Write-Host '================================================================' -ForegroundColor Yellow
python main.py defense

Write-Host ''
Write-Host '================================================================' -ForegroundColor Magenta
Write-Host '  PHASE 9: NATURAL CORRUPTIONS - GAP 1 ~2hrs' -ForegroundColor Magenta
Write-Host '================================================================' -ForegroundColor Magenta
python main.py corruptions --models yolov8x yolov11x rtdetr-l --max-images 200

Write-Host ''
Write-Host '================================================================' -ForegroundColor Magenta
Write-Host '  PHASE 10: SEGMENTATION UNDER ATTACK ~1hr' -ForegroundColor Magenta
Write-Host '================================================================' -ForegroundColor Magenta
python main.py seg-attack --models yolov8x-seg yolov11x-seg --max-images 100

Write-Host ''
Write-Host '================================================================' -ForegroundColor Magenta
Write-Host '  PHASE 11: TRANSFER ATTACKS - black box ~30min' -ForegroundColor Magenta
Write-Host '================================================================' -ForegroundColor Magenta
python main.py transfer

Write-Host ''
Write-Host '================================================================' -ForegroundColor Magenta
Write-Host '  PHASE 12: CROSS-DATASET VOC - GAP 5 ~30min' -ForegroundColor Magenta
Write-Host '================================================================' -ForegroundColor Magenta
python main.py cross-voc --models yolov8x rtdetr-l

Write-Host ''
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host '  PHASE 13: STATISTICAL ANALYSIS - bootstrap CI + effect sizes ~5min' -ForegroundColor Cyan
Write-Host '================================================================' -ForegroundColor Cyan
python main.py statistics

Write-Host ''
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host '  PHASE 14: PAPER ASSETS - ALL figures + tables + samples ~2min' -ForegroundColor Cyan
Write-Host '================================================================' -ForegroundColor Cyan
python main.py paper-assets

Write-Host ''
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host '  PIPELINE COMPLETE - ALL 12 PHASES' -ForegroundColor Cyan
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host ''
Write-Host '  Results:'
Write-Host '    results/evaluation/       7 model metrics + predictions'
Write-Host '    results/ablation/         Ablation A0-A5 + baselines'
Write-Host '    results/real_viz/         Publication figures on COCO'
Write-Host '    results/theory/           Proposition P1,P3,P4,P5 verification'
Write-Host '    results/adversarial/      3 models x 5 attacks + conformal tables'
Write-Host '    results/defense/          Pareto + AUROC + baseline comparison'
Write-Host '    results/corruptions/      fog/blur/noise/jpeg x 3 severity'
Write-Host '    results/segmentation/     bbox vs mask vulnerability'
Write-Host '    results/transfer/         Black-box transferability'
Write-Host '    results/cross_dataset/    COCO -> VOC2012 real shift'
Write-Host '    results/statistics/       Bootstrap CI + effect sizes + AUROC comparison'
Write-Host ''
Write-Host '    paper_assets/             ALL publication-ready figures + LaTeX tables'
Write-Host ''
