@echo off
REM ================================================================
REM  COMPLETE PIPELINE — From zero to paper
REM  Run from: C:\Users\Alberto\Desktop\CV
REM  Estimated time: ~4-5 hours total on RTX 4070
REM ================================================================

echo ================================================================
echo   PHASE 0: SETUP (5 min)
echo ================================================================
python main.py setup
python main.py download

echo ================================================================
echo   PHASE 1: EDA + VISUALIZATIONS (3 min)
echo ================================================================
python main.py viz-all

echo ================================================================
echo   PHASE 2: MODEL EVALUATION on COCO val2017 (90 min)
echo   7 models x 5000 images = 35000 inferences
echo ================================================================
python main.py eval-all

echo ================================================================
echo   PHASE 3: TRUE BOX CALIBRATION (5 min, no GPU)
echo   Bias correction + IoU guarantee on real predictions
echo ================================================================
python visualize_box_calibration.py --models yolov8x yolov11x rtdetr-l detr-resnet101

echo ================================================================
echo   PHASE 4: ABLATION + BASELINES + CROSS-DATASET (5 min, no GPU)
echo   6 ablation configs x 4 baselines x 5 shift levels
echo ================================================================
python main.py ablation --models yolov8x yolov11x rtdetr-l detr-resnet101

echo ================================================================
echo   PHASE 5: REAL COCO VISUALIZATIONS (3 min, no GPU)
echo   GT vs Predictions vs Calibrated on real photos
echo ================================================================
python main.py real-viz --models yolov8x rtdetr-l --n-images 8

echo ================================================================
echo   PHASE 6: THEORETICAL PROPOSITIONS (3 min, no GPU)
echo   Verify 4 propositions on real data
echo ================================================================
python theoretical_framework.py

echo ================================================================
echo   PHASE 7: ADVERSARIAL ATTACKS (60 min, GPU required)
echo   Real gradient attacks + 6-phase conformal evaluation
echo ================================================================
python main.py adversarial --models yolov8x yolov11x --attacks fgsm pgd-20 pgd-50 dag tog-v --max-images 100

echo ================================================================
echo   PHASE 8: ADVERSARIAL on DETR (30 min, GPU required)
echo ================================================================
python main.py adversarial --models detr-resnet101 --attacks fgsm pgd-20 --max-images 50

echo ================================================================
echo   PIPELINE COMPLETE
echo ================================================================
echo.
echo   Results:
echo     results/eda/              23 EDA + conformal plots
echo     results/attacks/          5 Setting 2 plots
echo     results/evaluation/       7 model metrics + predictions + 6 plots
echo     results/box_calibration/  IoU correction figures + bias vectors
echo     results/ablation/         Ablation + baselines + cross-dataset
echo     results/real_viz/         Real COCO image visualizations
echo     results/theory/           Proposition verification + figures
echo     results/adversarial/      Attack results + 3-way comparison
echo.
pause
