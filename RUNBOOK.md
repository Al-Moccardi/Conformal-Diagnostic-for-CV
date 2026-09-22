# RUNBOOK — reproducing every number of the revised paper

Two tiers:

* **Tier A (CPU, minutes):** regenerate every table of the revised manuscript from the saved predictions shipped in `results/`. No GPU, no dataset download beyond the COCO annotation file.
* **Tier B (GPU, ~40 GPU-hours on an RTX 4070):** regenerate the saved predictions themselves (clean evaluation, attacks, corruptions).

All commands are run from the repository root. Python 3.10+, `pip install -r requirements.txt`. Random seed 42 everywhere (calibration/test split, attack subsets, bootstrap).

---

## 0. Setup

```bash
git clone https://github.com/Al-Moccardi/Conformal-Diagnostic-for-CV.git
cd Conformal-Diagnostic-for-CV
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# GPU only (Tier B): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Annotations are needed by every script (matching against ground truth):

```bash
python main.py download --dataset coco        # writes data/coco/annotations/instances_val2017.json and the images
# Tier A only needs the annotation file; if you already have it:
mkdir -p data/coco/annotations && cp /path/to/instances_val2017.json data/coco/annotations/
```

Clean per-model predictions (`results/evaluation/predictions/<model>_bbox.json`, ≈ 50–240 MB each) are **not** in the zip. Either run Tier B step 1, or obtain them from the authors and drop them in place. Tier A scripts marked † need them; the others only need `results/adversarial/full_results.json` (included).

---

## Tier A — post-hoc tables from saved predictions

> **v8 note.** The manuscript no longer reads generated macros: the values of Tables 8 and 10 are the stored in-sample runs listed below (baseline fitted on the clean versions of the evaluated images; proxy false alarms counted on the calibration stream), and the paper says so. `scripts/rebuttal/oos_abstention_monitor.py` (three disjoint clean sets, held-out false alarms) is an optional out-of-sample variant that the paper does not use.

| # | Command | Produces | Paper |
|---|---|---|---|
| A1 | `python scripts/rebuttal/alpha_sweep.py` † | `results/revision/alpha_sweep_tau.json` | Table 4, τ column (0.047 / 0.042 / 0.203 / 0.387) |
| A2 | `python scripts/rebuttal/active_filter_eval.py` † | `results/revision/table4_active_filter.json` | Table 5 (post-threshold coverage, n per pair) |
| A3 | `python scripts/rebuttal/recal_eta.py` † | `results/revision/prop2_recalibration_eta.json` | Table 6 (AR, Cov clean-τ, Cov recalibrated-τ; the script also prints η, which the revised paper no longer reports) |
| A4 | `python scripts/rebuttal/ge2_table6_coverage_fix.py` † | `results/revision/table6_corrected.json` | Table 8 (selective predictor operating curve; the stored run of the paper is `table6_fixedcode_check.json`, which also holds the τ_A = 0.31 row) |
| A5 | `python scripts/rebuttal/proxy_monitor.py` † | `results/revision/table7b_proxy_monitor.json` | Table 10 (label-free monitor, Rules A/B) |
| A6 | `python scripts/rebuttal/check_tables.py` † | stdout | Tables 11–12 (class-conditional and multi-IoU clean values) |
| A7 | `python scripts/rebuttal/table11_pvalues.py` † | `results/revision/table11_pvalues_onesample.json` | Table 13 (one-sample KS, clean D_n, per-image AUROC) |
| A8 | `python scripts/rebuttal/one_pair_per_image.py` † | `results/revision/one_pair_per_image.json` | Section 3.3 (pooled vs one-pair-per-image Cov^CP) |
| A9 | `python scripts/rebuttal/effect_sizes_active.py` † | `results/revision/effect_sizes_active_filter.json` | Cohen's d, Welch p, image-bootstrap CIs (Table 5 caption, §5.4, Figure 1 right) |
| A12 | `python scripts/rebuttal/table4_all100.py` † | `results/revision/table4_all100.json` | Table 5 over ALL 100 attacked images (empty output = coverage 0), Cohen's d vs clean coverage of the same 100 images |
| A13 | `python scripts/rebuttal/all100_effect_sizes.py` † | `results/revision/effect_sizes_all100.json` | Image-bootstrap CIs over 100 images (Figure 1 right), Table 6 rescaled to 100 images |
| A14 | `python paper/regen_fig_v7.py` (reads A13's JSON) | `paper/fig_coverage_gap_severity_v7.png` | Figure 1 |
| A15 | `python scripts/rebuttal/realised_budget.py` | stdout | Table 2 (realised L∞ / RMS per pair, from the recorded distances in `full_results.json`) |
| A11 | `python main.py statistics` † | `results/statistics/statistical_analysis.json` | Table 15 (per-pair AUROC of every signal), Table 14 AUROC columns |

Expected run time per script: 10 s – 3 min on a laptop CPU (A3 and A9 are the slowest).

Expected key outputs (to check your run):

```
A1  yolov8x tau=0.0471  yolov11x 0.0424  rtdetr-l 0.2030  detr-resnet101 0.3870
A2  yolov8x clean post_macro=0.861  pgd-20 0.214 (n=100)  tog-v 0.143 (n=98)
A3  yolov8x pgd-20  AR=0.262 Cov(cleanCP)=0.214 Cov(recalCP)=0.239 eta=0.0007
A4  tau=0.3: rej_clean 0.06 rej_adv 0.82 cov_clean_ret 0.735 cov_adv_ret 0.659
A5  pgd-20: rule A delay 19 FA 0 | rule B delay 8 FA 17/181
A7  yolov8x D_adv=0.2411 p=5.5e-16  D_clean=0.0142 ; yolov11x D_adv=0.089 p=0.20
A8  yolov8x cov_cp_pooled=0.9004  cov_cp_onepair=0.9016
A9  detr-resnet101 pgd-20 d=0.02 p=0.87 ; yolov8x tog-v d=2.78
```

Values that are read directly from stored files (no script): Table 14 coverage from `results/corruptions/natural_corruption_results.json` (`clean_coverage` = 0.832 for YOLOv8x); transfer AUROC from `results/transfer/transfer_results.json`; DETR/APGD from `results/revision/apgd_detr_results_v2.json`; prediction-set sizes from `results/revision/aps_ablation_results.json`.

---

## Tier B — regenerating the saved predictions (GPU)

| # | Command | Produces | Time (RTX 4070) |
|---|---|---|---|
| B1 | `python main.py eval-all` | `results/evaluation/predictions/*_bbox.json`, `metrics_summary.json` (Table 4 mAP/ECE/Brier) | ~1 h |
| B2 | `python main.py adversarial --models yolov8x yolov11x detr-resnet101 --attacks fgsm pgd-20 pgd-50 dag tog-v --max-images 200` | `results/adversarial/full_results.json` (per-image adversarial predictions) | ~35 h |
| B3 | `python scripts/attacks/run_apgd_detr.py --n-images 100 --steps 100 --restarts 5` | `results/revision/apgd_detr_results_v2.json` | ~4 h |
| B4 | `python main.py corruptions` | `results/corruptions/natural_corruption_results.json` | ~2 h |
| B5 | `python main.py transfer` | `results/transfer/transfer_results.json` | ~1 h |
| B6 | `python main.py defense` | `results/defense/` (figures) | minutes |

Attack generation fails on some images (numerical overflow or empty output); the attacked subsets therefore contain 30–100 images per pair, as reported in the paper's n columns. Re-running B2 reproduces the same subsets under seed 42.

After Tier B, run Tier A again; every table should reproduce to the printed digit.

### Population of the adversarial means (v3 correction)

`run_adversarial_eval.py` stores only the detections of an attacked image; an image on which the attacked detector returns nothing at conf 0.01 has no entry, and `active_filter_eval.py` / `recal_eta.py` averaged coverage only over the images that have one (`n` = 30–100 per pair). Every attack was generated on all 100 test images (see `results/adversarial/attack_samples/`, 100 files per YOLO pair), so a missing image is a fully successful attack, not a failed one. `table4_all100.py` and `all100_effect_sizes.py` count such images with coverage 0 and average over all 100; the manuscript (v3) reports those values and gives the number of images with surviving output as `n_det`. Since a missing image contributes 0 to every per-image mean and no true positive to the recalibration, the earlier per-image means rescale exactly by `n_det/100`.

---

## Where each paper number comes from

| Paper item | File | Field |
|---|---|---|
| Table 2 | `results/adversarial/full_results.json` | `mean_l_inf`, `mean_l2` per pair (`scripts/rebuttal/realised_budget.py`) |
| Table 4 mAP / ECE | `results/evaluation/metrics_summary.json` | `<model>.mAP@[0.5:0.95]`, `.cal_ECE` |
| Table 4 τ | `results/revision/alpha_sweep_tau.json` | `<model>.tau_eq11[alpha=0.10]` |
| Table 5 | `results/revision/table4_all100.json` (all 100 images) | `<model>.<attack>.post100`, `.n_present`, `.cohens_d` |
| Table 6 | `results/revision/effect_sizes_all100.json` | `<model>.<attack>.table5.{AR,cov_cleanCP,cov_recalCP,eta}` (rescaled to 100 images) |
| Table 7 (PGD steps) | `results/adversarial/full_results.json` (`pgd-20`, `pgd-50`), `results/revision/apgd_detr_results_v2.json`; PGD-5/10 from the original sweep run | pre-threshold coverage |
| Table 8 | `results/revision/table6_fixedcode_check.json` (grid rows also in `table6_active_filter.json`) | per τ_A, incl. the calibrated 0.31 row |
| Table 9 | `python main.py revision` (runtime monitor block) → `results/revision/fig_runtime_monitor.png` | windowed coverage |
| Table 10 | `results/revision/table7b_proxy_monitor.json` | rules A/B |
| Table 11 | `scripts/rebuttal/check_tables.py` output | per-class clean / PGD-20 |
| Table 12 | same | multi-IoU profile |
| Table 13 | `results/revision/table11_pvalues_onesample.json` | `D_adv_onesample`, `D_clean_onesample`, `auroc_image_ks` |
| Table 14 | `results/corruptions/natural_corruption_results.json` | `clean_coverage`, `corruptions.<name>_s<k>.coverage/.auroc` |
| Table 15 | `results/statistics/statistical_analysis.json` (`auroc_comparison`, `cp_specific_signals`), `results/reviewer_fixes/reviewer_fixes_v2.json` (squeezing) | means over the eleven pairs of the revised paper (DETR-R101/PGD-20 excluded): 0.832 / 0.824 / 0.806 / 0.762 / 0.704 |
| §6.6 transfer | `results/transfer/transfer_results.json` | `auroc`, `mean_confidence` |
| §3.3 check | `results/revision/one_pair_per_image.json` | `cov_cp_pooled`, `cov_cp_onepair_cal_onepair_test` |
| Effect sizes / CIs | `results/revision/effect_sizes_all100.json` | `cohens_d`, `welch_p`, `gap_ci` (clean bar of Fig. 1: `effect_sizes_active_filter.json`) |

---

## What was wrong in the submitted code and how it was fixed

1. `src/calibration/adaptive_conformal.py`, `calibrate()`: the calibration scores were built from **every** raw detection, with score 1.0 for unmatched ones. Since 91–98 % of raw detections are unmatched, the (1−α)-quantile was 1.0 and `conf_threshold` collapsed to 0. Fix: `matched = [m for m in all_matched if m["correct"]]` before computing any quantile (confidence and localisation).
2. `src/calibration/conformal_defense.py`: `SelectivePredictor._uses_anomaly` was `conf_threshold < 0.001` and `ConformalAttackDetector.fit_thresholds` set `threshold_uses_filtering = mean abstention > 0.01`; both silently replaced the anomaly score of Eq. 12 by the per-image filter rate as soon as τ > 0. Fix: both are now constant (`True` / `False`), so the abstention decision always uses the composite score.

With the fixes, `python main.py conformal` reproduces the τ values of Table 4, and `scripts/rebuttal/ge2_table6_coverage_fix.py` reproduces Table 8 exactly.

---

## Troubleshooting

* `FileNotFoundError: data/coco/annotations/instances_val2017.json` — run `python main.py download` or copy the annotation file (Tier A needs only this file).
* `FileNotFoundError: results/evaluation/predictions/yolov8x_bbox.json` — clean predictions are not in the zip; run Tier B step 1 or request them.
* `git` warnings about `index.lock` on synced folders (OneDrive/Dropbox): pause sync or work on a local clone.
* Windows: use `run_all.ps1` (PowerShell 5, one command per line) instead of `run.sh`.
