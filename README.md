# Lightweight Texture-Based Defect Detection — Multi-Scale LBP + GLCM + SVM vs CNN baselines (MVTec AD)

VIT Chennai · BCSE417L · Machine Vision project. Full specification: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

## Overview

Can a lightweight, CPU-only, hand-engineered pipeline (LBP histograms ± GLCM statistics → SVM) detect surface defects on
MVTec AD **textures** (carpet, grid, leather, tile, wood) competitively with a CNN, while training and running much
faster and being explainable (region heatmaps + feature-level importance)?

| ID | Method | Features |
|---|---|---|
| P1 | Baseline: single-scale uniform LBP (P=8,R=1) | 59-d |
| P2 | Multi-scale LBP, R=1,2,3 | 177-d |
| P3 | P2 + GLCM (24-d) → standardize → SelectKBest(MI, k=100) | 201 → 100 |
| B1 | Frozen ResNet-18 features + same SVM grid (reference) | 512-d |
| B2 | Fine-tuned ResNet-18 + Grad-CAM (reference) | — |

Everything runs on CPU. All outputs go to `outputs/` (figures, tables, models, gallery, `index.html`).

## Setup (already done once; repeat only on a new machine)

> **The C: drive must never be written to.** Every terminal must source `scripts\env_d.ps1`.

```powershell
cd "D:\MV Project\texture-defect-detection-lbp-svm"
. .\scripts\env_d.ps1                                   # redirects TEMP, caches, Jupyter dirs to D:\MV Project\_sys
py -3.10 -m venv .venv                                  # any Python 3.10-3.12 works
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
setx CLAUDE_CODE_TMPDIR "D:\MV Project\_sys\tmp"        # one-time, only if you use Claude Code
```

## How to run

**Open a fresh PowerShell and run these three lines first, every time:**

```powershell
cd "D:\MV Project\texture-defect-detection-lbp-svm"
. .\scripts\env_d.ps1
.\.venv\Scripts\Activate.ps1
```

### 1. Tests (≈ 1 min; the split test needs the dataset)
```powershell
python -m pytest tests -q
```

### 2. Quick smoke test (2 categories, 1 seed, 1 CNN epoch → `outputs_quick\`)
```powershell
python main.py --config configs/quick.yaml
```
Add `--skip-cnn` to skip B1/B2 (much faster, but figures that need B1/B2 drop those columns).

### 3. Full run (5 categories × 5 seeds, cold cache → `outputs\`) — long (hours; B2 fine-tuning and ablations dominate)
```powershell
python main.py --config configs/textures.yaml --rebuild-cache
```

### 4. 512-px size ablation (A5 / fig31) — after the full run
```powershell
python main.py --config configs/extended_ablation_512.yaml
python main.py --config configs/textures.yaml --stage figures      # re-render so fig31 appears
python main.py --config configs/textures.yaml --stage report       # refresh index.html
```

### 5. Run one stage at a time (each stage is resumable and reads what earlier stages saved)
```powershell
python main.py --config configs/textures.yaml --stage data        # T10 + split manifests
python main.py --config configs/textures.yaml --stage features    # LBP/GLCM/ResNet feature caches (data_cache\)
python main.py --config configs/textures.yaml --stage train       # P1-P3, B1, B2 -> T1,T2,T3,T6,T9 + models
python main.py --config configs/textures.yaml --stage latency     # T4, T8, T0_environment, T0_targets
python main.py --config configs/textures.yaml --stage localize    # patch-SVM heatmaps + Grad-CAM -> T5
python main.py --config configs/textures.yaml --stage explain     # permutation importance, Cohen's d
python main.py --config configs/textures.yaml --stage ablate      # A1-A4, A6 -> T7
python main.py --config configs/textures.yaml --stage figures     # fig01-fig43
python main.py --config configs/textures.yaml --stage gallery     # per-image result sheets
python main.py --config configs/textures.yaml --stage report      # outputs\index.html
```
Options: `--seeds 42` (override seeds), `--skip-cnn`, `--rebuild-cache`.

### 6. Notebooks (never retrain; they only show saved results)
```powershell
python scripts/make_notebooks.py
jupyter nbconvert --to notebook --execute --inplace notebooks/00_dataset_explorer.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/01_full_comparison.ipynb
```

### 7. Look at the results
```powershell
start outputs\index.html                          # everything in one page
explorer outputs\figures                          # 43 figures
explorer outputs\gallery                          # one sheet per test image
Import-Csv outputs\tables\T1_metrics_per_method.csv | Format-Table -AutoSize
Import-Csv outputs\tables\T4_inference_latency.csv | Format-Table -AutoSize
Import-Csv outputs\tables\T0_targets.csv | Format-Table -AutoSize
Get-Content (Get-ChildItem outputs\logs\run_*.log | Sort LastWriteTime | Select -Last 1) -Tail 40
python scripts/update_readme.py                   # copies the real T1/T4 numbers into the Results section below
```
If one figure fails, the run continues and the log shows `figNN FAILED: <reason>`; fix and rerun `--stage figures`.

## Results

<!-- RESULTS:START -->
_Not filled yet. Run the full pipeline, then `python scripts/update_readme.py`._
<!-- RESULTS:END -->

## Project structure

```
main.py                   CLI (stages: data features train latency localize explain ablate figures gallery report)
configs/                  textures.yaml (full) · quick.yaml (smoke) · extended_ablation_512.yaml
scripts/                  env_d.ps1 (D: redirection) · make_notebooks.py · update_readme.py
src/                      config, utils, timing, data_loader, preprocessing, features_lbp, features_glcm,
                          cnn_baselines, classifier, evaluate, evaluate_latency, localize, explain, ablations,
                          visualize (43 figures), report (gallery + index.html)
tests/                    test_shapes, test_split, test_localize
notebooks/                00_dataset_explorer, 01_full_comparison
outputs/                  figures/ figures_data/ tables/ models/ heatmaps/ gallery/ logs/ index.html  (git-ignored)
data_cache/               cached features (git-ignored)
dataset/                  MVTec AD (read-only, git-ignored)
```

## Binding deviations from the proposal (see plan §2.2)
D1 uniform LBP uses 59 bins (`nri_uniform`) · D2 P2 uses (8,1),(8,2),(8,3) = 177-d (literal 54-d version in ablation A1) ·
D3 GLCM averaged over d∈{1,2} → 24-d · D4 primary size 256 px (512 in A5) · D5 P3 uses SelectKBest-MI (PCA in A3) ·
D6 own stratified 70/30 split (supervised re-formulation) · D7 CNN baselines B1/B2 added.

## Limitations
1. Supervised re-formulation of an unsupervised benchmark; own split → numbers are **not** comparable to MVTec leaderboards.
2. Textures only; object classes are out of scope.
3. The image-level SVM does not localize; heatmaps come from separate patch SVMs trained on training-image masks (a proxy).
4. B2 Grad-CAM thresholds were fitted on images B2 had trained on.
5. B2 is evaluated on one seed; P1–P3/B1 on five.
6. 256 px (not 512) and the LBP bin layout deviate from the proposal; ablations A5/A1 quantify both.
7. Timings are machine-specific, single-thread, image decoding excluded.
8. Small per-category test sets (e.g. wood has 19 normal images in total).
