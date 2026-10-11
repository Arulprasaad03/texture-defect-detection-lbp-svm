# IMPLEMENTATION PLAN — Lightweight Texture-Based Defect Detection (Multi-Scale LBP + GLCM + SVM) vs CNN Baselines on MVTec AD

> **Audience:** the coding assistant that will implement this project.
> **Rule of this document:** every decision is already made. Implement exactly what is written. If something is genuinely missing or contradictory, **stop and ask the user** — do not guess, do not substitute, do not "improve".
> **Sources used:** `docs/Machine_Vision_Project_Proposal.docx`, `docs/Lightweight_Texture-Based_Defect_Detection_...pptx`, `README.md`, and the dataset on disk (`dataset/MVTec AD/`, verified).
> **Scope of this file:** only this file (`IMPLEMENTATION_PLAN.md`) was written when the plan was produced. All other files listed below are to be **created by you during implementation**.

---

## 0. HARD RULES (read first, never violate)

### 0.1 D: drive only — the C: drive must never be written to
The C: drive is full/unusable (writes fail with `No space left on device`). **Nothing** — code, venv, caches, temp files, logs, pip/torch downloads, matplotlib config, Jupyter runtime files, `__pycache__`, model weights, outputs — may be written to `C:`.

1. Project root (the only repo): `D:\MV Project\texture-defect-detection-lbp-svm\` (called `ROOT` below).
2. System-side scratch root (outside the repo, never committed): `D:\MV Project\_sys\` (called `SYS` below). Create these sub-folders: `tmp`, `pip_cache`, `mpl`, `torch`, `joblib`, `cache`, `pycache`, `jupyter\config`, `jupyter\data`, `jupyter\runtime`, `ipython`, `userbase`.
3. Create `ROOT\scripts\env_d.ps1` that sets, for the current PowerShell session, **all** of the following, then prints each value:
   ```
   TEMP, TMP, TMPDIR                  = SYS\tmp
   PIP_CACHE_DIR                      = SYS\pip_cache
   MPLCONFIGDIR                       = SYS\mpl
   TORCH_HOME                         = SYS\torch
   JOBLIB_TEMP_FOLDER                 = SYS\joblib
   XDG_CACHE_HOME, HF_HOME            = SYS\cache
   PYTHONPYCACHEPREFIX                = SYS\pycache
   PYTHONUSERBASE                     = SYS\userbase
   JUPYTER_CONFIG_DIR                 = SYS\jupyter\config
   JUPYTER_DATA_DIR                   = SYS\jupyter\data
   JUPYTER_RUNTIME_DIR                = SYS\jupyter\runtime
   IPYTHONDIR                         = SYS\ipython
   OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS = 1   (needed for fair timing, see §7)
   ```
   Every command in this project must be run **after** `. .\scripts\env_d.ps1`.
4. Python environment: virtual env at `ROOT\.venv` (Python 3.11). Find Python with `py -3.11 --version` / `where python`. Running an interpreter that already lives on C: only to *create* the venv is allowed (it only reads). If no Python 3.10–3.12 exists at all, install Python 3.11 with the silent installer using `TargetDir=D:\Python311` and `/quiet`; never install to C:.
5. `src/utils.py` must contain `assert_on_d(path)` which raises `RuntimeError` if the resolved absolute path starts with `C:` (case-insensitive). It must be called on: every output directory, the cache directory, the config's `dataset_root`, and on `tempfile.gettempdir()` at program start. `main.py` must call a `check_environment()` first that (a) asserts `TEMP`, `TMP`, `MPLCONFIGDIR`, `TORCH_HOME`, `JOBLIB_TEMP_FOLDER`, `PIP_CACHE_DIR` all start with `D:`, (b) asserts `sys.prefix` starts with `D:`, and aborts with a clear message otherwise.
6. **Claude Code itself** (the assistant's own temp/scratchpad/config location) is outside the repo and cannot be changed by this plan's code. Required one-time **user** action (tell the user in your first message, do not do it silently): run in PowerShell `setx CLAUDE_CODE_TMPDIR "D:\MV Project\_sys\tmp"` and restart Claude Code. Until confirmed, **never** use the harness-provided scratchpad directory if it is on C:; use `SYS\tmp` for every temporary file and pass `-I` to any ad-hoc `python` call that reads untrusted files. Do not change `CLAUDE_CONFIG_DIR` (it would move the user's login/memory).
7. Add to `ROOT\.gitignore` (append, do not delete existing lines): `.venv/`, `data_cache/`, `outputs/`, `outputs_quick/`, `*.npz`, `*.pkl`, `*.joblib`, `*.pt`, `.ipynb_checkpoints/`, `_sys/`. (`dataset/` is also ignored — it is 6,644 files; never commit it.)

### 0.2 Other hard rules
- Never write inside `dataset/`. It is read-only input.
- Never `git commit`, `git push`, or delete anything unless the user asks.
- No GPU code paths. Everything runs on CPU. (`torch` CPU build only.)
- No feature/SVM logic inside notebooks. Notebooks only call `src/` functions and show saved figures.
- All randomness is seeded (§5.2). A rerun must reproduce every table value (training/latency seconds excepted — those vary by machine).
- Never fabricate or "fill in" a number. Every number in tables/figures comes from code that ran. If a target (e.g. "F1 > 0.90") is not met, report the real value and flag it as "target not met".
- Do not add features, models, datasets, or figures that are not in this plan. Do not remove any that are.

---

## 1. Project in one paragraph

**Research question:** Can a *lightweight, CPU-only, hand-engineered* machine-vision pipeline (LBP texture histograms ± GLCM statistics → SVM) detect surface defects on MVTec AD textures with accuracy competitive with a CNN, while training and running **orders of magnitude faster** and being **inherently explainable** (we can mark the defect region and say which texture features caused the decision)?

**Three proposed approaches (the core comparison, from the proposal):**
| ID | Name | Features | Dim |
|---|---|---|---|
| **P1** | Baseline: single-scale LBP | uniform LBP (P=8, R=1) | 59 |
| **P2** | Improved: multi-scale LBP | LBP at R=1,2,3 concatenated | 177 |
| **P3** | Advanced: multi-scale LBP + GLCM + feature selection | P2 (177) + GLCM Haralick (24) = 201 → standardize → SelectKBest(MI, k=100) | 100 (after selection) |

**Two heavy-model reference baselines (added by the user's objective "show superiority over CNN/heavy models"):**
| ID | Name | Description |
|---|---|---|
| **B1** | Frozen CNN features + SVM | ImageNet ResNet-18, frozen, 512-d global-avg-pool features → same scaler + SVM grid as P1–P3 |
| **B2** | Fine-tuned CNN | ImageNet ResNet-18 fully fine-tuned on CPU, 2-class head; Grad-CAM for explanation |

**What we claim/measure:** (1) accuracy/F1/AUC of P1<P2<P3 (hypothesis H1), (2) P-pipelines train and infer far faster than B1/B2 at comparable F1 (H2), (3) P-pipelines are explainable at the region level and feature level (H3). Hypotheses are tested honestly; failing is a valid result.

**Output philosophy:** this is a *visual* research project — ~43 figures + a per-image gallery (§11). Every claim is backed by an image.

---

## 2. What the documents say, and the decisions taken where they conflict

### 2.1 Facts taken from the proposal/PPT (keep)
- Title: *Lightweight Texture-Based Defect Detection Using Multi-Scale LBP, GLCM and SVM on MVTec AD*. VIT Chennai, course BCSE417L. Binary task: Normal=0 / Defective=1.
- Preprocessing: grayscale → resize → Gaussian blur 3×3 σ=0.8 → min–max normalization.
- Classifier: SVM linear + RBF, grid `C ∈ {0.1,1,10,100}`, `γ ∈ {scale, auto, 0.01}`, 5-fold stratified CV.
- Metrics: accuracy, precision, recall, F1, confusion matrix, latency (ms/img), FPS, RAM.
- Targets (not guarantees): F1 > 0.90 on textures; latency < 30 ms/img on CPU (> 33 FPS); P2 improves over P1 by +8–15 %.
- Fusion requires z-score standardization before concatenation (LBP ∈ [0,1], GLCM contrast ≫ 1).

### 2.2 Inconsistencies found in the proposal and the binding decisions
| # | Proposal says | Problem | **Decision (binding)** |
|---|---|---|---|
| D1 | P1 = "rotation-invariant uniform LBP, 59-dim" | Rotation-invariant uniform with P=8 gives **10** bins; **59** bins is the *non-rotation-invariant uniform* mapping | Use skimage `method="nri_uniform"` → **59 bins** for P=8. Call it "uniform LBP (59 bins)" everywhere. |
| D2 | P2 = (8,1),(16,2),(24,3) → 177-dim | nri_uniform gives 59/243/555 bins for P=8/16/24; ri-uniform gives 10/18/26 = 54. **No standard binning yields 177** with those P values | Use **(P,R) = (8,1),(8,2),(8,3)** all `nri_uniform` → 59×3 = **177**. Ablation A1 (§9.1) additionally reports the proposal's literal (8,1),(16,2),(24,3) with `uniform` (54-dim) so the deviation is transparently quantified. |
| D3 | GLCM "6 stats × 4 angles at d∈{1,2}" = 24 | 6×4×2 = 48, not 24 | Compute all 48, then **average over the two distances** → **24** (6 props × 4 angles). Fused total = 177+24 = **201**. |
| D4 | Image size 512×512 | 4× slower, no accuracy need for textures | **Primary size = 256×256**. A 512 ablation (A5, `--size-ablation`) runs P1–P3 at 512 on seed 42 and reports both. |
| D5 | P3 "PCA 95 % **or** SelectKBest-MI" | Must pick one for the main result | **Main P3 = StandardScaler → SelectKBest(mutual_info_classif, k=100)** (keeps features interpretable). PCA(95 %) is an ablation (A3). |
| D6 | MVTec AD train folder has only normal images | A supervised SVM needs both classes | Pool images and make **our own stratified 70/30 split** (§4.3). State this in the report: the benchmark's unsupervised protocol is repurposed as supervised binary classification. |
| D7 | "Compare with CNN/heavy models" appears as motivation only | No CNN in the plan | Add B1 and B2 (§5.6). They are *references*, not the "three approaches". |

---

## 3. Dataset (verified on disk)

```
dataset/MVTec AD/
  <category>/
    train/good/*.png                    (normal only)
    test/good/*.png                     (normal)
    test/<defect_type>/*.png            (defective)
    ground_truth/<defect_type>/<id>_mask.png   (binary mask, white = defect)
  license.txt, readme.txt
```
- 15 categories exist. **Primary scope = the 5 textures:** `carpet, grid, leather, tile, wood`. Objects are out of scope (limitation, §13).
- Verified counts (textures):

| category | train/good | test/good | test/defective | defect types |
|---|---|---|---|---|
| carpet | 280 | 28 | 89 | color, cut, hole, metal_contamination, thread |
| grid | 264 | 21 | 57 | bent, broken, glue, metal_contamination, thread |
| leather | 245 | 32 | 92 | color, cut, fold, glue, poke |
| tile | 230 | 33 | 84 | crack, glue_strip, gray_stroke, oil, rough |
| wood | 247 | 19 | 60 | color, combined, hole, liquid, scratch |

- Pool per category = train/good (label 0) + test/good (label 0) + all test/defect (label 1). Pool sizes: carpet 397, grid 342, leather 369, tile 347, wood 326 → **1,781 images** (1,399 normal / 382 defective).
- Images are PNG, ~1024×1024 (grid is single-channel; always `.convert("RGB")` on load then derive gray).
- Masks are matched by name: `test/<type>/NNN.png` ↔ `ground_truth/<type>/NNN_mask.png`. Normal images have an all-zero mask (do not look for a file).

### 3.1 Image ID convention
`img_id = f"{category}/{split_origin}/{defect_type}/{stem}"`, e.g. `carpet/test/cut/003`; for normal ones `defect_type="good"`, `split_origin ∈ {train,test}`.

---

## 4. Experimental protocol

### 4.1 Granularity
One model **per category per pipeline** (5 categories × {P1,P2,P3,B1} × 5 seeds, plus B2 once per category on seed 42). Headline numbers are the **macro average over the 5 categories**, mean ± std over seeds.

### 4.2 Seeds
`seeds = [42, 43, 44, 45, 46]` for P1, P2, P3, B1. **B2 runs on seed 42 only** (CPU fine-tune time); in tables B2 std is shown as `—`. `--quick` uses `[42]`.

### 4.3 Split (per category, per seed) — written to `outputs/tables/split_manifest_seed<seed>.csv`
1. Pool as in §3. Stratify on the string key `defect_type` (good / each defect type).
2. `sklearn.model_selection.train_test_split(test_size=0.30, stratify=<defect_type key>, random_state=seed)`.
3. Result ≈ 70 % train / 30 % test. Test split is **never** used for fitting anything (scaler, selector, SVM, grid search, threshold, patch model, CNN epochs). It is used only for final metrics, figures, and the gallery.
4. Inside train, hyper-parameter search uses 5-fold `StratifiedKFold(shuffle=True, random_state=seed)` on the label.
5. A unit test must assert: train∩test = ∅ for every (category, seed), both classes present in both splits, and the 5-category union equals 1,781 images.

### 4.4 Class imbalance
Test set is ~78 % normal. Therefore: `class_weight="balanced"` in every SVC, weighted cross-entropy in B2, **positive class = Defective**, headline metrics = F1 (defect), ROC-AUC, balanced accuracy; accuracy is reported but never used alone. Decision threshold is fixed at 0 on SVM `decision_function` (0.5 on B2 softmax). No threshold tuning on the test set.

---

## 5. Methods (exact specification)

### 5.1 Image loading
`load_rgb(path) -> np.ndarray uint8 (H,W,3)` via Pillow. Record `read_decode_seconds` separately (§7).

### 5.2 Determinism
`utils.set_seed(seed)` sets `random`, `numpy`, `torch` (`manual_seed`, `use_deterministic_algorithms(True, warn_only=True)`). Every sklearn object gets `random_state=seed` where supported.

### 5.3 Preprocessing (classical pipelines) — `preprocessing.preprocess(rgb, size=256) -> np.ndarray uint8 (size,size)`
Order, fixed:
1. Gray: `0.299R + 0.587G + 0.114B` (BT.601), float32.
2. Resize to `size×size`, bilinear (`skimage.transform.resize(order=1, anti_aliasing=True, preserve_range=True)`).
3. Gaussian blur `scipy.ndimage.gaussian_filter(sigma=0.8, truncate=1.25)` (≈3×3 kernel).
4. Min–max normalize to [0,1]: `(x−min)/(max−min+1e-8)`.
5. Convert back to `uint8`: `np.rint(x*255)`. **All downstream LBP/GLCM use this uint8 image.**
`demo_chain(rgb) -> dict` returns every intermediate (for fig03).

### 5.4 Feature extraction

**LBP** (`features_lbp.py`)
- `lbp_codes(img_u8, P, R, method) -> uint8/uint16 map (size,size)`: pad image by 3 px with `np.pad(mode="reflect")`, run `skimage.feature.local_binary_pattern`, crop back with `[3:-3, 3:-3]`.
- `lbp_hist(codes, n_bins) -> float32` normalized so it sums to 1 (`np.bincount(...,minlength=n_bins)/codes.size`).
- `extract_p1(img) -> (59,)`: `nri_uniform`, P=8, R=1.
- `extract_p2(img) -> (177,)`: `[h(8,1) ‖ h(8,2) ‖ h(8,3)]`, each `nri_uniform`, 59 bins.
- `extract_p2_literal(img) -> (54,)` for ablation A1: `uniform` at (8,1),(16,2),(24,3) → 10+18+26.
- Fixed bin order = skimage output order. Feature names list `LBP_R{1,2,3}_bin{0..58}`.

**GLCM** (`features_glcm.py`)
- Quantize: `img_q = img_u8 // 8` (→ 32 levels, 0..31).
- `skimage.feature.graycomatrix(img_q, distances=[1,2], angles=[0, π/4, π/2, 3π/4], levels=32, symmetric=True, normed=True)`.
- Props in this exact order: `contrast, dissimilarity, homogeneity, energy, correlation, ASM`. For each prop: `graycoprops(...)` gives shape (2,4) → **mean over distance axis** → 4 values (angles 0°,45°,90°,135°).
- `extract_glcm(img) -> (24,)` ordered `[contrast×4, dissimilarity×4, homogeneity×4, energy×4, correlation×4, ASM×4]`. Names `GLCM_<prop>_<angle>`.

**Combined raw vector** `extract_p3_raw(img) -> (201,) = [P2(177) ‖ GLCM(24)]`.

### 5.5 Pipelines as sklearn `Pipeline` objects (`classifier.py`)
All fitted **inside** GridSearchCV so scaler/selector never see validation folds.
```
P1: StandardScaler → SVC
P2: StandardScaler → SVC
P3: StandardScaler → SelectKBest(mutual_info_classif, k=100)  → SVC      # MI random_state=seed
B1: StandardScaler → SVC
SVC(class_weight="balanced", cache_size=500, random_state=seed)
```
Grid (identical for every pipeline), one `GridSearchCV(scoring="f1", cv=5-fold stratified, n_jobs=1, refit=True)` over a list of two dicts:
```
{"svm__kernel":["linear"], "svm__C":[0.1,1,10,100]}
{"svm__kernel":["rbf"],    "svm__C":[0.1,1,10,100], "svm__gamma":["scale","auto",0.01]}
```
(16 configurations). `n_jobs=1` is mandatory so training time is comparable (§7). The best estimator is the model used for all test metrics. Save per fit: best params, best CV F1, `fit_seconds` (whole GridSearchCV incl. refit), `n_support_vectors`.

### 5.6 Heavy-model reference baselines (`cnn_baselines.py`)
- Input transform: RGB → resize to 224×224 (bilinear) → ToTensor → ImageNet mean/std. Load weights `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` (downloads into `TORCH_HOME` on D:). If the download fails, **stop and tell the user**; do not substitute random weights.
- `torch.set_num_threads(1)` for all timing runs (§7). `device="cpu"`.
- **B1:** `model.fc = Identity()`, `eval()`, `no_grad`, batch of 16 for extraction; per-image extraction time recorded as batch_time/batch_size and also single-image latency separately (§7). 512-d features cached like LBP features. Then the §5.5 B1 pipeline.
- **B2:** fine-tune all layers. `model.fc = Linear(512, 2)`. Adam, `lr=1e-4`, `weight_decay=1e-4`, batch 16, **10 epochs**, loss `CrossEntropyLoss(weight=[w0,w1])` with `w_c = N / (2·N_c)` from the training split, augmentation = random horizontal flip and random vertical flip only, last-epoch weights used (no test-based checkpoint selection). Records wall-clock `finetune_seconds` and per-epoch seconds + loss. Prediction = argmax; score = softmax prob of Defective. Seed 42 only.
- **Grad-CAM (B2):** hook on `layer4[-1]` output; CAM for the *Defective* logit; ReLU; bilinear upsample to 256×256; min–max per image. Implemented manually (no extra package).
- Model stats for T8: `n_params`, `fp32_size_MB` (state_dict file), `GFLOPs` via `torch.utils.flop_counter.FlopCounterMode` at 1×3×224×224.

### 5.7 Region-level localization / explanation for P1, P2, P3 (`localize.py`)
The image-level SVMs cannot say *where*. Therefore a **separate patch-level SVM per (category, pipeline)** is trained with the same feature family, using the ground-truth masks of **training images only**.

1. Work on the preprocessed 256×256 uint8 image. Window = **64×64**, training stride **32**, inference stride **16** (13×13 = 169 windows/image).
2. **Efficient window features (this is also a speed point):** compute the LBP code maps for the 3 scales **once per image** (§5.4, with reflect padding), then each window's histogram is `bincount` of the code-map crop (normalized to sum 1). GLCM is computed per window on the quantized crop (same params as §5.4). P1 window feature = 59, P2 = 177, P3 = 201 raw.
3. Window label from the 256×256 mask (mask resized with nearest): `frac = mean(mask[window])`; **positive if frac ≥ 0.25; negative if frac == 0; windows with 0 < frac < 0.25 are discarded.** Good images contribute only negatives.
4. From the category's train split, hold out **20 % of train images** (stratified by label, `random_state=seed`) as `loc_val`; the remaining 80 % = `loc_train`.
5. Training windows from `loc_train` (stride 32). Randomly subsample negatives to at most **3× the number of positives** (`random_state=seed`). Model: same Pipeline family without GridSearch: `StandardScaler → [SelectKBest(MI, k=100) for P3 only] → SVC(kernel="linear", C=1, class_weight="balanced")`.
6. **Heatmap:** score each window with `decision_function`; each pixel's value = mean of scores of windows covering it; then `gaussian_filter(sigma=8)`. Output float32 256×256, named `heat_P{1,2,3}`.
7. **Binarization threshold τ** per (category, pipeline): the value among 50 evenly spaced quantiles of `loc_val` heatmap pixels (quantile 0.50 … 0.995) that maximizes pixel-level F1 against `loc_val` masks. Fitted once, applied unchanged to the test split.
8. **Predicted regions:** `heat > τ` → remove connected components < 100 px (`scipy.ndimage.label`) → bounding boxes with `scipy.ndimage.find_objects` (no OpenCV).
9. **Localization metrics** on the **test split** (all test images; normal ones have empty masks), computed at 128×128 (nearest for masks, bilinear for heatmaps): pixel ROC-AUC, Dice@τ, IoU@τ. Also the same metrics for **B2 Grad-CAM** (its τ fitted the same way on `loc_val`, using the fine-tuned B2 on those images — B2 was trained on them, state this as a caveat in the figure footnote).
10. **Display normalisation** (per category, per method): `vmin = 1st percentile`, `vmax = 99.5th percentile` of all test-split heatmap pixels of that method/category; colormap `jet`; overlay alpha 0.45 over the *original RGB resized to 256×256*.

### 5.8 Training-free explanation: LBP rarity map (`explain.py`)
For each category, from **training-split normal images only**, compute the mean LBP histogram of each of the 3 scales. For a test image, per-pixel rarity = `−log(h_normal_R1[code_R1] + 1e-6)` averaged over the 3 scales, then `gaussian_filter(sigma=4)`. This needs no classifier and illustrates *why* texture features find defects. It is an explanation figure only (fig38), not a fourth approach, not in metric tables.

### 5.9 Global feature explanations
- **Group permutation importance** (fig39) on the test split for each pipeline's final image-level model: groups for P3 = `LBP_R1, LBP_R2, LBP_R3, GLCM_contrast, GLCM_dissimilarity, GLCM_homogeneity, GLCM_energy, GLCM_correlation, GLCM_ASM`; for P2 the three LBP groups; for P1 one group (so P1 omitted from the plot). Permute the raw feature columns of the group (same row permutation inside the group), `n_repeats=20`, metric drop in F1. Seed 42 only.
- **Per-bin discrimination** (fig40): for each category and each of the 177 LBP bins, Cohen's d between normal and defective train-split feature values.

---

## 6. Repository layout (create exactly this)

```
texture-defect-detection-lbp-svm/
├── main.py                       # CLI entry (currently empty file — fill it)
├── requirements.txt
├── README.md                     # update at the end (§12 step 13)
├── IMPLEMENTATION_PLAN.md        # this file — do not edit
├── .gitignore                    # append lines per §0.1.7
├── configs/
│   ├── textures.yaml             # default (full run)
│   ├── quick.yaml                # smoke test
│   └── extended_ablation_512.yaml# 512-px size ablation
├── scripts/
│   ├── env_d.ps1                 # §0.1.3
│   └── make_notebooks.py         # builds the two .ipynb with nbformat
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── timing.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features_lbp.py
│   ├── features_glcm.py
│   ├── cnn_baselines.py
│   ├── classifier.py
│   ├── evaluate.py
│   ├── localize.py
│   ├── explain.py
│   ├── ablations.py
│   ├── visualize.py              # one function per figure
│   └── report.py                 # index.html gallery of all outputs
├── tests/
│   ├── test_shapes.py
│   ├── test_split.py
│   └── test_localize.py
├── notebooks/
│   ├── 00_dataset_explorer.ipynb
│   └── 01_full_comparison.ipynb
├── docs/                         # existing, untouched
├── dataset/                      # existing, read-only
├── data_cache/                   # git-ignored
└── outputs/                      # git-ignored; the deliverable
    ├── figures/                  # figNN_*.png (300 dpi)
    ├── figures_data/             # figNN_*.csv  (the exact numbers plotted)
    ├── tables/                   # T*.csv
    ├── models/                   # *.joblib, b2_seed42_<cat>.pt
    ├── gallery/<category>/       # per-image result sheets (§11.4)
    ├── logs/                     # run_<timestamp>.log, timings_raw.csv
    └── index.html                # local viewer of everything
```
`outputs_quick/` has the same sub-structure and is used only by `--quick`.

### 6.1 `requirements.txt`
```
numpy>=1.26,<2.3
scipy>=1.11
pandas>=2.0
scikit-image>=0.22
scikit-learn>=1.3
pillow>=10
matplotlib>=3.8
seaborn>=0.13
pyyaml>=6.0
tqdm>=4.66
joblib>=1.3
psutil>=5.9
pytest>=8.0
jupyter>=1.0
ipykernel>=6.0
nbformat>=5.9
nbconvert>=7.0
# CPU torch is installed separately (see §12 step 0):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 6.2 Module responsibilities and required public functions
| File | Must define |
|---|---|
| `config.py` | `@dataclass Config` (all keys of §7.1); `load_config(path)`; resolves all paths to absolute and calls `assert_on_d` |
| `utils.py` | `set_seed`, `assert_on_d`, `check_environment`, `get_logger`, `cache_path(category, kind, size)`, `save_fig(fig, name)` (300 dpi PNG + closes fig), `save_csv_for_fig(df, name)` |
| `timing.py` | `Timer(stage, log)` context manager using `time.perf_counter_ns`; `TimingLog` with `.add(stage, seconds, **tags)` and `.to_csv()`; `peak_rss_mb()` via psutil |
| `data_loader.py` | `discover_categories(root)`, `build_pool(root, category) -> DataFrame[img_id, path, mask_path, label, defect_type, origin]`, `make_split(pool, seed) -> (train_df, test_df)`, `load_mask(row, size)`, `dataset_stats_table()` |
| `preprocessing.py` | `load_rgb`, `preprocess`, `demo_chain` |
| `features_lbp.py` | `lbp_codes`, `lbp_hist`, `extract_p1`, `extract_p2`, `extract_p2_literal`, `lbp_feature_names` |
| `features_glcm.py` | `extract_glcm`, `glcm_matrices(img, d, angle)` (for fig06), `glcm_feature_names` |
| `cnn_baselines.py` | `resnet_features(paths)`, `finetune_resnet(train_df, seed, log) -> (model, stats)`, `predict_resnet`, `gradcam(model, rgb)`, `model_stats(model)` |
| `classifier.py` | `build_pipeline(name, seed)`, `PARAM_GRID`, `train_model(X, y, name, seed, log) -> (best_pipeline, info_dict)` |
| `evaluate.py` | `metrics(y, score, pred)`, `measure_latency(...)` (§7.3), `evaluate_all(...)` |
| `localize.py` | everything in §5.7: `window_features`, `train_patch_model`, `heatmap`, `fit_threshold`, `regions`, `localization_metrics` |
| `explain.py` | `rarity_map`, `group_permutation_importance`, `bin_cohens_d` |
| `ablations.py` | A1–A6 (§9) |
| `visualize.py` | `fig01_...` … `fig43_...` as listed in §11 |
| `report.py` | `build_index_html()` — a static HTML page that lists every figure (with caption) and table |

### 6.3 Feature cache (the biggest speed lever)
- File: `data_cache/<category>_<kind>_<size>.npz` with arrays `ids` (str), `X` (float32, n×d), `t_read`, `t_pre`, `t_feat` (float64 seconds per image, measured single-threaded at extraction time), `names`.
- `kind ∈ {p1, p2, p3raw, p2literal, glcm, resnet18}`. (`p3raw` = 201 columns; `p1`, `p2` may also be derived by slicing but each must be stored with its **own measured extraction time**: P1 time = scale (8,1) only; P2 = three scales; P3raw = three scales + GLCM; i.e. times are not shared. Implement by timing each extractor in its own loop pass.)
- `extract_cached(category, kind, size, df)` loads the cache if all ids are present, otherwise computes the missing ones with `tqdm` and updates the file. `--rebuild-cache` deletes the cache for the given config first. **The first full run must be a cold run** (no cache), because training-time figures are built from the stored per-image times.

---

## 7. Fair timing protocol (this is what proves the "faster" claim)

### 7.1 Config (`configs/textures.yaml`, exact content)
```yaml
dataset_root: "D:/MV Project/texture-defect-detection-lbp-svm/dataset/MVTec AD"
output_root:  "D:/MV Project/texture-defect-detection-lbp-svm/outputs"
cache_root:   "D:/MV Project/texture-defect-detection-lbp-svm/data_cache"
categories: [carpet, grid, leather, tile, wood]
seeds: [42, 43, 44, 45, 46]
b2_seeds: [42]
image_size: 256
test_size: 0.30
cv_folds: 5
glcm_levels: 32
mi_k: 100
cnn:
  input_size: 224
  epochs: 10
  batch_size: 16
  lr: 0.0001
  weight_decay: 0.0001
patch:
  window: 64
  stride_train: 32
  stride_infer: 16
  pos_frac: 0.25
  neg_ratio: 3
  loc_val_frac: 0.20
  heat_sigma: 8
latency:
  n_images: 200
  warmup: 20
  sample_seed: 0
gallery_seed: 42
run_ablations: true
run_gallery: true
```
`quick.yaml` = same but `output_root: ".../outputs_quick"`, `categories: [carpet, leather]`, `seeds: [42]`, `cnn.epochs: 1`, `run_ablations: false`, `run_gallery: false`. `extended_ablation_512.yaml` = same as textures.yaml but `image_size: 512`, `seeds: [42]`, `b2_seeds: []`, `output_root: ".../outputs/size512"`, `cache_root: ".../data_cache"` (cache keys contain the size so no clash).

### 7.2 Training time (per category, per pipeline, per seed) — stored in `T3`
- **P1/P2/P3:** `train_seconds = Σ_{train images}(t_read-excluded: t_pre + t_feat) + fit_seconds(GridSearchCV incl. scaler/selector inside)`. Read/decode time is reported as its own column (identical across methods) and **not** added, so the comparison is about the algorithm, not disk.
- **B1:** `Σ (t_pre + resnet forward time) + fit_seconds`.
- **B2:** `finetune_seconds` (all epochs, incl. its own preprocessing/augmentation). No feature cache applies.
- Headline "total training time" = sum over the 5 categories, averaged over seeds. Per-stage breakdown columns: `preprocess, features, scaler_selector_svm_grid, cnn_forward, cnn_finetune`.

### 7.3 Inference latency — stored in `T4`
- Pooled sample: 200 test-split images (seed-42 split, 5 categories pooled), chosen with `np.random.default_rng(sample_seed)`. Images are **pre-loaded as uint8 RGB arrays in RAM** (decode time excluded and reported separately as `t_read_ms`).
- For each method a single function `predict_one(rgb_uint8, models[category]) -> (label, score)` that does *everything*: preprocess → features → scaler/selector → SVM decision (or CNN preprocess → forward → softmax).
- Protocol: batch size 1, 1 CPU thread (`torch.set_num_threads(1)`, env threads = 1), `warmup=20` calls discarded, then 200 timed calls with `time.perf_counter_ns`. Report **median, mean, p95 (ms)**, `FPS = 1000/median`, and a stage breakdown (preprocess / features / scaler+selector / svm or cnn forward) measured by timing each stage on the same 200 images.
- Extra B2 row: "B2 (all CPU threads)" with `torch.set_num_threads(os.cpu_count())`, to show the comparison is not an artefact of single-threading. Label it clearly.
- `peak_rss_mb`: process RSS peak (psutil) measured in a fresh subprocess per method running the 200-image loop.
- Record CPU model (`platform.processor()`), core count, RAM, Python, numpy, sklearn, torch versions into `outputs/tables/T0_environment.csv`. Figures 25–29 must print the CPU name in the footer.

### 7.4 Honest-reporting rules
- Show both "total time incl. one-off feature extraction" and "retrain time on cached features" (the latter = `fit_seconds` only) in fig25's second panel, labelled.
- Never plot a speed-up without the matching F1 next to it (fig28/fig29 exist for that purpose).

---

## 8. Metrics & statistics (`evaluate.py`)
Per (category, method, seed) on the test split: `accuracy, balanced_accuracy, precision, recall, f1, roc_auc, pr_auc (average_precision), mcc, TN, FP, FN, TP`. Aggregates: macro mean over categories, then mean ± std over seeds (std across seeds of the macro mean). Per-defect-type recall (fig18) = recall restricted to test rows of that defect type.
Statistical tests (`T9_stat_tests.csv`): paired Wilcoxon signed-rank (two-sided) on the 25 (category×seed) F1 values for **P2 vs P1** and **P3 vs P2** and **P3 vs B1**; report W, p, mean difference. B2 is excluded (one seed). With `--quick` the test is skipped and the file says `n<... skipped`.
Targets table `T0_targets.csv`: for each proposal target (F1 > 0.90 per texture category for best pipeline; latency < 30 ms for P1–P3; P2−P1 F1 gain in [+0.08, +0.15]) write `value, target, met (yes/no)`. Plain facts, no spin.

---

## 9. Ablations (`ablations.py`, all use cached features → cheap; seed list = main seeds; B2 excluded)
| ID | What | Output |
|---|---|---|
| A1 | LBP scales: {R1}, {R2}, {R3}, {R1+R2}, {R1+R2+R3 (=P2)}, and proposal-literal 54-d (8,1)(16,2)(24,3) `uniform` | fig21 |
| A2 | Feature groups: LBP-only(P2), GLCM-only(24), P2+GLCM all 201 (no selection), P3 (MI k=100) | fig22 (left) |
| A3 | Selector variants on 201: none, SelectKBest-MI k=100 (=P3), PCA(0.95 variance) | fig22 (right) |
| A4 | MI k sweep: k ∈ {10,25,50,100,150,201}: F1 and inference ms for each | fig23 |
| A5 | Image size 256 vs 512 (P1–P3, seed 42): F1 and latency | fig31 (needs the 512 run; if `outputs/size512` is missing the function logs a warning and skips) |
| A6 | Learning curve: train fraction ∈ {0.1,0.25,0.5,0.75,1.0} of train split (stratified, seeded): F1 and `fit_seconds` for P1,P2,P3,B1 | fig24 |
All use the same §5.5 grid and the same test split. Save raw numbers to `outputs/tables/T7_ablations.csv` (columns: `ablation, variant, category, seed, f1, roc_auc, dim, fit_seconds, latency_ms`).

---

## 10. Tables (all in `outputs/tables/`, CSV, UTF-8, `.4f` floats)
| File | Content |
|---|---|
| `T0_environment.csv` | machine/library versions; `T0_targets.csv` targets met/not met |
| `T1_metrics_per_method.csv` | rows P1,P2,P3,B1,B2; columns each metric as `mean±std` (macro over categories, over seeds) + raw mean/std columns |
| `T2_metrics_per_category.csv` | category × method × metric (mean, std over seeds) |
| `T3_training_time.csv` | per (category, method, seed): stage seconds + total; plus aggregate rows |
| `T4_inference_latency.csv` | method, median/mean/p95 ms, FPS, stage breakdown, peak_rss_mb, t_read_ms |
| `T5_localization_metrics.csv` | category × method {P1,P2,P3,B2-GradCAM}: pixel AUROC, Dice, IoU, τ |
| `T6_best_params.csv` | best kernel/C/gamma, best CV-F1, n_support_vectors per (category, method, seed) |
| `T7_ablations.csv` | §9 |
| `T8_resources.csv` | feature dim, n_params (CNN) / n_support_vectors (SVM), model file size MB, GFLOPs (CNN only, else `N/A`), GPU required (no/no/no/no/no — all CPU), training data needed |
| `T9_stat_tests.csv` | §8 |
| `T10_dataset_stats.csv` | per category: n_normal, n_defect, per-defect-type counts, train/test split counts |
| `split_manifest_seed<seed>.csv` | per image: id, category, label, defect_type, split |

---

## 11. Figures (every one mandatory) — filenames are exact, saved in `outputs/figures/` at 300 dpi, with the plotted numbers also saved as `outputs/figures_data/<same name>.csv` (for image-only figures save the list of image ids used)

### 11.0 Style rules (apply to all)
- Method colours (colour-blind safe, fixed everywhere): **P1 `#0072B2`, P2 `#E69F00`, P3 `#009E73`, B1 `#CC79A7`, B2 `#D55E00`**; normal = `#4C78A8` grey-blue, defective = `#E45756`. GT contours = lime `#32CD32`; predicted boxes = red `#FF0000`.
- Font sizes: title 14, axis labels 12, ticks 10, annotations 9. Every axis has label **with units**; every figure has a title; legends never cover data; value labels on bars. Image panels have no ticks, small caption on top (e.g. "P3 heatmap").
- Error bars = std over seeds (omit for B2). Footer line in small grey text: `CPU: <name> | 1 thread | n_seeds=<k>`.
- Size: single plots 8×5 in; multi-panel figures sized by grid at ~3 in per image panel.

### 11.1 Figure list
| File | Content (exact) |
|---|---|
| **Dataset & preprocessing** | |
| `fig01_dataset_montage.png` | 5 rows (categories) × 5 cols: [normal \| defect type A \| same with GT contour \| defect type B \| same with GT contour]; A,B = first two defect types alphabetically, image = first sorted test file |
| `fig02_class_balance_split.png` | stacked bars per category: train normal/defect, test normal/defect (seed 42); counts annotated |
| `fig03_preprocessing_chain.png` | one carpet defect image, 7 panels: original RGB, gray, resized 256, Gaussian, min–max, LBP(8,1) code map, LBP(8,3) code map |
| `fig04_lbp_maps_hist.png` | per category row: normal image, defect image, their LBP(8,1) maps, and overlaid mean histogram (normal vs defect, train split, 59 bins) |
| `fig05_multiscale_histograms.png` | 5 rows × 3 cols (R=1,2,3): mean normal vs mean defect histogram with ±1 std band |
| `fig06_glcm_matrices.png` | 5 rows × 4 cols: [normal θ=0°, defect θ=0°, normal θ=90°, defect θ=90°] GLCM (d=1, 32×32, log-scaled colour) |
| `fig07_glcm_haralick_delta.png` | grouped bars per category: relative change (defect mean − normal mean)/std for the 6 Haralick props (angle-averaged) |
| `fig08_pca_variance_mi_top20.png` | left: cumulative explained variance of the 201 standardized features (95 % line + component count); right: top-20 MI scores (feature names) — pooled over the 5 categories, seed 42 train split |
| `fig09_feature_space_pca2d.png` | 5 rows × 3 cols (P1,P2,P3 standardized features): 2-D PCA scatter normal vs defect (train split) |
| **Classification results** | |
| `fig10_confusion_P1.png`, `fig11_confusion_P2.png`, `fig12_confusion_P3.png` | 6 panels each (5 categories + pooled): row-normalised confusion matrix with raw counts, seed 42 |
| `fig13_confusion_B1_B2.png` | same layout, B1 on top row, B2 on bottom row |
| `fig14_roc_curves.png` | 2×3 panels (5 categories + pooled); five method curves, AUC in legend, seed 42 |
| `fig15_pr_curves.png` | same layout, precision–recall, AP in legend, prevalence line |
| `fig16_percategory_f1_bar.png` | grouped bars: F1 per category (5 groups × 5 methods) + macro group; error bars; dashed line at 0.90 target |
| `fig17_metric_summary_heatmap.png` | annotated heatmap, rows = methods, cols = accuracy, bal-acc, precision, recall, F1, AUC, MCC (macro mean) |
| `fig18_defecttype_recall_heatmap.png` | rows = `category/defect_type` (25), cols = 5 methods, cell = recall (mean over seeds) |
| `fig19_seed_variability_box.png` | box + points: macro F1 per seed for P1,P2,P3,B1 (B2 as single marker) |
| `fig20_score_distributions.png` | 5 rows × 3 cols (P1,P2,P3): histogram/violin of SVM decision values, normal vs defect, threshold line at 0 |
| **Ablations** | |
| `fig21_ablation_scales.png` | A1 bars (F1) + dimension annotation |
| `fig22_ablation_features_selector.png` | A2 (left) and A3 (right) bars |
| `fig23_ablation_k_sweep.png` | A4: F1 vs k (left axis) and inference ms vs k (right axis) |
| `fig24_learning_curve.png` | A6: two panels — F1 vs train fraction; fit seconds vs train fraction (log-y), 4 methods |
| **Efficiency (the main claim)** | |
| `fig25_training_time_bars.png` | **horizontal stacked bars**, one per method (P1,P2,P3,B1,B2), stages coloured, total seconds written at bar end; left panel linear x, right panel log x; second pair of bars "retrain on cached features" for P1–P3,B1 |
| `fig26_inference_latency_bars.png` | stacked stage bars (ms/img) per method (B2 single-thread and B2 all-threads rows), vertical dashed line at 30 ms, FPS annotated, log x |
| `fig27_speedup_bars.png` | for P1,P2,P3,B1: `×` speed-up vs B2 for training time and for inference latency (two panels, log y, values labelled) |
| `fig28_accuracy_vs_latency.png` | scatter: x = latency (ms, log), y = macro F1; marker area ∝ model size (MB); all 5 methods labelled; Pareto frontier line |
| `fig29_accuracy_vs_traintime.png` | same with x = total train seconds (log) |
| `fig30_resource_table.png` | rendered table image from T8 (dim, params/SVs, size MB, GFLOPs, RAM MB, GPU needed) |
| `fig31_size_ablation_256_512.png` | A5: F1 and latency for 256 vs 512 for P1–P3 |
| **Explainability / localization** | |
| `fig32_localization_carpet.png` … `fig36_localization_wood.png` (5 files, one per category: carpet, grid, leather, tile, wood) | 6 rows × 7 cols. Rows = 6 test-split defect images (seed 42), chosen round-robin over defect types in alphabetical order, taking the first sorted image of each type not yet used. Cols: ① original ② GT mask contour (lime) ③ P1 heatmap ④ P2 heatmap ⑤ P3 heatmap ⑥ B2 Grad-CAM ⑦ P3 overlay with predicted red boxes + GT lime contour + text `P3 score=…, pred=Defective/Normal`. Column headers on the top row; footnote about B2 caveat (§5.7.9) |
| `fig37_localization_metrics.png` | grouped bars (pixel AUROC, Dice, IoU) per category × {P1,P2,P3,B2 Grad-CAM} |
| `fig38_lbp_rarity_maps.png` | 10 rows (2 defect images per category) × 4 cols: original, rarity map, overlay, GT contour |
| `fig39_permutation_importance.png` | horizontal bars of F1 drop per feature group, panels for P2 and P3 (pooled over categories, seed 42), error bars over 20 repeats |
| `fig40_lbp_bin_discrimination.png` | heatmap: rows = categories, cols = 177 LBP bins (scale separators), colour = Cohen's d |
| `fig41_failure_cases_P3.png` | top 8 false negatives and top 8 false positives of P3 pooled (seed 42) ranked by |score|: image with P3 heatmap overlay, GT contour, score, category/defect type in caption |
| `fig42_rescued_by_P3.png` | up to 8 test images wrong for P1 but right for P3 (defects only; prefer largest score gain): P1 heatmap vs P3 heatmap overlays side by side |
| `fig43_pipeline_diagram.png` | block diagram drawn with matplotlib patches: P1/P2/P3 path (image → preprocess → LBP/GLCM → scaler → selector → SVM → label + heatmap) and B2 path (image → ResNet-18 → softmax), each block annotated with its measured median ms from T4 |

### 11.2 Required image conventions
- Original image always resized to 256×256 RGB for overlays. Heatmaps per §5.7.10.
- Every localization figure states in the footer: `window 64, stride 16; τ fitted on held-out train images`.

### 11.3 Tables-as-images
Only `fig30`. All other tables are CSV (and shown in notebook 01 via `pandas`).

### 11.4 Per-image gallery (`run_gallery: true`)
For **every test-split image** of every category (seed 42) save `outputs/gallery/<category>/<defect_type>__<stem>__<TP|FP|FN|TN>.png` (120 dpi, 1×5 panels): original | GT contour | P3 heatmap overlay with red boxes | B2 Grad-CAM | text panel listing P1, P2, P3, B1, B2 predictions and scores, and ms/img for P3 vs B2. Outcome tag refers to **P3**. ≈ 534 files. Also write `outputs/gallery/index.csv` (path, outcome, per-method prediction).

### 11.5 `outputs/index.html`
`report.build_index_html()` writes a single static HTML page (relative links, no external resources) with: T1 and T4 as HTML tables, then every figure in numeric order with its caption, then links to gallery folders. Open it in a browser = the whole result in one place.

---

## 12. Build order — implement strictly in this order; every step has an acceptance check

0. **Environment** — create `SYS` folders, `scripts/env_d.ps1`, venv at `ROOT\.venv`, `. .\scripts\env_d.ps1`, `pip install -r requirements.txt`, `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`. Tell the user to run the `setx CLAUDE_CODE_TMPDIR` line (§0.1.6). *Check:* `python -c "import sys,tempfile;print(sys.prefix, tempfile.gettempdir())"` prints D: paths only; `import skimage, sklearn, torch, torchvision` works.
1. **Scaffold** — folders, empty modules with docstrings, `configs/*.yaml` (content per §7.1), `.gitignore` lines, `config.py`, `utils.py` (incl. `assert_on_d`, `check_environment`), `timing.py`. *Check:* `pytest tests -q` skeleton passes; `check_environment()` aborts if `TEMP` is set to a C: path (test it once by temporarily overriding).
2. **Data** — `data_loader.py`, `T10`, split manifests, `tests/test_split.py`. *Check:* pool sizes 397/342/369/347/326, total 1,781, 1,399 normal / 382 defective, no train/test overlap, masks found for all 382 defective images.
3. **Preprocessing + fig01–fig03.** *Check:* output dtype uint8, shape (256,256), values span 0–255.
4. **Features** — LBP, GLCM, cache. *Check (tests/test_shapes.py):* P1 → (59,), P2 → (177,), literal → (54,), GLCM → (24,), P3raw → (201,); histograms sum to 1; rerun with warm cache returns identical arrays; cache files exist under `data_cache` on D:. Generate fig04–fig07.
5. **Classifier** — §5.5, run P1, P2, P3 for seed 42 on one category first. *Check:* grid has 16 configs; `fit_seconds` recorded; model saved to `outputs/models`. Then fig08, fig09.
6. **Evaluation (P1–P3 all categories & seeds)** — T1, T2, T3(P part), T6, T9. Figures fig10–12, fig14–fig20.
7. **CNN baselines** — B1 (all seeds), then B2 (seed 42). Add to all tables and the figures that include B1/B2 (fig13 and re-render fig14–fig20). *Check:* B2 epoch losses logged, finetune seconds recorded, Grad-CAM shape (256,256).
8. **Latency & resources** — T4, T8, `T0_environment`, fig25–fig30. *Check:* 200 timed calls per method; median < p95; CPU name printed in footers.
9. **Localization** — §5.7 for P1–P3 + Grad-CAM, T5, fig32–fig37. *Check (tests/test_localize.py):* 64×64 window count at stride 16 on 256×256 = 169; heatmap shape (256,256) float32; Dice/IoU in [0,1]; pixel AUROC finite for every category.
10. **Explainability extras** — fig38, fig39, fig40, fig41, fig42.
11. **Ablations** — A1–A4, A6 (+ A5 via `python main.py --config configs/extended_ablation_512.yaml`), T7, fig21–fig24, fig31, fig43, T0_targets.
12. **Gallery + index.html** — §11.4, §11.5.
13. **Notebooks + README** — `scripts/make_notebooks.py` builds:
    - `00_dataset_explorer.ipynb`: runs in < 2 min; shows T10, fig01–fig07.
    - `01_full_comparison.ipynb`: reads `outputs/tables/*.csv` and displays *saved* figures with `IPython.display.Image`, with a markdown paragraph of interpretation per section (accuracy, efficiency, explainability, ablations, limitations). **It must not retrain anything.**
    Execute both with `jupyter nbconvert --to notebook --execute --inplace` (env script sourced, so Jupyter dirs are on D:). Update `README.md`: overview, how to run (§13), the results table T1/T4 copied from the actual CSVs, project structure, and the limitations list.
14. **Final verification** — run the full command (§13). Then run the checklist in §15 and report to the user: list of files created, T1 and T4 numbers, targets met/not met, anything skipped, total wall time.

---

## 13. Run commands (PowerShell, from `ROOT`)
```powershell
. .\scripts\env_d.ps1
.\.venv\Scripts\Activate.ps1
python -m pytest tests -q
python main.py --config configs/quick.yaml               # smoke test, target < 10 min, writes outputs_quick/
python main.py --config configs/textures.yaml --rebuild-cache   # full cold run (all figures, tables, gallery)
python main.py --config configs/extended_ablation_512.yaml      # A5 only (size ablation)
python scripts/make_notebooks.py
jupyter nbconvert --to notebook --execute --inplace notebooks/00_dataset_explorer.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/01_full_comparison.ipynb
```
`main.py` arguments (exactly these): `--config PATH` (required), `--rebuild-cache`, `--stage {all,data,features,train,eval,latency,localize,explain,ablate,figures,gallery,report}` (default `all`; each stage must be resumable and read what earlier stages saved), `--skip-cnn` (omit B1/B2 everywhere; figures drop their columns), `--seeds` (override list, e.g. `--seeds 42`). `main.py` must print the stage name, a `tqdm` bar for every loop (desc such as `LBP-P2 carpet/train`, unit `img`) and, at the end of each stage, elapsed seconds. Everything is also written to `outputs/logs/run_<timestamp>.log`.

Expected scale (informational, not a requirement): cold feature extraction for 1,781 images × P1/P2/P3raw/ResNet is a few to ~20 minutes on a laptop CPU; SVM grids are seconds–minutes; B2 fine-tuning ≈ tens of minutes for 5 categories (this slowness *is* a result); gallery ≈ a few minutes.

---

## 14. Limitations to state in README/notebook (do not hide)
1. Supervised binary re-formulation of an unsupervised benchmark (D6); our split differs from the official protocol, so numbers are **not comparable** to published MVTec AD leaderboards.
2. Textures only; object classes (cable, screw, transistor…) are expected to be poor for pure texture features.
3. Image-level SVM does not localize; the region maps come from separate patch SVMs trained on GT masks of the training images and are a localization *proxy*, not segmentation SOTA.
4. B2 Grad-CAM thresholds were fitted on images B2 had trained on (§5.7.9).
5. B2 evaluated on one seed only; B1/P-pipelines on five.
6. 256-px primary size deviates from the proposal's 512 (D4) and LBP bin layout deviates from the proposal's literal (D2); both quantified by ablations A5/A1.
7. Timings are machine-specific, single-thread, image decode excluded.
8. Small test sets per category (e.g. wood: 19 normal images in total) → wide confidence; this is why 5 seeds are used.

---

## 15. Definition of done (check every line before reporting completion)
- [ ] Nothing was written to C: (verify: no `C:` path in logs, config, or any saved file; `check_environment()` passes).
- [ ] `pytest tests -q` all green.
- [ ] All 43 figure files exist in `outputs/figures/` (fig31 may be absent only if the 512 run was not executed — then say so), each with a matching CSV or image-id list in `figures_data/`.
- [ ] Tables T0…T10 and split manifests exist and are non-empty.
- [ ] `outputs/gallery/` contains one PNG per seed-42 test image of the 5 categories and `index.csv`.
- [ ] `outputs/index.html` opens locally and lists everything.
- [ ] Both notebooks executed with embedded outputs, no retraining inside.
- [ ] README updated with real numbers copied from T1/T4 only.
- [ ] The final message to the user lists: files created, headline table (F1, AUC, train s, ms/img, FPS for P1/P2/P3/B1/B2), which proposal targets were met/not met, deviations D1–D7 recap, and anything skipped with the reason.

---

## 16. Manual terminal commands (for the user to run and inspect each step by hand)

> Added at the user's request. Every block assumes PowerShell opened in `ROOT` (`D:\MV Project\texture-defect-detection-lbp-svm`). **Always run the first three lines in every new terminal.** Nothing here touches `C:`.

### 16.0 One-time setup (run once, then restart Claude Code)
```powershell
setx CLAUDE_CODE_TMPDIR "D:\MV Project\_sys\tmp"      # moves Claude Code's temp off C: (§0.1.6)
```

### 16.1 Every new terminal (mandatory)
```powershell
cd "D:\MV Project\texture-defect-detection-lbp-svm"
. .\scripts\env_d.ps1                  # sets TEMP/TMP/caches/threads to D: and prints them
.\.venv\Scripts\Activate.ps1           # activate the venv (after step 0 has created it)
```

### 16.2 Per-step commands to see the result yourself
| Step | Run | What you should see |
|---|---|---|
| 0 Environment | `python -c "import sys,tempfile;print(sys.prefix, tempfile.gettempdir())"` ; `python -c "import skimage, sklearn, torch, torchvision; print(torch.__version__, torch.cuda.is_available())"` | only `D:` paths; torch CPU version, `False` |
| 1 Scaffold | `python -m pytest tests -q` ; `python -c "from src.utils import check_environment; check_environment(); print('env ok')"` ; `$env:TEMP='C:\Windows\Temp'; python -c "from src.utils import check_environment; check_environment()"` (must abort — then re-run `. .\scripts\env_d.ps1`) | tests pass; `env ok`; abort message for the C: case |
| 2 Data | `python main.py --config configs/quick.yaml --stage data` ; `python -m pytest tests/test_split.py -q` ; `Import-Csv outputs_quick\tables\T10_dataset_stats.csv \| Format-Table` | pool sizes 397/342/369/347/326, split manifests in `tables/` |
| 3 Preprocessing | `python main.py --config configs/quick.yaml --stage figures` ; `explorer outputs_quick\figures` | fig01–fig03 PNGs open in Explorer |
| 4 Features | `python main.py --config configs/quick.yaml --stage features` ; `python -m pytest tests/test_shapes.py -q` ; `Get-ChildItem data_cache` | `.npz` caches on D:, shapes 59/177/54/24/201 |
| 5 Classifier | `python main.py --config configs/quick.yaml --stage train` ; `Get-ChildItem outputs_quick\models` | `*.joblib` models, fit seconds in the log |
| 6 Evaluation | `python main.py --config configs/quick.yaml --stage eval` ; `Import-Csv outputs_quick\tables\T1_metrics_per_method.csv \| Format-Table` | T1/T2/T3/T6 tables, fig10–fig20 |
| 7 CNN | `python main.py --config configs/quick.yaml --stage train` (B1/B2 included unless `--skip-cnn`) | B2 epoch losses printed, `b2_seed42_<cat>.pt` |
| 8 Latency | `python main.py --config configs/quick.yaml --stage latency` ; `Import-Csv outputs_quick\tables\T4_inference_latency.csv \| Format-Table` | median < p95, FPS, CPU name in fig25–29 footers |
| 9 Localization | `python main.py --config configs/quick.yaml --stage localize` ; `python -m pytest tests/test_localize.py -q` | fig32–fig37, T5 |
| 10 Explain | `python main.py --config configs/quick.yaml --stage explain` | fig38–fig42 |
| 11 Ablations | `python main.py --config configs/textures.yaml --stage ablate` ; `python main.py --config configs/extended_ablation_512.yaml` | T7, fig21–fig24, fig31, fig43 |
| 12 Gallery + report | `python main.py --config configs/textures.yaml --stage gallery` ; `python main.py --config configs/textures.yaml --stage report` ; `start outputs\index.html` | browser page with everything |
| 13 Notebooks | `python scripts/make_notebooks.py` ; `jupyter nbconvert --to notebook --execute --inplace notebooks/00_dataset_explorer.ipynb` ; same for `01_full_comparison.ipynb` ; `jupyter notebook` | executed notebooks |
| 14 Final | full commands of §13, then `python -m pytest tests -q` | checklist §15 |

### 16.3 Handy inspection commands (any time)
```powershell
explorer outputs\figures                         # browse figures
explorer outputs\gallery                         # browse per-image result sheets
Import-Csv outputs\tables\T1_metrics_per_method.csv | Format-Table -AutoSize
Import-Csv outputs\tables\T4_inference_latency.csv  | Format-Table -AutoSize
Get-Content outputs\logs\run_*.log -Tail 40      # latest log lines
Get-ChildItem outputs\figures | Measure-Object   # figure count (expect 43)
git status                                       # nothing is ever committed without your request
```
Quick smoke test (< 10 min): `python main.py --config configs/quick.yaml` → results in `outputs_quick\`.
Full cold run: `python main.py --config configs/textures.yaml --rebuild-cache` → results in `outputs\`.

*End of plan.*
