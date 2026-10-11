"""Build the two notebooks with nbformat. Notebooks only display saved outputs; they never retrain."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)

SETUP = '''import sys, pathlib
import pandas as pd
from IPython.display import Image, display, Markdown
ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
OUT = ROOT / "outputs"
T = OUT / "tables"
def show(name):
    p = OUT / "figures" / (name + ".png")
    display(Image(filename=str(p))) if p.exists() else print("missing:", p)
'''


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_code_cell(c) if k == "code" else nbf.v4.new_markdown_cell(c) for k, c in cells]
    nbf.write(nb, NB / name)


build("00_dataset_explorer.ipynb", [
    ("md", "# Dataset explorer\nMVTec AD textures (carpet, grid, leather, tile, wood), the split, preprocessing, "
           "LBP and GLCM features. Everything here is read from `outputs/`; no feature or SVM logic lives in "
           "this notebook."),
    ("code", SETUP),
    ("code", 'pd.read_csv(T / "T10_dataset_stats.csv")'),
    ("md", "## Montage and class balance"), ("code", 'show("fig01_dataset_montage"); show("fig02_class_balance_split")'),
    ("md", "## Preprocessing chain"), ("code", 'show("fig03_preprocessing_chain")'),
    ("md", "## LBP"), ("code", 'show("fig04_lbp_maps_hist"); show("fig05_multiscale_histograms")'),
    ("md", "## GLCM"), ("code", 'show("fig06_glcm_matrices"); show("fig07_glcm_haralick_delta")'),
])

build("01_full_comparison.ipynb", [
    ("md", "# Full comparison: P1 / P2 / P3 vs B1 / B2\nThis notebook only reads saved tables and figures. The "
           "numbers printed below come from `outputs/tables`; interpret them honestly - a target that is not met "
           "is reported as such."),
    ("code", SETUP),
    ("md", "## 1. Accuracy\nF1 of the defect class, ROC-AUC and balanced accuracy are the headline metrics because the "
           "test set is ~78 % normal. Compare P1 -> P2 -> P3 (hypothesis H1) and the CNN references."),
    ("code", 'pd.read_csv(T / "T1_metrics_per_method.csv").loc[:, lambda d: ~d.columns.str.endswith(("_mean","_std"))]'),
    ("code", 'display(pd.read_csv(T / "T2_metrics_per_category.csv").head(30)); display(pd.read_csv(T / "T9_stat_tests.csv"))'),
    ("code", 'for f in ["fig10_confusion_P1","fig11_confusion_P2","fig12_confusion_P3","fig13_confusion_B1_B2","fig14_roc_curves","fig15_pr_curves","fig16_percategory_f1_bar","fig17_metric_summary_heatmap","fig18_defecttype_recall_heatmap","fig19_seed_variability_box","fig20_score_distributions"]: show(f)'),
    ("md", "## 2. Efficiency\nTraining time excludes image decoding; inference is batch size 1 on one CPU thread. Never "
           "read a speed-up without the matching F1 (figs 28/29)."),
    ("code", 'display(pd.read_csv(T / "T3_training_time.csv").groupby(["method"]).mean(numeric_only=True)); display(pd.read_csv(T / "T4_inference_latency.csv")); display(pd.read_csv(T / "T8_resources.csv")); display(pd.read_csv(T / "T0_targets.csv"))'),
    ("code", 'for f in ["fig25_training_time_bars","fig26_inference_latency_bars","fig27_speedup_bars","fig28_accuracy_vs_latency","fig29_accuracy_vs_traintime","fig30_resource_table"]: show(f)'),
    ("md", "## 3. Explainability\nHeatmaps come from separate patch-level SVMs (a localization proxy); Grad-CAM thresholds "
           "for B2 were fitted on images B2 had trained on."),
    ("code", 'display(pd.read_csv(T / "T5_localization_metrics.csv")); [show(f"fig3{i}_localization_{c}") for i, c in zip(range(2,7), ["carpet","grid","leather","tile","wood"])]; [show(f) for f in ["fig37_localization_metrics","fig38_lbp_rarity_maps","fig39_permutation_importance","fig40_lbp_bin_discrimination","fig41_failure_cases_P3","fig42_rescued_by_P3","fig43_pipeline_diagram"]]'),
    ("md", "## 4. Ablations"),
    ("code", 'for f in ["fig21_ablation_scales","fig22_ablation_features_selector","fig23_ablation_k_sweep","fig24_learning_curve","fig31_size_ablation_256_512"]: show(f)'),
    ("md", "## 5. Limitations\n1. Supervised re-formulation of an unsupervised benchmark; own 70/30 split, numbers are not "
           "comparable to leaderboards.\n2. Textures only.\n3. Image-level SVM does not localize; patch SVMs are a proxy.\n"
           "4. B2 Grad-CAM thresholds fitted on training images of B2.\n5. B2 evaluated on one seed.\n"
           "6. 256-px size and LBP bin layout deviate from the proposal (ablations A5/A1 quantify this).\n"
           "7. Timings are machine-specific, single-thread, decode excluded.\n8. Small per-category test sets."),
])
print("notebooks written to", NB)
