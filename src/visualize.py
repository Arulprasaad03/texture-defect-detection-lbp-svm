"""All 43 figures (plan §11). Stage 'figures'. Each fig function reads saved tables/caches; nothing is retrained
except the cheap PCA/MI diagnostics of fig08/fig09."""
from __future__ import annotations

import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.feature_selection import SelectKBest, mutual_info_classif  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,  # noqa: E402
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .data_loader import build_pool, load_mask, make_split  # noqa: E402
from .explain import normal_mean_hists, rarity_map  # noqa: E402
from .features_glcm import PROPS, glcm_feature_names, glcm_matrices  # noqa: E402
from .features_lbp import extract_cached, feature_names, lbp_codes  # noqa: E402
from .localize import regions  # noqa: E402
from .preprocessing import demo_chain, load_rgb, preprocess  # noqa: E402
from .utils import output_root, save_csv_for_fig, save_fig  # noqa: E402

COL = {"P1": "#0072B2", "P2": "#E69F00", "P3": "#009E73", "B1": "#CC79A7", "B2": "#D55E00"}
NORMAL_C, DEFECT_C, LIME, RED = "#4C78A8", "#E45756", "#32CD32", "#FF0000"
ALL_M = ["P1", "P2", "P3", "B1", "B2"]
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 9})


class Ctx:
    def __init__(self, cfg):
        self.cfg, self.seed, self.out = cfg, cfg.seeds[0], output_root()
        self.tbl = self.out / "tables"
        self._split, self._heat = {}, {}
        self.preds = pd.read_csv(self.tbl / "predictions.csv")
        self.raw = pd.read_csv(self.tbl / "T2_raw_per_seed.csv")
        self.methods = [m for m in ALL_M if m in set(self.raw.method)]
        self.p42 = self.preds[self.preds.seed == self.seed]

    def table(self, name):
        return pd.read_csv(self.tbl / name)

    def split(self, cat):
        if cat not in self._split:
            pool = build_pool(self.cfg.dataset_root, cat)
            self._split[cat] = (pool,) + make_split(pool, self.seed, self.cfg.test_size)
        return self._split[cat]

    def feats(self, cat, kind, df):
        return extract_cached(cat, kind, self.cfg.image_size, df, self.cfg.cache_root)[0]

    def heat(self, cat):
        if cat not in self._heat:
            z = np.load(self.out / "heatmaps" / f"{cat}.npz", allow_pickle=True)
            d = {k: z[k] for k in z.files}
            d["idx"] = {i: n for n, i in enumerate(d["ids"])}
            self._heat[cat] = d
        return self._heat[cat]

    @property
    def cats(self):
        return list(self.cfg.categories)


# ---------------------------------------------------------------- helpers
def rgb256(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB").resize((256, 256), Image.BILINEAR))


def mask256(row):
    return load_mask(row, 256)


def contour(ax, mask, color=LIME):
    if mask.any():
        ax.contour(mask, levels=[0.5], colors=color, linewidths=1.2)


def noaxis(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def overlay(ax, rgb, heat, vmin, vmax):
    ax.imshow(rgb)
    ax.imshow(heat, cmap="jet", alpha=0.45, vmin=vmin, vmax=vmax)


def prange(arr):
    return float(np.percentile(arr, 1)), float(np.percentile(arr, 99.5))


def blank(ax, text="n/a"):
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9, color="grey")
    noaxis(ax)


def finish(fig, name, df):
    save_csv_for_fig(df, name)
    save_fig(fig, name)


def ids_df(ids):
    return pd.DataFrame({"img_id": list(ids)})


def first_by_type(df, t):
    return df[df.defect_type == t].sort_values("img_id").iloc[0]


# ---------------------------------------------------------------- dataset & preprocessing
def fig01(ctx):
    fig, axs = plt.subplots(5, 5, figsize=(15, 15))
    used = []
    for r, cat in enumerate(ctx.cats):
        pool = ctx.split(cat)[0]
        test = pool[pool.origin == "test"]
        normal = test[test.label == 0].sort_values("img_id").iloc[0]
        types = sorted(pool[pool.label == 1].defect_type.unique())[:2]
        panels = [("normal", normal, False)]
        for t in types:
            row = first_by_type(test, t)
            panels += [(t, row, False), (f"{t} + GT", row, True)]
        for c, (ttl, row, gt) in enumerate(panels):
            axs[r, c].imshow(rgb256(row.path))
            if gt:
                contour(axs[r, c], mask256(row))
            noaxis(axs[r, c], f"{cat}: {ttl}")
            used.append(row.img_id)
    fig.suptitle("Fig 1 - MVTec AD texture categories: normal vs defects (lime = ground truth)", fontsize=14)
    finish(fig, "fig01_dataset_montage", ids_df(used))


def fig02(ctx):
    rows = []
    for cat in ctx.cats:
        _, tr, te = ctx.split(cat)
        rows.append(dict(category=cat, train_normal=(tr.label == 0).sum(), train_defect=(tr.label == 1).sum(),
                         test_normal=(te.label == 0).sum(), test_defect=(te.label == 1).sum()))
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    for off, sp in ((-0.2, "train"), (0.2, "test")):
        n, d = df[f"{sp}_normal"].values, df[f"{sp}_defect"].values
        ax.bar(x + off, n, 0.38, color=NORMAL_C, label="normal" if sp == "train" else None)
        ax.bar(x + off, d, 0.38, bottom=n, color=DEFECT_C, label="defective" if sp == "train" else None)
        for xi, a, b in zip(x + off, n, d):
            ax.text(xi, a / 2, str(a), ha="center", color="white", fontsize=9)
            ax.text(xi, a + b / 2, str(b), ha="center", color="white", fontsize=9)
            ax.text(xi, a + b + 2, sp, ha="center", fontsize=8)
    ax.set_xticks(x, df.category)
    ax.set_ylabel("number of images (count)")
    ax.set_xlabel("category (left bar = train split, right bar = test split)")
    ax.set_title("Fig 2 - Class balance and 70/30 stratified split (seed 42)")
    ax.legend(loc="upper right")
    finish(fig, "fig02_class_balance_split", df)


def fig03(ctx):
    pool = ctx.split("carpet")[0]
    row = pool[pool.label == 1].sort_values("img_id").iloc[0]
    ch = demo_chain(load_rgb(row.path), 256)
    u8 = ch["uint8"]
    panels = [("original RGB", ch["rgb"], None), ("gray", ch["gray"], "gray"), ("resized 256", ch["resized"], "gray"),
              ("Gaussian 3x3 s=0.8", ch["blur"], "gray"), ("min-max [0,1]", ch["minmax"], "gray"),
              ("LBP (8,1) code map", lbp_codes(u8, 8, 1, "nri_uniform"), "nipy_spectral"),
              ("LBP (8,3) code map", lbp_codes(u8, 8, 3, "nri_uniform"), "nipy_spectral")]
    fig, axs = plt.subplots(1, 7, figsize=(21, 3.4))
    for ax, (t, im, cm) in zip(axs, panels):
        ax.imshow(im, cmap=cm)
        noaxis(ax, t)
    fig.suptitle(f"Fig 3 - Preprocessing chain and LBP code maps ({row.img_id})", fontsize=14)
    finish(fig, "fig03_preprocessing_chain", ids_df([row.img_id]))


def _first(df, label):
    return df[df.label == label].sort_values("img_id").iloc[0]


def fig04(ctx):
    fig, axs = plt.subplots(5, 5, figsize=(16, 15))
    rows = []
    for r, cat in enumerate(ctx.cats):
        _, tr, _ = ctx.split(cat)
        n, d = _first(tr, 0), _first(tr, 1)
        un, ud = preprocess(load_rgb(n.path)), preprocess(load_rgb(d.path))
        axs[r, 0].imshow(rgb256(n.path))
        axs[r, 1].imshow(rgb256(d.path))
        axs[r, 2].imshow(lbp_codes(un, 8, 1, "nri_uniform"), cmap="nipy_spectral")
        axs[r, 3].imshow(lbp_codes(ud, 8, 1, "nri_uniform"), cmap="nipy_spectral")
        for c, t in enumerate(["normal", "defect", "LBP(8,1) normal", "LBP(8,1) defect"]):
            noaxis(axs[r, c], f"{cat}: {t}")
        X = ctx.feats(cat, "p1", tr)
        hn, hd = X[tr.label.values == 0].mean(0), X[tr.label.values == 1].mean(0)
        axs[r, 4].plot(hn, color=NORMAL_C, label="normal")
        axs[r, 4].plot(hd, color=DEFECT_C, label="defect")
        axs[r, 4].set_xlabel("LBP bin (index)")
        axs[r, 4].set_ylabel("mean frequency (fraction)")
        axs[r, 4].legend()
        rows += [dict(category=cat, bin=i, normal=a, defect=b) for i, (a, b) in enumerate(zip(hn, hd))]
    fig.suptitle("Fig 4 - LBP(8,1) maps and mean histograms (train split)", fontsize=14)
    finish(fig, "fig04_lbp_maps_hist", pd.DataFrame(rows))


def fig05(ctx):
    fig, axs = plt.subplots(5, 3, figsize=(15, 14))
    rows = []
    for r, cat in enumerate(ctx.cats):
        _, tr, _ = ctx.split(cat)
        X, y = ctx.feats(cat, "p2", tr), tr.label.values
        for c in range(3):
            sl = slice(59 * c, 59 * (c + 1))
            for lab, col, nm in ((0, NORMAL_C, "normal"), (1, DEFECT_C, "defect")):
                m, s = X[y == lab][:, sl].mean(0), X[y == lab][:, sl].std(0)
                axs[r, c].plot(m, color=col, label=nm)
                axs[r, c].fill_between(range(59), m - s, m + s, color=col, alpha=0.2)
                rows += [dict(category=cat, scale=c + 1, cls=nm, bin=i, mean=a, std=b)
                         for i, (a, b) in enumerate(zip(m, s))]
            axs[r, c].set_title(f"{cat} R={c + 1}")
            axs[r, c].set_xlabel("LBP bin (index)")
            axs[r, c].set_ylabel("frequency (fraction)")
            axs[r, c].legend()
    fig.suptitle("Fig 5 - Multi-scale LBP histograms, mean +/- 1 std", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    finish(fig, "fig05_multiscale_histograms", pd.DataFrame(rows))


def fig06(ctx):
    fig, axs = plt.subplots(5, 4, figsize=(13, 16))
    ids = []
    for r, cat in enumerate(ctx.cats):
        _, tr, _ = ctx.split(cat)
        n, d = _first(tr, 0), _first(tr, 1)
        ids += [n.img_id, d.img_id]
        un, ud = preprocess(load_rgb(n.path)), preprocess(load_rgb(d.path))
        for c, (im, ang, t) in enumerate([(un, 0, "normal 0 deg"), (ud, 0, "defect 0 deg"),
                                           (un, np.pi / 2, "normal 90 deg"), (ud, np.pi / 2, "defect 90 deg")]):
            axs[r, c].imshow(np.log(glcm_matrices(im, 1, ang) + 1e-8), cmap="viridis")
            axs[r, c].set_title(f"{cat}: {t}", fontsize=9)
            axs[r, c].set_xlabel("gray level j")
            axs[r, c].set_ylabel("gray level i")
    fig.suptitle("Fig 6 - GLCM (d=1, 32 levels, log scale)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    finish(fig, "fig06_glcm_matrices", ids_df(ids))


def fig07(ctx):
    rows = []
    for cat in ctx.cats:
        _, tr, _ = ctx.split(cat)
        X, y = ctx.feats(cat, "glcm", tr), tr.label.values
        for i, p in enumerate(PROPS):
            v = X[:, 4 * i:4 * i + 4].mean(1)
            rows.append(dict(category=cat, prop=p, delta=(v[y == 1].mean() - v[y == 0].mean()) / (v.std() + 1e-12)))
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="category", y="delta", hue="prop", ax=ax)
    ax.set_ylabel("(defect mean - normal mean) / std (z-units)")
    ax.set_xlabel("category")
    ax.set_title("Fig 7 - Haralick property shift between defective and normal textures")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(loc="best", ncol=3)
    finish(fig, "fig07_glcm_haralick_delta", df)


def fig08(ctx):
    Xs, ys = [], []
    for cat in ctx.cats:
        _, tr, _ = ctx.split(cat)
        Xs.append(ctx.feats(cat, "p3raw", tr))
        ys.append(tr.label.values)
    X, y = np.vstack(Xs), np.concatenate(ys)
    Z = StandardScaler().fit_transform(X)
    cum = np.cumsum(PCA().fit(Z).explained_variance_ratio_)
    mi = mutual_info_classif(Z, y, random_state=ctx.seed)
    names = feature_names("p3raw")
    top = np.argsort(mi)[::-1][:20]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].plot(np.arange(1, len(cum) + 1), cum, color=COL["P3"])
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    axs[0].axhline(0.95, color="grey", ls="--")
    axs[0].axvline(k95, color="grey", ls=":")
    axs[0].set_title(f"cumulative variance (95% at {k95} components)")
    axs[0].set_xlabel("number of principal components (count)")
    axs[0].set_ylabel("cumulative explained variance (fraction)")
    axs[1].barh([names[i] for i in top][::-1], mi[top][::-1], color=COL["P3"])
    axs[1].set_title("top-20 mutual information")
    axs[1].set_xlabel("mutual information with label (nats)")
    axs[1].set_ylabel("feature")
    fig.suptitle("Fig 8 - PCA variance and MI ranking of the 201 features (5 categories pooled, seed 42)", fontsize=14)
    df = pd.DataFrame({"feature": [names[i] for i in top], "mi": mi[top]})
    finish(fig, "fig08_pca_variance_mi_top20", df)


def fig09(ctx):
    fig, axs = plt.subplots(5, 3, figsize=(14, 18))
    rows = []
    for r, cat in enumerate(ctx.cats):
        _, tr, _ = ctx.split(cat)
        y = tr.label.values
        for c, (m, k) in enumerate((("P1", "p1"), ("P2", "p2"), ("P3", "p3raw"))):
            Z = StandardScaler().fit_transform(ctx.feats(cat, k, tr))
            if m == "P3":
                Z = SelectKBest(mutual_info_classif, k=100).fit_transform(Z, y)
            P = PCA(2, random_state=ctx.seed).fit_transform(Z)
            for lab, col, nm in ((0, NORMAL_C, "normal"), (1, DEFECT_C, "defect")):
                axs[r, c].scatter(P[y == lab, 0], P[y == lab, 1], s=8, color=col, alpha=0.6, label=nm)
            axs[r, c].set_title(f"{cat} - {m}")
            axs[r, c].set_xlabel("PC1 (a.u.)")
            axs[r, c].set_ylabel("PC2 (a.u.)")
            axs[r, c].legend()
            rows += [dict(category=cat, pipeline=m, pc1=a, pc2=b, label=l) for (a, b), l in zip(P, y)]
    fig.suptitle("Fig 9 - 2-D PCA of standardized features (train split)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    finish(fig, "fig09_feature_space_pca2d", pd.DataFrame(rows))


# ---------------------------------------------------------------- classification results
def _cm_panel(ax, y, pred, title, m):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({rn[i, j]:.0%})", ha="center", va="center",
                    color="white" if rn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks([0, 1], ["Normal", "Defective"])
    ax.set_yticks([0, 1], ["Normal", "Defective"])
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title, fontsize=10)


def _conf_fig(ctx, methods, name, title):
    methods = [m for m in methods if m in ctx.methods]
    if not methods:
        return
    fig, axs = plt.subplots(len(methods), 6, figsize=(21, 3.8 * len(methods)), squeeze=False)
    rows = []
    for r, m in enumerate(methods):
        d = ctx.p42[ctx.p42.method == m]
        for c, cat in enumerate(ctx.cats + ["pooled"]):
            s = d if cat == "pooled" else d[d.category == cat]
            _cm_panel(axs[r, c], s.label.values, s.pred.values, f"{m} - {cat}", m)
            tn, fp, fn, tp = confusion_matrix(s.label, s.pred, labels=[0, 1]).ravel()
            rows.append(dict(method=m, category=cat, TN=tn, FP=fp, FN=fn, TP=tp))
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    finish(fig, name, pd.DataFrame(rows))


def fig10(ctx): _conf_fig(ctx, ["P1"], "fig10_confusion_P1", "Fig 10 - Confusion matrices, P1 (seed 42)")
def fig11(ctx): _conf_fig(ctx, ["P2"], "fig11_confusion_P2", "Fig 11 - Confusion matrices, P2 (seed 42)")
def fig12(ctx): _conf_fig(ctx, ["P3"], "fig12_confusion_P3", "Fig 12 - Confusion matrices, P3 (seed 42)")
def fig13(ctx): _conf_fig(ctx, ["B1", "B2"], "fig13_confusion_B1_B2", "Fig 13 - Confusion matrices, B1 (top) and B2 (bottom)")


def _curve_fig(ctx, kind, name, title):
    fig, axs = plt.subplots(2, 3, figsize=(17, 10))
    rows = []
    for ax, cat in zip(axs.ravel(), ctx.cats + ["pooled"]):
        for m in ctx.methods:
            d = ctx.p42[ctx.p42.method == m]
            d = d if cat == "pooled" else d[d.category == cat]
            y, s = d.label.values, d.score.values
            if kind == "roc":
                fpr, tpr, _ = roc_curve(y, s)
                a = roc_auc_score(y, s)
                ax.plot(fpr, tpr, color=COL[m], label=f"{m} (AUC {a:.3f})")
            else:
                p, r_, _ = precision_recall_curve(y, s)
                a = average_precision_score(y, s)
                ax.plot(r_, p, color=COL[m], label=f"{m} (AP {a:.3f})")
            rows.append(dict(category=cat, method=m, area=a))
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=0.6)
            ax.set_xlabel("false positive rate (fraction)")
            ax.set_ylabel("true positive rate (fraction)")
        else:
            ax.axhline((y == 1).mean(), color="k", ls="--", lw=0.6, label="prevalence")
            ax.set_xlabel("recall (fraction)")
            ax.set_ylabel("precision (fraction)")
        ax.set_title(cat)
        ax.legend(loc="lower right" if kind == "roc" else "lower left", fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    finish(fig, name, pd.DataFrame(rows))


def fig14(ctx): _curve_fig(ctx, "roc", "fig14_roc_curves", "Fig 14 - ROC curves (seed 42)")
def fig15(ctx): _curve_fig(ctx, "pr", "fig15_pr_curves", "Fig 15 - Precision-recall curves (seed 42)")


def fig16(ctx):
    raw = ctx.raw
    per = raw.groupby(["category", "method"]).f1.agg(["mean", "std"]).reset_index()
    macro = raw.groupby(["method", "seed"]).f1.mean().groupby("method").agg(["mean", "std"]).reset_index()
    macro.insert(0, "category", "MACRO")
    df = pd.concat([per, macro], ignore_index=True).fillna({"std": 0})
    groups = ctx.cats + ["MACRO"]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    w = 0.8 / len(ctx.methods)
    for i, m in enumerate(ctx.methods):
        d = df[df.method == m].set_index("category").loc[groups]
        x = np.arange(len(groups)) + i * w - 0.4 + w / 2
        ax.bar(x, d["mean"], w, yerr=None if m == "B2" else d["std"], color=COL[m], label=m, capsize=2)
        for xi, v in zip(x, d["mean"]):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, rotation=90)
    ax.axhline(0.90, color="k", ls="--", lw=1, label="target 0.90")
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 of defect class (fraction), mean +/- std over seeds")
    ax.set_xlabel("category")
    ax.set_title("Fig 16 - F1 per category and macro average")
    ax.legend(ncol=6, loc="lower center")
    finish(fig, "fig16_percategory_f1_bar", df)


def fig17(ctx):
    cols = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]
    t1 = ctx.table("T1_metrics_per_method.csv").set_index("method")
    d = t1.loc[ctx.methods, [c + "_mean" for c in cols]]
    d.columns = ["accuracy", "bal-acc", "precision", "recall", "F1", "AUC", "MCC"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(d, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "score"})
    ax.set_xlabel("metric")
    ax.set_ylabel("method")
    ax.set_title("Fig 17 - Macro-averaged metrics (mean over categories and seeds)")
    finish(fig, "fig17_metric_summary_heatmap", d.reset_index())


def fig18(ctx):
    d = ctx.preds[ctx.preds.label == 1].copy()
    d["row"] = d.category + "/" + d.defect_type
    piv = d.groupby(["row", "method"]).pred.mean().unstack()[ctx.methods]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(piv) + 2))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "recall"})
    ax.set_xlabel("method")
    ax.set_ylabel("category / defect type")
    ax.set_title("Fig 18 - Recall per defect type (mean over seeds)")
    finish(fig, "fig18_defecttype_recall_heatmap", piv.reset_index())


def fig19(ctx):
    mac = ctx.raw.groupby(["method", "seed"]).f1.mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    order = [m for m in ctx.methods if m != "B2"]
    sns.boxplot(data=mac[mac.method.isin(order)], x="method", y="f1", order=order, palette=COL, ax=ax)
    sns.stripplot(data=mac[mac.method.isin(order)], x="method", y="f1", order=order, color="k", size=4, ax=ax)
    if "B2" in ctx.methods:
        v = mac[mac.method == "B2"].f1.values
        ax.scatter([len(order)] * len(v), v, marker="D", color=COL["B2"], s=60, label="B2 (single seed)")
        ax.set_xticks(range(len(order) + 1), order + ["B2"])
        ax.legend()
    ax.set_ylabel("macro F1 per seed (fraction)")
    ax.set_xlabel("method")
    ax.set_title("Fig 19 - Seed variability of macro F1")
    finish(fig, "fig19_seed_variability_box", mac)


def fig20(ctx):
    fig, axs = plt.subplots(5, 3, figsize=(15, 16))
    rows = []
    for r, cat in enumerate(ctx.cats):
        for c, m in enumerate(["P1", "P2", "P3"]):
            d = ctx.p42[(ctx.p42.method == m) & (ctx.p42.category == cat)]
            for lab, col, nm in ((0, NORMAL_C, "normal"), (1, DEFECT_C, "defect")):
                axs[r, c].hist(d[d.label == lab].score, bins=20, color=col, alpha=0.6, label=nm)
            axs[r, c].axvline(0, color="k", ls="--")
            axs[r, c].set_title(f"{cat} - {m}")
            axs[r, c].set_xlabel("SVM decision value (a.u.)")
            axs[r, c].set_ylabel("images (count)")
            axs[r, c].legend()
            rows.append(d[["img_id", "method", "score", "label"]])
    fig.suptitle("Fig 20 - SVM decision-value distributions (seed 42, threshold 0)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    finish(fig, "fig20_score_distributions", pd.concat(rows))


# ---------------------------------------------------------------- ablations
def _t7(ctx, abl):
    return ctx.table("T7_ablations.csv").query("ablation == @abl")


def _bar_variants(ax, d, title, color):
    g = d.groupby("variant", sort=False).agg(f1=("f1", "mean"), sd=("f1", "std"), dim=("dim", "first")).reset_index()
    ax.bar(range(len(g)), g.f1, yerr=g.sd.fillna(0), color=color, capsize=3)
    for i, (v, dm) in enumerate(zip(g.f1, g.dim)):
        ax.text(i, v + 0.01, f"{v:.3f}\n{dm}-d", ha="center", fontsize=8)
    ax.set_xticks(range(len(g)), g.variant, rotation=25, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 of defect class (fraction), mean over categories x seeds")
    ax.set_title(title)
    return g


def fig21(ctx):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    g = _bar_variants(ax, _t7(ctx, "A1"), "Fig 21 - A1: LBP scale ablation", COL["P2"])
    ax.set_xlabel("LBP scale set")
    finish(fig, "fig21_ablation_scales", g)


def fig22(ctx):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
    g1 = _bar_variants(axs[0], _t7(ctx, "A2"), "A2: feature groups", COL["P3"])
    g2 = _bar_variants(axs[1], _t7(ctx, "A3"), "A3: selector on 201 features", COL["P3"])
    axs[0].set_xlabel("feature set")
    axs[1].set_xlabel("selector")
    fig.suptitle("Fig 22 - Feature-group and selector ablations", fontsize=14)
    finish(fig, "fig22_ablation_features_selector", pd.concat([g1.assign(ab="A2"), g2.assign(ab="A3")]))


def fig23(ctx):
    d = _t7(ctx, "A4").copy()
    d["k"] = d.variant.str.replace("k=", "").astype(int)
    g = d.groupby("k").agg(f1=("f1", "mean"), sd=("f1", "std"), lat=("latency_ms", "mean")).reset_index()
    t4 = ctx.table("T4_inference_latency.csv").set_index("method")
    ext = float(t4.loc["P3", ["preprocess_ms", "features_ms"]].sum()) if "P3" in t4.index else 0.0
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(g.k, g.f1, yerr=g.sd.fillna(0), color=COL["P3"], marker="o", label="F1")
    ax.set_xlabel("selected features k (count)")
    ax.set_ylabel("F1 of defect class (fraction)")
    ax2 = ax.twinx()
    ax2.plot(g.k, g.lat + ext, color=COL["B2"], marker="s", ls="--", label="inference ms")
    ax2.set_ylabel("inference latency incl. extraction (ms per image)")
    ax.set_title("Fig 23 - A4: SelectKBest k sweep")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")
    finish(fig, "fig23_ablation_k_sweep", g)


def fig24(ctx):
    d = _t7(ctx, "A6").copy()
    d["method"] = d.variant.str.split(" ").str[0]
    d["frac"] = d.variant.str.split("=").str[1].astype(float)
    g = d.groupby(["method", "frac"]).agg(f1=("f1", "mean"), sd=("f1", "std"), fit=("fit_seconds", "mean")).reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for m, s in g.groupby("method"):
        axs[0].errorbar(s.frac, s.f1, yerr=s.sd.fillna(0), marker="o", color=COL[m], label=m)
        axs[1].plot(s.frac, s.fit, marker="o", color=COL[m], label=m)
    axs[0].set_xlabel("fraction of training split used (fraction)")
    axs[0].set_ylabel("F1 of defect class (fraction)")
    axs[1].set_xlabel("fraction of training split used (fraction)")
    axs[1].set_ylabel("SVM grid-search fit time (s, log)")
    axs[1].set_yscale("log")
    for a in axs:
        a.legend()
    fig.suptitle("Fig 24 - A6: learning curve", fontsize=14)
    finish(fig, "fig24_learning_curve", g)


# ---------------------------------------------------------------- efficiency
def _train_stage_table(ctx):
    t3 = ctx.table("T3_training_time.csv")
    cols = ["preprocess", "features", "scaler_selector_svm_grid", "cnn_forward", "cnn_finetune"]
    per_seed = t3.groupby(["method", "seed"])[cols + ["total"]].sum().reset_index()
    return per_seed.groupby("method")[cols + ["total"]].mean().reindex(ctx.methods)


STAGE_C = {"preprocess": "#8da0cb", "features": "#66c2a5", "scaler_selector_svm_grid": "#fc8d62",
           "cnn_forward": "#e78ac3", "cnn_finetune": "#a6761d"}


def fig25(ctx):
    T = _train_stage_table(ctx)
    labels = list(T.index) + [f"{m} (retrain, cached)" for m in T.index if m != "B2"]
    stages = {c: list(T[c].values) + [T.loc[m, c] if c == "scaler_selector_svm_grid" else 0.0
                                      for m in T.index if m != "B2"] for c in STAGE_C}
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, log in zip(axs, (False, True)):
        left = np.zeros(len(labels))
        for c, vals in stages.items():
            ax.barh(labels, vals, left=left, color=STAGE_C[c], label=c)
            left += np.array(vals)
        for i, t in enumerate(left):
            ax.text(t * 1.02 if log else t + left.max() * 0.01, i, f"{t:.1f}s", va="center", fontsize=9)
        if log:
            ax.set_xscale("log")
            ax.set_xlim(left[left > 0].min() * 0.5, left.max() * 4)
        ax.invert_yaxis()
        ax.set_xlabel("training time summed over 5 categories (s)" + (" - log scale" if log else ""))
        ax.set_ylabel("method")
    axs[0].legend(loc="lower right")
    fig.suptitle("Fig 25 - Training time per method (incl. one-off feature extraction; image decode excluded)",
                 fontsize=14)
    finish(fig, "fig25_training_time_bars", pd.DataFrame(stages, index=labels).reset_index())


LAT_LABEL = {"B2": "B2 (1 thread)", "B2_allthreads": "B2 (all CPU threads)"}
LAT_C = {"preprocess_ms": "#8da0cb", "features_ms": "#66c2a5", "scaler_selector_ms": "#fc8d62", "svm_ms": "#e78ac3"}


def fig26(ctx):
    t4 = ctx.table("T4_inference_latency.csv")
    labels = [LAT_LABEL.get(m, m) for m in t4.method]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    left = np.zeros(len(t4))
    for c, col in LAT_C.items():
        ax.barh(labels, t4[c], left=left, color=col, label=c)
        left += t4[c].values
    for i, (t, fps) in enumerate(zip(t4.median_ms, t4.fps)):
        ax.text(t * 1.1, i, f"{t:.1f} ms ({fps:.1f} FPS)", va="center",
                fontsize=9)
    ax.axvline(30, color="k", ls="--", label="30 ms target")
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("latency per image (ms, log) - batch 1, decode excluded")
    ax.set_ylabel("method")
    ax.set_title("Fig 26 - Inference latency with stage breakdown")
    ax.legend(loc="lower right")
    finish(fig, "fig26_inference_latency_bars", t4)


def fig27(ctx):
    T = _train_stage_table(ctx)
    t4 = ctx.table("T4_inference_latency.csv").set_index("method")
    if "B2" not in T.index or "B2" not in t4.index:
        raise RuntimeError("B2 missing - fig27 needs B2")
    ms = [m for m in ("P1", "P2", "P3", "B1") if m in T.index]
    tr = [T.loc["B2", "total"] / T.loc[m, "total"] for m in ms]
    la = [t4.loc["B2", "median_ms"] / t4.loc[m, "median_ms"] for m in ms]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, v, t in ((axs[0], tr, "training time"), (axs[1], la, "inference latency")):
        ax.bar(ms, v, color=[COL[m] for m in ms])
        ax.set_yscale("log")
        for i, x in enumerate(v):
            ax.text(i, x * 1.1, f"{x:.1f}x", ha="center")
        ax.set_ylabel(f"speed-up of {t} vs B2 (x, log)")
        ax.set_xlabel("method")
        ax.set_title(t)
    fig.suptitle("Fig 27 - Speed-up relative to the fine-tuned CNN (B2)", fontsize=14)
    finish(fig, "fig27_speedup_bars", pd.DataFrame({"method": ms, "train_speedup": tr, "latency_speedup": la}))


def _pareto(xs, ys):
    pts = sorted(zip(xs, ys))
    out, best = [], -1
    for x, y in pts:
        if y > best:
            out.append((x, y))
            best = y
    return out


def _scatter(ctx, xs, xlabel, name, title):
    t1 = ctx.table("T1_metrics_per_method.csv").set_index("method")
    t8 = ctx.table("T8_resources.csv").set_index("method")
    ms = [m for m in ctx.methods if m in xs]
    x = [xs[m] for m in ms]
    y = [t1.loc[m, "f1_mean"] for m in ms]
    size = [30 + 40 * t8.loc[m, "model_size_MB"] if m in t8.index else 60 for m in ms]
    fig, ax = plt.subplots(figsize=(9, 6))
    for m, xi, yi, s in zip(ms, x, y, size):
        ax.scatter(xi, yi, s=s, color=COL[m], alpha=0.8, edgecolor="k")
        ax.annotate(m, (xi, yi), textcoords="offset points", xytext=(8, 6))
    pf = _pareto(x, y)
    ax.plot(*zip(*pf), "k--", lw=1, label="Pareto frontier")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("macro F1 (fraction)")
    ax.set_title(title + "\n(marker area ~ model size in MB)")
    ax.legend(loc="lower right")
    sizes = [t8.loc[m, "model_size_MB"] if m in t8.index else np.nan for m in ms]
    finish(fig, name, pd.DataFrame({"method": ms, "x": x, "f1": y, "size_MB": sizes}))


def fig28(ctx):
    t4 = ctx.table("T4_inference_latency.csv").set_index("method")
    _scatter(ctx, {m: t4.loc[m, "median_ms"] for m in t4.index if m in ALL_M}, "median latency (ms per image, log)",
             "fig28_accuracy_vs_latency", "Fig 28 - Accuracy vs inference latency")


def fig29(ctx):
    T = _train_stage_table(ctx)
    _scatter(ctx, {m: T.loc[m, "total"] for m in T.index}, "total training time (s, log)",
             "fig29_accuracy_vs_traintime", "Fig 29 - Accuracy vs training time")


def fig30(ctx):
    t8 = ctx.table("T8_resources.csv")
    t4 = ctx.table("T4_inference_latency.csv").set_index("method")
    t8["peak_RAM_MB"] = [t4.loc[m, "peak_rss_mb"] if m in t4.index else np.nan for m in t8.method]
    show = t8.copy()
    for c in show.columns:
        show[c] = show[c].map(lambda v: f"{v:.2f}" if isinstance(v, float) else v)
    fig, ax = plt.subplots(figsize=(17, 0.6 * len(show) + 1.8))
    ax.axis("off")
    tb = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 1.8)
    ax.set_title("Fig 30 - Resource comparison")
    finish(fig, "fig30_resource_table", t8)


def fig31(ctx):
    p512 = ctx.out / "size512" / "tables"
    if not (p512 / "T2_raw_per_seed.csv").exists():
        raise RuntimeError("outputs/size512 missing - run the 512 config for fig31")
    r512 = pd.read_csv(p512 / "T2_raw_per_seed.csv")
    l512 = pd.read_csv(p512 / "T4_inference_latency.csv").set_index("method")
    r256 = ctx.raw[ctx.raw.seed == ctx.seed]
    l256 = ctx.table("T4_inference_latency.csv").set_index("method")
    rows = []
    for m in ("P1", "P2", "P3"):
        rows.append(dict(method=m, size=256, f1=r256[r256.method == m].groupby("seed").f1.mean().mean(),
                         latency_ms=l256.loc[m, "median_ms"]))
        rows.append(dict(method=m, size=512, f1=r512[(r512.method == m) & (r512.seed == ctx.seed)].f1.mean(),
                         latency_ms=l512.loc[m, "median_ms"]))
    df = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, lab in ((axs[0], "f1", "macro F1 (fraction)"), (axs[1], "latency_ms", "median latency (ms per image)")):
        sns.barplot(data=df, x="method", y=col, hue="size", ax=ax)
        ax.set_ylabel(lab)
        ax.set_xlabel("method")
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f" if col == "f1" else "%.1f", fontsize=8)
    fig.suptitle("Fig 31 - A5: 256 px vs 512 px (seed 42)", fontsize=14)
    finish(fig, "fig31_size_ablation_256_512", df)


# ---------------------------------------------------------------- localization / explainability
def _pick_rows(te, n=6):
    d = te[te.label == 1]
    types = sorted(d.defect_type.unique())
    used, rows = set(), []
    while len(rows) < n:
        progressed = False
        for t in types:
            cand = d[(d.defect_type == t) & (~d.img_id.isin(used))].sort_values("img_id")
            if len(cand):
                rows.append(cand.iloc[0])
                used.add(cand.iloc[0].img_id)
                progressed = True
            if len(rows) == n:
                break
        if not progressed:
            break
    return rows


def _boxes(ax, heat, tau):
    for sl in regions(heat, tau):
        ax.add_patch(Rectangle((sl[1].start, sl[0].start), sl[1].stop - sl[1].start, sl[0].stop - sl[0].start,
                               fill=False, edgecolor=RED, lw=1.5))


def _loc_fig(ctx, cat, num):
    H = ctx.heat(cat)
    te = ctx.split(cat)[2]
    rows = _pick_rows(te)
    rng = {m: prange(H[f"heat_{m}"]) for m in ("P1", "P2", "P3")}
    has_cam = "heat_B2" in H
    if has_cam:
        rng["B2"] = prange(H["heat_B2"])
    heads = ["original", "GT mask", "P1 heatmap", "P2 heatmap", "P3 heatmap", "B2 Grad-CAM", "P3 overlay + boxes"]
    fig, axs = plt.subplots(len(rows), 7, figsize=(21, 3 * len(rows) + 1))
    p3 = ctx.p42[(ctx.p42.method == "P3") & (ctx.p42.category == cat)].set_index("img_id")
    for r, row in enumerate(rows):
        i = H["idx"][row.img_id]
        img, mask = rgb256(row.path), mask256(row)
        axs[r, 0].imshow(img)
        axs[r, 1].imshow(np.zeros((256, 256)), cmap="gray", vmin=0, vmax=1)
        axs[r, 1].imshow(mask, cmap="gray", alpha=0)
        contour(axs[r, 1], mask)
        for c, m in ((2, "P1"), (3, "P2"), (4, "P3")):
            axs[r, c].imshow(H[f"heat_{m}"][i], cmap="jet", vmin=rng[m][0], vmax=rng[m][1])
        if has_cam:
            axs[r, 5].imshow(H["heat_B2"][i], cmap="jet", vmin=rng["B2"][0], vmax=rng["B2"][1])
        else:
            blank(axs[r, 5], "B2 not run")
        overlay(axs[r, 6], img, H["heat_P3"][i], *rng["P3"])
        _boxes(axs[r, 6], H["heat_P3"][i], H["tau"][2])
        contour(axs[r, 6], mask)
        s = p3.loc[row.img_id]
        axs[r, 6].text(4, 12, f"P3 score={s.score:.2f}, pred={'Defective' if s.pred else 'Normal'}", color="white",
                       fontsize=8, bbox=dict(facecolor="black", alpha=0.6, pad=1))
        for c in range(7):
            noaxis(axs[r, c], heads[c] if r == 0 else None)
        axs[r, 0].set_ylabel(row.defect_type, fontsize=9)
    fig.suptitle(f"Fig {num} - Localization on {cat} (seed 42 test defects)", fontsize=14)
    fig.text(0.5, 0.005, "window 64, stride 16; tau fitted on held-out train images. B2 Grad-CAM tau was fitted on "
             "images B2 had trained on (caveat).", ha="center", fontsize=8, color="grey")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    finish(fig, f"fig{num}_localization_{cat}", ids_df([r.img_id for r in rows]))


def _make_loc(cat, num):
    return lambda ctx: _loc_fig(ctx, cat, num)


def fig37(ctx):
    t5 = ctx.table("T5_localization_metrics.csv")
    fig, axs = plt.subplots(1, 3, figsize=(19, 5))
    cols = {"P1": COL["P1"], "P2": COL["P2"], "P3": COL["P3"], "B2-GradCAM": COL["B2"]}
    for ax, met in zip(axs, ("pixel_auroc", "dice", "iou")):
        sns.barplot(data=t5, x="category", y=met, hue="method", palette=cols, ax=ax)
        ax.set_ylabel({"pixel_auroc": "pixel ROC-AUC (fraction)", "dice": "Dice (fraction)",
                       "iou": "IoU (fraction)"}[met])
        ax.set_xlabel("category")
        ax.set_title(met)
    fig.suptitle("Fig 37 - Localization metrics on the test split (128x128)", fontsize=14)
    finish(fig, "fig37_localization_metrics", t5)


def fig38(ctx):
    fig, axs = plt.subplots(10, 4, figsize=(12, 30))
    ids, r = [], 0
    for cat in ctx.cats:
        _, tr, te = ctx.split(cat)
        Xp2 = ctx.feats(cat, "p2", tr)
        hists = normal_mean_hists(Xp2[tr.label.values == 0])
        for row in te[te.label == 1].sort_values("img_id").iloc[:2].itertuples():
            img, mask = rgb256(row.path), mask256(row._asdict())
            rar = rarity_map(preprocess(load_rgb(row.path)), hists)
            axs[r, 0].imshow(img)
            axs[r, 1].imshow(rar, cmap="jet")
            overlay(axs[r, 2], img, rar, *prange(rar))
            axs[r, 3].imshow(img)
            contour(axs[r, 3], mask)
            for c, t in enumerate(["original", "LBP rarity", "overlay", "GT contour"]):
                noaxis(axs[r, c], f"{cat}: {t}" if r % 2 == 0 or c == 0 else t)
            ids.append(row.img_id)
            r += 1
    fig.suptitle("Fig 38 - Training-free LBP rarity maps", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    finish(fig, "fig38_lbp_rarity_maps", ids_df(ids))


def fig39(ctx):
    d = ctx.table("permutation_importance.csv")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, p in zip(axs, ("P2", "P3")):
        s = d[d.pipeline == p].sort_values("mean_drop")
        ax.barh(s.group, s.mean_drop, xerr=s.std_drop, color=COL[p], capsize=3)
        ax.set_xlabel("F1 drop after permuting the group (fraction)")
        ax.set_ylabel("feature group")
        ax.set_title(f"{p} (pooled over categories, seed 42)")
    fig.suptitle("Fig 39 - Group permutation importance (20 repeats)", fontsize=14)
    finish(fig, "fig39_permutation_importance", d)


def fig40(ctx):
    d = ctx.table("lbp_bin_cohens_d.csv").pivot(index="category", columns="bin", values="cohens_d").loc[ctx.cats]
    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(d.values, cmap="coolwarm", aspect="auto", vmin=-np.abs(d.values).max(), vmax=np.abs(d.values).max())
    for x in (58.5, 117.5):
        ax.axvline(x, color="k", lw=1)
    ax.set_yticks(range(len(d)), d.index)
    ax.set_xlabel("LBP bin (index; separators between R=1 | R=2 | R=3)")
    ax.set_ylabel("category")
    fig.colorbar(im, label="Cohen's d (defect - normal)")
    ax.set_title("Fig 40 - Per-bin discrimination (train split)")
    finish(fig, "fig40_lbp_bin_discrimination", d.reset_index())


def _case_panel(ax, ctx, cat, img_id, method="P3"):
    te = ctx.split(cat)[2].set_index("img_id")
    row = te.loc[img_id]
    H = ctx.heat(cat)
    rgb = rgb256(row.path)
    h = H[f"heat_{method}"][H["idx"][img_id]]
    overlay(ax, rgb, h, *prange(H[f"heat_{method}"]))
    contour(ax, mask256(row))
    noaxis(ax)
    return row


def fig41(ctx):
    d = ctx.p42[ctx.p42.method == "P3"]
    fn = d[(d.label == 1) & (d.pred == 0)].assign(a=lambda x: x.score.abs()).sort_values("a", ascending=False).head(8)
    fp = d[(d.label == 0) & (d.pred == 1)].assign(a=lambda x: x.score.abs()).sort_values("a", ascending=False).head(8)
    fig, axs = plt.subplots(2, 8, figsize=(24, 7))
    for r, (df, nm) in enumerate(((fn, "FN"), (fp, "FP"))):
        for c in range(8):
            if c < len(df):
                x = df.iloc[c]
                _case_panel(axs[r, c], ctx, x.category, x.img_id)
                axs[r, c].set_title(f"{nm} {x.category}/{x.defect_type}\nscore={x.score:.2f}", fontsize=8)
            else:
                blank(axs[r, c], "none")
    fig.suptitle("Fig 41 - P3 failure cases: false negatives (top) and false positives (bottom)", fontsize=14)
    finish(fig, "fig41_failure_cases_P3", pd.concat([fn.assign(kind="FN"), fp.assign(kind="FP")]))


def fig42(ctx):
    a = ctx.p42[ctx.p42.method == "P1"].set_index("img_id")
    b = ctx.p42[ctx.p42.method == "P3"].set_index("img_id")
    j = a.join(b, lsuffix="_p1", rsuffix="_p3", how="inner")
    j = j[(j.label_p1 == 1) & (j.pred_p1 == 0) & (j.pred_p3 == 1)]
    j = j.assign(gain=j.score_p3 - j.score_p1).sort_values("gain", ascending=False).head(8)
    fig, axs = plt.subplots(4, 4, figsize=(14, 14))
    for ax in axs.ravel():
        ax.axis("off")
    for k, (img_id, x) in enumerate(j.iterrows()):
        r, c = divmod(k, 2)
        for off, m in ((0, "P1"), (1, "P3")):
            ax = axs[r, 2 * c + off]
            ax.axis("on")
            _case_panel(ax, ctx, x.category_p1, img_id, m)
            ax.set_title(f"{m} - {x.category_p1}/{x.defect_type_p1}", fontsize=8)
    fig.suptitle("Fig 42 - Defects missed by P1 but found by P3 (P1 vs P3 heatmaps)", fontsize=14)
    finish(fig, "fig42_rescued_by_P3", j.reset_index()[["img_id", "score_p1", "score_p3", "gain"]]
           if len(j) else ids_df([]))


def fig43(ctx):
    t4 = ctx.table("T4_inference_latency.csv").set_index("method")
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, t, ms, color):
        ax.add_patch(FancyBboxPatch((x, y), w, 0.9, boxstyle="round,pad=0.05", fc=color, ec="k", alpha=0.85))
        ax.text(x + w / 2, y + 0.45, t if ms is None else f"{t}\n{ms:.2f} ms", ha="center", va="center", fontsize=9)

    def arrow(x0, x1, y):
        ax.annotate("", xy=(x1, y + 0.45), xytext=(x0, y + 0.45), arrowprops=dict(arrowstyle="->"))
    p = t4.loc["P3"] if "P3" in t4.index else None
    g = lambda k: None if p is None else float(p[k])  # noqa: E731
    ys = 4
    ax.text(0.1, ys + 1.2, "P1 / P2 / P3 path (ms shown for P3)", fontsize=11, weight="bold")
    xs = [0.1, 2.5, 5.4, 8.6, 11.0, 13.4]
    labs = [("image", None), ("preprocess", g("preprocess_ms")), ("LBP / GLCM", g("features_ms")),
            ("scaler + selector", g("scaler_selector_ms")), ("SVM", g("svm_ms")), ("label + heatmap", None)]
    for x, (t, ms), w in zip(xs, labs, [2.0, 2.4, 2.6, 2.1, 2.0, 2.4]):
        box(x, ys, w, t, ms, COL["P3"])
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + 2.0, b, ys)
    ys = 1.5
    ax.text(0.1, ys + 1.2, "B2 path (ResNet-18 fine-tuned)", fontsize=11, weight="bold")
    q = t4.loc["B2"] if "B2" in t4.index else None
    h = lambda k: None if q is None else float(q[k])  # noqa: E731
    for x, (t, ms), w in zip([0.1, 2.5, 5.4, 8.6], [("image", None), ("resize + normalize", h("preprocess_ms")),
                                                   ("ResNet-18 + softmax", h("features_ms")), ("label", None)],
                             [2.0, 2.4, 2.6, 2.0]):
        box(x, ys, w, t, ms, COL["B2"])
    for a, b in zip([0.1, 2.5, 5.4], [2.5, 5.4, 8.6]):
        arrow(a + 2.0, b, ys)
    ax.set_title("Fig 43 - Pipeline diagram with measured median stage times (single thread)")
    finish(fig, "fig43_pipeline_diagram", t4.reset_index())


FIGS = [("fig01", fig01), ("fig02", fig02), ("fig03", fig03), ("fig04", fig04), ("fig05", fig05), ("fig06", fig06),
        ("fig07", fig07), ("fig08", fig08), ("fig09", fig09), ("fig10", fig10), ("fig11", fig11), ("fig12", fig12),
        ("fig13", fig13), ("fig14", fig14), ("fig15", fig15), ("fig16", fig16), ("fig17", fig17), ("fig18", fig18),
        ("fig19", fig19), ("fig20", fig20), ("fig21", fig21), ("fig22", fig22), ("fig23", fig23), ("fig24", fig24),
        ("fig25", fig25), ("fig26", fig26), ("fig27", fig27), ("fig28", fig28), ("fig29", fig29), ("fig30", fig30),
        ("fig31", fig31)] + [(f"fig{n}", None) for n in range(32, 37)] + [
    ("fig37", fig37), ("fig38", fig38), ("fig39", fig39), ("fig40", fig40), ("fig41", fig41), ("fig42", fig42),
    ("fig43", fig43)]


def run(cfg, log, **_):
    ctx = Ctx(cfg)
    loc = {32 + i: _make_loc(c, 32 + i) for i, c in enumerate(["carpet", "grid", "leather", "tile", "wood"])}
    for key, fn in FIGS:
        num = int(key[3:])
        if fn is None:
            cat = ["carpet", "grid", "leather", "tile", "wood"][num - 32]
            if cat not in ctx.cats:
                continue
            fn = loc[num]
        try:
            fn(ctx)
            log.info(f"{key} done")
        except Exception as e:  # keep going; report at the end
            plt.close("all")
            log.info(f"{key} FAILED: {e}\n{traceback.format_exc(limit=3)}")
