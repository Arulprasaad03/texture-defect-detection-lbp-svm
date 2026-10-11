"""Per-image gallery (plan §11.4) and the static index.html (§11.5). Stages 'gallery' and 'report'."""
from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .data_loader import build_pool, make_split  # noqa: E402
from .localize import regions  # noqa: E402
from .utils import output_root  # noqa: E402
from .visualize import contour, mask256, noaxis, overlay, prange, rgb256  # noqa: E402


def _outcome(label, pred):
    return ("TP" if pred else "FN") if label else ("FP" if pred else "TN")


def build_gallery(cfg, log):
    out = output_root()
    seed = cfg.gallery_seed
    preds = pd.read_csv(out / "tables" / "predictions.csv")
    preds = preds[preds.seed == seed]
    t4 = pd.read_csv(out / "tables" / "T4_inference_latency.csv").set_index("method")
    ms_p3 = float(t4.loc["P3", "median_ms"]) if "P3" in t4.index else float("nan")
    ms_b2 = float(t4.loc["B2", "median_ms"]) if "B2" in t4.index else float("nan")
    index = []
    for cat in cfg.categories:
        te = make_split(build_pool(cfg.dataset_root, cat), seed, cfg.test_size)[1]
        z = np.load(out / "heatmaps" / f"{cat}.npz", allow_pickle=True)
        idx = {i: n for n, i in enumerate(z["ids"])}
        rng3 = prange(z["heat_P3"])
        has_b2 = "heat_B2" in z.files
        rngb = prange(z["heat_B2"]) if has_b2 else None
        gdir = out / "gallery" / cat
        gdir.mkdir(parents=True, exist_ok=True)
        pc = preds[preds.category == cat].set_index(["img_id", "method"])
        for row in te.itertuples():
            i = idx[row.img_id]
            img, mask = rgb256(row.path), mask256(row._asdict())
            p3 = pc.loc[(row.img_id, "P3")]
            tag = _outcome(row.label, int(p3.pred))
            fig, axs = plt.subplots(1, 5, figsize=(16, 3.4))
            axs[0].imshow(img)
            axs[1].imshow(img)
            contour(axs[1], mask)
            overlay(axs[2], img, z["heat_P3"][i], *rng3)
            for sl in regions(z["heat_P3"][i], z["tau"][2]):
                axs[2].add_patch(Rectangle((sl[1].start, sl[0].start), sl[1].stop - sl[1].start,
                                           sl[0].stop - sl[0].start, fill=False, edgecolor="red", lw=1.5))
            if has_b2:
                axs[3].imshow(z["heat_B2"][i], cmap="jet", vmin=rngb[0], vmax=rngb[1])
            else:
                axs[3].text(0.5, 0.5, "B2 not run", ha="center", va="center")
            lines, rec = [f"true: {'Defective' if row.label else 'Normal'} ({row.defect_type})"], {}
            for m in ("P1", "P2", "P3", "B1", "B2"):
                if (row.img_id, m) in pc.index:
                    r = pc.loc[(row.img_id, m)]
                    lines.append(f"{m}: {'Defective' if r.pred else 'Normal'}  score={r.score:.2f}")
                    rec[m] = int(r.pred)
            lines.append(f"P3 {ms_p3:.1f} ms vs B2 {ms_b2:.1f} ms / img")
            axs[4].axis("off")
            axs[4].text(0, 0.95, "\n".join(lines), va="top", fontsize=9, family="monospace")
            for a, t in zip(axs[:4], ["original", "GT contour", f"P3 overlay [{tag}]", "B2 Grad-CAM"]):
                noaxis(a, t)
            path = gdir / f"{row.defect_type}__{Path(row.img_id).name}__{tag}.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            index.append(dict(path=str(path.relative_to(out)), outcome=tag, **{f"pred_{k}": v for k, v in rec.items()}))
        log.info(f"gallery {cat}: {len(te)} images")
    pd.DataFrame(index).to_csv(out / "gallery" / "index.csv", index=False)


def build_index_html(cfg):
    out = output_root()
    parts = ["<html><head><meta charset='utf-8'><title>Texture defect detection - results</title>"
             "<style>body{font-family:sans-serif;margin:2em}img{max-width:100%}table{border-collapse:collapse}"
             "td,th{border:1px solid #999;padding:3px 8px;font-size:13px}</style></head><body>",
             "<h1>Lightweight texture-based defect detection (LBP + GLCM + SVM) vs CNN baselines</h1>"]
    for name, title in (("T1_metrics_per_method.csv", "T1 - metrics per method (macro, mean±std over seeds)"),
                        ("T4_inference_latency.csv", "T4 - inference latency (ms per image, 1 thread)"),
                        ("T0_targets.csv", "T0 - proposal targets met / not met")):
        f = out / "tables" / name
        if f.exists():
            df = pd.read_csv(f)
            if name.startswith("T1"):
                df = df[[c for c in df.columns if not c.endswith(("_mean", "_std"))]]
            parts.append(f"<h2>{html.escape(title)}</h2>{df.to_html(index=False, float_format=lambda v: f'{v:.4f}')}")
    parts.append("<h2>Figures</h2>")
    for f in sorted((out / "figures").glob("fig*.png")):
        cap = f.stem.replace("_", " ")
        parts.append(f"<h3>{html.escape(cap)}</h3><a href='figures/{f.name}'><img src='figures/{f.name}'></a>")
    parts.append("<h2>Tables</h2><ul>")
    for f in sorted((out / "tables").glob("*.csv")):
        parts.append(f"<li><a href='tables/{f.name}'>{f.name}</a></li>")
    parts.append("</ul><h2>Per-image gallery</h2><ul>")
    g = out / "gallery"
    if g.exists():
        parts.append("<li><a href='gallery/index.csv'>gallery/index.csv</a></li>")
        for d in sorted(p for p in g.iterdir() if p.is_dir()):
            parts.append(f"<li><a href='gallery/{d.name}/'>gallery/{d.name}/</a> ({len(list(d.glob('*.png')))} images)</li>")
    parts.append("</ul></body></html>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")


def run(cfg, log, stage="report", **_):
    if stage == "gallery":
        if cfg.run_gallery:
            build_gallery(cfg, log)
        else:
            log.info("run_gallery is false: gallery skipped")
    else:
        build_index_html(cfg)
        log.info(f"index written: {output_root() / 'index.html'}")
