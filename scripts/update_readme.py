"""Copy the real T1/T4/T0_targets numbers from outputs/tables into the README Results block."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
readme = ROOT / "README.md"


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


t1 = pd.read_csv(T / "T1_metrics_per_method.csv")
t1 = t1[["method", "f1", "roc_auc", "balanced_accuracy", "precision", "recall"]]
t4 = pd.read_csv(T / "T4_inference_latency.csv")[["method", "median_ms", "p95_ms", "fps", "peak_rss_mb"]]
t3 = pd.read_csv(T / "T3_training_time.csv").groupby(["method", "seed"])["total"].sum().groupby("method").mean()
t3 = t3.rename("train_seconds_total_5_categories").reset_index()
tg = pd.read_csv(T / "T0_targets.csv")
block = ("<!-- RESULTS:START -->\n### T1 - macro metrics (mean±std over seeds)\n" + md(t1) +
         "\n\n### Training time (s, summed over categories, mean over seeds)\n" + md(t3) +
         "\n\n### T4 - inference latency (1 thread, batch 1)\n" + md(t4) +
         "\n\n### Proposal targets\n" + md(tg) + "\n<!-- RESULTS:END -->")
s = readme.read_text(encoding="utf-8")
a, b = s.index("<!-- RESULTS:START -->"), s.index("<!-- RESULTS:END -->") + len("<!-- RESULTS:END -->")
readme.write_text(s[:a] + block + s[b:], encoding="utf-8")
print("README results updated")
