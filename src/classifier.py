"""sklearn pipelines P1/P2/P3/B1 and the shared SVM grid search."""
from __future__ import annotations

import time
from functools import partial

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PARAM_GRID = [
    {"svm__kernel": ["linear"], "svm__C": [0.1, 1, 10, 100]},
    {"svm__kernel": ["rbf"], "svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", "auto", 0.01]},
]


def build_pipeline(name: str, seed: int, mi_k: int = 100, selector=None) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if name == "P3":
        steps.append(("select", SelectKBest(partial(mutual_info_classif, random_state=seed), k=mi_k)))
    if selector is not None:  # ablation hook (replaces/adds a selector step)
        steps.append(("select", selector))
    steps.append(("svm", SVC(class_weight="balanced", cache_size=500, random_state=seed)))
    return Pipeline(steps)


def train_model(X, y, name: str, seed: int, log=None, cv_folds: int = 5, mi_k: int = 100, selector=None):
    pipe = build_pipeline(name, seed, mi_k, selector)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, PARAM_GRID, scoring="f1", cv=cv, n_jobs=1, refit=True)
    t0 = time.perf_counter_ns()
    gs.fit(X, y)
    fit_s = (time.perf_counter_ns() - t0) / 1e9
    best = gs.best_estimator_
    p = gs.best_params_
    info = dict(best_kernel=p["svm__kernel"], best_C=p["svm__C"], best_gamma=p.get("svm__gamma", ""),
                best_cv_f1=float(gs.best_score_), fit_seconds=fit_s,
                n_support_vectors=int(best.named_steps["svm"].n_support_.sum()),
                n_configs=len(gs.cv_results_["params"]))
    if log is not None:
        log.add("fit", fit_s, pipeline=name, seed=seed)
    return best, info
