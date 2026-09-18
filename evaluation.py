"""Split comparison.

Two evaluations with the same estimator, fold count and test set sizes. The
only difference is whether one person's segments can appear in both train and
test. Those have to stay matched or the comparison means nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_SPLITS = 8


def make_model():
    """Kept simple on purpose.

    We want the gap between two protocols. A stronger classifier raises both
    numbers and makes the gap harder to read.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def _fit_predict(X, y, train_idx, test_idx):
    model = make_model()
    model.fit(X[train_idx], y[train_idx])
    return model.predict(X[test_idx]), y[test_idx]


def evaluate_random(X, y, n_splits: int = N_SPLITS, seed: int = 0) -> float:
    """Random split over segments, ignoring who performed them.

    Overstates performance because the same person appears in train and test.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds, trues = [], []
    for tr, te in kf.split(X):
        p, t = _fit_predict(X, y, tr, te)
        preds.append(p)
        trues.append(t)
    return f1_score(
        np.concatenate(trues), np.concatenate(preds),
        average="macro", labels=np.unique(y), zero_division=0,
    )


def participant_score(y_true, y_pred) -> float:
    """Macro-F1 for one person over the classes they actually perform.

    Using the full label set would score someone against classes missing from
    their ground truth, so a person doing fewer distinct actions looks worse
    even when the model handles them perfectly.
    """
    y_true = np.asarray(y_true)
    return f1_score(
        y_true, y_pred, average="macro",
        labels=np.unique(y_true), zero_division=0,
    )


def evaluate_grouped(X, y, groups, n_splits: int = N_SPLITS):
    """Leave-participants-out.

    GroupKFold with k folds costs k fits instead of one per participant, and
    each participant still lands in exactly one test fold so everyone gets a
    score.

    Returns (overall_macro_f1, {participant_id: macro_f1}).
    """
    gkf = GroupKFold(n_splits=n_splits)
    per_participant: dict[str, float] = {}
    preds, trues = [], []

    for tr, te in gkf.split(X, y, groups):
        p, t = _fit_predict(X, y, tr, te)
        preds.append(p)
        trues.append(t)
        held = np.asarray(groups)[te]
        for pid in np.unique(held):
            mask = held == pid
            per_participant[pid] = participant_score(t[mask], p[mask])

    overall = f1_score(
        np.concatenate(trues), np.concatenate(preds),
        average="macro", labels=np.unique(y), zero_division=0,
    )
    return overall, per_participant


@dataclass
class Spread:
    n: int
    mean: float
    std: float
    worst_id: str
    worst: float
    best_id: str
    best: float

    @property
    def range(self) -> float:
        return self.best - self.worst

    def __str__(self) -> str:
        return (
            f"n={self.n}  mean={self.mean:.3f}  sd={self.std:.3f}  "
            f"worst={self.worst:.3f} ({self.worst_id})  "
            f"best={self.best:.3f} ({self.best_id})  range={self.range:.3f}"
        )


def spread(per_participant: dict[str, float]) -> Spread:
    s = pd.Series(per_participant).sort_values()
    return Spread(
        n=len(s), mean=float(s.mean()), std=float(s.std()),
        worst_id=str(s.index[0]), worst=float(s.iloc[0]),
        best_id=str(s.index[-1]), best=float(s.iloc[-1]),
    )


def compare(X, y, groups, n_splits: int = N_SPLITS, seed: int = 0) -> dict:
    """Run both protocols and return the headline figures."""
    rand = evaluate_random(X, y, n_splits=n_splits, seed=seed)
    grp, per = evaluate_grouped(X, y, groups, n_splits=n_splits)
    sp = spread(per)
    return {
        "random_split_macro_f1": rand,
        "grouped_split_macro_f1": grp,
        "gap": rand - grp,
        "per_participant": per,
        "spread": sp,
    }
