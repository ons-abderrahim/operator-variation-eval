#!/usr/bin/env python3
"""Analyse Assembly101 fine-grained annotations. No feature access required.

Three results:

1. Do the official train/validation/test splits keep operators apart?
2. How much do operators differ in pace and in the verbs they use?
3. Can an operator be identified from their label stream alone? If yes, the
   signal a model could latch onto is present in the data before any pixels
   are involved.

Usage:
    python scripts/annotation_analysis.py --ann-dir path/to/fine-grained-annotations
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from participants import build_index, participant_id, recording_from_video

FPS = 30.0
OUT = Path(__file__).resolve().parent / "results"
USECOLS = ["video", "start_frame", "end_frame", "verb_id", "verb_cls"]


def load_split(path: Path) -> pd.DataFrame:
    """Load one split and collapse camera views down to physical segments.

    Every action is annotated once per camera, so the raw row count is roughly
    12x the number of real segments. Splitting randomly over raw rows would put
    the same physical action in train and test under two different views, which
    is a second leak on top of the operator one.
    """
    raw = pd.read_csv(path, usecols=USECOLS)
    raw["recording"] = raw["video"].map(recording_from_video)
    raw["participant"] = raw["recording"].map(participant_id)
    bad = raw["participant"].isna().sum()
    if bad:
        print(f"  warning: {bad} rows failed participant parse, dropped")
        raw = raw.dropna(subset=["participant"])
    seg = raw.drop_duplicates(subset=["recording", "start_frame", "end_frame"])
    return seg.reset_index(drop=True), len(raw)


def overlap_report(splits: dict[str, pd.DataFrame]) -> str:
    train = set(splits["train"]["participant"])
    lines = ["## Operator overlap in the official splits", ""]
    for other in ["validation", "test"]:
        if other not in splits:
            continue
        d = splits[other]
        shared = train & set(d["participant"])
        recs_seen = d[d["participant"].isin(train)]["recording"].nunique()
        recs = d["recording"].nunique()
        segs_seen = int(d["participant"].isin(train).sum())
        lines += [
            f"**train vs {other}**",
            "",
            f"- operators present in both: **{len(shared)} of {d['participant'].nunique()}**",
            f"- {other} recordings from an operator seen in train: "
            f"**{recs_seen}/{recs}** ({100 * recs_seen / max(recs, 1):.1f}%)",
            f"- {other} segments from an operator seen in train: "
            f"**{segs_seen:,}/{len(d):,}** ({100 * segs_seen / max(len(d), 1):.1f}%)",
            "",
        ]
    return "\n".join(lines)


def variation(df: pd.DataFrame):
    df = df.copy()
    df["n"] = df["end_frame"] - df["start_frame"]
    df = df[df["n"] > 0]
    dur = df.groupby("participant")["n"].median().sort_values() / FPS
    mix = pd.crosstab(df["participant"], df["verb_cls"], normalize="index")
    glob = (df["verb_cls"].value_counts(normalize=True)
            .reindex(mix.columns).fillna(0).values)

    def js(p, q, eps=1e-12):
        p, q = p + eps, q + eps
        m = 0.5 * (p + q)
        kl = lambda a, b: float(np.sum(a * np.log(a / b)))
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    div = pd.Series([js(mix.loc[i].values, glob) for i in mix.index],
                    index=mix.index).sort_values()
    return df, dur, div, mix


def identify_operator(df: pd.DataFrame, seed: int = 0) -> dict:
    """Predict which operator produced a recording, from labels only.

    Features are the recording's verb histogram plus its median action
    duration. No appearance information of any kind.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    hist = pd.crosstab(df["recording"], df["verb_cls"], normalize="index")
    meta = df.groupby("recording").agg(participant=("participant", "first"),
                                       med=("n", "median"))
    pace = (meta.loc[hist.index, ["med"]].values / FPS)
    y = meta.loc[hist.index, "participant"].values
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    def acc(X):
        return float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())

    chance = 1.0 / len(np.unique(y))
    return {
        "n_recordings": len(y),
        "n_operators": int(len(np.unique(y))),
        "chance": chance,
        "pace_only": acc(pace),
        "verbs_only": acc(hist.values),
        "both": acc(np.hstack([hist.values, pace])),
    }


def plots(dur, div, ident, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, PINK, ORANGE = "#4C72B0", "#D4326B", "#DD8452"

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].bar(range(len(dur)), dur.values, color=BLUE)
    ax[0].axhline(dur.median(), color=PINK, ls="--", lw=1.3,
                  label=f"median {dur.median():.2f}s")
    ax[0].set(title=f"Median action duration by operator "
                    f"({dur.max() / dur.min():.2f}x spread)",
              xlabel=f"operator (sorted, n={len(dur)})", ylabel="seconds")
    ax[0].legend(frameon=False)
    ax[1].bar(range(len(div)), div.values, color=ORANGE)
    ax[1].set(title="Divergence of verb mix from the population",
              xlabel=f"operator (sorted, n={len(div)})",
              ylabel="Jensen-Shannon divergence")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "fig_operator_variation.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    names = ["pace\nonly", "verb mix\nonly", "verb mix\n+ pace"]
    vals = [ident["pace_only"], ident["verbs_only"], ident["both"]]
    ax.bar(names, vals, color=[ORANGE, BLUE, BLUE])
    ax.axhline(ident["chance"], color=PINK, ls="--", lw=1.3,
               label=f"chance (1/{ident['n_operators']})")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=10)
    ax.set(title=f"Identifying which of {ident['n_operators']} operators "
                 f"made a recording",
           ylabel="accuracy", ylim=(0, max(vals) * 1.25))
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "fig_operator_identification.png", dpi=150)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", required=True)
    args = ap.parse_args()
    ann = Path(args.ann_dir).expanduser()
    OUT.mkdir(parents=True, exist_ok=True)

    print("loading splits (collapsing camera views to physical segments)")
    splits, raw_rows = {}, 0
    for name in ["train", "validation", "test"]:
        path = ann / f"{name}.csv"
        if not path.exists():
            print(f"  {name}.csv missing, skipping")
            continue
        splits[name], n_raw = load_split(path)
        raw_rows += n_raw
        print(f"  {name:11s} rows {n_raw:>8,} -> segments {len(splits[name]):>7,}  "
              f"recordings {splits[name]['recording'].nunique():>4}  "
              f"operators {splits[name]['participant'].nunique():>3}")

    all_recs = sorted({r for d in splits.values() for r in d["recording"]})
    idx = build_index(all_recs)
    print("\n" + idx.summary())
    if idx.ambiguous:
        drop = set(idx.ambiguous)
        for k in splits:
            splits[k] = splits[k][~splits[k]["recording"].isin(drop)]

    overlap = overlap_report(splits)
    print("\n" + overlap)

    pooled = pd.concat(splits.values(), ignore_index=True)
    pooled, dur, div, _ = variation(pooled)
    ident = identify_operator(pooled)

    var_text = "\n".join([
        "## Operator variation from labels alone", "",
        f"- **{len(pooled):,}** physical segments "
        f"(from {raw_rows:,} annotation rows across 12 camera views), "
        f"**{pooled['participant'].nunique()}** operators, "
        f"**{pooled['verb_cls'].nunique()}** verbs",
        f"- median action duration ranges **{dur.min():.2f}s** to "
        f"**{dur.max():.2f}s**, a **{dur.max() / dur.min():.2f}x** spread "
        f"between the fastest and slowest operator",
        f"- verb-mix divergence from the population ranges "
        f"**{div.iloc[0]:.4f}** to **{div.iloc[-1]:.4f}**",
        "", "![operator variation](fig_operator_variation.png)", "",
        "## Operator identification from labels alone", "",
        f"Predicting which of **{ident['n_operators']}** operators produced a "
        f"recording, using only its verb histogram and median action duration. "
        f"No appearance information.", "",
        "| Features | Accuracy | vs chance |",
        "|---|---|---|",
        f"| pace only | {ident['pace_only']:.3f} | "
        f"{ident['pace_only'] / ident['chance']:.1f}x |",
        f"| verb mix only | {ident['verbs_only']:.3f} | "
        f"{ident['verbs_only'] / ident['chance']:.1f}x |",
        f"| verb mix + pace | **{ident['both']:.3f}** | "
        f"**{ident['both'] / ident['chance']:.1f}x** |",
        f"| chance | {ident['chance']:.3f} | 1.0x |",
        "",
        "Operator identity is recoverable from what people do and how fast "
        "they do it, before any pixels are considered. That is the signal a "
        "model can exploit when the same person appears on both sides of a "
        "split.", "",
        "![operator identification](fig_operator_identification.png)", "",
    ])
    print(var_text)

    plots(dur, div, ident, OUT)
    (OUT / "annotation_findings.md").write_text(
        "# Annotation-level findings\n\n"
        "Produced by `scripts/annotation_analysis.py`. Annotations only, no "
        "feature access.\n\n" + overlap + "\n" + var_text)
    print(f"wrote {OUT / 'annotation_findings.md'} and two figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
