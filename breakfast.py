"""Loader for MS-TCN-format datasets (Breakfast, 50Salads, GTEA).

Layout:
  <root>/features/<video>.npy        float, shape (D, T) or (T, D)
  <root>/groundTruth/<video>.txt     one action label per frame
  <root>/mapping.txt                 "<id> <label>" per line
  <root>/splits/*.bundle             official subject-disjoint folds

Frame-wise labels are converted to segments (contiguous runs), then features
are mean-pooled per segment, using the same harness as Assembly101.
"""
import os, re, glob
import numpy as np
import pandas as pd

# Breakfast: P03_cam01_P03_cereals | 50Salads: rgb-01-1 | GTEA: S1_Cheese_C1
SUBJ_PATTERNS = [
    (re.compile(r"^(P\d+)_"), "breakfast"),
    (re.compile(r"^rgb-(\d+)-"), "50salads"),
    (re.compile(r"^(S\d+)_"), "gtea"),
]


def subject_from_video(name):
    for pat, ds in SUBJ_PATTERNS:
        m = pat.match(name)
        if m:
            return m.group(1), ds
    return None, None


def load_mapping(root):
    path = os.path.join(root, "mapping.txt")
    m = {}
    for line in open(path):
        parts = line.strip().split()
        if len(parts) >= 2:
            m[parts[1]] = int(parts[0])
    return m


def frames_to_segments(labels):
    """Contiguous runs of identical frame labels -> (label, start, end)."""
    segs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            segs.append((labels[start], start, i))
            start = i
    return segs


def load_dataset(root, min_seg_frames=4, max_frames=16, verbose=True):
    mapping = load_mapping(root)
    gts = sorted(glob.glob(os.path.join(root, "groundTruth", "*.txt")))
    if verbose:
        print(f"mapping: {len(mapping)} classes | groundTruth files: {len(gts)}")

    X, rows, skipped = [], [], 0
    for gi, gt in enumerate(gts):
        vid = os.path.basename(gt)[:-4]
        subj, ds = subject_from_video(vid)
        if subj is None:
            skipped += 1
            continue
        feat_path = os.path.join(root, "features", vid + ".npy")
        if not os.path.exists(feat_path):
            skipped += 1
            continue

        F = np.load(feat_path)
        # MS-TCN stores (D, T); orient to (T, D) using the label count as truth.
        labels = [l.strip() for l in open(gt) if l.strip()]
        if F.shape[0] != len(labels) and F.shape[1] == len(labels):
            F = F.T
        T = min(len(labels), F.shape[0])
        labels, F = labels[:T], F[:T]

        for lab, s, e in frames_to_segments(labels):
            if e - s < min_seg_frames or lab not in mapping:
                continue
            idx = np.linspace(s, e - 1, min(max_frames, e - s)).astype(int)
            X.append(F[idx].mean(0))
            rows.append((vid, subj, mapping[lab], lab, s, e))
        if verbose and gi % 200 == 0:
            print(f"  {gi}/{len(gts)}", end="\r")

    df = pd.DataFrame(rows, columns=["recording", "participant", "verb",
                                     "verb_name", "start", "end"])
    df["n_frames"] = df["end"] - df["start"]
    X = np.vstack(X).astype(np.float32)
    if verbose:
        print(f"\nsegments {len(df):,} | participants {df['participant'].nunique()} "
              f"| classes {df['verb'].nunique()} | X {X.shape} | skipped {skipped} files")
        v = df.groupby("participant").size()
        print(f"segments/participant: min {v.min()} median {int(v.median())} max {v.max()}")
    return X, df


# ------------------------------------------------------------------ fixture
def make_fixture(root, n_subj=8, per_subj=4, dim=64, n_cls=10, seed=0):
    """Fake MS-TCN tree so the loader is tested before the real 30GB lands."""
    rng = np.random.default_rng(seed)
    os.makedirs(f"{root}/features", exist_ok=True)
    os.makedirs(f"{root}/groundTruth", exist_ok=True)
    names = [f"act_{i:02d}" for i in range(n_cls)]
    with open(f"{root}/mapping.txt", "w") as f:
        for i, n in enumerate(names):
            f.write(f"{i} {n}\n")
    proto = rng.normal(0, .3, (n_cls, dim))
    for s in range(n_subj):
        style = rng.normal(0, 1.0, dim)
        for k in range(per_subj):
            vid = f"P{s+3:02d}_cam01_P{s+3:02d}_recipe{k}"
            labels, T = [], 0
            while T < 400:
                c = int(rng.integers(0, n_cls)); L = int(rng.integers(15, 60))
                labels += [names[c]] * L; T += L
            idx = np.array([names.index(l) for l in labels])
            F = proto[idx] + style + rng.normal(0, .6, (len(labels), dim))
            np.save(f"{root}/features/{vid}.npy", F.T.astype(np.float32))  # (D,T)
            open(f"{root}/groundTruth/{vid}.txt", "w").write("\n".join(labels))
    return root


if __name__ == "__main__":
    root = make_fixture("/tmp/bf_fixture")
    X, df = load_dataset(root)
    assert df["participant"].nunique() == 8
    assert X.shape[0] == len(df)
    assert X.shape[1] == 64
    assert df["n_frames"].min() >= 4
    print("\nsample rows:"); print(df.head(3).to_string(index=False))
    print("\nsubject parsing:")
    for v in ["P03_cam01_P03_cereals", "rgb-01-1", "S1_Cheese_C1", "junk"]:
        print(f"  {v:26s} -> {subject_from_video(v)}")
    print("\nPASS")
