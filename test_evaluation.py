"""Calibration tests for the split comparison.

Plants an operator effect that is known to be there and checks the harness
finds it. Without this, a small gap on real data could just mean the harness
cannot detect gaps.
"""
import numpy as np
import pytest

from evaluation import (compare, evaluate_grouped, evaluate_random,
                            participant_score, spread)


def synth(n_participants=24, n_classes=12, dim=48, per=180,
          style_scale=1.10, proto_scale=0.30, noise=0.55, seed=0):
    """Segments carrying a shared class signal plus a persistent per-person offset.

    style_scale=0 removes the operator effect entirely, which gives us a
    negative control as well as a positive one.
    """
    rng = np.random.default_rng(seed)
    proto = rng.normal(0, proto_scale, (n_classes, dim))
    X, y, g = [], [], []
    for i in range(n_participants):
        style = rng.normal(0, style_scale, dim) if style_scale else np.zeros(dim)
        pref = rng.dirichlet(np.ones(n_classes) * 0.5)
        labels = rng.choice(n_classes, per, p=pref)
        X.append(proto[labels] + style + rng.normal(0, noise, (per, dim)))
        y.append(labels)
        g.append(np.full(per, f"90{i:02d}"))
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


def test_positive_control_detects_planted_operator_effect():
    X, y, g = synth()
    rand = evaluate_random(X, y)
    grp, _ = evaluate_grouped(X, y, g)
    assert rand > grp, (
        f"harness blind to a planted operator effect: random={rand:.3f} "
        f"grouped={grp:.3f}. No result from it can be trusted."
    )


def test_negative_control_shows_little_gap_without_an_operator_effect():
    """With no per-person style, the two protocols should roughly agree.

    A large gap here would mean the gap is an artefact of the split mechanics
    rather than of operator identity.
    """
    X, y, g = synth(style_scale=0.0, seed=7)
    rand = evaluate_random(X, y)
    grp, _ = evaluate_grouped(X, y, g)
    assert abs(rand - grp) < 0.10, f"unexplained gap: {rand:.3f} vs {grp:.3f}"


def test_every_participant_receives_a_score():
    X, y, g = synth(n_participants=24)
    _, per = evaluate_grouped(X, y, g)
    assert set(per) == set(np.unique(g))


def test_per_participant_scoring_ignores_classes_the_person_never_performs():
    """A narrow repertoire must not look like poor model performance.

    Both people are handled perfectly by the model. One performs six classes,
    the other three. Their scores must be equal. Averaging over the union of
    labels would give the second person 0.5 and misattribute it to the model.
    """
    broad_true = np.array([0, 1, 2, 3, 4, 5])
    narrow_true = np.array([0, 1, 2])
    assert participant_score(broad_true, broad_true.copy()) == pytest.approx(1.0)
    assert participant_score(narrow_true, narrow_true.copy()) == pytest.approx(1.0)


def test_participant_score_still_penalises_real_errors():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 1])   # class 2 always wrong
    assert participant_score(y_true, y_pred) < 0.75


def test_protocols_are_matched_on_fold_count():
    """Same k for both, otherwise training-set sizes differ and the
    comparison stops being about grouping."""
    X, y, g = synth(n_participants=16, per=120)
    out = compare(X, y, g, n_splits=4)
    assert out["gap"] == pytest.approx(
        out["random_split_macro_f1"] - out["grouped_split_macro_f1"]
    )


def test_spread_reports_the_extremes():
    s = spread({"a": 0.10, "b": 0.50, "c": 0.30})
    assert s.n == 3
    assert (s.worst_id, s.best_id) == ("a", "b")
    assert s.range == pytest.approx(0.40)
