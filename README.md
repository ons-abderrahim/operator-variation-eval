# Assembly101 Operator Variation

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)

Measuring how much action recognition accuracy on assembly video depends on
having already seen the person doing the work.

---

## 📌 Overview

Benchmarks for procedural activity split data by recording or by object.
Segments from the same participant end up in both train and test, so a model
can learn one person's habits and get credit for it.

On a production line that credit disappears. A new operator starts, or an
existing one works differently after a shift change, and the model meets
someone it has never seen.

This repo measures that. Four findings so far, all on the real annotations:

1. Assembly101's official splits contain **no operator holdout at all**. Every
   test recording belongs to someone in the training set.
2. Operator identity is **recoverable from the label stream** at 5.5x chance,
   before any appearance model is involved.
3. A model with **no visual input matches the published video baseline** on verb
   recognition, so much of what the benchmark measures is procedural context.
4. The **operator gap is zero** while individual operators differ by 19 points,
   which is the case for reporting per-operator scores rather than a mean. The
   same harness does find a 2.8 point within-recording leak, so the operator
   null is an absence rather than a blind spot.

---

## 🎯 Status

| Component | State |
|---|---|
| Participant structure | Done, runs without dataset access |
| Data quality audit | Done, two issues found |
| Operator overlap in official splits | Done, measured |
| Operator variation from labels | Done, measured |
| Operator identification from labels | Done, measured |
| Split comparison on procedural structure | Done, two model classes |
| Per-operator spread | Done, measured |
| Evaluation harness | Done, 18 tests passing |
| Same comparison on TSM visual features | Blocked, access request pending |
| Qualitative review of worst operator | Blocked, needs video |

Everything marked done is measured on the real annotations, which are public.
The TSM feature store is gated, so the visual version of the comparison has
not run. The notebook watermarks any figure it produces from synthetic data.

---

## ❓ Why the standard benchmark misses this

Assembly101 splits recordings 60/15/25 for train, validation and test. The
splits are built around toy novelty: 25 of 101 toys are shared across all
three splits, with 20 and 16 unseen toys held out for validation and test.

Operator identity is not a controlled variable. Measured directly on the
annotations, all 48 operators appear in the training split and every single
validation and test recording belongs to one of them (see Results).

The benchmark answers whether a model generalises to a new product. It cannot
answer whether it generalises to a new person.

---

## 🔬 Method

Three protocols, matched on model, fold count and test set size. The only
thing that changes is what is allowed to appear on both sides of the split.

| Protocol | Split | Leaks |
|---|---|---|
| A | `KFold` shuffled over segments | Recording and operator |
| B | `GroupKFold` on recording | Operator only |
| C | `GroupKFold` on operator id | Nothing |

Two protocols would conflate two different leaks. Adding B means the operator
effect can be isolated: A to B measures the within-recording leak, B to C
measures the operator leak on its own.

Two classifiers, both run through all three protocols: a standardised
multinomial logistic regression and gradient boosted trees. Reporting both
matters, because the leak turns out to depend on capacity and a single model
would have hidden that.

Two feature sets:

| Features | Question | State |
|---|---|---|
| Duration, gap, position, toy, neighbouring actions | Is procedural structure operator dependent? | Done |
| Mean-pooled TSM visual features | Is appearance operator dependent? | Blocked on access |

Reported per protocol: macro-F1, weighted-F1, top-1 and top-k accuracy, the two
gaps, and the spread of per-operator scores under protocol C. macro-F1 alone is
misleading on a label set with 81x class imbalance, so it is never reported on
its own.

---

## 📊 Results

Measured on the full fine-grained annotations: 1,013,523 annotation rows
collapsing to **84,255 physical segments**, **48 operators**, **24 verbs**.

### Operators are not held out at all

| | train vs validation | train vs test |
|---|---|---|
| Operators in both | 37 of 37 | 41 of 41 |
| Recordings from an operator seen in train | 62/62 (100%) | 88/88 (100%) |
| Segments from an operator seen in train | 15,603 (100%) | 21,759 (100%) |

Training contains all 48 operators. Every recording in validation and test
belongs to someone the model has already trained on. The split is not merely
non-disjoint in operators, it has no operator holdout whatsoever.

### Operators differ measurably

- Median action duration ranges **0.67s to 1.43s**, a **2.15x** spread between
  the fastest and slowest operator.
- Verb-mix divergence from the population ranges **0.0024 to 0.0227**, roughly
  a tenfold range in how idiosyncratic someone's action mix is.

![operator variation](results/fig_operator_variation.png)

### Operator identity is recoverable from labels alone

Predicting which of 48 operators produced a recording, from its verb histogram
and median action duration. No pixels involved.

| Features | Accuracy | vs chance |
|---|---|---|
| pace only | 0.069 | 3.3x |
| verb mix only | 0.111 | 5.3x |
| verb mix + pace | **0.114** | **5.5x** |
| chance | 0.021 | 1.0x |

![operator identification](results/fig_operator_identification.png)

The signal exists in the data before any appearance model sees it, and the
splits make it available on both sides.

### Split comparison on procedural structure

The visual features are still gated, so this runs the protocol comparison on
features derived from annotations only: action duration, gap to the previous
action, relative and absolute position in the recording, duration normalised
within the recording, toy identity, and the identity of the two actions either
side. **This is not the vision experiment.** It asks a narrower question: is the
procedural structure of the work operator dependent?

Label space is the 18 verbs these features can represent. The six `attempt to X`
classes are excluded because an attempted screw has the same duration and the
same neighbouring actions as a successful one, so the distinction lives in the
video and nowhere in these features. The 24 verb numbers are reported below too.

Gradient boosted trees, 5 fold, 18 verbs:

| Protocol | macro-F1 | weighted-F1 | top-1 |
|---|---|---|---|
| Random segments | 0.546 | 0.619 | 0.632 |
| Recordings held out | 0.518 | 0.599 | 0.616 |
| Operators held out | 0.515 | 0.598 | 0.614 |

| Decomposition | Gap |
|---|---|
| Within-recording leak | **+0.028** |
| Operator leak, isolated | +0.004 |

**A leak exists, and it is not the operator one.** Holding out whole recordings
costs 2.8 points of macro-F1. Holding out whole operators costs nothing beyond
that. The harness detects a real leak sitting immediately next to the one it
fails to find, which is what makes the operator null credible rather than
ambiguous.

### The leak scales with model capacity

The same comparison with plain logistic regression on a narrower feature set:

| Protocol | macro-F1 |
|---|---|
| Random segments | 0.3275 |
| Recordings held out | 0.3270 |
| Operators held out | 0.3275 |

A linear model finds no leak at all. A boosted one finds 2.8 points. Whatever a
production model does at scale, the within-recording leak gets worse with
capacity, not better.

### Full metric set

Operators held out, 18 verbs, 3 fold. Top-k is reported because macro-F1 over an
imbalanced label set is the harshest single number available and says little on
its own.

| Metric | Value | Chance |
|---|---|---|
| macro-F1 | 0.508 | |
| weighted-F1 | 0.594 | |
| top-1 | 0.610 | 0.056 |
| top-2 | 0.757 | 0.111 |
| top-3 | 0.836 | 0.167 |
| top-5 | 0.920 | 0.278 |

macro-F1 is dragged down by rare classes. Support ranges from 15,969 segments
for `pick up` to 197 for `shake`, an 81x imbalance, and macro-F1 weights all 18
equally. Head classes score well: `clap` 0.796, `unscrew` 0.736, `screw` 0.660,
`pick up` 0.621.

### Against the published visual baseline

On the **full 24 verb** space, no pixels of any kind:

| Model | Input | top-1 |
|---|---|---|
| This work, operators held out | duration and neighbouring labels | 0.576 |
| This work, random split | duration and neighbouring labels | 0.593 |
| Assembly101 TSM, fixed and egocentric | full video | 0.585 |
| Assembly101 TSM, egocentric | full video | 0.470 |
| Assembly101 TSM, fixed views | full video | 0.640 |

A model that never sees an image matches the published fixed plus egocentric
video baseline and beats the egocentric one by twelve points. A large part of
what this verb benchmark measures is recoverable from procedural context alone,
so a model can post a respectable number while learning little about what hands
and objects are doing.

That is a third way the benchmark flatters a model, alongside the absent
operator holdout and the operator identity leaking through the labels.

### The mean hides the spread

Operators held out, 18 verbs:

| | |
|---|---|
| Per-operator mean | 0.518 |
| Worst | 0.404 (operator 9074) |
| Best | 0.599 (operator 9036) |
| Range | 0.195 |

![split comparison](results/fig_split_comparison.png)

A 19 point range under a protocol whose aggregate says operator identity does
not matter. Two checks that this is not noise:

- A permutation test shuffling which operator each recording belongs to, group
  sizes preserved, gives a null range averaging 0.097. The observed range
  exceeds all 12 permutations, p = 0.077, which is the floor at 12 permutations.
  Suggestive rather than significant.
- Per-operator scores correlate at Spearman 0.83 (p = 3e-13) between logistic
  regression and gradient boosting. Operator 9074 is hardest under every model
  configuration tried, and 8 of the hardest 10 overlap. Operator difficulty is a
  property of the people, not an artefact of one classifier.

### What this rules out

There are two candidate sources of an operator gap in a vision model: what
people do and when, or how they look doing it. This isolates the first and finds
no aggregate effect, using a model that demonstrably can detect a leak. If the
TSM experiment shows a gap, it comes from appearance and motion style.

### Still pending

| Metric | Value |
|---|---|
| Macro-F1 on TSM visual features, all three protocols | pending features |
| Per-operator spread on visual features | pending features |
| Qualitative review of the worst operator's footage | needs video access |

Reproduce all of the above with:

```bash
python annotation_analysis.py --ann-dir path/to/fine-grained-annotations
```

---

## 🔍 Data quality

Two problems, both found while building the operator grouping and both
invisible to anyone splitting by toy.

### Issue 1: camera views inflate the row count 12x

Each physical action is annotated once per camera. The `video` column carries a
view suffix, so train.csv holds 566,855 rows for 47,252 real segments. Splitting
randomly over rows puts the same physical action in train and test under two
different cameras, which is a second leak independent of the operator one.
`load_split` collapses views before anything else happens.

### Issue 2: conflicting operator ids

Recording names carry the operator id twice:

```
nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724
                       ^^^^     ^^^^
```

In 7 of 362 recordings the two ids disagree. Since the point of this study is
to keep a person's footage out of training, a wrong assignment puts back the
exact leak being measured.

**Alias, 6 recordings.** All carry `9065` first and `9095` second. `9095` never
appears as a primary id anywhere, so these six are one group either way and
only the label changes. Kept under `9065`.

**Unresolvable, 1 recording.**
`nusar-2021_action_both_9072-a14_9071_user_id_2021-02-11_104901` names `9072`
and `9071`. Both are real participants with their own recordings on that date,
so the filename gives no way to choose. Excluded rather than guessed. That is
0.28% of recordings, against the risk of putting one person's work into
training while holding that same person out.

Check it without downloading the dataset:

```bash
python check_participants.py
```

```
input recordings    362
recordings usable   361
participants        48
recordings/person   min 5  median 7  max 14
benign aliases      6
ambiguous, dropped  1
unparsed            0
```

---

## ⚙️ Design choices

**Verbs, not 1380 fine-grained actions.** Per-operator macro-F1 over 1380
classes is driven by classes where a person has one or two segments, so the
score moves on sampling noise instead of operator identity. 24 verbs give every
operator over a thousand segments, enough support per class to be stable.

**18 verbs for the main table, 24 for the baseline comparison.** The six
`attempt to X` classes are excluded from the main results because these features
provably cannot represent them: an attempt and a success share duration,
position and neighbouring actions, and differ only in outcome, which is visible
in the video alone. Both label spaces are reported, and the comparison against
the published baseline uses the full 24 so it is like for like.

**Top-k alongside macro-F1.** Not instead of it. macro-F1 weights `shake`
(197 segments) equally with `pick up` (15,969), so it understates a model that
handles the common cases well. Reporting one number would mean choosing which
story to tell, so all of them are in the table with the chance rate beside.

**Each person scored on their own classes.** `participant_score` averages over
the classes that participant actually performs. Using the full label set would
penalise someone for classes missing from their ground truth, which measures
how varied their work is rather than how well the model handles them.

**Controls before results.** A positive control plants a known operator effect
and checks the harness finds it. A negative control removes the effect and
checks the gap goes away. Without both, the zero gap reported above would be
uninterpretable, since an instrument that cannot detect a gap also returns zero.

**A permutation null for the spread.** The per-operator range is compared
against shuffling which operator each recording belongs to, keeping group sizes
fixed. A range means nothing without knowing what an arbitrary grouping of the
same data produces.

**Second dataset.** `breakfast.py` runs the same harness on Breakfast
(52 subjects, around 33 videos each), whose official protocol is
subject-disjoint. Assembly101 controls for new objects and ignores operators.
Breakfast does the opposite.

---

## ⚡ Quick start

```bash
pip install -r requirements.txt

python check_participants.py           # no dataset access needed
python annotation_analysis.py --ann-dir path/to/fine-grained-annotations
pytest -q                              # harness controls
```

Full experiment: `assembly101_operator_variation.ipynb`. It detects
whether dataset access is available and falls back to a synthetic stand-in
with the same structure, watermarking every synthetic figure.

---

## 📁 Structure

Flat on purpose, so the layout survives any upload method.

```
participants.py           id parsing, alias and ambiguity handling
evaluation.py             three protocols, per operator scoring, spread
breakfast.py              MS-TCN format loader (Breakfast, 50Salads, GTEA)
check_participants.py     participant audit, no dataset access needed
annotation_analysis.py    overlap, variation and identification
test_participants.py      8 tests
test_evaluation.py        10 tests, positive and negative controls
results/                  figures and findings
DATA_INTEGRITY.md         notes on the two data issues
assembly101_operator_variation.ipynb
```

---

## 📄 Data
Assembly101 is CC BY-NC 4.0 and obtained from the maintainers. This is a
personal, non-commercial project.

> Sener et al., *Assembly101: A Large-Scale Multi-View Video Dataset for
> Understanding Procedural Activities*, CVPR 2022.
