# Assembly101 Operator Variation

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)

How much of a model's score on assembly video comes from having already seen the
person doing the work?

---

## 📌 Overview

Benchmarks for assembly video usually split the data by recording or by object.
That means clips of the same worker end up in both the training set and the test
set. The model can learn one person's habits and get rewarded for it.

On a real production line that reward disappears. A new operator starts, or
someone works differently after a shift change, and the model meets a person it
has never seen.

Four things came out of this work, all measured on the real annotations:

1. The official splits have **no operator held out at all**. Every test
   recording comes from someone in the training set.
2. You can tell **which operator made a recording at 5.5 times chance**, using
   only the labels. No video needed.
3. A model with **no video input matches the published video baseline** on verb
   recognition. So a lot of this benchmark can be solved from context alone.
4. The **operator gap is zero**, but single operators still differ by 19 points.
   The same setup does find a 2.8 point leak between recordings, so the zero is
   a real absence and not a blind spot.

---

## 🎯 Status

| Part | State |
|---|---|
| Operator structure | Done, runs with no dataset access |
| Data quality check | Done, found two problems |
| Operator overlap in the official splits | Done, measured |
| How much operators differ | Done, measured |
| Telling operators apart from labels | Done, measured |
| Split comparison on timing and context | Done, two kinds of model |
| Test suite | Done, 18 tests passing |
| Same comparison on TSM video features | Blocked, access request still open |
| Looking at the hardest operator's footage | Blocked, needs video |

Everything marked done uses the public annotation files. The video features are
behind a gate, so that half has not run. The notebook marks any figure made from
fake data so it cannot be mistaken for a result.

---

## ❓ Why the standard benchmark misses this

Assembly101 splits its recordings 60/15/25 into train, validation and test. The
splits are built around new toys: 25 of the 101 toys show up in all three
splits, and 20 and 16 unseen toys are kept back for validation and test.

Nobody controls for who is doing the work. So the benchmark can tell you if a
model handles a new product. It cannot tell you if it handles a new person.

---

## 🔬 Method

Three splits, with the same model, the same number of folds and the same test
set sizes. The only thing that changes is what is allowed to show up on both
sides.

| Split | What it is | What leaks |
|---|---|---|
| A | Shuffle all segments | The recording and the operator |
| B | Keep whole recordings together | The operator only |
| C | Keep whole operators together | Nothing |

Two splits would mix up two different problems. Adding B separates them. Going
from A to B shows how much comes from seeing the same recording. Going from B to
C shows how much comes from seeing the same person.

Two kinds of model, both run through all three splits: plain logistic regression
and gradient boosted trees. Running both matters, because the leak turns out to
depend on how strong the model is. One model alone would have hidden that.

Two sets of inputs:

| Inputs | Question | State |
|---|---|---|
| Duration, gaps, position, toy, and the actions either side | Does the order and timing of the work depend on the operator? | Done |
| TSM video features | Does what the operator looks like depend on the operator? | Blocked |

Reported for each split: macro-F1, weighted-F1, top-1 and top-k accuracy, both
gaps, and how far apart single operators land under split C. macro-F1 on its own
is misleading here because the classes are very uneven, so it never appears
alone.

---

## 📊 Results

All measured on the real annotations. 1,013,523 annotation rows become
**84,255 real segments**, from **48 operators**, over **24 verbs**.

### Operators are never held out

| | train vs validation | train vs test |
|---|---|---|
| Operators in both | 37 of 37 | 41 of 41 |
| Recordings from an operator seen in training | 62 of 62 (100%) | 88 of 88 (100%) |
| Segments from an operator seen in training | 15,603 (100%) | 21,759 (100%) |

All 48 operators are in the training split. Every recording in validation and
test belongs to someone the model has already trained on. This is not a partial
overlap. There is no operator held out at all.

### Operators work differently

- The slowest operator takes **2.15 times** as long per action as the fastest,
  0.67 seconds against 1.43 seconds.
- How far an operator's mix of verbs sits from the group average ranges from
  **0.0024 to 0.0227**, about a tenfold spread.

![operator variation](results/fig_operator_variation.png)

### You can tell operators apart from the labels alone

Guessing which of 48 operators made a recording, using only the mix of verbs and
how fast they work. No images at all.

| Inputs | Accuracy | Against chance |
|---|---|---|
| Speed only | 0.069 | 3.3x |
| Verb mix only | 0.111 | 5.3x |
| Verb mix and speed | **0.114** | **5.5x** |
| Chance | 0.021 | 1.0x |

![operator identification](results/fig_operator_identification.png)

The signal is already sitting in the labels before any model looks at a picture.
And the splits hand it to the model on both sides.

### Split comparison

Since the video features are blocked, this runs on inputs taken from the
annotations: how long an action lasts, the gap before it, where it sits in the
recording, the toy, and which actions come just before and after.

**This is not the video experiment.** It asks a smaller question. Does the order
and timing of the work depend on who is doing it?

The label set is the 18 verbs these inputs can actually tell apart. The six
`attempt to X` verbs are left out, because an attempted screw lasts the same time
and sits between the same actions as a successful one. The difference is only
visible in the video. The 24 verb numbers are below as well.

Gradient boosted trees, 5 folds, 18 verbs:

| Split | macro-F1 | weighted-F1 | top-1 |
|---|---|---|---|
| Shuffle all segments | **0.546** | **0.619** | **0.632** |
| Keep recordings together | 0.518 | 0.599 | 0.616 |
| Keep operators together | 0.515 | 0.598 | 0.614 |

| Where the gap comes from | Size |
|---|---|
| Seeing the same recording | **+0.028** |
| Seeing the same operator | +0.004 |

**There is a leak, and it is not the operator one.** Keeping whole recordings
together costs 2.8 points. Keeping whole operators together costs nothing on top
of that. The setup finds a real leak sitting right next to the one it does not
find, which is what makes the operator result believable.

### The leak grows with model strength

The same comparison with plain logistic regression:

| Split | macro-F1 |
|---|---|
| Shuffle all segments | 0.3275 |
| Keep recordings together | 0.3270 |
| Keep operators together | 0.3275 |

A simple model finds no leak anywhere. A stronger one finds 2.8 points. So
bigger models make the recording leak worse, not better.

### All the numbers

Operators held out, 18 verbs, 3 folds. top-k is here because macro-F1 is the
harshest single number on an uneven label set and says very little by itself.

| Measure | Value | Chance |
|---|---|---|
| macro-F1 | 0.508 | |
| weighted-F1 | 0.594 | |
| top-1 | 0.610 | 0.056 |
| top-2 | 0.757 | 0.111 |
| top-3 | **0.836** | 0.167 |
| top-5 | 0.920 | 0.278 |

macro-F1 is pulled down by rare verbs. `pick up` has 15,969 segments and `shake`
has 197, an 81 times difference, and macro-F1 counts all 18 the same. The common
verbs do well: `clap` 0.796, `unscrew` 0.736, `screw` 0.660, `pick up` 0.621.

### Against the published video baseline

On the **full 24 verbs**, with no images at all:

| Model | Input | top-1 |
|---|---|---|
| This work, operators held out | Timing and nearby labels | 0.576 |
| This work, shuffled split | Timing and nearby labels | 0.593 |
| Assembly101 TSM, fixed and head cameras | Full video | 0.585 |
| Assembly101 TSM, head cameras only | Full video | 0.470 |
| Assembly101 TSM, fixed cameras | Full video | 0.640 |

A model that never sees a single image matches the published video result, and
beats the head camera version by twelve points. Much of what this benchmark
rewards is just knowing the order of the work. A model can score well here while
learning very little about hands and parts.

That is a third way the benchmark makes a model look better than it is, next to
the missing operator holdout and the operator signal in the labels.

### The average hides the spread

Operators held out, 18 verbs:

| | |
|---|---|
| Average per operator | 0.518 |
| Worst | 0.404 (operator 9074) |
| Best | 0.599 (operator 9036) |
| Spread | 0.195 |

![split comparison](results/fig_split_comparison.png)

A 19 point spread, under a split whose overall number says the operator does not
matter. Two checks that this is not just noise:

- Shuffling which operator each recording belongs to, keeping the group sizes
  the same, gives an average spread of 0.097. The real spread beats all 12
  shuffles, which puts p at 0.077. That is the lowest p you can get with 12
  shuffles, so it points the right way without being proof.
- Per operator scores line up at Spearman 0.83 between the two kinds of model.
  Operator 9074 is the hardest one under every setup tried, and 8 of the hardest
  10 are the same. So some operators really are harder, and it is not an
  accident of one model.

### Not done yet

| | |
|---|---|
| Same three splits on TSM video features | Waiting on access |
| Per operator spread on video features | Waiting on access |
| Watching the hardest operator's footage | Needs video |

---

## 🔍 Data quality

Two problems, both found while grouping the data by operator, and both invisible
to anyone splitting by toy.

### Problem 1: every action is annotated 12 times

Each action is labelled once per camera. The `video` column carries the camera
name, so `train.csv` has 566,855 rows for 47,252 real segments. Shuffling those
rows puts the same action in training and test from two different angles, which
is a second leak on top of the operator one. `load_split` merges the cameras
before anything else happens.

### Problem 2: the operator id contradicts itself

Recording names carry the operator id twice:

```
nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724
                       ^^^^     ^^^^
```

In 7 of 362 recordings the two do not match. Since the whole point here is to
keep one person's clips out of training, getting this wrong puts the leak
straight back in.

**Same person, different label. 6 recordings.** All say `9065` first and `9095`
second. `9095` never appears as a first id anywhere, so these six are one group
either way and only the name changes. Kept as `9065`.

**Cannot be decided. 1 recording.**
`nusar-2021_action_both_9072-a14_9071_user_id_2021-02-11_104901` names `9072` and
`9071`. Both are real operators with their own recordings, so the file name gives
no way to pick. Dropped instead of guessed. That is 0.28% of the recordings,
against the risk of putting one person's work into training while holding that
same person out.

Check it yourself without downloading the dataset:

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

**Verbs, not 1,380 actions.** With 1,380 classes, a per operator score is driven
by classes where that person has one or two segments, so it moves with random
noise instead of with the operator. 24 verbs give every operator over a thousand
segments, which is enough to be stable.

**18 verbs in the main table, 24 for the baseline comparison.** The six
`attempt to X` verbs are left out of the main table because these inputs cannot
tell them apart from the successful version. An attempt and a success last the
same time, sit in the same place, and have the same actions around them. Only the
outcome differs, and that is in the video. Both label sets are reported, and the
comparison against the published baseline uses all 24 so it is a fair match.

**top-k next to macro-F1, not instead of it.** macro-F1 counts `shake` with 197
segments the same as `pick up` with 15,969, so it makes a model look worse than
it is at the common cases. Showing one number would mean picking which story to
tell, so every number is in the table with the chance rate beside it.

**A simple model on purpose.** The point is the difference between splits, not
the highest possible score. A stronger model lifts every number and makes the
difference harder to read. Both models are reported for exactly that reason.

**Checks before results.** One check plants an operator effect that is definitely
there and confirms the setup finds it. Another removes the effect and confirms
the gap disappears. Without both, the zero above would mean nothing, because a
setup that cannot find a gap also returns zero.

**A shuffle test for the spread.** The spread between operators is compared
against shuffling which operator each recording belongs to, with group sizes
kept the same. A spread means nothing until you know what a random grouping of
the same data gives you.

---

## ⚡ Quick start

```bash
pip install -r requirements.txt

python check_participants.py           # no dataset access needed
python annotation_analysis.py --ann-dir path/to/fine-grained-annotations
pytest -q                              # the checks
```

`check_participants.py` works straight away. It pulls a public list of recording
names and reproduces the 48 operators and both id problems, with no login and no
download.

The full walkthrough is in `assembly101_operator_variation.ipynb`. It notices
whether you have dataset access and falls back to stand-in data with the same
shape, marking every figure it makes that way.

---

## 📁 Structure

Flat on purpose, so nothing breaks depending on how the files get uploaded.

```
participants.py           operator ids, and the two id problems
evaluation.py             the three splits, per operator scores, spread
breakfast.py              loader for Breakfast, 50Salads and GTEA
check_participants.py     operator check, no dataset access needed
annotation_analysis.py    overlap, how operators differ, telling them apart
test_participants.py      8 tests
test_evaluation.py        10 tests, including the two checks
results/                  figures and written up findings
DATA_INTEGRITY.md         notes on the two data problems
assembly101_operator_variation.ipynb
```

---

## 📄 Data

No dataset files are included here.

Assembly101 is released under CC BY-NC 4.0 and comes from its authors. This is a
personal project and is not used commercially.

> Sener et al., *Assembly101: A Large-Scale Multi-View Video Dataset for
> Understanding Procedural Activities*, CVPR 2022.
