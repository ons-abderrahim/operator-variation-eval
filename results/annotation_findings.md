# Annotation-level findings

Produced by `scripts/annotation_analysis.py`. Annotations only, no feature access.

## Operator overlap in the official splits

**train vs validation**

- operators present in both: **37 of 37**
- validation recordings from an operator seen in train: **62/62** (100.0%)
- validation segments from an operator seen in train: **15,603/15,603** (100.0%)

**train vs test**

- operators present in both: **41 of 41**
- test recordings from an operator seen in train: **88/88** (100.0%)
- test segments from an operator seen in train: **21,759/21,759** (100.0%)

## Operator variation from labels alone

- **84,255** physical segments (from 1,013,523 annotation rows across 12 camera views), **48** operators, **24** verbs
- median action duration ranges **0.67s** to **1.43s**, a **2.15x** spread between the fastest and slowest operator
- verb-mix divergence from the population ranges **0.0024** to **0.0227**

![operator variation](fig_operator_variation.png)

## Operator identification from labels alone

Predicting which of **48** operators produced a recording, using only its verb histogram and median action duration. No appearance information.

| Features | Accuracy | vs chance |
|---|---|---|
| pace only | 0.069 | 3.3x |
| verb mix only | 0.111 | 5.3x |
| verb mix + pace | **0.114** | **5.5x** |
| chance | 0.021 | 1.0x |

Operator identity is recoverable from what people do and how fast they do it, before any pixels are considered. That is the signal a model can exploit when the same person appears on both sides of a split.

![operator identification](fig_operator_identification.png)
