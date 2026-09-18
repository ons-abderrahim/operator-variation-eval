# Data integrity notes

## Conflicting participant identifiers

Assembly101 recording names encode the participant id twice:

    nusar-2021_action_both_<pid_a>-<toy>_<pid_b>_user_id_<date>_<time>

In 7 of the 362 released recordings `pid_a != pid_b`.

### Resolution rule

If the second id never appears as a primary id anywhere in the corpus, it
cannot be an independent participant, so the conflict is a labelling alias and
the grouping is unaffected. If both ids are primary ids elsewhere, both are
real people and the assignment cannot be recovered from the filename.

| Case | Count | Action |
|---|---|---|
| `9065` / `9095` | 6 | Kept under `9065`. `9095` has no recordings of its own. |
| `9072` / `9071` | 1 | Excluded. Both are real participants recording on 2021-02-11. |

### Why exclusion rather than a guess

A leave-one-participant-out split only works if the held-out person's footage
is actually absent from training. Assigning this recording to the wrong
participant would put several minutes of one person's work into training while
that same person is being held out, which is the leak this project measures.
Dropping 0.28% of recordings costs less than that.

If the maintainers confirm the correct identity, the recording can be restored
by adding an override to `build_index`.

## Frame rate

Annotation frame indices correspond to video extracted at 30 fps. Confirm
alignment against the feature store before trusting any duration in seconds.
