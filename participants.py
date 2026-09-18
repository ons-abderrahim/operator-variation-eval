"""Participant identity for Assembly101 recordings.

Recording names encode the participant id twice:

    nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724
                           ^^^^     ^^^^

7 of the 362 recordings have the two ids disagreeing. Getting those wrong puts
a person's footage back into training while they are held out, so they are
resolved explicitly here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

RECORDING_RE = re.compile(
    r"^nusar-(?P<year>\d{4})_action_both_"
    r"(?P<pid_a>\d+)-(?P<toy>[a-z0-9]+)_"
    r"(?P<pid_b>\d+)_user_id_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d+)$"
)


def recording_from_video(video: str) -> str:
    """Strip the view suffix that annotation CSVs append to the recording name.

    The `video` column looks like
    `nusar-2021_action_both_9011-b06b_9011_user_id_2021-02-01_154253/C10095_rgb.mp4`.
    Each physical action segment is repeated once per camera, 12 views on most
    recordings, so this is also the key for de-duplicating to real segments.
    """
    return str(video).split("/")[0]


def parse_recording(name: str) -> dict | None:
    """Parsed fields of a recording name, or None if it does not match.

    Returns None instead of raising so callers can count failures.
    """
    m = RECORDING_RE.match(recording_from_video(str(name).strip()))
    return m.groupdict() if m else None


def participant_id(name: str) -> str | None:
    """Primary participant id (the first of the two encoded ids)."""
    p = parse_recording(name)
    return p["pid_a"] if p else None


@dataclass
class ParticipantIndex:
    """Resolved participant assignment over a set of recording names.

    assignment : recording name -> participant id, usable recordings
    aliased    : ids disagree, second id is not a participant elsewhere.
                 Same grouping either way, only the label differs.
    ambiguous  : ids disagree and both are real participants. Cannot be
                 resolved from the filename, so excluded.
    unparsed   : names not matching the expected pattern.
    """

    assignment: dict[str, str] = field(default_factory=dict)
    aliased: dict[str, tuple[str, str]] = field(default_factory=dict)
    ambiguous: dict[str, tuple[str, str]] = field(default_factory=dict)
    unparsed: list[str] = field(default_factory=list)

    @property
    def participants(self) -> set[str]:
        return set(self.assignment.values())

    def summary(self) -> str:
        n_rec = len(self.assignment)
        counts = {}
        for pid in self.assignment.values():
            counts[pid] = counts.get(pid, 0) + 1
        v = sorted(counts.values())
        med = v[len(v) // 2] if v else 0
        return (
            f"recordings usable   {n_rec}\n"
            f"participants        {len(counts)}\n"
            f"recordings/person   min {min(v) if v else 0}  "
            f"median {med}  max {max(v) if v else 0}\n"
            f"benign aliases      {len(self.aliased)}\n"
            f"ambiguous, dropped  {len(self.ambiguous)}\n"
            f"unparsed            {len(self.unparsed)}"
        )


def build_index(recording_names) -> ParticipantIndex:
    """Resolve participant ids over a collection of recording names.

    Rule
    ----
    If the two encoded ids agree, use it.
    If they differ:
      * second id never appears as a primary id anywhere -> benign alias, keep
        the recording under the primary id (same grouping either way).
      * second id IS a primary id elsewhere -> both are real participants, the
        assignment is unresolvable, drop the recording.
    """
    idx = ParticipantIndex()
    parsed = {}
    for name in recording_names:
        p = parse_recording(name)
        if p is None:
            idx.unparsed.append(name)
        else:
            parsed[name] = p

    primaries = {p["pid_a"] for p in parsed.values()}

    for name, p in parsed.items():
        a, b = p["pid_a"], p["pid_b"]
        if a == b:
            idx.assignment[name] = a
        elif b in primaries:
            idx.ambiguous[name] = (a, b)
        else:
            idx.aliased[name] = (a, b)
            idx.assignment[name] = a
    return idx
