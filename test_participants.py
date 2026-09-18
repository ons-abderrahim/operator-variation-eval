import pytest

from participants import build_index, parse_recording, participant_id

REAL = [
    "nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724",
    "nusar-2021_action_both_9011-b06b_9011_user_id_2021-02-01_154253",
    "nusar-2021_action_both_9071-b06b_9071_user_id_2021-02-11_100739",
    "nusar-2021_action_both_9072-a15_9072_user_id_2021-02-11_110655",
]
# Both ids are independent participants -> unresolvable.
AMBIGUOUS = "nusar-2021_action_both_9072-a14_9071_user_id_2021-02-11_104901"
# 9095 appears nowhere else as a primary id -> harmless alias of 9065.
ALIASED = "nusar-2021_action_both_9065-a17_9095_user_id_2021-02-17_114124"


def test_parses_expected_fields():
    p = parse_recording(REAL[0])
    assert p["pid_a"] == "9011"
    assert p["pid_b"] == "9011"
    assert p["toy"] == "a01"
    assert p["date"] == "2021-02-01"


@pytest.mark.parametrize("bad", ["", "garbage", "nusar-2021_action_both_x-a01"])
def test_malformed_names_return_none_rather_than_raise(bad):
    assert parse_recording(bad) is None
    assert participant_id(bad) is None


def test_agreeing_ids_are_assigned():
    idx = build_index(REAL)
    assert len(idx.assignment) == 4
    assert idx.participants == {"9011", "9071", "9072"}
    assert not idx.ambiguous and not idx.aliased


def test_alias_is_kept_under_primary_id():
    idx = build_index(REAL + [ALIASED])
    assert ALIASED in idx.aliased
    assert idx.assignment[ALIASED] == "9065"
    assert "9095" not in idx.participants


def test_ambiguous_recording_is_dropped_not_guessed():
    idx = build_index(REAL + [AMBIGUOUS])
    assert AMBIGUOUS in idx.ambiguous
    assert AMBIGUOUS not in idx.assignment
    assert idx.ambiguous[AMBIGUOUS] == ("9072", "9071")


def test_ambiguity_depends_on_the_rest_of_the_corpus():
    """The same name is resolvable in isolation and unresolvable in context.

    With no other 9071 recordings the second id is just an alias. Once 9071 is
    a real participant, the assignment becomes a coin flip and we refuse it.
    """
    alone = build_index([AMBIGUOUS])
    assert AMBIGUOUS in alone.assignment

    with_context = build_index([AMBIGUOUS, REAL[2]])
    assert AMBIGUOUS in with_context.ambiguous


def test_unparsed_names_are_reported():
    idx = build_index(REAL + ["not-a-recording"])
    assert idx.unparsed == ["not-a-recording"]
    assert len(idx.assignment) == 4


def test_video_column_with_view_suffix_is_handled():
    """Annotation CSVs append a camera view to the recording name."""
    from participants import recording_from_video
    v = REAL[0] + "/C10095_rgb.mp4"
    assert recording_from_video(v) == REAL[0]
    assert participant_id(v) == "9011"
    assert parse_recording(v)["toy"] == "a01"


def test_index_builds_from_video_column_entries():
    videos = [r + "/HMC_21110305_mono10bit.mp4" for r in REAL]
    idx = build_index(videos)
    assert idx.participants == {"9011", "9071", "9072"}
    assert not idx.unparsed
