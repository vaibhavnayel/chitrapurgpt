"""The collections picker's logic (kb_picker.py). The chat wiring is tested in tests/test_agent.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kb_picker  # noqa: E402
from kb_picker import ALL, box_id  # noqa: E402

SUNBEAM = {"id": "sunbeam", "name": "The Chitrapur Sunbeam", "description": "the magazine of Shri Chitrapur Math"}
KSA = {"id": "ksa", "name": "Kanara Saraswat", "description": ""}
OTHER = {"id": "other", "name": "Other", "description": "x"}
KBS = [SUNBEAM, KSA]
BOTH = {"sunbeam", "ksa"}


def settings(all_=False, **boxes):
    return {ALL: all_, **{box_id(k): v for k, v in boxes.items()}}


def test_offered_only_with_more_than_one_collection():
    assert not kb_picker.offered([])
    assert not kb_picker.offered([SUNBEAM])
    assert kb_picker.offered(KBS)


def test_widgets():
    ws = kb_picker.widgets(KBS, {"ksa"})
    assert [(w.id, w.label, w.initial) for w in ws] == [
        (ALL, "All my collections", False), ("kb:sunbeam", "The Chitrapur Sunbeam", False),
        ("kb:ksa", "Kanara Saraswat", True)]
    assert ws[1].description == "the magazine of Shri Chitrapur Math" and ws[2].description is None
    assert ws[0].initial is False and kb_picker.widgets(KBS, BOTH)[0].initial is True
    assert all(w.to_dict()["type"] == "checkbox" for w in ws)


def test_new_or_old_chat_without_a_choice_searches_everything():
    assert kb_picker.from_settings(KBS, None) == BOTH
    assert kb_picker.from_settings(KBS, {}) == BOTH
    assert kb_picker.from_settings(KBS, {"something_else": 1}) == BOTH


def test_saved_choice():
    assert kb_picker.from_settings(KBS, settings(sunbeam=False, ksa=True)) == {"ksa"}
    assert kb_picker.from_settings(KBS, settings(True, sunbeam=False, ksa=False)) == BOTH, "All wins"
    assert kb_picker.from_settings(KBS, settings(sunbeam=False, ksa=False)) == set(), "nothing ticked stays empty"


def test_saved_choice_against_changed_access():
    # "All" saved: collections gained since are searched too.
    assert kb_picker.from_settings([*KBS, OTHER], settings(True, sunbeam=True, ksa=True)) == {"sunbeam", "ksa", "other"}
    # A subset saved: a collection gained since stays off.
    assert kb_picker.from_settings([*KBS, OTHER], settings(sunbeam=False, ksa=True)) == {"ksa"}
    # Only collections they lost were ticked: back to all of theirs (not an empty search).
    assert kb_picker.from_settings([SUNBEAM, OTHER], settings(sunbeam=False, ksa=True)) == {"sunbeam", "other"}


def test_update():
    # Untick one box (All still ticked in the submitted form): the boxes win.
    assert kb_picker.update(KBS, settings(True, sunbeam=True, ksa=False), BOTH) == {"sunbeam"}
    # Tick All while a subset was chosen: everything, whatever the boxes say.
    assert kb_picker.update(KBS, settings(True, sunbeam=False, ksa=True), {"ksa"}) == BOTH
    # Tick the missing box by hand: everything.
    assert kb_picker.update(KBS, settings(False, sunbeam=True, ksa=True), {"ksa"}) == BOTH
    # Untick only All: start from nothing.
    assert kb_picker.update(KBS, settings(False, sunbeam=True, ksa=True), BOTH) == set()
    # Untick All and a box together: the boxes win.
    assert kb_picker.update(KBS, settings(False, sunbeam=True, ksa=False), BOTH) == {"sunbeam"}
    # Untick everything by hand.
    assert kb_picker.update(KBS, settings(False, sunbeam=False, ksa=False), {"ksa"}) == set()
    # No change at all.
    assert kb_picker.update(KBS, settings(False, sunbeam=False, ksa=True), {"ksa"}) == {"ksa"}
    # A previous choice naming a collection they no longer have.
    assert kb_picker.update(KBS, settings(False, sunbeam=True, ksa=False), {"sunbeam", "gone"}) == {"sunbeam"}


def test_describe():
    assert kb_picker.describe(KBS, BOTH) == "all your collections"
    assert kb_picker.describe(KBS, {"ksa"}) == "Kanara Saraswat"
    three = [*KBS, OTHER, {"id": "z", "name": "Zed", "description": ""}]
    assert kb_picker.describe(three, {"sunbeam", "ksa"}) == "The Chitrapur Sunbeam and Kanara Saraswat"
    assert kb_picker.describe(three, {"sunbeam", "ksa", "other"}) == "The Chitrapur Sunbeam, Kanara Saraswat and Other"
