"""The collections picker: people with access to more than one collection choose which ones a chat searches.

Shown as chat settings (the settings button in the message box): an "All my collections" tick box, then one per
collection. A new chat searches all of them. The choice is kept with the chat (Chainlit saves chat settings in the
thread), so a reopened chat searches what it searched before; with "All" ticked it also searches collections the
person gained since. Access itself (who may search what) is still set by admins at /admin; this only narrows it.

The settings panel is submitted as a whole, so `update` works out what changed against the previous choice:
ticking "All" selects everything, unticking only "All" clears the boxes, and otherwise the individual boxes win
("All" then shows whether every box is ticked).
"""
from __future__ import annotations

from chainlit.input_widget import Checkbox

ALL = "kb_all"
PREFIX = "kb:"


def box_id(kb_id: str) -> str:
    return PREFIX + kb_id


def offered(kbs: list[dict]) -> bool:
    """The picker is only for people with more than one collection."""
    return len(kbs) > 1


def widgets(kbs: list[dict], selected: set[str]) -> list[Checkbox]:
    ids = [k["id"] for k in kbs]
    out = [Checkbox(id=ALL, label="All my collections", initial=all(i in selected for i in ids),
                    description="Search every collection you have access to.")]
    out += [Checkbox(id=box_id(k["id"]), label=k["name"], initial=k["id"] in selected,
                     description=k.get("description") or None) for k in kbs]
    return out


def from_settings(kbs: list[dict], settings: dict | None) -> set[str]:
    """The collections a chat searches, from its saved settings. No saved choice (a new chat, or one from before
    the picker): all of them. A saved choice that matches none of the person's collections any more: all of them."""
    ids = [k["id"] for k in kbs]
    settings = settings or {}
    if ALL not in settings or settings.get(ALL):
        return set(ids)
    chosen = {i for i in ids if settings.get(box_id(i))}
    if chosen:
        return chosen
    ticked_any = any(key.startswith(PREFIX) and value for key, value in settings.items())
    # Nothing ticked on purpose stays empty (they're asked to tick one); ticked collections they've since lost
    # access to fall back to all of theirs.
    return set(ids) if ticked_any else set()


def update(kbs: list[dict], settings: dict, previous: set[str]) -> set[str]:
    """The new choice after the person submits the panel (may be empty: they unticked everything)."""
    ids = [k["id"] for k in kbs]
    previous = {i for i in previous if i in ids}
    was_all = previous == set(ids)
    ticked = {i for i in ids if settings.get(box_id(i))}
    all_ticked = bool(settings.get(ALL))
    if all_ticked and not was_all:
        return set(ids)                  # they ticked "All"
    if not all_ticked and was_all and ticked == previous:
        return set()                     # they unticked only "All": start from nothing
    return ticked


def describe(kbs: list[dict], selected: set[str]) -> str:
    """'all your collections' or 'Sunbeam and KSA', in the person's order."""
    names = [k["name"] for k in kbs if k["id"] in selected]
    if len(names) == len(kbs):
        return "all your collections"
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]
