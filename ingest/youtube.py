"""The Youtube Channel collection, from the two CSV files the channel's admin exports: the video list (one row per
video, with its description) and the playlist list (one row per video in each playlist).

Each public or unlisted video becomes one entry (title, date, length, playlists, link and description), and so does
each public or unlisted playlist (its videos in order, with links). Private and deleted videos and private playlists
are left out. Videos that appear only in the playlist file (the channel's video export misses some) are included
with what that file has: no description.
"""
from __future__ import annotations

import csv
import io
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime

from ingest.sources import Entry

SOURCE_NAME = "YouTube"
CHANNEL = "the Shri Chitrapur Math YouTube channel"
VIDEO_COLUMNS = {"Title", "Privacy Status", "Published Date", "URL", "Description"}
PLAYLIST_COLUMNS = {"Playlist Name", "Playlist Privacy", "Playlist URL", "Position in Playlist", "Video Title",
                    "Video Privacy", "Published Date", "Video URL"}
SHOWN = {"public", "unlisted"}


class CsvProblem(ValueError):
    """Something the admin needs to fix in the files (the message is shown to them)."""


def read_csv(text: str) -> tuple[list[str], list[dict]]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    rows = [{(k or "").strip(): (v or "").strip() for k, v in row.items() if k is not None} for row in reader]
    return [(c or "").strip() for c in reader.fieldnames or []], rows


def sort_files(files: list[tuple[str, str]]) -> tuple[list[dict], list[dict]]:
    """(video rows, playlist rows) from (file name, text) pairs, told apart by their columns."""
    videos, playlists, _ = read_files(files)
    return videos, playlists


def read_files(files: list[tuple[str, str]]) -> tuple[list[dict], list[dict], list[dict]]:
    """sort_files, plus what each file was: [{"name", "what", "rows"}] (shown as the lists last sent)."""
    videos = playlists = None
    sent = []
    for name, text in files:
        columns, rows = read_csv(text)
        if PLAYLIST_COLUMNS <= set(columns):
            playlists = rows
            sent.append({"name": name, "what": "playlist list", "rows": len(rows)})
        elif VIDEO_COLUMNS <= set(columns):
            videos = rows
            sent.append({"name": name, "what": "video list", "rows": len(rows)})
        else:
            raise CsvProblem(f"{name} doesn't look like either YouTube list: it has no "
                             f"{'Playlist Name' if 'Playlist Name' not in columns else 'Title'} column.")
    if videos is None or playlists is None:
        missing = "video list (youtube_master_catalog…)" if videos is None else "playlist list (youtube_playlists_contents…)"
        raise CsvProblem(f"Please choose both files. The {missing} is missing.")
    return videos, playlists, sorted(sent, key=lambda f: f["what"] != "video list")


def video_id(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path.strip("/") or None
    return (urllib.parse.parse_qs(parsed.query).get("v") or [None])[0]


def playlist_id(url: str) -> str | None:
    return (urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("list") or [None])[0]


def _privacy(value: str) -> str:
    return (value or "").split("/")[0].strip().lower()


def _date(value: str) -> tuple[str, str]:
    """("2026-03-04", "4 March 2026") from an ISO timestamp; blanks if it can't be read."""
    try:
        d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "", ""
    return d.date().isoformat(), f"{d.day} {d:%B %Y}"


def _length(seconds: str) -> str:
    try:
        s = int(float(seconds))
    except ValueError:
        return ""
    if s <= 0:
        return ""
    h, m = divmod(round(s / 60), 60)
    return f"{h} h {m} min" if h else f"{max(m, 1)} min"


def _unlisted_note(verb: str = "watch") -> str:
    return f" It is unlisted: it isn't shown publicly on the channel, but anyone with the link can {verb} it."


def build_entries(videos: list[dict], playlists: list[dict]) -> tuple[list[Entry], list[str]]:
    """The entries, and notes for the admin."""
    notes: list[str] = []
    info: dict[str, dict] = {}  # video id -> what we know
    left_out = Counter()
    no_link = 0
    for r in videos:
        vid = video_id(r.get("URL", ""))
        if not vid:
            no_link += 1
            continue
        info[vid] = {"title": r.get("Title", ""), "privacy": _privacy(r.get("Privacy Status", "")),
                     "date": r.get("Published Date", ""), "length": _length(r.get("Duration (s)", "")),
                     "format": r.get("Content Format", ""), "description": r.get("Description", ""),
                     "url": f"https://www.youtube.com/watch?v={vid}", "in_video_list": True}

    lists: dict[str, dict] = {}
    for r in playlists:
        pid, vid = playlist_id(r.get("Playlist URL", "")), video_id(r.get("Video URL", ""))
        if not pid or not vid:
            no_link += 1
            continue
        pl = lists.setdefault(pid, {"name": r.get("Playlist Name", ""), "privacy": _privacy(r.get("Playlist Privacy", "")),
                                    "url": f"https://www.youtube.com/playlist?list={pid}", "items": []})
        try:
            pos = int(float(r.get("Position in Playlist") or 0))
        except ValueError:
            pos = 0
        pl["items"].append((pos, vid))
        if vid not in info:
            info[vid] = {"title": r.get("Video Title", ""), "privacy": _privacy(r.get("Video Privacy", "")),
                         "date": r.get("Published Date", ""), "length": _length(r.get("Duration (s)", "")),
                         "format": r.get("Content Format", ""), "description": "",
                         "url": f"https://www.youtube.com/watch?v={vid}", "in_video_list": False}

    shown_lists = {pid: pl for pid, pl in lists.items() if pl["privacy"] in SHOWN and pl["items"]}
    member_of: dict[str, list[str]] = defaultdict(list)
    for pid, pl in shown_lists.items():
        for _, vid in sorted(pl["items"]):
            if pl["name"] not in member_of[vid]:
                member_of[vid].append(pl["name"])

    entries: list[Entry] = []
    only_in_playlists = 0
    for vid, v in info.items():
        if v["privacy"] not in SHOWN:
            left_out[v["privacy"] or "unknown"] += 1
            continue
        if not v["in_video_list"]:
            only_in_playlists += 1
        iso, nice = _date(v["date"])
        kind = "live stream" if v["format"] == "Live Stream" else "short" if v["format"] == "Short" else "video"
        lines = [v["title"],
                 f"A {kind} on {CHANNEL}" + (f", published {nice}" if nice else "")
                 + (f", {v['length']} long" if v["length"] else "") + "."
                 + (_unlisted_note() if v["privacy"] == "unlisted" else ""),
                 f"Link: {v['url']}"]
        if member_of.get(vid):
            lines.append("Playlists: " + "; ".join(member_of[vid]))
        if v["description"]:
            lines += ["", "Description:", v["description"]]
        metadata = {"source": SOURCE_NAME, "published_date": iso or None, "title": v["title"], "url": v["url"],
                    "kind": "YouTube " + kind, "privacy": v["privacy"]}
        if v["length"]:
            metadata["length"] = v["length"]
        if member_of.get(vid):
            metadata["playlists"] = "; ".join(member_of[vid])
        entries.append(Entry(key=f"video:{vid}", title=v["title"], text="\n".join(lines), metadata=metadata))

    for pid, pl in shown_lists.items():
        items = [(pos, vid) for pos, vid in sorted(pl["items"]) if info.get(vid, {}).get("privacy") in SHOWN]
        if not items:
            continue
        dates = sorted(d for d in (_date(info[vid]["date"])[0] for _, vid in items) if d)
        lines = [f"Playlist: {pl['name']}",
                 f"A playlist on {CHANNEL} with {len(items)} video{'s' if len(items) != 1 else ''}"
                 + (f", published {_date(info[items[0][1]]['date'])[1]}" if len(dates) == 1 else
                    f", published between {dates[0]} and {dates[-1]}" if dates else "") + "."
                 + (_unlisted_note("open") if pl["privacy"] == "unlisted" else ""),
                 f"Link: {pl['url']}", "", "Videos:"]
        for n, (_, vid) in enumerate(items, 1):
            v = info[vid]
            nice = _date(v["date"])[1]
            lines.append(f"{n}. {v['title']}" + (f" ({nice})" if nice else "")
                         + (" [unlisted]" if v["privacy"] == "unlisted" else "") + f" {v['url']}")
        metadata = {"source": SOURCE_NAME, "published_date": dates[-1] if dates else None,
                    "title": f"Playlist: {pl['name']}", "url": pl["url"], "kind": "YouTube playlist",
                    "privacy": pl["privacy"], "videos": len(items)}
        entries.append(Entry(key=f"playlist:{pid}", title=f"Playlist: {pl['name']}", text="\n".join(lines),
                             metadata=metadata))

    videos_n = sum(1 for e in entries if e.key.startswith("video:"))
    notes.append(f"{videos_n:,} videos ({sum(1 for e in entries if e.metadata.get('privacy') == 'unlisted' and e.key.startswith('video:')):,} "
                 f"of them unlisted) and {len(entries) - videos_n} playlists will be searchable.")
    if left_out:
        notes.append("Left out: " + ", ".join(f"{n:,} {k}" for k, n in sorted(left_out.items())) + " videos.")
    hidden_lists = sorted(pl["name"] for pl in lists.values() if pl["privacy"] not in SHOWN)
    if hidden_lists:
        notes.append("Private playlists left out: " + ", ".join(hidden_lists) + ".")
    if only_in_playlists:
        notes.append(f"{only_in_playlists:,} videos appear only in the playlist file, not in the video list, so "
                     f"they have no description.")
    if no_link:
        notes.append(f"{no_link:,} rows had no video or playlist link and were skipped.")
    return entries, notes
