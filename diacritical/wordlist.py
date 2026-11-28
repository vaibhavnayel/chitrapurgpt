"""The Diacritical word list as the dashboard (admin_words.py) changes it: the list table, uploads (reviews) and the
decisions read from their filled reports, change batches with undo, and the pending list. Plain psycopg 3 (sync; the
pages call it through asyncio.to_thread).

Which list: env DIACRITICAL_WORDS_LIST = "trial" (default) or "live".
  trial -> table diacritic_mappings_trial, a copy of the real list made on first use (same columns, own ids kept,
           own sequence); reset_trial() copies it again. The translator never reads it.
  live  -> table diacritic_mappings, the real list the translator reads on every request.
The database is DIACRITICAL_DATABASE_URL (never ChitrapurGPT's DATABASE_URL; same guard as diacritical_mount.py).

Support tables (created if missing; every row has list_name 'trial' or 'live', so the two never mix):
  dia_review   an upload (kind document | list) with its items (harvest.review_document / review_list), or a filled
               pending list (kind pending); 'decisions' holds what the uploaded decisions said until Apply;
               status open | applied | undone | cancelled
  dia_batch    one change to the list (kind review | pending | undo | reset), who and when, undone or not
  dia_change   the entries a batch changed, with the values before and after (what undo puts back)
  dia_pending  words waiting for the Diacriticals team (with notes, and 'also': other ways of typing the word, added
               with it when it is answered Add), until resolved
The old diacritic_mapping_changes table (used by the clean-up scripts) is never touched.
Every write takes a per-list advisory lock; Apply works out its plan again under the lock against the list as it is
then, and refuses if that isn't what the confirm page showed (sheets.fingerprint).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import timedelta, timezone

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from . import harvest, sheets

LIVE_TABLE = "diacritic_mappings"
TRIAL_TABLE = "diacritic_mappings_trial"
_SCHEMA_LOCK = 727275
_WRITE_LOCK = 727276
NOW = "(now() at time zone 'utc')"
IST = timezone(timedelta(hours=5, minutes=30))


class NotConfigured(Exception):
    pass


class WordListError(Exception):
    """Something the person needs to know (the message is for people); nothing was changed."""


class Changed(WordListError):
    """The list or the pending list changed after the confirm page was shown: nothing was applied."""


def list_name(environ=os.environ) -> str:
    return "live" if (environ.get("DIACRITICAL_WORDS_LIST") or "").strip().lower() == "live" else "trial"


def table(name: str) -> str:
    return LIVE_TABLE if name == "live" else TRIAL_TABLE


def database_url(environ=os.environ) -> str:
    from diacritical_mount import _same_database
    url = (environ.get("DIACRITICAL_DATABASE_URL") or "").strip()
    if not url:
        raise NotConfigured("DIACRITICAL_DATABASE_URL is not set")
    if environ.get("DATABASE_URL") and _same_database(url, environ["DATABASE_URL"]):
        raise NotConfigured("DIACRITICAL_DATABASE_URL is ChitrapurGPT's own DATABASE_URL; refusing to use it")
    return url


_ready: set = set()


@contextmanager
def connect(environ=os.environ):
    """A connection (rows as dicts) with the tables ready. Commits when the block ends, rolls back on an error."""
    url = database_url(environ)
    name = list_name(environ)
    with psycopg.connect(url, row_factory=dict_row, connect_timeout=15) as conn:
        if (url, name) not in _ready:
            ensure_tables(conn, name)
            conn.commit()
            _ready.add((url, name))
        yield conn


def _t(name: str):
    return sql.Identifier(table(name))


def ensure_tables(conn, name: str) -> None:
    """The support tables, and in trial mode the trial copy of the real list (made once)."""
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK,))
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS dia_review (
          id serial PRIMARY KEY, list_name text NOT NULL, kind text NOT NULL, title text NOT NULL,
          files jsonb NOT NULL DEFAULT '[]', created_by text, created_at timestamp NOT NULL DEFAULT {NOW},
          status text NOT NULL DEFAULT 'open', items jsonb NOT NULL, decisions jsonb NOT NULL DEFAULT '{{}}',
          accepted_by text, accepted_at timestamp, batch_id int, updated_at timestamp);
        CREATE TABLE IF NOT EXISTS dia_batch (
          id serial PRIMARY KEY, list_name text NOT NULL, kind text NOT NULL, review_id int, summary text NOT NULL,
          created_by text, created_at timestamp NOT NULL DEFAULT {NOW}, undone_by text, undone_at timestamp,
          undo_of int);
        CREATE TABLE IF NOT EXISTS dia_change (
          id serial PRIMARY KEY, batch_id int NOT NULL REFERENCES dia_batch(id), list_name text NOT NULL,
          op text NOT NULL, mapping_id int, old_plain text, old_result text, new_plain text, new_result text);
        CREATE INDEX IF NOT EXISTS dia_change_batch ON dia_change(batch_id);
        CREATE INDEX IF NOT EXISTS dia_change_mapping ON dia_change(list_name, mapping_id);
        CREATE TABLE IF NOT EXISTS dia_pending (
          id serial PRIMARY KEY, list_name text NOT NULL, plain_text text NOT NULL, suggested text NOT NULL DEFAULT '',
          reason text NOT NULL DEFAULT '', source text NOT NULL DEFAULT '', page text NOT NULL DEFAULT '',
          context text NOT NULL DEFAULT '', added_by text, added_at timestamp NOT NULL DEFAULT {NOW},
          resolved_by text, resolved_at timestamp, resolution text, review_id int);
        CREATE INDEX IF NOT EXISTS dia_pending_plain ON dia_pending(list_name, plain_text);
        ALTER TABLE dia_pending ADD COLUMN IF NOT EXISTS notes text NOT NULL DEFAULT '';
        ALTER TABLE dia_pending ADD COLUMN IF NOT EXISTS resolved_batch int;
        ALTER TABLE dia_pending ADD COLUMN IF NOT EXISTS also text NOT NULL DEFAULT '';
    """)
    if name == "trial" and not _exists(conn, TRIAL_TABLE):
        conn.execute(sql.SQL("""
            CREATE TABLE {} (id serial PRIMARY KEY, plain_text varchar(255) UNIQUE NOT NULL,
              diacritic_text varchar(255) NOT NULL, created_at timestamp, updated_at timestamp)""").format(_t("trial")))
        _copy_live_to_trial(conn)


def _exists(conn, name: str) -> bool:
    return conn.execute("SELECT to_regclass(%s) IS NOT NULL AS x", (f"public.{name}",)).fetchone()["x"]


def _copy_live_to_trial(conn) -> int:
    conn.execute(sql.SQL("INSERT INTO {} (id, plain_text, diacritic_text, created_at, updated_at) "
                         "SELECT id, plain_text, diacritic_text, created_at, updated_at FROM {}")
                 .format(_t("trial"), _t("live")))
    conn.execute(sql.SQL("SELECT setval(pg_get_serial_sequence(%s, 'id'), COALESCE((SELECT max(id) FROM {}), 1), "
                         "(SELECT count(*) > 0 FROM {}))").format(_t("trial"), _t("trial")), (TRIAL_TABLE,))
    return conn.execute(sql.SQL("SELECT count(*) AS n FROM {}").format(_t("trial"))).fetchone()["n"]


def _lock(conn, name: str) -> None:
    conn.execute("SELECT pg_advisory_xact_lock(%s, %s)", (_WRITE_LOCK, 1 if name == "live" else 2))


# ---------- the list ----------

def entries(conn, name: str) -> list[dict]:
    return conn.execute(sql.SQL("SELECT id, plain_text, diacritic_text, created_at, updated_at FROM {} "
                                "ORDER BY plain_text").format(_t(name))).fetchall()


def mappings(conn, name: str) -> dict[str, str]:
    return {r["plain_text"]: r["diacritic_text"] for r in
            conn.execute(sql.SQL("SELECT plain_text, diacritic_text FROM {}").format(_t(name)))}


def pending_count(conn, name: str) -> int:
    return conn.execute("SELECT count(*) AS n FROM dia_pending WHERE list_name = %s AND resolved_at IS NULL",
                        (name,)).fetchone()["n"]


def counts(conn, name: str) -> dict:
    rows = entries(conn, name)
    return {"entries": len(rows), "words": len({harvest.small_result(r["diacritic_text"]) for r in rows}),
            "pending": pending_count(conn, name)}


def _new_batch(conn, name: str, kind: str, by: str, summary: str, review_id: int | None = None,
               undo_of: int | None = None) -> int:
    return conn.execute("INSERT INTO dia_batch (list_name, kind, review_id, summary, created_by, undo_of) "
                        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                        (name, kind, review_id, summary, by, undo_of)).fetchone()["id"]


def _log(conn, name: str, batch_id: int, op: str, mapping_id, old_plain=None, old_result=None, new_plain=None,
         new_result=None) -> None:
    conn.execute("INSERT INTO dia_change (batch_id, list_name, op, mapping_id, old_plain, old_result, new_plain, "
                 "new_result) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                 (batch_id, name, op, mapping_id, old_plain, old_result, new_plain, new_result))


def _add(conn, name: str, batch_id: int, plain: str, result: str, entry_id: int | None = None) -> int:
    if entry_id is None:
        new_id = conn.execute(sql.SQL(f"INSERT INTO {{}} (plain_text, diacritic_text, created_at, updated_at) "
                                      f"VALUES (%s, %s, {NOW}, {NOW}) RETURNING id").format(_t(name)),
                              (plain, result)).fetchone()["id"]
    else:  # undoing a delete: the entry comes back with its own id
        new_id = conn.execute(sql.SQL(f"INSERT INTO {{}} (id, plain_text, diacritic_text, created_at, updated_at) "
                                      f"VALUES (%s, %s, %s, {NOW}, {NOW}) RETURNING id").format(_t(name)),
                              (entry_id, plain, result)).fetchone()["id"]
    _log(conn, name, batch_id, "add", new_id, new_plain=plain, new_result=result)
    return new_id


def _update(conn, name: str, batch_id: int, row: dict, plain: str, result: str) -> None:
    conn.execute(sql.SQL(f"UPDATE {{}} SET plain_text = %s, diacritic_text = %s, updated_at = {NOW} WHERE id = %s")
                 .format(_t(name)), (plain, result, row["id"]))
    _log(conn, name, batch_id, "update", row["id"], row["plain_text"], row["diacritic_text"], plain, result)


def _delete(conn, name: str, batch_id: int, row: dict) -> None:
    conn.execute(sql.SQL("DELETE FROM {} WHERE id = %s").format(_t(name)), (row["id"],))
    _log(conn, name, batch_id, "delete", row["id"], row["plain_text"], row["diacritic_text"])


def _locked(conn, name: str, *, entry_id=None, plain=None) -> dict | None:
    if entry_id is not None:
        return conn.execute(sql.SQL("SELECT * FROM {} WHERE id = %s FOR UPDATE").format(_t(name)),
                            (entry_id,)).fetchone()
    return conn.execute(sql.SQL("SELECT * FROM {} WHERE plain_text = %s FOR UPDATE").format(_t(name)),
                        (plain,)).fetchone()


def _resolve_pending(conn, name: str, plains, by: str, resolution: str, batch: int) -> int:
    if not plains:
        return 0
    return conn.execute(f"UPDATE dia_pending SET resolved_by = %s, resolved_at = {NOW}, resolution = %s, "
                        "resolved_batch = %s WHERE list_name = %s AND resolved_at IS NULL AND plain_text = ANY(%s)",
                        (by, resolution, batch, name, list(plains))).rowcount


# ---------- uploads, decisions and Apply ----------

def create_review(conn, name: str, kind: str, title: str, files: list[dict], by: str, items: dict) -> int:
    return conn.execute("INSERT INTO dia_review (list_name, kind, title, files, created_by, items) "
                        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                        (name, kind, title, Jsonb(files), by, Jsonb(items))).fetchone()["id"]


def get_review(conn, name: str, review_id: int) -> dict | None:
    return conn.execute("SELECT r.*, b.undone_at FROM dia_review r LEFT JOIN dia_batch b ON b.id = r.batch_id "
                        "WHERE r.id = %s AND r.list_name = %s", (review_id, name)).fetchone()


def uploads(conn, name: str, limit: int = 30) -> list[dict]:
    """Recent uploads of documents and lists (newest first), with whether decisions are waiting to be applied. The
    counts of an upload made before one row per word are those of its regrouped items (harvest.regroup)."""
    rows = conn.execute("""
        SELECT r.id, r.kind, r.title, r.files, r.created_by, r.created_at, r.status, r.accepted_at, r.updated_at,
               r.batch_id, r.items->'counts' AS counts, r.decisions <> '{}'::jsonb AS decided, b.undone_at,
               r.kind = 'document' AND COALESCE((r.items->>'grouped')::int, 0) < %s AS regroup
        FROM dia_review r LEFT JOIN dia_batch b ON b.id = r.batch_id
        WHERE r.list_name = %s AND r.kind IN ('document', 'list') ORDER BY r.id DESC LIMIT %s""",
                        (harvest.GROUPED, name, limit)).fetchall()
    for r in rows:
        if r.pop("regroup"):
            items = conn.execute("SELECT items FROM dia_review WHERE id = %s", (r["id"],)).fetchone()["items"]
            r["counts"] = harvest.regroup(items)["counts"]
    return rows


def _day(t) -> str:
    """A database time (UTC) as an IST date: '20 Nov 2026'."""
    return t.replace(tzinfo=timezone.utc).astimezone(IST).strftime("%-d %b %Y") if t else ""


def applied_note(rv: dict) -> str:
    note = f"These decisions were already applied on {_day(rv['accepted_at'])}"
    if rv.get("undone_at"):
        note += f" (and undone on {_day(rv['undone_at'])})"
    return note + ". To decide again, upload the document again for a new report."


def store_report_answers(conn, name: str, filled: dict, by: str, file_name: str) -> int:
    """Read a filled report's decisions and keep them on its review (a new upload replaces them). Returns the review
    id. WordListError (for people) if the report doesn't belong here or was already applied."""
    rv = conn.execute("SELECT * FROM dia_review WHERE id = %s FOR UPDATE", (filled["review"],)).fetchone() \
        if filled.get("review") else None
    if rv is None or rv["kind"] not in ("document", "list"):
        raise WordListError("This report doesn't belong to any upload here (perhaps it was made on another copy of "
                            "the dashboard). Please download the report again from “Your uploads”.")
    if rv["list_name"] != name or filled.get("list") != name:
        raise WordListError(_other_list(filled.get("list") or rv["list_name"]))
    made = filled.get("made") or ""
    if made and made != rv["created_at"].isoformat(timespec="seconds"):
        raise WordListError("This report doesn't match upload " + str(rv["id"]) + " here (perhaps it was made on "
                            "another copy of the dashboard). Please download the report again from “Your uploads”.")
    if rv["status"] != "open":
        full = get_review(conn, rv["list_name"], rv["id"])
        raise WordListError(applied_note(full) if rv["status"] in ("applied", "undone") else
                            "This upload is closed, so its decisions can't be applied.")
    answers = sheets.answers_from_report(rv["items"], filled)
    answers.update(file=file_name, by=by, kind="report")
    conn.execute(f"UPDATE dia_review SET decisions = %s, updated_at = {NOW} WHERE id = %s",
                 (Jsonb(answers), rv["id"]))
    return rv["id"]


def store_pending_answers(conn, name: str, filled: dict, by: str, file_name: str) -> int:
    """Keep a filled pending list's rows as a 'pending' upload until Apply. Returns its id."""
    if filled["list"] != name:
        raise WordListError(_other_list(filled["list"]))
    answers = sheets.answers_from_pending(filled) | {"file": file_name, "by": by, "kind": "pending"}
    return conn.execute("INSERT INTO dia_review (list_name, kind, title, files, created_by, items, decisions) "
                        "VALUES (%s, 'pending', %s, %s, %s, '{}', %s) RETURNING id",
                        (name, "Pending list answers", Jsonb([{"name": file_name}]), by, Jsonb(answers))
                        ).fetchone()["id"]


def _other_list(made_on: str) -> str:
    if made_on == "trial":
        return ("This file was made on the trial copy of the list, but the dashboard now works on the real list. "
                "Please upload the document again for a new report.")
    return ("This file was made on the real list, but this dashboard works on the trial copy. Please upload it on the "
            "real dashboard.")


def cancel(conn, name: str, review_id: int) -> None:
    """Cancel on the confirm page: a report's uploaded decisions are forgotten (the upload waits for decisions
    again); a pending-list upload is closed."""
    rv = conn.execute("SELECT kind, status FROM dia_review WHERE id = %s AND list_name = %s FOR UPDATE",
                      (review_id, name)).fetchone()
    if rv is None or rv["status"] != "open":
        return
    if rv["kind"] == "pending":
        conn.execute(f"UPDATE dia_review SET status = 'cancelled', updated_at = {NOW} WHERE id = %s", (review_id,))
    else:
        conn.execute(f"UPDATE dia_review SET decisions = '{{}}', updated_at = {NOW} WHERE id = %s", (review_id,))


def plan_for(conn, name: str, rv: dict) -> dict:
    """What Apply would do now (sheets.plan / plan_pending against the list and pending list as they are), with
    'pending_new' (items not already waiting), 'pending_off' (open pending items it answers or removes) and
    'fingerprint'."""
    current = mappings(conn, name)
    open_rows = pending(conn, name)
    answers = rv["decisions"] or {}
    if rv["kind"] == "pending":
        p = sheets.plan_pending(answers, {r["id"]: r for r in open_rows}, current)
    else:
        source = ", ".join(f["name"] for f in rv["files"]) + f" (upload {rv['id']})"
        if answers.get("cells") is not None:
            # read again from the cells as uploaded, so decisions stored before a change in how they are read (one
            # row per word, a correct spelling being enough) are read the way they are now
            answers = answers | sheets.answers_from_report(rv["items"], {"cells": answers["cells"],
                                                                         "extra": answers.get("extra_cells") or []})
        p = sheets.plan(rv["items"], answers, current, source)
    waiting = {(r["plain_text"], r["suggested"]) for r in open_rows}
    new, seen = [], set()
    for x in p["pending"]:
        if (x["plain"], x["suggested"]) not in waiting | seen:
            new.append(x)
            seen.add((x["plain"], x["suggested"]))
    p["pending_new"] = new
    p["pending_already"] = len(p["pending"]) - len(new)
    touched = {c["plain"] for k in ("add", "update") for c in p[k]} | set(p["resolve"])
    off = {r["id"] for r in open_rows if r["plain_text"] in touched} | set(p.get("remove", [])) | set(p.get("done", []))
    p["pending_off"] = len(off)
    p["fingerprint"] = sheets.fingerprint(p)
    return p


def apply(conn, name: str, review_id: int, by: str, fingerprint: str) -> dict:
    """Apply an upload's decisions in one transaction: the changes to the list (a batch with its change rows), new
    pending items, pending items answered, removed or with new notes. The plan is worked out again under the lock;
    if it isn't what the confirm page showed (fingerprint), nothing is applied (Changed). Returns {'batch', 'plan'}."""
    _lock(conn, name)
    rv = conn.execute("SELECT * FROM dia_review WHERE id = %s AND list_name = %s FOR UPDATE",
                      (review_id, name)).fetchone()
    if rv is None or rv["status"] != "open" or not rv["decisions"]:
        raise WordListError("There is nothing waiting to be applied here (perhaps it was applied or cancelled "
                            "already).")
    p = plan_for(conn, name, rv)
    if p["fingerprint"] != fingerprint:
        raise Changed("The word list or the pending list changed after this page was shown, so nothing was applied. "
                      "Please check the summary again.")
    pending_upload = rv["kind"] == "pending"
    parts = [f"{len(p[k])} {label}" for k, label in (("add", "added"), ("update", "changed"), ("delete", "deleted"))
             if p[k]]
    ways = sum(1 for c in p["add"] if c.get("also_for"))
    if ways:
        parts[0] += f" ({ways} of them other ways of typing)" if ways > 1 else " (1 of them another way of typing)"
    if p["pending_new"]:
        parts.append(f"{len(p['pending_new'])} put on the pending list")
    if p.get("remove"):
        parts.append(f"{len(p['remove'])} removed from the pending list")
    what = ", ".join(parts) if parts else "nothing changed in the list"
    title = (f"Pending list answers ({rv['files'][0]['name']})" if pending_upload else
             f"Report {review_id} ({rv['title']})")
    batch = _new_batch(conn, name, "pending" if pending_upload else "review", by, f"{title}: {what}",
                       review_id=review_id)
    for c in p["delete"]:
        _delete(conn, name, batch, _locked(conn, name, plain=c["plain"]))
    for c in p["update"]:
        _update(conn, name, batch, _locked(conn, name, plain=c["plain"]), c["plain"], c["result"])
    for c in p["add"]:
        _add(conn, name, batch, c["plain"], c["result"])
    changed = {c["plain"] for k in ("add", "update") for c in p[k]}
    where = "the pending list" if pending_upload else f"report {review_id}"
    _resolve_pending(conn, name, changed, by, f"added or changed from {where} (change {batch})", batch)
    _resolve_pending(conn, name, set(p["resolve"]) - changed, by, f"confirmed as it is in {where}", batch)
    for pid in p.get("remove", []):
        conn.execute(f"UPDATE dia_pending SET resolved_by = %s, resolved_at = {NOW}, resolution = 'removed', "
                     "resolved_batch = %s WHERE id = %s AND list_name = %s AND resolved_at IS NULL",
                     (by, batch, pid, name))
    for pid, notes in (p.get("notes") or {}).items():
        conn.execute("UPDATE dia_pending SET notes = %s WHERE id = %s AND list_name = %s", (notes, int(pid), name))
    for pid, also in (p.get("also_set") or {}).items():
        conn.execute("UPDATE dia_pending SET also = %s WHERE id = %s AND list_name = %s", (also, int(pid), name))
    for item in p["pending"]:
        if item not in p["pending_new"] and item.get("notes"):   # already waiting: its notes get the new remark
            conn.execute("UPDATE dia_pending SET notes = CASE WHEN notes = '' THEN %s ELSE notes || ' · ' || %s END "
                         "WHERE list_name = %s AND plain_text = %s AND suggested = %s AND resolved_at IS NULL "
                         "AND position(%s in notes) = 0",
                         (item["notes"], item["notes"], name, item["plain"], item["suggested"], item["notes"]))
        if item not in p["pending_new"] and item.get("also"):  # already waiting: it takes the other ways of typing too
            for row in conn.execute("SELECT id, also FROM dia_pending WHERE list_name = %s AND plain_text = %s AND "
                                    "suggested = %s AND resolved_at IS NULL",
                                    (name, item["plain"], item["suggested"])).fetchall():
                merged = sheets.also_text(", ".join(x for x in (row["also"], item["also"]) if x), item["plain"])
                if merged != row["also"]:
                    conn.execute("UPDATE dia_pending SET also = %s WHERE id = %s", (merged, row["id"]))
    for item in p["pending_new"]:
        conn.execute("INSERT INTO dia_pending (list_name, plain_text, suggested, reason, source, page, context, notes, "
                     "also, added_by, review_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                     (name, item["plain"], item["suggested"], item["reason"], item["source"], item["page"],
                      item["context"], item.get("notes") or "", item.get("also") or "", by, review_id))
    conn.execute(f"UPDATE dia_review SET status = 'applied', accepted_by = %s, accepted_at = {NOW}, batch_id = %s "
                 "WHERE id = %s", (by, batch, review_id))
    return {"batch": batch, "plan": p}


# ---------- pending ----------

def pending(conn, name: str, *, resolved: bool = False, limit: int = 2000) -> list[dict]:
    if resolved:
        return conn.execute("SELECT * FROM dia_pending WHERE list_name = %s AND resolved_at IS NOT NULL "
                            "ORDER BY resolved_at DESC LIMIT %s", (name, limit)).fetchall()
    return conn.execute("SELECT * FROM dia_pending WHERE list_name = %s AND resolved_at IS NULL "
                        "ORDER BY plain_text, id", (name,)).fetchall()


# ---------- history and undo ----------

def batches(conn, name: str, limit: int = 200) -> list[dict]:
    return conn.execute("""
        SELECT b.*, count(c.id) FILTER (WHERE c.op = 'add') AS adds,
               count(c.id) FILTER (WHERE c.op = 'update') AS updates,
               count(c.id) FILTER (WHERE c.op = 'delete') AS deletes
        FROM dia_batch b LEFT JOIN dia_change c ON c.batch_id = b.id
        WHERE b.list_name = %s GROUP BY b.id ORDER BY b.id DESC LIMIT %s""", (name, limit)).fetchall()


def get_batch(conn, name: str, batch_id: int) -> dict | None:
    return conn.execute("SELECT * FROM dia_batch WHERE id = %s AND list_name = %s", (batch_id, name)).fetchone()


def changes(conn, batch_id: int) -> list[dict]:
    return conn.execute("SELECT * FROM dia_change WHERE batch_id = %s ORDER BY id", (batch_id,)).fetchall()


def undo(conn, name: str, batch_id: int, by: str) -> int:
    """Put back what a batch changed, as a new 'undo' batch, in one transaction; the pending words it answered go back
    on the pending list (words it put on the pending list stay there). Refused (WordListError naming the words) if
    any of its entries was changed again since."""
    _lock(conn, name)
    b = conn.execute("SELECT * FROM dia_batch WHERE id = %s AND list_name = %s FOR UPDATE",
                     (batch_id, name)).fetchone()
    if b is None:
        raise WordListError("That change doesn't exist.")
    if b["undone_at"]:
        raise WordListError("That change was already undone.")
    if b["kind"] in ("undo", "reset"):
        raise WordListError("This kind of change can't be undone.")
    if conn.execute("SELECT 1 FROM dia_batch WHERE list_name = %s AND kind = 'reset' AND id > %s",
                    (name, batch_id)).fetchone():
        raise WordListError("The trial was started again from the real list after this change, so it can't be undone.")
    rows = changes(conn, batch_id)
    if not rows:
        raise WordListError("That change didn't change any words, so there is nothing to undo.")
    later = []
    for c in rows:
        now = _locked(conn, name, entry_id=c["mapping_id"])
        if c["op"] in ("add", "update"):
            if now is None or (now["plain_text"], now["diacritic_text"]) != (c["new_plain"], c["new_result"]):
                later.append(c["new_plain"])
        elif now is not None or _locked(conn, name, plain=c["old_plain"]) is not None:
            later.append(c["old_plain"])
    if later:
        raise WordListError("These words were changed again after this, so it can't be undone (change them by hand "
                            "instead): " + ", ".join(sorted(set(later))))
    new = _new_batch(conn, name, "undo", by, f"Undid change {batch_id}: {b['summary']}", undo_of=batch_id)
    for c in reversed(rows):
        if c["op"] == "add":
            _delete(conn, name, new, _locked(conn, name, entry_id=c["mapping_id"]))
        elif c["op"] == "update":
            _update(conn, name, new, _locked(conn, name, entry_id=c["mapping_id"]), c["old_plain"], c["old_result"])
        else:
            _add(conn, name, new, c["old_plain"], c["old_result"], entry_id=c["mapping_id"])
    conn.execute(f"UPDATE dia_batch SET undone_by = %s, undone_at = {NOW} WHERE id = %s", (by, batch_id))
    conn.execute("UPDATE dia_pending SET resolved_at = NULL, resolved_by = NULL, resolution = NULL, "
                 "resolved_batch = NULL WHERE list_name = %s AND resolved_batch = %s", (name, batch_id))
    if b["review_id"]:
        conn.execute(f"UPDATE dia_review SET status = 'undone', updated_at = {NOW} "
                     "WHERE id = %s AND status = 'applied'", (b["review_id"],))
    return new


def reset_trial(conn, by: str) -> int:
    """Start the trial again: the trial copy becomes the real list again (logged as a 'reset' batch; earlier trial
    changes can't be undone after it). The pending list and uploads are kept."""
    _lock(conn, "trial")
    conn.execute(sql.SQL("DELETE FROM {}").format(_t("trial")))
    n = _copy_live_to_trial(conn)
    return _new_batch(conn, "trial", "reset", by, f"Started the trial again from the real list ({n:,} entries)")

