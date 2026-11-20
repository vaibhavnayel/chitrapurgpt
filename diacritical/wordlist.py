"""The Diacritical word list as the dashboard (admin_words.py) changes it: the list table, reviews, change batches with
undo, and the pending list. Plain psycopg 3 (sync; the pages call it through asyncio.to_thread).

Which list: env DIACRITICAL_WORDS_LIST = "trial" (default) or "live".
  trial -> table diacritic_mappings_trial, a copy of the real list made on first use (same columns, own ids kept,
           own sequence); reset_trial() copies it again. The translator never reads it.
  live  -> table diacritic_mappings, the real list the translator reads on every request.
The database is DIACRITICAL_DATABASE_URL (never ChitrapurGPT's DATABASE_URL; same guard as diacritical_mount.py).

Support tables (created if missing; every row has list_name 'trial' or 'live', so the two never mix):
  dia_review   an upload's review: its items (harvest.review_document / review_list), the choices, its status
  dia_batch    one change to the list (kind review | edit | pending | undo | reset), who and when, undone or not
  dia_change   the entries a batch changed, with the values before and after (what undo puts back)
  dia_pending  words waiting for the Diacriticals team, until resolved
The old diacritic_mapping_changes table (used by the clean-up scripts) is never touched.
Every write takes a per-list advisory lock, and checks the entries it changes against what the page showed.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from . import harvest
from .cleaning import clean_plain, clean_result
from .diacritics import translate_text

LIVE_TABLE = "diacritic_mappings"
TRIAL_TABLE = "diacritic_mappings_trial"
_SCHEMA_LOCK = 727275
_WRITE_LOCK = 727276
NOW = "(now() at time zone 'utc')"


class NotConfigured(Exception):
    pass


class WordListError(Exception):
    """Something the person needs to know (the message is for people); nothing was changed."""


class Conflict(WordListError):
    """Entries changed since the review or form was made. words: [(plain word, what it is now or None)]."""
    def __init__(self, words):
        self.words = words
        listed = ", ".join(w for w, _ in words[:20]) + (" …" if len(words) > 20 else "")
        super().__init__(f"Some words were changed in the list since this was made: {listed}")


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


def entry(conn, name: str, entry_id: int) -> dict | None:
    return conn.execute(sql.SQL("SELECT * FROM {} WHERE id = %s").format(_t(name)), (entry_id,)).fetchone()


def entry_by_plain(conn, name: str, plain: str) -> dict | None:
    return conn.execute(sql.SQL("SELECT * FROM {} WHERE plain_text = %s").format(_t(name)), (plain,)).fetchone()


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


def _resolve_pending(conn, name: str, plains, by: str, resolution: str) -> int:
    if not plains:
        return 0
    return conn.execute(f"UPDATE dia_pending SET resolved_by = %s, resolved_at = {NOW}, resolution = %s "
                        "WHERE list_name = %s AND resolved_at IS NULL AND plain_text = ANY(%s)",
                        (by, resolution, name, list(plains))).rowcount


def save_entry(conn, name: str, by: str, *, plain: str, result: str, entry_id: int | None = None,
               expected: tuple[str, str] | None = None, pending_id: int | None = None) -> int:
    """Add an entry (entry_id None) or change one, as one batch. expected: (plain, result) the form showed; if the
    entry changed since, nothing is saved (Conflict). Saving from a pending item resolves it. Returns the batch id."""
    plain, result = clean_plain(plain), clean_result(result)
    if not plain or not result:
        raise WordListError("Please fill in the plain word and the result.")
    _lock(conn, name)
    other = _locked(conn, name, plain=plain)
    if other and other["id"] != entry_id:
        raise WordListError(f"“{plain}” is already in the list (as {harvest.describe(other['diacritic_text'])}).")
    kind = "pending" if pending_id else "edit"
    if entry_id is None:
        batch = _new_batch(conn, name, kind, by, f"Added {plain} → {harvest.describe(result)}")
        _add(conn, name, batch, plain, result)
    else:
        row = _locked(conn, name, entry_id=entry_id)
        if row is None:
            raise WordListError("This entry was deleted meanwhile. Nothing was saved.")
        if expected and (row["plain_text"], row["diacritic_text"]) != expected:
            raise Conflict([(row["plain_text"], row["diacritic_text"])])
        if (row["plain_text"], row["diacritic_text"]) == (plain, result):
            raise WordListError("Nothing was changed.")
        what = (f"Changed {plain}: {harvest.describe(row['diacritic_text'])} → {harvest.describe(result)}"
                if row["plain_text"] == plain else f"Changed {row['plain_text']} → {plain} ({harvest.describe(result)})")
        batch = _new_batch(conn, name, kind, by, what)
        _update(conn, name, batch, row, plain, result)
    resolved = {plain}
    if pending_id:
        item = conn.execute("SELECT plain_text FROM dia_pending WHERE id = %s AND list_name = %s",
                            (pending_id, name)).fetchone()
        if item:
            resolved.add(item["plain_text"])
            conn.execute(f"UPDATE dia_pending SET resolved_by = %s, resolved_at = {NOW}, resolution = %s "
                         "WHERE id = %s AND resolved_at IS NULL", (by, f"saved in the list (change {batch})",
                                                                   pending_id))
    _resolve_pending(conn, name, resolved, by, f"saved in the list (change {batch})")
    return batch


def delete_entry(conn, name: str, by: str, entry_id: int, expected: tuple[str, str] | None = None) -> int:
    _lock(conn, name)
    row = _locked(conn, name, entry_id=entry_id)
    if row is None:
        raise WordListError("This entry was already deleted.")
    if expected and (row["plain_text"], row["diacritic_text"]) != expected:
        raise Conflict([(row["plain_text"], row["diacritic_text"])])
    batch = _new_batch(conn, name, "edit", by, f"Deleted {row['plain_text']} ({harvest.describe(row['diacritic_text'])})")
    _delete(conn, name, batch, row)
    return batch


# ---------- reviews ----------

def create_review(conn, name: str, kind: str, title: str, files: list[dict], by: str, items: dict) -> int:
    decisions = harvest.default_decisions(items)
    return conn.execute("INSERT INTO dia_review (list_name, kind, title, files, created_by, items, decisions) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                        (name, kind, title, Jsonb(files), by, Jsonb(items), Jsonb(decisions))).fetchone()["id"]


def get_review(conn, name: str, review_id: int) -> dict | None:
    return conn.execute("SELECT * FROM dia_review WHERE id = %s AND list_name = %s", (review_id, name)).fetchone()


def reviews(conn, name: str, status: str, limit: int = 50) -> list[dict]:
    return conn.execute("SELECT id, kind, title, files, created_by, created_at, status, accepted_by, accepted_at, "
                        "batch_id, items->'counts' AS counts FROM dia_review WHERE list_name = %s AND status = %s "
                        "ORDER BY id DESC LIMIT %s", (name, status, limit)).fetchall()


def save_decisions(conn, name: str, review_id: int, decisions: dict) -> None:
    n = conn.execute(f"UPDATE dia_review SET decisions = %s, updated_at = {NOW} "
                     "WHERE id = %s AND list_name = %s AND status = 'open'",
                     (Jsonb(decisions), review_id, name)).rowcount
    if not n:
        raise WordListError("This review is already finished, so its choices can't be changed.")


def discard_review(conn, name: str, review_id: int, by: str) -> None:
    n = conn.execute(f"UPDATE dia_review SET status = 'discarded', accepted_by = %s, accepted_at = {NOW} "
                     "WHERE id = %s AND list_name = %s AND status = 'open'", (by, review_id, name)).rowcount
    if not n:
        raise WordListError("This review is already finished.")


def accept_review(conn, name: str, review_id: int, by: str) -> dict:
    """Apply the review's choices in one transaction: changes to the list (a 'review' batch with its change rows),
    new pending items, and pending items resolved by the change. Each entry is checked against the review's snapshot
    first; if any changed meanwhile, nothing is applied (Conflict). Returns {'batch', 'plan', 'resolved'}."""
    _lock(conn, name)
    rv = conn.execute("SELECT * FROM dia_review WHERE id = %s AND list_name = %s FOR UPDATE",
                      (review_id, name)).fetchone()
    if rv is None or rv["status"] != "open":
        raise WordListError("This review is already finished.")
    source = ", ".join(f["name"] for f in rv["files"]) + f" (review {review_id})"
    p = harvest.plan(rv["items"], rv["decisions"], source)
    if p["undecided"]:
        raise WordListError(f"{len(p['undecided'])} rows still need a choice.")
    if p["clashing"]:
        raise WordListError("Two rows want to change the same word differently: " + ", ".join(p["clashing"]))
    rows, conflicts = {}, []
    for op in ("update", "delete"):
        for c in p[op]:
            row = _locked(conn, name, plain=c["plain"])
            if row is None or row["diacritic_text"] != c["old"]:
                conflicts.append((c["plain"], row["diacritic_text"] if row else None))
            rows[c["plain"]] = row
    for c in p["add"]:
        row = _locked(conn, name, plain=c["plain"])
        if row is not None:
            conflicts.append((c["plain"], row["diacritic_text"]))
    if conflicts:
        raise Conflict(conflicts)
    parts = [f"{len(p[k])} {label}" for k, label in (("add", "added"), ("update", "changed"), ("delete", "deleted"),
                                                    ("pending", "sent to pending")) if p[k]]
    batch = _new_batch(conn, name, "review", by, f"Review {review_id} ({rv['title']}): "
                       + (", ".join(parts) if parts else "nothing changed"), review_id=review_id)
    for c in p["delete"]:
        _delete(conn, name, batch, rows[c["plain"]])
    for c in p["update"]:
        _update(conn, name, batch, rows[c["plain"]], c["plain"], c["result"])
    for c in p["add"]:
        _add(conn, name, batch, c["plain"], c["result"])
    open_pending = {(r["plain_text"], r["suggested"]) for r in conn.execute(
        "SELECT plain_text, suggested FROM dia_pending WHERE list_name = %s AND resolved_at IS NULL", (name,))}
    changed = {c["plain"] for k in ("add", "update") for c in p[k]}
    resolved = _resolve_pending(conn, name, changed, by, f"added or changed in review {review_id}")
    resolved += _resolve_pending(conn, name, set(p["resolve"]) - changed, by,
                                 f"confirmed as it is in review {review_id}")
    for item in p["pending"]:
        if (item["plain"], item["suggested"]) in open_pending:
            continue  # already waiting, with the same suggestion
        conn.execute("INSERT INTO dia_pending (list_name, plain_text, suggested, reason, source, page, context, "
                     "added_by, review_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                     (name, item["plain"], item["suggested"], item["reason"], item["source"], item["page"],
                      item["context"], by, review_id))
        open_pending.add((item["plain"], item["suggested"]))
    conn.execute(f"UPDATE dia_review SET status = 'accepted', accepted_by = %s, accepted_at = {NOW}, batch_id = %s "
                 "WHERE id = %s", (by, batch, review_id))
    return {"batch": batch, "plan": p, "resolved": resolved}


def refresh_review(conn, name: str, review_id: int, words: list[tuple[str, str | None]]) -> None:
    """After a Conflict: the rows for these words get the list's current value, a note, and their choice cleared, so
    they are decided again."""
    rv = conn.execute("SELECT items, decisions FROM dia_review WHERE id = %s AND list_name = %s FOR UPDATE",
                      (review_id, name)).fetchone()
    if rv is None:
        return
    now = dict(words)
    items, decisions = rv["items"], rv["decisions"]
    current = mappings(conn, name)
    for section in items["sections"].values():
        for item in section:
            if item.get("plain") not in now or "id" not in item:
                continue
            value = now[item["plain"]]
            item["changed_meanwhile"] = harvest.describe(value) if value else "deleted"
            if "entry" in item:
                item["entry"] = value
            if item.get("kind") == "clash":
                typed = harvest.plain_typed(item["form"])
                item["gives"] = translate_text(typed, current)
            if item.get("kind") in ("new", "lnew") and value is not None:
                item["now_in_list"] = value
            d = decisions.get(item["id"]) or {}
            d.pop("choice", None)
            decisions[item["id"]] = d
    conn.execute(f"UPDATE dia_review SET items = %s, decisions = %s, updated_at = {NOW} WHERE id = %s",
                 (Jsonb(items), Jsonb(decisions), review_id))


# ---------- pending ----------

def pending(conn, name: str, *, resolved: bool = False, limit: int = 2000) -> list[dict]:
    if resolved:
        return conn.execute("SELECT * FROM dia_pending WHERE list_name = %s AND resolved_at IS NOT NULL "
                            "ORDER BY resolved_at DESC LIMIT %s", (name, limit)).fetchall()
    return conn.execute("SELECT * FROM dia_pending WHERE list_name = %s AND resolved_at IS NULL "
                        "ORDER BY plain_text, id", (name,)).fetchall()


def pending_item(conn, name: str, item_id: int) -> dict | None:
    return conn.execute("SELECT * FROM dia_pending WHERE id = %s AND list_name = %s", (item_id, name)).fetchone()


def remove_pending(conn, name: str, item_id: int, by: str) -> str:
    row = conn.execute(f"UPDATE dia_pending SET resolved_by = %s, resolved_at = {NOW}, resolution = 'removed' "
                       "WHERE id = %s AND list_name = %s AND resolved_at IS NULL RETURNING plain_text",
                       (by, item_id, name)).fetchone()
    if row is None:
        raise WordListError("That word is no longer on the pending list.")
    return row["plain_text"]


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
    """Put back what a batch changed, as a new 'undo' batch, in one transaction. Refused (WordListError naming the
    words) if any of its entries was changed again since."""
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
    return new


def reset_trial(conn, by: str) -> int:
    """Start the trial again: the trial copy becomes the real list again (logged as a 'reset' batch; earlier trial
    changes can't be undone after it). The pending list and reviews are kept."""
    _lock(conn, "trial")
    conn.execute(sql.SQL("DELETE FROM {}").format(_t("trial")))
    n = _copy_live_to_trial(conn)
    return _new_batch(conn, "trial", "reset", by, f"Started the trial again from the real list ({n:,} entries)")

