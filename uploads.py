"""Turn files a user attaches to a message into content the agent can read.

  images (jpeg/png/gif/webp)   image blocks, downscaled to <= 1568 px on the long edge (what the API does anyway)
  PDF                          a native document block (the model sees text and page images); text if too big
  xlsx, docx, csv, text        extracted text
  anything else / failures     a note to the agent and a note for the user

Type is decided from the MIME type, then the file extension, then the file's magic bytes, then "is it UTF-8 text".
This module does no network calls.
"""
from __future__ import annotations

import base64
import csv
import io
import mimetypes
import os
import zipfile
from dataclasses import dataclass, field
from xml.etree import ElementTree

import filetype

IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
PDF_MIME = "application/pdf"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TEXT_MIMES = {"application/json", "application/xml", "application/x-yaml", "application/yaml", "application/csv"}

MAX_FILE_BYTES = 50 * 1024 * 1024        # anything bigger isn't read at all
MAX_IMAGE_EDGE = 1568
MAX_IMAGES = 10
MAX_PDF_NATIVE_BYTES = 10 * 1024 * 1024  # larger PDFs (or more pages) go as extracted text
MAX_PDF_NATIVE_PAGES = 100
MAX_PDF_NATIVE_TOTAL_BYTES = 15 * 1024 * 1024  # per message, to stay well under the 32 MB request limit
MAX_TEXT_CHARS_PER_FILE = 100_000
MAX_TEXT_CHARS_PER_MESSAGE = 200_000
SUPPORTED = "images (JPEG, PNG, GIF, WebP), PDFs, Word (.docx), Excel (.xlsx), CSV and text files"


class Unsupported(Exception):
    """The file can't be read; the message is shown to the user and the agent."""


@dataclass
class Converted:
    blocks: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # for the user


def _size(n: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024


def detect_mime(name: str, mime: str | None, path: str) -> str:
    mime = (mime or "").split(";")[0].strip().lower()
    if mime and mime != "application/octet-stream":
        return mime
    guessed = mimetypes.guess_type(name or "")[0]
    if guessed:
        return guessed
    try:
        kind = filetype.guess(path)
    except (OSError, TypeError):
        kind = None
    if kind:
        return kind.mime
    try:
        with open(path, "rb") as f:
            head = f.read(65536)
        if b"\x00" not in head:
            head.decode("utf-8")
            return "text/plain"
    except (OSError, UnicodeDecodeError):
        pass
    return mime or "application/octet-stream"


def _image_block(path: str, mime: str) -> dict:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            img.load()
            fmt = img.format
            if max(img.size) > MAX_IMAGE_EDGE:
                img.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
                out_fmt = fmt if fmt in ("JPEG", "PNG", "GIF", "WEBP") else "PNG"
                if out_fmt == "JPEG" and img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format=out_fmt, quality=90) if out_fmt == "JPEG" else img.save(buf, format=out_fmt)
                data, mime = buf.getvalue(), Image.MIME[out_fmt]
            else:
                with open(path, "rb") as f:
                    data = f.read()
                mime = Image.MIME.get(fmt, mime)
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise Unsupported(f"the image couldn't be opened ({type(e).__name__})") from e
    if mime not in IMAGE_MIMES:
        raise Unsupported(f"{mime} images aren't supported")
    return {"type": "image", "base64": base64.b64encode(data).decode(), "mime_type": mime}


def _pdf_page_count(path: str) -> int:
    from pypdf import PdfReader

    reader = PdfReader(path)
    if reader.is_encrypted and not reader.decrypt(""):
        raise Unsupported("the PDF is password-protected")
    return len(reader.pages)


def _pdf_text(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    if reader.is_encrypted and not reader.decrypt(""):
        raise Unsupported("the PDF is password-protected")
    parts, total = [], 0
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        parts.append(f"--- page {i} ---\n{text}")
        total += len(text)
        if total > MAX_TEXT_CHARS_PER_FILE:
            break
    return "\n".join(parts)


def _xlsx_text(path: str) -> str:
    from openpyxl import load_workbook

    try:
        wb = load_workbook(path, read_only=True, data_only=True)
    except Exception as e:  # noqa: BLE001  (openpyxl raises many types for bad files)
        raise Unsupported(f"the spreadsheet couldn't be opened ({type(e).__name__})") from e
    out, total = [], 0
    try:
        for ws in wb.worksheets:
            buf = io.StringIO()
            writer = csv.writer(buf)
            for row in ws.iter_rows(values_only=True):
                if any(v is not None for v in row):
                    writer.writerow(["" if v is None else v for v in row])
                if total + buf.tell() > MAX_TEXT_CHARS_PER_FILE:
                    break
            out.append(f"## Sheet: {ws.title}\n{buf.getvalue()}")
            total += buf.tell()
            if total > MAX_TEXT_CHARS_PER_FILE:
                break
    finally:
        wb.close()
    return "\n".join(out)


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_text(path: str) -> str:
    try:
        with zipfile.ZipFile(path) as z:
            root = ElementTree.fromstring(z.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as e:
        raise Unsupported(f"the Word document couldn't be opened ({type(e).__name__})") from e

    def para(p) -> str:
        return "".join(t.text or "" for t in p.iter(f"{_W}t"))

    lines = []
    body = root.find(f"{_W}body")
    for child in (body if body is not None else []):
        if child.tag == f"{_W}p":
            lines.append(para(child))
        elif child.tag == f"{_W}tbl":
            for tr in child.iter(f"{_W}tr"):
                cells = [" ".join(para(p) for p in tc.iter(f"{_W}p")).strip() for tc in tr.iter(f"{_W}tc")]
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def _plain_text(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read(MAX_TEXT_CHARS_PER_FILE * 4 + 4)
    return data.decode("utf-8", errors="replace")


def _is_text(mime: str) -> bool:
    return mime.startswith("text/") or mime in TEXT_MIMES or mime.endswith("+json") or mime.endswith("+xml")


def to_content(text: str, elements: list) -> tuple[str | list[dict], list[str]]:
    """HumanMessage content for a message and its attachments, plus notes to show the user.
    With no attachments the content is the plain string, exactly as before."""
    if not elements:
        return text, []
    out = Converted()
    images = 0
    pdf_bytes = 0
    text_budget = MAX_TEXT_CHARS_PER_MESSAGE
    for el in elements:
        name = getattr(el, "name", None) or "attachment"
        path = getattr(el, "path", None)
        try:
            if not path or not os.path.exists(path):
                raise Unsupported("the file wasn't received")
            size = os.path.getsize(path)
            mime = detect_mime(name, getattr(el, "mime", None), path)
            label = f'Attachment "{name}" ({mime}, {_size(size)})'
            if size > MAX_FILE_BYTES:
                raise Unsupported(f"it is {_size(size)}; the limit is {_size(MAX_FILE_BYTES)}")
            if mime in IMAGE_MIMES or (mime.startswith("image/") and mime not in ("image/svg+xml",)):
                if images >= MAX_IMAGES:
                    raise Unsupported(f"only the first {MAX_IMAGES} images in a message are read")
                block = _image_block(path, mime)
                images += 1
                out.blocks += [{"type": "text", "text": label + ":"}, block]
                continue
            if mime == PDF_MIME:
                pages = _pdf_page_count(path)
                if (size <= MAX_PDF_NATIVE_BYTES and pages <= MAX_PDF_NATIVE_PAGES
                        and pdf_bytes + size <= MAX_PDF_NATIVE_TOTAL_BYTES):
                    with open(path, "rb") as f:
                        data = f.read()
                    pdf_bytes += size
                    out.blocks += [{"type": "text", "text": f"{label}, {pages} page(s):"},
                                   {"type": "file", "base64": base64.b64encode(data).decode(),
                                    "mime_type": PDF_MIME, "extras": {"title": name}}]
                    continue
                extracted = _pdf_text(path)
                out.notes.append(f'"{name}" is large ({pages} pages, {_size(size)}), so I read its text only, '
                                 "without page images.")
            elif mime == XLSX_MIME or name.lower().endswith(".xlsx"):
                extracted = _xlsx_text(path)
            elif mime == DOCX_MIME or name.lower().endswith(".docx"):
                extracted = _docx_text(path)
            elif _is_text(mime):
                extracted = _plain_text(path)
            else:
                raise Unsupported(f"{mime} files aren't supported")
            if not extracted.strip():
                raise Unsupported("no text could be found in it")
            limit = min(MAX_TEXT_CHARS_PER_FILE, text_budget)
            if limit <= 0:
                raise Unsupported("this message already has as much attached text as I can read")
            text_budget -= min(len(extracted), limit)
            if len(extracted) > limit:
                out.notes.append(f'"{name}" is long, so I read only its first {limit:,} characters.')
                extracted = extracted[:limit] + f"\n[... truncated: the first {limit:,} characters are shown]"
            out.blocks.append({"type": "text", "text": f"<attachment name=\"{name}\" type=\"{mime}\">\n"
                                                       f"{extracted}\n</attachment>"})
        except Unsupported as e:
            out.blocks.append({"type": "text", "text": f'[Attachment "{name}" could not be read: {e}]'})
            out.notes.append(f'I couldn\'t read "{name}": {e}. I can read {SUPPORTED}.')
        except Exception as e:  # noqa: BLE001  (a broken file must never break the turn)
            out.blocks.append({"type": "text", "text": f'[Attachment "{name}" could not be read: {type(e).__name__}]'})
            out.notes.append(f'I couldn\'t read "{name}" (it may be damaged). I can read {SUPPORTED}.')
    if text and text.strip():
        out.blocks.append({"type": "text", "text": text})
    return out.blocks, out.notes


def resume_note(elements: list[dict]) -> str:
    """Stands in for a message's attachments when a chat is reopened (the files aren't kept)."""
    names = ", ".join(f"{e.get('name') or 'attachment'} ({e.get('mime') or 'unknown type'})" for e in elements)
    return f"[The user attached: {names}. The file isn't available in this reopened chat.]"
