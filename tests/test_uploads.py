"""Offline tests for uploads.py: every file type, plus oversize, unsupported and damaged files.

  .venv/bin/python -m pytest tests/test_uploads.py -q
"""
import base64
import io
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uploads  # noqa: E402

NOTE_PDF = ROOT / "eval" / "fixtures" / "note.pdf"


def el(path, name=None, mime=None):
    return SimpleNamespace(path=str(path), name=name or Path(path).name, mime=mime)


def texts(blocks):
    return [b["text"] for b in blocks if b["type"] == "text"]


def test_no_attachments_keeps_plain_string():
    assert uploads.to_content("hello", []) == ("hello", [])


def test_large_image_is_downscaled(tmp_path):
    p = tmp_path / "photo.jpg"
    Image.new("RGB", (4000, 3000), "orange").save(p, "JPEG")
    blocks, notes = uploads.to_content("What is this?", [el(p, mime="image/jpeg")])
    assert notes == []
    label, image, question = blocks
    assert label["text"].startswith('Attachment "photo.jpg" (image/jpeg')
    assert image["type"] == "image" and image["mime_type"] == "image/jpeg"
    assert max(Image.open(io.BytesIO(base64.b64decode(image["base64"]))).size) == uploads.MAX_IMAGE_EDGE
    assert question == {"type": "text", "text": "What is this?"}


def test_small_png_sent_as_is_and_octet_stream_sniffed(tmp_path):
    p = tmp_path / "upload"  # no extension, octet-stream: detected from the bytes
    Image.new("RGBA", (64, 32), (0, 0, 0, 0)).save(p, "PNG")
    blocks, notes = uploads.to_content("", [el(p, mime="application/octet-stream")])
    assert notes == [] and blocks[1]["mime_type"] == "image/png"
    assert base64.b64decode(blocks[1]["base64"]) == p.read_bytes()
    assert len(blocks) == 2, "no empty text block for an empty message"


def test_pdf_native_block():
    blocks, notes = uploads.to_content("Translate this", [el(NOTE_PDF, mime="application/pdf")])
    assert notes == []
    assert blocks[0]["text"].endswith("1 page(s):")
    assert blocks[1]["type"] == "file" and blocks[1]["mime_type"] == "application/pdf"
    assert blocks[1]["extras"] == {"title": "note.pdf"}
    assert base64.b64decode(blocks[1]["base64"]) == NOTE_PDF.read_bytes()


def test_big_pdf_falls_back_to_text(monkeypatch):
    monkeypatch.setattr(uploads, "MAX_PDF_NATIVE_PAGES", 0)
    blocks, notes = uploads.to_content("", [el(NOTE_PDF, mime="application/pdf")])
    assert "Navaratri" in blocks[0]["text"] and blocks[0]["text"].startswith('<attachment name="note.pdf"')
    assert "read its text only" in notes[0]


def test_xlsx(tmp_path):
    from openpyxl import Workbook
    wb = Workbook()
    wb.active.title = "Schedule"
    wb.active.append(["date", "event"])
    wb.active.append(["2026-10-12", "Navaratri"])
    wb.create_sheet("Empty")
    p = tmp_path / "s.xlsx"
    wb.save(p)
    blocks, notes = uploads.to_content("", [el(p, mime="application/octet-stream")])
    assert notes == []
    assert "## Sheet: Schedule\ndate,event\r\n2026-10-12,Navaratri" in blocks[0]["text"]


def test_docx(tmp_path):
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xml = (f'<w:document xmlns:w="{w}"><w:body><w:p><w:r><w:t>Namaskar </w:t></w:r><w:r><w:t>all</w:t></w:r></w:p>'
           '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>a</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>b</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
           '</w:body></w:document>')
    p = tmp_path / "letter.docx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("word/document.xml", xml)
    blocks, notes = uploads.to_content("", [el(p)])
    assert notes == [] and "Namaskar all\na | b" in blocks[0]["text"]


def test_csv_and_utf8_text_without_extension(tmp_path):
    csv_path = ROOT / "eval" / "fixtures" / "schedule.csv"
    txt = tmp_path / "notes"
    txt.write_text("ಶ್ರೀ ಚಿತ್ರಾಪುರ ಮಠ", encoding="utf-8")
    blocks, notes = uploads.to_content("q", [el(csv_path, mime="text/csv"), el(txt, mime="application/octet-stream")])
    assert notes == []
    assert "Chandi Havan" in blocks[0]["text"] and "ಶ್ರೀ ಚಿತ್ರಾಪುರ ಮಠ" in blocks[1]["text"]


def test_unsupported_damaged_missing_and_oversize(tmp_path, monkeypatch):
    binary = tmp_path / "data.bin"
    binary.write_bytes(bytes(range(256)))
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"\x89PNG not really")
    big = tmp_path / "big.txt"
    big.write_text("x" * 2000)
    monkeypatch.setattr(uploads, "MAX_FILE_BYTES", 1000)
    blocks, notes = uploads.to_content("q", [el(binary, mime="application/octet-stream"), el(broken, mime="image/png"),
                                             el(tmp_path / "gone.pdf", mime="application/pdf"), el(big, mime="text/plain")])
    agent_notes = texts(blocks)
    assert agent_notes[0].startswith('[Attachment "data.bin" could not be read: application/octet-stream files aren')
    assert "couldn't be opened" in agent_notes[1]
    assert "wasn't received" in agent_notes[2]
    assert "the limit is 1000 bytes" in agent_notes[3]
    assert agent_notes[4] == "q"
    assert len(notes) == 4 and all(n.startswith("I couldn't read") for n in notes)
    assert "I can read images" in notes[0]


def test_text_caps_and_image_count(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "MAX_TEXT_CHARS_PER_FILE", 50)
    monkeypatch.setattr(uploads, "MAX_TEXT_CHARS_PER_MESSAGE", 80)
    monkeypatch.setattr(uploads, "MAX_IMAGES", 1)
    long_text = tmp_path / "long.txt"
    long_text.write_text("y" * 500)
    img = tmp_path / "a.png"
    Image.new("RGB", (10, 10)).save(img)
    blocks, notes = uploads.to_content("", [el(long_text), el(long_text, name="again.txt"), el(long_text, name="third.txt"),
                                            el(img), el(img, name="b.png")])
    t = texts(blocks)
    assert "[... truncated: the first 50 characters are shown]" in t[0]
    assert "[... truncated: the first 30 characters are shown]" in t[1], "the per-message budget applies"
    assert "as much attached text as I can read" in t[2]
    assert "only the first 1 images" in t[-1]
    assert sum(b["type"] == "image" for b in blocks) == 1
    assert len(notes) == 4


def test_resume_note():
    assert uploads.resume_note([{"name": "a.pdf", "mime": "application/pdf"}]) == (
        "[The user attached: a.pdf (application/pdf). The file isn't available in this reopened chat.]")
