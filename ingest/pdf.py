"""PDF access for ingest, on pypdfium2 (Apache-2.0/BSD-3; PyMuPDF would be AGPL).

pdfium isn't thread-safe, so every call goes through one process-wide lock. Pages are rendered one at a time and
the document is opened from a file, so memory stays flat however long the issue is.
"""
from __future__ import annotations

import ctypes
import io
import re
import threading
from collections import Counter
from dataclasses import dataclass, field

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

LOCK = threading.Lock()
LONG_EDGE = 1568          # px; the size the API resizes images to anyway
MAX_PNG_BYTES = 3_500_000  # larger pages go as JPEG (base64 must stay under the API's 5 MB image limit)
PREVIEW_LONG_EDGE = 1200   # px; the page image shown on the review page
MIN_CLEAN_CHARS = 200      # a page's text layer is used as a check only with at least this much clean text

# Pre-Unicode Devanagari fonts: their text layer extracts as Latin garbage ("³ees Jew Yetcee"). Matched on the
# font name with the subset prefix and punctuation removed, lowercased. Unicode fonts (Mangal, Nirmala) are fine.
LEGACY_FONT_PREFIXES = ("akruti", "sanskrittext", "sanskrit99", "shree", "kruti", "shivaji", "chanakya",
                        "walkmanchanakya", "dvtt", "devlys", "kundli")

_SUBSET = re.compile(r"^[A-Z]{6}\+")


def font_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _SUBSET.sub("", name or "").lower())


def is_legacy_font(name: str) -> bool:
    return font_key(name).startswith(LEGACY_FONT_PREFIXES)


@dataclass
class TextLayer:
    """A page's embedded text, split by font kind."""
    clean_text: str = ""      # text in ordinary (non-legacy) fonts: the check compares the page read against it
    clean_chars: int = 0      # non-space characters in clean_text
    legacy_chars: int = 0     # characters in legacy Devanagari fonts (garbage when extracted; ignored)
    fonts: Counter = field(default_factory=Counter)

    @property
    def checkable(self) -> bool:
        return self.clean_chars >= MIN_CLEAN_CHARS

    @property
    def kind(self) -> str:
        if self.checkable:
            return "clean"
        if self.clean_chars or self.legacy_chars:
            return "thin"   # some text, but too little clean text to check against
        return "none"       # no text layer (a scan, or a page that is all pictures)


@dataclass
class PageImage:
    data: bytes
    media_type: str
    width: int
    height: int


class PdfFile:
    """One PDF on disk. Use as a context manager, or call close()."""

    def __init__(self, path: str):
        with LOCK:
            self._doc = pdfium.PdfDocument(path)
            self.page_count = len(self._doc)
        self._font_buf = ctypes.create_string_buffer(512)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        with LOCK:
            if self._doc is not None:
                self._doc.close()
                self._doc = None

    def _render(self, page_no: int, long_edge: int):
        page = self._doc[page_no - 1]
        try:
            w, h = page.get_size()
            bitmap = page.render(scale=long_edge / max(w, h, 1))
            try:
                return bitmap.to_pil().convert("RGB")
            finally:
                bitmap.close()
        finally:
            page.close()

    def render(self, page_no: int) -> PageImage:
        """The page for the reader: LONG_EDGE px on the long side, PNG, or JPEG if the PNG would be too big."""
        with LOCK:
            img = self._render(page_no, LONG_EDGE)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        if buf.tell() <= MAX_PNG_BYTES:
            return PageImage(buf.getvalue(), "image/png", *img.size)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return PageImage(buf.getvalue(), "image/jpeg", *img.size)

    def preview(self, page_no: int) -> bytes:
        """A smaller JPEG of the page for the review page."""
        with LOCK:
            img = self._render(page_no, PREVIEW_LONG_EDGE)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    def text_layer(self, page_no: int) -> TextLayer:
        """The page's text, character by character, keeping only characters in ordinary fonts. pdfium's own
        spacing (generated spaces and line breaks) separates words; legacy-font characters become spaces."""
        layer = TextLayer()
        out: list[str] = []
        with LOCK:
            page = self._doc[page_no - 1]
            textpage = page.get_textpage()
            try:
                tp = textpage.raw
                fonts_by_obj: dict[int, str] = {}
                for i in range(textpage.count_chars()):
                    code = pdfium_c.FPDFText_GetUnicode(tp, i)
                    ch = chr(code) if code else " "
                    if ch.isspace() or pdfium_c.FPDFText_IsGenerated(tp, i) == 1:
                        out.append("\n" if ch in "\r\n" else " ")
                        continue
                    obj = pdfium_c.FPDFText_GetTextObject(tp, i)
                    key = ctypes.cast(obj, ctypes.c_void_p).value or 0
                    font = fonts_by_obj.get(key)
                    if font is None:
                        n = pdfium_c.FPDFText_GetFontInfo(tp, i, self._font_buf, len(self._font_buf), None)
                        font = self._font_buf.raw[:max(n - 1, 0)].decode("utf-8", "replace")
                        fonts_by_obj[key] = font
                    layer.fonts[_SUBSET.sub("", font)] += 1
                    if is_legacy_font(font):
                        layer.legacy_chars += 1
                        out.append(" ")
                    else:
                        layer.clean_chars += 1
                        out.append(ch)
            finally:
                textpage.close()
                page.close()
        layer.clean_text = "".join(out)
        return layer


def page_count(path: str) -> int:
    with PdfFile(path) as pdf:
        return pdf.page_count


def is_pdf_header(head: bytes) -> bool:
    """PDF files start with "%PDF-" (some writers put a few bytes of junk first; readers allow 1 KB of it)."""
    return b"%PDF-" in head[:1024]


def extract_pages(src: str, dst: str, pages: list[int]) -> None:
    """Write a new PDF with some pages of `src` (1-based), e.g. to cut a small test fixture from an issue."""
    with LOCK:
        source = pdfium.PdfDocument(src)
        out = pdfium.PdfDocument.new()
        try:
            out.import_pages(source, [p - 1 for p in pages])
            out.save(dst)
        finally:
            out.close()
            source.close()
