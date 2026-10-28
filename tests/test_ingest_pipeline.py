"""The ingest pipeline without network or database: page check and Sonnet fallback, the issue map and contiguous
block ids, range expansion and the checker, the repair round, contents parsing, assembly and part splitting.
A fake LLM plays back scripted replies; PDFs are generated here (no magazine content in the repo)."""
import asyncio
import ctypes
import json
import sys
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from ingest import assemble as asm  # noqa: E402
from ingest import llm as ingest_llm  # noqa: E402
from ingest.llm import Usage  # noqa: E402
from ingest.pages import (MIN_RECALL, READER_MODEL, REREAD_MODEL, parse_page_read, read_page, recall)  # noqa: E402
from ingest.pdf import PageImage, PdfFile, TextLayer, extract_pages, is_legacy_font, is_pdf_header  # noqa: E402
from ingest.split import (BlockIndex, SplitError, check_split, content_blocks, find_contents, furniture_lines,  # noqa: E402
                          issue_map, parse_contents, parse_split, split_issue)


class FakeLLM:
    """Replies come from `script`: a list of strings, or a function (model, messages) -> str."""

    def __init__(self, script):
        self.script, self.calls = script, []

    async def complete(self, *, model, messages, system=None, max_tokens=16000, thinking=None):
        self.calls.append({"model": model, "messages": messages, "system": system, "thinking": thinking})
        reply = self.script(model, messages) if callable(self.script) else self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, Usage(model=model, input_tokens=1000, output_tokens=500, cost=0.001, stop_reason="end_turn")


def read_json(*blocks, printed=None):
    return json.dumps({"printed_page_number": printed, "new_articles_starting_here": [], "notes": "",
                       "blocks": [{"type": t, "text": x, "continued_from_previous_page": False,
                                   "continues_on_next_page": False} for t, x in blocks]})


class FakePdf:
    def __init__(self, layer: TextLayer):
        self.layer = layer

    def render(self, page):
        return PageImage(b"\x89PNG fake", "image/png", 10, 10)

    def text_layer(self, page):
        return self.layer


BODY = ("The Rathotsava at Shirali was celebrated with great devotion this year. Devotees from many towns pulled the "
        "chariot through the streets while bhajans were sung. In the evening a cultural programme was held in the "
        "Math courtyard, where children recited shlokas and elders narrated stories of earlier festivals. "
        "Volunteers served prasad to everyone, and the celebrations ended late at night with a deepotsava. ")
LAYER = TextLayer(clean_text="The Chitrapur Sunbeam 7 August 2024\n" + BODY, clean_chars=len(BODY))


# ---------- the page check and the Sonnet fallback ----------

def test_good_read_is_kept_without_a_second_opinion():
    llm = FakeLLM([read_json(("page_furniture", "The Chitrapur Sunbeam 7 August 2024"), ("body", BODY))])
    r = asyncio.run(read_page(FakePdf(LAYER), 7, llm, '"The Chitrapur Sunbeam"'))
    assert r.model == READER_MODEL and r.recall == 1.0 and not r.flags and not r.reread
    assert [c["model"] for c in llm.calls] == [READER_MODEL] and llm.calls[0]["thinking"] is False
    assert '"The Chitrapur Sunbeam"' in llm.calls[0]["messages"][0]["content"][1]["text"]


def test_low_recall_goes_to_sonnet_and_the_better_read_wins():
    half = BODY[: len(BODY) // 2]
    llm = FakeLLM([read_json(("body", half)), read_json(("page_furniture", "The Chitrapur Sunbeam 7 August 2024"),
                                                        ("body", BODY))])
    r = asyncio.run(read_page(FakePdf(LAYER), 7, llm, "x"))
    assert [c["model"] for c in llm.calls] == [READER_MODEL, REREAD_MODEL]
    assert llm.calls[1]["thinking"] is None, "Sonnet at its default effort"
    assert r.model == REREAD_MODEL and r.recall == 1.0 and r.reread and not r.flags
    assert r.attempts[0].recall < MIN_RECALL


def test_still_bad_after_sonnet_becomes_a_flag_and_keeps_the_better_read():
    half, third = BODY[: len(BODY) // 2], BODY[: len(BODY) // 3]
    llm = FakeLLM([read_json(("body", half)), read_json(("body", third))])
    r = asyncio.run(read_page(FakePdf(LAYER), 7, llm, "x"))
    assert r.model == READER_MODEL, "Haiku's read had more of the text"
    assert len(r.flags) == 1 and "page 7: the read has only" in r.flags[0] and "missing e.g." in r.flags[0]


def test_invalid_json_is_retried_then_reread():
    llm = FakeLLM(["not json", "still {not json",
                   read_json(("page_furniture", "The Chitrapur Sunbeam 7 August 2024"), ("body", BODY))])
    r = asyncio.run(read_page(FakePdf(LAYER), 3, llm, "x"))
    assert [c["model"] for c in llm.calls] == [READER_MODEL, READER_MODEL, REREAD_MODEL]
    assert r.model == REREAD_MODEL and not r.flags


def test_empty_read_of_a_page_with_text_is_reread_and_a_blank_page_is_fine():
    llm = FakeLLM([read_json(("page_furniture", "7")), read_json(("body", BODY))])
    r = asyncio.run(read_page(FakePdf(LAYER), 7, llm, "x"))
    assert r.model == REREAD_MODEL
    blank = FakePdf(TextLayer())
    llm = FakeLLM([read_json()])
    r = asyncio.run(read_page(blank, 2, llm, "x"))
    assert r.model == READER_MODEL and not r.flags and r.layer_kind == "none"


def test_a_page_that_cant_be_read_is_flagged():
    llm = FakeLLM([RuntimeError("overloaded")] * 3)
    r = asyncio.run(read_page(FakePdf(LAYER), 9, llm, "x"))
    assert r.read is None and r.flags == ["page 9: couldn't be read (RuntimeError: overloaded)"]


def test_recall_tolerates_text_layer_quirks_but_catches_a_dropped_paragraph():
    read = json.loads(read_json(("body", "Shrī GuruPādukā Pūjā on the 1st of September, participants welcome")))
    read = parse_page_read(json.dumps(read))
    layer = "Shr GuruP duk P j ī ā ā ū ā on the 1 st of September, parocipants welcome"
    assert recall(layer, read) == 1.0, "accent debris, a split ordinal and a ligature misspelling"
    assert recall("instancesthat noted", parse_page_read(read_json(("body", "instances that noted")))) == 1.0
    page = BODY + " Kumarika Pujana at Karla drew many families from Mumbai, Bengaluru and Mangaluru."
    full = parse_page_read(read_json(("body", page)))
    dropped = parse_page_read(read_json(("body", BODY)))
    assert recall(page, full) == 1.0 and recall(page, dropped) < MIN_RECALL, "a dropped sentence is caught"
    assert recall(page + " " + page, full) == 1.0, "text drawn twice (faux bold) in the layer"
    assert recall("Seva 300.00 and 350.00", parse_page_read(read_json(("body", "Seva 300.00")))) < MIN_RECALL, \
        "numbers must match exactly"


def test_parse_page_read_normalises():
    r = parse_page_read("```json\n" + json.dumps({"printed_page_number": 12, "blocks": [
        {"type": "Headline", "text": "A"}, {"type": "sidebar", "text": "B", "continues_on_next_page": 1}]}) + "\n```")
    assert r["printed_page_number"] == "12" and [b["type"] for b in r["blocks"]] == ["headline", "body"]
    assert r["blocks"][1]["continues_on_next_page"] is True
    with pytest.raises(ValueError):
        parse_page_read('{"blocks": [{"type": "body"}]}')


# ---------- the issue map, contiguous ids, contents ----------

READS = {
    1: parse_page_read(read_json(("page_furniture", "VOL. XXXI AUGUST 2024"), ("headline", "THE CHITRAPUR SUNBEAM"),
                                 ("photo", "a swamiji"))),
    2: parse_page_read(read_json(("headline", "THE CHITRAPUR SUNBEAM :- AUGUST 2024"), ("table",
        "| CONTENTS | Page No. |\n|---|---|\n| Prayers | 3 |\n| A Long Teaching | 4 |\n| Seva Rates | 6 |"),
        ("sponsor_note", "SHRI CHITRAPUR MATH, SHIRALI – 581 354"), ("page_furniture", "The Chitrapur Sunbeam 2 August 2024"),
        printed="2")),
    3: parse_page_read(read_json(("page_furniture", "The Chitrapur Sunbeam 3 August 2024"), ("headline", "PRAYERS"),
                                 ("verse", "ॐ नमः"), ("body", "Prayer text."), printed="3")),
    4: parse_page_read(read_json(("headline", "A LONG TEACHING"), ("byline", "– H.H. Swamiji"),
                                 ("body", "First part."), ("page_furniture", "The Chitrapur Sunbeam 4 August 2024"),
                                 printed="4")),
    5: parse_page_read(read_json(("body", "Second part."), ("caption", "Swamiji at Shirali, 5-8-2024"),
                                 ("photo", "a crowd"), printed="5")),
    6: parse_page_read(read_json(("headline", "SEVA RATES"), ("table", "| Seva | Rate |\n|---|---|\n| Puja | 300 |"),
                                 ("advert", "BUY SAREES"), printed="6")),
}


def test_content_blocks_are_numbered_contiguously_without_furniture():
    blocks = content_blocks(READS)
    assert [b.id for b in blocks if b.page in (1, 4)] == ["p1.b0", "p1.b1", "p4.b0", "p4.b1", "p4.b2"]
    assert all(b.type != "page_furniture" for b in blocks)
    imap = issue_map(blocks, READS)
    assert "=== PDF page 4 (printed page 4) ===" in imap and "p4.b0 [headline] A LONG TEACHING" in imap
    assert "The Chitrapur Sunbeam 4 August 2024" not in imap, "furniture isn't in the map"
    assert furniture_lines(READS)[0] == "VOL. XXXI AUGUST 2024", "the cover's masthead first (it has the date)"


def test_contents_entries_skip_headers_and_impossible_pages():
    page, text, entries = find_contents(READS)
    assert page == 2 and [(e.title, e.page) for e in entries] == [("Prayers", 3), ("A Long Teaching", 4),
                                                                   ("Seva Rates", 6)]
    assert [(e.title, e.page) for e in parse_contents("| Item | Page No. |\n|---|---|\n| Editorial | 5 |")] == \
        [("Editorial", 5)]
    dotted = parse_contents("Contents\nEditorial ........ 3\nFestivals in May   12\nAddress: Shirali 581 354", 60)
    assert [(e.title, e.page) for e in dotted] == [("Editorial", 3), ("Festivals in May", 12)]


# ---------- ranges and the checker ----------

def test_ranges_are_read_tolerantly():
    index = BlockIndex(content_blocks(READS))
    ids = lambda orders: [index.blocks[o].id for o in orders]  # noqa: E731
    assert ids(index.expand("p4.b0-p5.b1")[0]) == ["p4.b0", "p4.b1", "p4.b2", "p5.b0", "p5.b1"]
    assert ids(index.expand("p4.b0 – p4.b2")[0]) == ["p4.b0", "p4.b1", "p4.b2"], "en dash and spaces"
    assert ids(index.expand("p3.b1")[0]) == ["p3.b1"], "a single id"
    assert ids(index.expand("p5.b1-p4.b0")[0])[0] == "p4.b0", "written backwards"
    orders, adjusted = index.expand("p4.b3")
    assert ids(orders) == ["p4.b2"] and adjusted, "one past the end of a page: its last block"
    orders, adjusted = index.expand("p4.b9-p5.b2")
    assert ids(orders) == ["p5.b0", "p5.b1", "p5.b2"] and adjusted, "a start past the end: the next page"
    with pytest.raises(SplitError):
        index.expand("page four")
    with pytest.raises(SplitError):
        index.expand("p40.b0-p41.b0")


GOOD_SPLIT = {"issue": {"label": "August 2024", "published_date": "01-08-2024"},
              "articles": [{"title": "Cover", "kind": "cover", "ranges": ["p1.b0-p1.b1"]},
                           {"title": "Contents", "kind": "contents", "ranges": ["p2.b0-p2.b2"]},
                           {"title": "Prayers", "kind": "article", "toc_title": "Prayers", "ranges": ["p3.b0-p3.b2"]},
                           {"title": "A Long Teaching", "kind": "article", "author": "H.H. Swamiji",
                            "toc_title": "A Long Teaching", "ranges": ["p4.b0-p5.b2"]},
                           {"title": "Seva Rates", "kind": "notice", "toc_title": "Seva Rates",
                            "ranges": ["p6.b0-p6.b1"]}],
              "excluded": [{"range": "p6.b2", "why": "advert"}], "notes": ""}


def test_checker_on_a_good_and_a_bad_split():
    blocks = content_blocks(READS)
    entries = find_contents(READS)[2]
    f = check_split(parse_split(json.dumps(GOOD_SPLIT)), blocks, entries)
    assert not f.problems and f.contents_matched == 3 == f.contents_total
    bad = json.loads(json.dumps(GOOD_SPLIT))
    bad["articles"][3]["ranges"] = ["p4.b0-p4.b2"]                       # p5 left out
    bad["articles"][4]["ranges"] = ["p6.b0-p6.b2"]                       # the advert in an article and excluded
    bad["articles"][2]["ranges"] = ["p3.b0-p3.b2", "nonsense"]
    bad["articles"][2]["title"] = bad["articles"][2]["toc_title"] = "Something else"
    f = check_split(parse_split(json.dumps(bad)), blocks, entries)
    assert [blocks[o].id for o in f.unassigned] == ["p5.b0", "p5.b1", "p5.b2"]
    assert [blocks[o].id for o in f.doubled] == ["p6.b2"] and f.bad_ranges == ["nonsense"]
    assert [e.title for e in f.missing_entries] == ["Prayers"] and f.contents_matched == 2


def test_repair_round_sends_the_findings_back_and_keeps_the_better_answer():
    blocks = content_blocks(READS)
    bad = json.loads(json.dumps(GOOD_SPLIT))
    bad["articles"][3]["ranges"] = ["p4.b0-p4.b2"]
    llm = FakeLLM([json.dumps(bad), "```json\n" + json.dumps(GOOD_SPLIT) + "\n```"])
    outcome = asyncio.run(split_issue(blocks, READS, llm, '"The Chitrapur Sunbeam"'))
    assert outcome.rounds == 2 and not outcome.findings.problems and outcome.first_findings["unassigned"] == 3
    repair = llm.calls[1]["messages"]
    assert repair[0]["content"][0]["cache_control"] == {"type": "ephemeral"}, "the map is cached for the repair"
    assert repair[1] == {"role": "assistant", "content": json.dumps(bad)}
    assert "p5.b0 [body] Second part." in repair[2]["content"] and "complete corrected JSON" in repair[2]["content"]
    assert "COVER AND RUNNING HEADERS/FOOTERS\nVOL. XXXI AUGUST 2024" in llm.calls[0]["messages"][0]["content"][0]["text"]
    assert '"The Chitrapur Sunbeam"' in llm.calls[0]["system"]
    # A worse repair is ignored; a good first answer needs no repair.
    llm = FakeLLM([json.dumps(bad), json.dumps({"articles": []})])
    assert asyncio.run(split_issue(blocks, READS, llm, "x")).findings.summary()["unassigned"] == 3
    llm = FakeLLM([json.dumps(GOOD_SPLIT)])
    assert asyncio.run(split_issue(blocks, READS, llm, "x")).rounds == 1 and len(llm.calls) == 1


# ---------- assembly ----------

def test_assembly_text_pages_and_metadata():
    blocks = content_blocks(READS)
    arts = asm.assemble(parse_split(json.dumps(GOOD_SPLIT)), blocks)
    teaching = arts[3]
    assert teaching.text == "A LONG TEACHING\n\n– H.H. Swamiji\n\nFirst part.\n\nSecond part.\n\nSwamiji at Shirali, 5-8-2024"
    assert "a crowd" not in teaching.text, "photo descriptions are left out, captions kept"
    assert (teaching.start_page, teaching.end_page) == (4, 5) and teaching.author == "H.H. Swamiji"
    assert arts[4].text == "SEVA RATES\n\n| Seva | Rate |\n|---|---|\n| Puja | 300 |", "tables stay Markdown"
    assert arts[0].text == "THE CHITRAPUR SUNBEAM" and arts[0].excluded, "almost no text: excluded to start with"
    info = asm.issue_info(parse_split(json.dumps(GOOD_SPLIT)), "Sunbeam")
    assert info.source == "Sunbeam-2024-August.txt" and info.published_date == "01-08-2024"
    meta = asm.doc_metadata(teaching, info)
    assert list(meta) == ["source", "published_date", "title", "author", "summary", "start_line", "end_line",
                          "start_page", "end_page"]
    assert meta["start_line"] is None and meta["end_line"] is None and meta["start_page"] == 4


def test_long_articles_become_parts_and_nothing_is_dropped():
    para = "word " * 3000  # ~750 tokens by any count
    reads = {p: parse_page_read(read_json(("headline", "BIG") if p == 1 else ("body", para), ("body", para)))
             for p in range(1, 13)}
    blocks = content_blocks(reads)
    split = parse_split(json.dumps({"articles": [{"title": "Big", "kind": "article", "ranges": ["p1.b0-p12.b1"]}]}))
    arts = asm.assemble(split, blocks, max_tokens=8000)
    assert len(arts) >= 2 and [a.title for a in arts][0] == f"Big (part 1 of {len(arts)})"
    assert all(asm.count_tokens(a.text) <= 8000 for a in arts)
    assert sum(a.text.count("word") for a in arts) == sum(b.text.count("word") for b in blocks)
    assert arts[0].start_page == 1 and arts[-1].end_page == 12


def test_issue_names_and_double_issues():
    assert asm.issue_months("July-August 2026") == (2026, ["July", "August"])
    assert asm.issue_months("May & June 2023") == (2023, ["May", "June"])
    assert asm.issue_months("Sept–Oct 2026") == (2026, ["September", "October"])
    assert asm.issue_months("", "01-02-2025") == (2025, ["February"])
    assert asm.source_name("Sunbeam", 2026, ["July", "August"]) == "Sunbeam-2026-July-August.txt"
    assert asm.source_name("KSA", 2025, ["February"]) == "KSA-2025-February.txt"
    assert asm.published_date_for(2026, ["July", "August"], "1st July") == "01-07-2026"


def test_summaries_are_written_once(monkeypatch):
    arts = [asm.Article(n=1, title="T", kind="article", author=None, toc_title=None, blocks=[], pdf_start=1, pdf_end=1,
                        start_page=1, end_page=1, text="Some text."),
            asm.Article(n=2, title="X", kind="other", author=None, toc_title=None, blocks=[], pdf_start=1, pdf_end=1,
                        start_page=1, end_page=1, text="x", excluded=True)]
    llm = FakeLLM(["  A short\nsummary. "])
    asyncio.run(asm.summarize(arts, llm, '"The Chitrapur Sunbeam"'))
    assert arts[0].summary == "A short summary." and arts[1].summary is None
    assert llm.calls[0]["model"] == asm.SUMMARY_MODEL and llm.calls[0]["thinking"] is False
    asyncio.run(asm.summarize(arts, llm, "x"))
    assert len(llm.calls) == 1, "already summarised"


# ---------- PDFs (generated here) ----------

def make_pdf(path: Path, pages: list[str]) -> None:
    pdf = pdfium.PdfDocument.new()
    font = pdfium_c.FPDFText_LoadStandardFont(pdf.raw, b"Helvetica")
    for text in pages:
        page = pdf.new_page(420, 595)
        for i, line in enumerate(text.splitlines()):
            obj = pdfium_c.FPDFPageObj_CreateTextObj(pdf.raw, font, 11.0)
            buf = ctypes.create_unicode_buffer(line + "\x00")
            pdfium_c.FPDFText_SetText(obj, ctypes.cast(ctypes.create_string_buffer(
                (line + "\x00").encode("utf-16-le")), pdfium_c.FPDF_WIDESTRING))
            del buf
            pdfium_c.FPDFPageObj_Transform(obj, 1, 0, 0, 1, 30, 560 - 14 * i)
            pdfium_c.FPDFPage_InsertObject(page.raw, obj)
        pdfium_c.FPDFPage_GenerateContent(page.raw)
        page.close()
    pdf.save(str(path))
    pdf.close()


def test_pdf_render_text_layer_and_page_cut(tmp_path):
    lines = "\n".join(f"Line {i}: the quick brown fox jumps over the lazy dog." for i in range(8))
    src = tmp_path / "issue.pdf"
    make_pdf(src, [lines, "Second page.", "Third page."])
    assert is_pdf_header(src.read_bytes()[:1024]) and not is_pdf_header(b"PK\x03\x04")
    with PdfFile(str(src)) as pdf:
        assert pdf.page_count == 3
        img = pdf.render(1)
        assert img.media_type == "image/png" and max(img.width, img.height) == 1568
        layer = pdf.text_layer(1)
        assert layer.checkable and layer.legacy_chars == 0 and "lazy dog" in layer.clean_text
        assert pdf.text_layer(2).kind == "thin" and pdf.preview(1)[:2] == b"\xff\xd8"
    cut = tmp_path / "cut.pdf"
    extract_pages(str(src), str(cut), [1, 3])
    with PdfFile(str(cut)) as pdf:
        assert pdf.page_count == 2 and "Third page" in pdf.text_layer(2).clean_text
    assert is_legacy_font("ABCDEF+AkrutiDevPriya,Bold") and is_legacy_font("SanskritText")
    assert is_legacy_font("Shivaji01,Bold") and not is_legacy_font("BookmanOldStyle") and not is_legacy_font("Mangal")


def test_prices_match_the_eval_table():
    import pricing
    for model, prices in ingest_llm.PRICES.items():
        assert pricing.PRICES[model] == prices, model
    assert ingest_llm.cost_usd("claude-haiku-5-5", 50_000, 1_000) == pytest.approx(0.0055)
    assert ingest_llm.cost_usd("claude-haiku-5-5", 120_000, 0) == pytest.approx(0.06), "long-prompt tier"


def test_the_worker_is_off_on_pr_previews_unless_asked(monkeypatch):
    from ingest.jobs import worker_enabled
    monkeypatch.delenv("INGEST_WORKER", raising=False)
    monkeypatch.delenv("IS_PULL_REQUEST", raising=False)
    assert worker_enabled()
    monkeypatch.setenv("IS_PULL_REQUEST", "true")  # previews share the live database
    assert not worker_enabled()
    monkeypatch.setenv("INGEST_WORKER", "1")
    assert worker_enabled()
    monkeypatch.delenv("IS_PULL_REQUEST")
    monkeypatch.setenv("INGEST_WORKER", "0")
    assert not worker_enabled()


def test_mostly_indic_articles_are_embedded_by_their_english_title_and_summary():
    from ingest.jobs import embedding_text, indic_share
    hindi = "अनन्त संसार-समुद्रमध्ये मग्न समभ्युद्धर। यह बहुत पहले से चल रहा है। " * 20 + "In every transaction, You are with me."
    english = "Swamiji arrived at Vittal on 29 July. " * 20 + "॥ ॐ ॥"
    assert indic_share(hindi) > 0.8 and indic_share(english) < 0.05 and indic_share("12 34") == 0.0
    assert embedding_text("Ashirvachana on Guru Purnima", "A discourse at Vittal.", hindi) == \
        "Ashirvachana on Guru Purnima\n\nA discourse at Vittal."
    assert embedding_text("Arrival", "Swamiji arrives.", english) == english, "English text is embedded as is"
    assert embedding_text("Ashirvachana", "  ", hindi) == hindi, "no summary: fall back to the text"
