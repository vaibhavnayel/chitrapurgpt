"""Regenerate the small upload fixtures used by the eval (no third-party libraries needed).

  note.pdf      one-page text PDF: a short note in romanised Konkani (Amchigele) with community terms
  schedule.csv  a made-up festival schedule
The image fixture is the app's own logo (public/logo_light.png).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

NOTE_LINES = [
    "Namaskar. Shirali Mathant Navaratri utsav sombara suru jata.",
    "Sakalla Chandi path ani sanje bhajan asa.",
    "Vijayadashami disa Swamijin ashirvachan ditale.",
    "Sagalyani yevun Devi prasad ghevka. - Sevakarte",
]

SCHEDULE = """date,event,place
2026-10-12,Navaratri begins (Ghatasthapana),Shri Chitrapur Math Shirali
2026-10-20,Chandi Havan,Shri Chitrapur Math Shirali
2026-10-21,Vijayadashami ashirvachan,Shri Chitrapur Math Shirali
2026-11-05,Kartika Deepotsava,Shri Chitrapur Math Shirali
"""


def _pdf(lines: list[str]) -> bytes:
    text_ops = ["BT", "/F1 14 Tf", "72 720 Td", "18 TL"]
    for line in lines:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        text_ops.append(f"({safe}) Tj T*")
    text_ops.append("ET")
    stream = "\n".join(text_ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


if __name__ == "__main__":
    (HERE / "note.pdf").write_bytes(_pdf(NOTE_LINES))
    (HERE / "schedule.csv").write_text(SCHEDULE)
    print("wrote note.pdf and schedule.csv")
