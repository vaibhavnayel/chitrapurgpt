"""Diacritical's Mappings page saves entries in the house style (diacritical/cleaning.py): Add, Edit and Upload."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from diacritical.cleaning import clean_plain, clean_result

ROOT = Path(__file__).resolve().parent.parent


def test_plain_text_loses_marks_and_outer_spaces_but_keeps_one_space_between_words():
    assert clean_plain("  Nāmāvalih  ") == "namavalih"
    assert clean_plain(" Parijñānāshrama   Trayodashī ") == "parijnanashrama trayodashi"
    assert clean_plain("pat\u035fhshala") == "pathshala"
    assert clean_plain("laddu-s") == "laddu-s"


def test_result_keeps_its_marks_in_the_house_style():
    assert clean_result("  ks͟hamā   yāchanā ") == "ks͟hamā yāchanā"
    assert clean_result(" {{}} {{Māṇāyī}} ") == "{{}}{{māṇāyī}}"
    assert clean_result("dus\u0332h\u0332kāl̃a") == "dus\u035fhkāl̃a"          # underline pair -> tie
    assert clean_result("t\u0323\u0332h\u0332") == "\u1e6d\u035fh"           # ṭ̲h̲ -> ṭ͟h
    assert clean_result("vṛks\u035f\u035fha") == "vṛks\u035fha"              # doubled tie
    assert clean_result("d\u0323") == "\u1e0d"                               # ḍ in pieces -> U+1E0D
    assert clean_result("{{nāmāvalī (h), nāmāvaliḥ (s)}}") == "{{nāmāvalī (h), nāmāvaliḥ (s)}}"


SCRIPT = textwrap.dedent("""
    import json, sys, types
    sys.modules.setdefault("docx", types.ModuleType("docx"))
    from diacritical import app as m
    m.configure("sqlite:///" + sys.argv[1], "secret", "u", "p", cookie_path="/", db_retries=1)
    c = m.app.test_client()
    c.post("/login", data={"username": "u", "password": "p"})
    out = {}
    r = c.post("/api/mappings", json={"plain_text": "  Pat\u035fhshala ", "diacritic_text": "  pāt\u0332h\u0332ashālā  "})
    out["add"] = [r.status_code, r.get_json()["plain_text"], r.get_json()["diacritic_text"]]
    r = c.post("/api/mappings", json={"plain_text": "pāthshala", "diacritic_text": "x"})
    out["duplicate"] = [r.status_code, r.get_json()["error"]]
    r = c.put(f"/api/mappings/{out['add'] and 1}", json={"plain_text": " Parijñānāshrama  Trayodashī ",
                                                         "diacritic_text": " {{}} {{parijñānāshrama trayodashī}} "})
    out["edit"] = [r.status_code, r.get_json()["plain_text"], r.get_json()["diacritic_text"]]
    path = sys.argv[1] + ".txt"
    open(path, "w", encoding="utf-8").write("tara ,  tārā \\nNāmāvalih,{{nāmāvalī (h), nāmāvaliḥ (s)}}\\n  ,x\\n")
    with open(path, "rb") as f:
        r = c.post("/api/mappings/upload", data={"file": (f, "m.txt"), "mode": "update"})
    out["upload"] = r.get_json()
    out["all"] = sorted([x["plain_text"], x["diacritic_text"]] for x in c.get("/api/mappings").get_json())
    r = c.post("/translate", json={"text": "Parijnanashrama Trayodashi, Tara"})
    out["translate"] = r.get_json()["result"]
    print(json.dumps(out))
""")


def test_add_edit_and_upload_save_the_house_style(tmp_path):
    run = subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path / "d.db")], cwd=ROOT, capture_output=True,
                         text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["add"] == [201, "pathshala", "pāt\u035fhashālā"]
    assert out["duplicate"] == [400, 'Mapping for "pathshala" already exists']
    assert out["edit"] == [200, "parijnanashrama trayodashi", "{{}}{{parijñānāshrama trayodashī}}"]
    assert out["upload"]["total"] == 2
    assert out["all"] == [["namavalih", "{{nāmāvalī (h), nāmāvaliḥ (s)}}"],
                          ["parijnanashrama trayodashi", "{{}}{{parijñānāshrama trayodashī}}"],
                          ["tara", "tārā"]]
    assert out["translate"] == "Parijñānāshrama Trayodashī, Tārā"
