"""The Mappings page's Type boxes (diacritical/static/js/mapping_form.js), checked with node when it's installed:
the JS unit test (tests/js/test_mapping_form.js; it also round-trips every entry of a list copy when one is given in
$MAPPINGS_CSV), and the page's house style matching cleaning.py on the server."""
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from diacritical.cleaning import clean_plain, clean_result

ROOT = Path(__file__).resolve().parent.parent
FORM_JS = ROOT / "diacritical" / "static" / "js" / "mapping_form.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

SAMPLES = [
    "  Nāmāvalih  ", " Parijñānāshrama   Trayodashī ", "pat\u035fhshala", "laddu-s",
    "  ks͟hamā   yāchanā ", " {{}} {{Māṇāyī}} ", "dus\u0332h\u0332kāl̃a", "t\u0323\u0332h\u0332", "vṛks\u035f\u035fha",
    "d\u0323", "{}}{{hebl̃ekar}}", "{{kākḍā}}}", "{{nāmāvalī (Hindi), nāmāvaliḥ (Sanskrit)}}", "{{gŏḍ}}{{God}}",
    "{{}}{{Kārtik,  kārtika}}", "KṚS͟HṆA", "a\tb\nc", "", "   ",
]


def _list_samples():
    path = os.environ.get("MAPPINGS_CSV", "")
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return [v for row in csv.DictReader(f) for v in (row["plain_text"], row["diacritic_text"])]


def test_node_unit_tests():
    r = subprocess.run([NODE, str(ROOT / "tests" / "js" / "test_mapping_form.js")], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "passed" in r.stdout


def test_page_house_style_matches_cleaning_py():
    samples = SAMPLES + _list_samples()
    script = ("const m = require(process.argv[1]);"
              "const xs = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              "process.stdout.write(JSON.stringify(xs.map(x => [m.cleanPlain(x), m.cleanResult(x)])));")
    r = subprocess.run([NODE, "-e", script, str(FORM_JS)], input=json.dumps(samples), capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    want = [[clean_plain(x), clean_result(x)] for x in samples]
    diffs = [(x, g, w) for x, g, w in zip(samples, got, want) if g != w]
    assert not diffs, diffs[:10]
