"""The house style for entries, applied whenever one is saved on the Mappings page (Add, Edit, Upload).

- Plain text: diacritical marks removed (nāmāvalih -> namavalih), spaces at the start and end removed, runs of
  spaces between words made one (so phrases such as "parijnanashram trayodashi" keep working), small letters.
- Diacritic text: spaces at the start and end removed, runs of spaces made one, spaces between {{ }} groups removed,
  broken brackets repaired (any run of { or } becomes {{ or }}, so "{}}{{x}}" becomes "{{}}{{x}}"), an underlined
  pair (s̲h̲, U+0332 under each letter) turned into the tie (s͟h, U+035F), a doubled tie made single, and letters
  stored in pieces joined into single characters (NFC: d + dot below becomes ḍ U+1E0D), small letters.
"""
import re
import unicodedata

_MARKS = "[\u0300-\u0331\u0333-\u036f]*"  # other marks a letter of an underlined pair may carry (ṭ̲h̲)


def clean_plain(text: str) -> str:
    no_marks = "".join(c for c in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_marks).strip().lower()


def clean_result(text: str) -> str:
    t = unicodedata.normalize("NFD", (text or "").strip())
    t = re.sub(rf"(\w{_MARKS})\u0332(\w{_MARKS})\u0332", "\\1\u035f\\2", t)
    t = re.sub("\u035f+", "\u035f", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\{+", "{{", t)
    t = re.sub(r"\}+", "}}", t)
    t = re.sub(r"\}\}\s+\{\{", "}}{{", t)
    return unicodedata.normalize("NFC", t).lower()
