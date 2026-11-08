"""Diacritical's translator: hyphenated words ("laddu-s") and phrases ("parijnanashram trayodashi") as entries of
their own, and that one-word entries behave as before."""
from diacritical.diacritics import phrase_key, translate_text

MAPPINGS = {
    "laddu": "laḍḍū",
    "laddu-s": "laḍḍū-s",
    "laddus": "laḍḍū-s",
    "parijnanashram": "{{}}{{parijñānāshram}}",
    "trayodashi": "{{}}{{trayodashī}}",
    "parijnanashram trayodashi": "{{}}{{parijñānāshrama trayodashī}}",
    "kshama": "ks͟hamā",
    "kshama- yachana": "ks͟hamā-yāchanā",
    "jaijaikara-s": "{{}}{{jaijaikāra-s}}",
    "namavali": "{{nāmāvalī (Hindi), nāmāvaliḥ (Sanskrit)}}",
    "akshay": "{{aks͟haya}}{{aks͟hay}}",
    "nayel": "{{}}{{nāyel}}",
    "vinati": "{{vinatī, vinati}}",
    "shankar": "shaṅkar",
    "rao": "rāo",
    "puja": "pūjā",
    "seva": "sevā",
    "puja-seva": "pūjā-sevā",
    "one two three": "ōne twō thrēē",
    "one two": "ōne-twō",
}


def t(text):
    return translate_text(text, MAPPINGS)


def test_one_word_entries_work_as_before():
    assert t("laddu Laddu LADDU") == "laḍḍū Laḍḍū LAḌḌŪ"
    assert t("akshay Akshay") == "aks͟haya Aks͟hay"
    assert t("nayel Nayel") == "Nāyel Nāyel"
    assert t("vinati Vinati") == "{{vinatī,vinati}} {{Vinatī,Vinati}}"
    assert t("Shankar Rao, unknown words.") == "Shaṅkar Rāo, unknown words."


def test_a_hyphenated_word_is_one_entry_and_keeps_the_typed_case():
    assert t("laddu-s Laddu-s LADDU-S laddus") == "laḍḍū-s Laḍḍū-s LAḌḌŪ-S laḍḍū-s"
    # an always-capital one starts with a capital, but the part after the hyphen doesn't
    assert t("jaijaikara-s Jaijaikara-s") == "Jaijaikāra-s Jaijaikāra-s"
    # words without an entry of their own are joined as they were
    assert t("co-operation laddu-x") == "co-operation laḍḍū-x"


def test_a_phrase_wins_over_its_words():
    assert t("Parijnanashram Trayodashi day") == "Parijñānāshrama Trayodashī day"
    assert t("parijnanashram trayodashi") == "Parijñānāshrama Trayodashī"
    assert t("Parijnanashram alone, Trayodashi alone") == "Parijñānāshram alone, Trayodashī alone"
    assert t("Parijnanashram  Trayodashi") == "Parijñānāshrama Trayodashī"  # two spaces


def test_the_longest_phrase_wins():
    assert t("one two three, one two, one") == "ōne twō thrēē, ōne-twō, one"


def test_a_phrase_does_not_run_across_lines_or_a_spaced_dash():
    assert t("Parijnanashram\nTrayodashi") == "Parijñānāshram\nTrayodashī"
    assert t("puja - seva, puja-seva") == "pūjā - sevā, pūjā-sevā"


def test_a_stored_key_matches_however_its_hyphen_was_spaced():
    assert phrase_key("Kshama- Yachana") == phrase_key("kshama-yachana") == "kshama-yachana"
    assert t("Kshama-Yachana, kshama") == "Ks͟hamā-Yāchanā, ks͟hamā"


def test_labels_in_brackets_keep_their_capital():
    assert t("namavali Namavali") == "{{nāmāvalī (Hindi),nāmāvaliḥ (Sanskrit)}} {{Nāmāvalī (Hindi),Nāmāvaliḥ (Sanskrit)}}"


def test_text_typed_with_marks_is_matched_without_them():
    assert t("Parijñānāshram Trayodashī") == "Parijñānāshrama Trayodashī"


def test_devanagari_is_left_to_its_own_conversion():
    out = t("लड्डू laddu-s")
    assert out.endswith(" laḍḍū-s") and "लड्डू" not in out
