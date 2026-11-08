"""
Diacritical text processing module.
Handles the core functionality of processing text with diacritics.
"""

import docx
import re
import unicodedata
import logging
from .devnagri import convert_devanagari_tokens

logger = logging.getLogger(__name__)

# Text processing functions
def read_docx(path: str) -> str:
    """Read text from a DOCX file"""
    doc = docx.Document(path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def generate_tokens(text: str) -> list[str]:
    """Split text into tokens"""
    tokens = re.split(r'([^\w\u0300-\u036f\s]|\s+)', text)
    tokens = [t for t in tokens if t]
    return tokens

def join_tokens(tokens: list[str]) -> str:
    """Join tokens back into text"""
    return ''.join(tokens)

def remove_diacritics(word: str) -> str:
    """Remove diacritics from a word"""
    normalized = unicodedata.normalize('NFKD', word)
    return ''.join(c for c in normalized if not unicodedata.combining(c))

# A hyphenated word ("laddu-s") or a phrase ("parijnanashram trayodashi") is one entry: its parts are joined by a
# hyphen (or its look-alikes) or by spaces on the same line.
_JOINER = r"[^\S\r\n]+|[-\u2010\u2011]"
_SPLIT_PARTS = re.compile(r"(\s+|[-\u2010\u2011])")


def _is_word(token: str) -> bool:
    return bool(re.fullmatch(r"[\w\u0300-\u036f]+", token))


def _is_joiner(token: str) -> bool:
    return bool(re.fullmatch(_JOINER, token))


def _has_parts(text: str) -> bool:
    return bool(re.search(r"[\s\-\u2010\u2011]", text.strip()))


def phrase_key(text: str) -> str:
    """The lookup key of a hyphenated word or phrase: no marks, lowercase, hyphen look-alikes as "-", no spaces around
    a hyphen, one space between words. "Kshama- Yachana" and "kshama-yachana" give the same key."""
    key = remove_diacritics(text).lower()
    key = re.sub(r"[\u2010\u2011]", "-", key)
    key = re.sub(r"\s*-\s*", "-", key)
    return re.sub(r"\s+", " ", key).strip()


def phrase_index(mappings: dict[str, str]) -> tuple[dict[str, str], int]:
    """{phrase key: stored key} for every entry that is a hyphenated word or a phrase, and the most words any of them
    has. Entries of one word are looked up as before."""
    index = {}
    for key in mappings:
        if _has_parts(key):
            pk = phrase_key(key)
            if len(re.findall(r"[\w\u0300-\u036f]+", pk)) > 1:
                index.setdefault(pk, key)
    longest = max((len(re.findall(r"[\w\u0300-\u036f]+", k)) for k in index), default=1)
    return index, longest


def _word_case(word: str, mapping: str) -> str:
    """The case of one typed word copied onto one word of the result (never fails)."""
    if len(word) > 1 and word.isupper():
        return mapping.upper()
    if word[:1].isupper():
        return mapping[:1].upper() + mapping[1:]
    return mapping.lower()


def _match_case_parts(word: str, mapping: str) -> str:
    """match_case for a hyphenated word or phrase: each typed word's case goes to the matching word of the result,
    so "Parijnanashram Trayodashi" gives "Parijñānāshrama Trayodashī" and "Laddu-s" gives "Laḍḍū-s"."""
    if word.isupper():
        return mapping.upper()
    if word.islower():
        return mapping.lower()
    typed = [w for w in _SPLIT_PARTS.split(word) if w and not _SPLIT_PARTS.fullmatch(w)]
    parts = _SPLIT_PARTS.split(mapping)
    n = 0
    for i, part in enumerate(parts):
        if part and not _SPLIT_PARTS.fullmatch(part):
            parts[i] = _word_case(typed[min(n, len(typed) - 1)], part)
            n += 1
    return "".join(parts)


def _capitalise(word: str) -> str:
    """The typed word as a name (for "always capital" entries): each word of a phrase starts with a capital, but not
    the part after a hyphen ("laddu-s" -> "Laddu-s")."""
    if not _has_parts(word):
        return word.title()
    return "".join(p[:1].upper() + p[1:].lower() for p in re.split(r"(\s+)", word))


def match_case(word: str, mapping: str) -> str:
    """Match the case of a mapping to a word"""
    if _has_parts(word):
        return _match_case_parts(word, mapping)
    if word.isupper():
        return mapping.upper()
    elif word.islower():
        return mapping.lower()
    elif word[0].isupper():
        return mapping[0].upper() + mapping[1:]
    else:
        raise ValueError(f"Cannot add diacritics to word: {word} with mapping: {mapping}")

def _match_case_choice(word: str, choice: str) -> str:
    """match_case for one choice, leaving a label in brackets as it's written: "nāmāvalī (Hindi)"."""
    text, label = re.fullmatch(r"(.*?)(\s*\(.*\))?", choice).groups()
    return match_case(word, text) + (label or "")


def handle_one_to_many_mappings(word: str, mapping: str) -> str:
    # Choices are separated by commas. Spaces inside a choice are kept (a phrase, or a label such as "(Hindi)").
    diacritics = [re.sub(r"\s+", " ", d).strip() for d in mapping.replace("{{", "").replace("}}", "").split(",")]
    if len(diacritics) == 1:
        # single word should display without curly brackets
        return _match_case_choice(word, diacritics[0])
    else:
        # multiple words
        return f"{{{{{','.join([_match_case_choice(word, m) for m in diacritics])}}}}}"

def add_diacritics(word: str, mappings: dict[str, str]) -> str:
    """
    Add diacritics to a word based on mappings

    cases:
    # 1: single word
    - we want to map a single word to a single word
    - eg "lalitā"

    # 2: multiple words
    - we want to map a single word to multiple words
    - eg "{{lalita, lalitā}}"

    # 3: 2 groups of multiple words
    - we want to map a single word to multiple words, but the mapping changes depending on the case of the word
    - if the word is lowercase, map it to the first group
    - if the word is uppercase, map it to the second group
    - eg "{{nayaka, nayak}}{{nayak}}"

    # 4: 2 groups, but only one group is used (Spellcheck)
    - This is mostly for proper nouns.
    - lowercase inputs are accidental and should always map to the second group ie become capitalized
    - eg "{{}}{{nāyel}}"
    nayel -> Nāyel
    Nayel -> Nāyel
    """
    
    mapping = mappings.get(word.lower(), word)
    return apply_mapping(word, mapping)


def apply_mapping(word: str, mapping: str) -> str:
    """The result for a typed word (or hyphenated word, or phrase) from its stored mapping; see add_diacritics."""
    if matches := re.findall(r'\{\{.*?\}\}', mapping):
        if len(matches) == 1:
            #case 2
            return handle_one_to_many_mappings(word, matches[0])
        elif len(matches) == 2:
            #case 3
            if word[0].islower():
                lower_case_mapping = handle_one_to_many_mappings(word, matches[0])
                if lower_case_mapping=="":
                    # case 4
                    return handle_one_to_many_mappings(_capitalise(word), matches[1])
                else:
                    return lower_case_mapping
            else:
                return handle_one_to_many_mappings(word, matches[1])
        else:
            raise ValueError(f"Invalid mapping: {mapping}")
    else:
        #case 1
        return match_case(word, mapping)
    
def make_mappings(tokens: list[str]) -> dict[str, str]:
    """Create mappings from tokens with diacritics"""
    mappings = {}
    for token in tokens:
        token_without_diacritics = remove_diacritics(token)
        if (token != token_without_diacritics) and not token.isspace():
            mappings[token_without_diacritics.lower()] = token.lower()
    return mappings

def find_phrase(tokens: list[str], i: int, index: dict[str, str], longest: int, skip_indices=None):
    """The longest hyphenated word or phrase in the list that starts at tokens[i]: (index after it, stored key), or
    None. Its words are joined by one hyphen ("laddu-s") or by spaces on the same line ("shankar rao"). A hyphen with
    spaces around it is a dash, so "puja - seva" isn't one."""
    skip = skip_indices or ()
    best = None
    words, end = 1, i + 1
    while words < longest:
        k = end + 1
        if k >= len(tokens) or end in skip or k in skip or not _is_joiner(tokens[end]) or not _is_word(tokens[k]):
            break
        words, end = words + 1, k + 1
        key = phrase_key("".join(tokens[i:end]))
        if key in index:
            best = (end, index[key])
    return best


def reconstruct_tokens(tokens: list[str], mappings: dict[str, str], skip_indices=None) -> list[str]:
    """Reconstruct tokens with diacritics; optionally skip applying mappings for indices. A hyphenated word or phrase
    that has its own entry wins over its single words (the longest one first)."""
    index, longest = phrase_index(mappings)
    translated_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if skip_indices and i in skip_indices:
            translated_tokens.append(token)
            i += 1
            continue
        found = find_phrase(tokens, i, index, longest, skip_indices) if index and _is_word(token) else None
        if found:
            end, key = found
            plain = "".join(remove_diacritics(t) for t in tokens[i:end])
            translated_tokens.append(apply_mapping(plain, mappings[key]))
            i = end
            continue
        i += 1
        token_without_diacritics = remove_diacritics(token)
        if token_without_diacritics.lower() not in mappings:
            translated_tokens.append(token_without_diacritics)
        else:
            translated_tokens.append(add_diacritics(token_without_diacritics, mappings))
    return translated_tokens

 

def verify(text: str, reconstructed_text: str, tokens: list[str], reconstructed_tokens: list[str]) -> bool:
    """Verify that the reconstructed text matches the original"""
    if text != reconstructed_text:
        if len(tokens) != len(reconstructed_tokens):
            logger.warning(f"tokens: {len(tokens)}, reconstructed_tokens: {len(reconstructed_tokens)}")

        if len(text) != len(reconstructed_text):
            logger.warning(f"text: {len(text)}, reconstructed_text: {len(reconstructed_text)}")

        for orig, recon in zip(tokens, reconstructed_tokens):
            if orig != recon:
                logger.warning(f"original: {orig}, reconstructed: {recon}")
        return False
    else:
        return True

def test_txt_file(path: str) -> bool:
    """Test the diacritics processing on a text file"""
    text = open(path, "r").read()
    tokens = generate_tokens(text)
    mappings = make_mappings(tokens)
    reconstructed_tokens = reconstruct_tokens(tokens, mappings)
    reconstructed_text = join_tokens(reconstructed_tokens)
    return verify(text, reconstructed_text, tokens, reconstructed_tokens)

def load_mappings_from_file(path: str) -> dict[str, str]:
    """Load mappings from a text file"""
    mappings = {}
    try:
        with open(path, "r") as f:
            for line in f:
                if ',' in line:
                    k, v = line.strip().split(",", 1)  # Split on first comma only
                    mappings[k.lower()] = v.lower()  # Convert key and value to lowercase
        logger.info(f"Loaded {len(mappings)} mappings from file {path}")
    except Exception as e:
        logger.error(f"Error loading mappings from file {path}: {e}")
    return mappings

def translate_text(text: str, mappings: dict[str, str]) -> str:
    """
    Translate text by converting Devanagari to Latin (with diacritics) and adding diacritics 
    to Latin text based on provided mappings.
    
    Processing:
    1. Devanagari tokens → Latin with diacritics (phonetic, already complete)
    2. Latin tokens → Apply database mappings for diacritics
    """
    tokens = generate_tokens(text)
    preprocessed_tokens, skip_indices = convert_devanagari_tokens(tokens)
    return join_tokens(reconstruct_tokens(preprocessed_tokens, mappings, skip_indices))