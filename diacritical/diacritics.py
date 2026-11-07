"""
Diacritical text processing module.
Handles the core functionality of processing text with diacritics.
"""

import docx
import re
import unicodedata
import logging
from devnagri import convert_devanagari_tokens

# Set up logging
logging.basicConfig(level=logging.INFO)
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

def match_case(word: str, mapping: str) -> str:
    """Match the case of a mapping to a word"""
    if word.isupper():
        return mapping.upper()
    elif word.islower():
        return mapping.lower()
    elif word[0].isupper():
        return mapping[0].upper() + mapping[1:]
    else:
        raise ValueError(f"Cannot add diacritics to word: {word} with mapping: {mapping}")

def handle_one_to_many_mappings(word: str, mapping: str) -> str:
    diacritics = mapping.replace("{{", "").replace("}}", "").replace(" ", "").split(",")
    if len(diacritics) == 1:
        # single word should display without curly brackets
        return match_case(word, diacritics[0])
    else:
        # multiple words
        return f"{{{{{','.join([match_case(word, m) for m in diacritics])}}}}}"

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
                    return handle_one_to_many_mappings(word.title(), matches[1])
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

def reconstruct_tokens(tokens: list[str], mappings: dict[str, str], skip_indices=None) -> list[str]:
    """Reconstruct tokens with diacritics; optionally skip applying mappings for indices."""
    translated_tokens = []
    for i, token in enumerate(tokens):
        if skip_indices and i in skip_indices:
            translated_tokens.append(token)
            continue
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