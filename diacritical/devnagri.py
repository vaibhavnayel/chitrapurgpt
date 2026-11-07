

def contains_devanagari(text):
    """Check if text contains any Devanagari characters (Unicode range U+0900 to U+097F)"""
    if not text:
        return False
    return any('\u0900' <= char <= '\u097F' for char in text)

def convert_devanagari_tokens(tokens):
    """
    Process a list of tokens and convert any Devanagari tokens to Latin characters with diacritics.
    
    Args:
        tokens: List of text tokens
        
    Returns:
        Tuple of (preprocessed_tokens, devanagari_indices) where:
        - preprocessed_tokens: List of tokens with Devanagari converted to Latin
        - devanagari_indices: Set of indices that were originally Devanagari (already have diacritics)
    """
    preprocessed_tokens = []
    devanagari_indices = set()
    
    for i, token in enumerate(tokens):
        if contains_devanagari(token):
            # Convert Devanagari characters to Latin (already includes diacritics)
            converted_token = map_characters(token)
            preprocessed_tokens.append(converted_token)
            devanagari_indices.add(i)
        else:
            # Keep non-Devanagari tokens as-is
            preprocessed_tokens.append(token)
    
    return preprocessed_tokens, devanagari_indices

def map_characters(input_string):
    transform_string = ''
    for i, char in enumerate(input_string):
        if (input_string[i] != "\u094D"):  # Halant
            if (i + 1 < len(input_string)) and (input_string[i + 1] < "\u093D") and (input_string[i] != " ") and (
                input_string[i] < "\u093D") and (input_string[i] != "\u0902") and (
                input_string[i] != "\u0903") and (input_string[i] != "\n") and (input_string[i] != "\u002D") and (
                input_string[i] != "\u0905") and (input_string[i] != "\u0906") and (
                input_string[i] != "\u0907") and (input_string[i] != "\u0908") and (
                input_string[i] != "\u0909") and (input_string[i] != "\u090A") and (
                input_string[i] != "\u090B") and (input_string[i] != "\u090C") and (
                input_string[i] != "\u090F") and (input_string[i] != "\u0910") and (
                input_string[i] != "\u0913") and (input_string[i] != "\u0914") and (
                input_string[i] != "\u007C") and (input_string[i] != "\u002E" and (input_string[i] != "\u000A") and ((input_string[i] > "\u0914") and (input_string[i] < "\u0962"))):
                transform_string += lookup_table.get(char, char) + 'a'
            else:
                transform_string += lookup_table.get(char, char)
    return transform_string

lookup_table = {
    'ं': 'ṁ',
    'ः': 'ḥ',
    'अ': 'a',
    'आ': 'ā',
    'इ': 'i',
    'ई': 'ī',
    'उ': 'u',
    'ऊ': 'ū',
    'ऋ': 'ṛ',
    'ऌ': 'ḷ',
    'ए': 'e',
    'ऐ': 'ai',
    'ओ': 'o',
    'औ': 'au',
    'क': 'k',
    'ख': 'kh',
    'ग': 'g',
    'घ': 'gh',
    'ङ': 'ṅ',
    'च': 'ch',
    'छ': 'c͟h',
    'ज': 'j',
    'झ': 'jh',
    'ञ': 'ñ',
    'ट': 'ṭ',
    'ठ': 't͟h',
    'ड': 'ḍ',
    'ढ': 'd͟h',
    'ण': 'ṇ',
    'त': 't',
    'थ': 'th',
    'द': 'd',
    'ध': 'dh',
    'न': 'n',
    'प': 'p',
    'फ': 'ph',
    'ब': 'b',
    'भ': 'bh',
    'म': 'm',
    'य': 'y',
    'र': 'r',
    'ल': 'l',
    'ळ': 'Ī',
    'व': 'v',
    'श': 'sh',
    'ष': 's͟h',
    'स': 's',
    'ह': 'h',
    'ॠ': 'ṝ',
    'ॡ': 'ḹ',
    'ॡ': 'ḹ',
    'ा': 'ā',
    'ि': 'i',
    'ी': 'ī',
    'ु': 'u',
    'ू': 'ū',
    'ृ': 'ṛ',
    'ॄ': 'ḷ',
    'े': 'e',
    'ै': 'ai',
    'ो': 'o',
    'ौ': 'au',
    '०': '0',
    '१': '1',
    '२': '2',
    '३': '3',
    '४': '4',
    '५': '5',
    '६': '6',
    '७': '7',
    '८': '8',
    '९': '9',
    'ॐ': 'oṁ',
    'ँ': 'a̐',
    'ऍ': 'ă',
    'ऑ': 'ŏ',
    'ॅ': 'ă',
    'ॉ': 'ŏ'
}
