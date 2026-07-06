import re
import sys
from typing import List

# Ensure stdout supports UTF-8 on Windows
if __name__ == "__main__" and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Phoneme mappings for standard Arabic letters
CHAR_TO_PHONEME = {
    'ء': '2', 'أ': '2', 'إ': '2', 'ؤ': '2', 'ئ': '2', 'آ': '2',
    'ب': 'b',
    'ت': 't',
    'ث': 'th',
    'ج': 'j',
    'ح': 'H',     # Pharyngeal Haa
    'خ': 'kh',
    'د': 'd',
    'ذ': 'dh',
    'ر': 'r',
    'ز': 'z',
    'س': 's',
    'ش': 'sh',
    'ص': 'S',     # Emphatic S
    'ض': 'D',     # Emphatic D
    'ط': 'T',     # Emphatic T
    'ظ': 'Z',     # Emphatic Z
    'ع': '3',     # Pharyngeal Ayn
    'غ': 'gh',
    'ف': 'f',
    'ق': 'q',
    'ك': 'k',
    'ل': 'l',
    'م': 'm',
    'ن': 'n',
    'ه': 'h',     # Plain H
    'و': 'w',     # Waw
    'ي': 'y',     # Yaa
    'ى': 'y',     # Alif Maqsura
    'ة': 't',     # Taa Marbuta
}

# Diacritics lists
FATHA = '\u064E'
DAMMA = '\u064F'
KASRA = '\u0650'
SUKUN = '\u0652'
SHADDAH = '\u0651'
SUPERSCRIPT_ALIF = '\u0670'

FATHATAYN = '\u064B'
DAMMATAYN = '\u064C'
KASRATAYN = '\u064D'

DIACRITICS = {FATHA, DAMMA, KASRA, SUKUN, SHADDAH, SUPERSCRIPT_ALIF, FATHATAYN, DAMMATAYN, KASRATAYN}
SILENT_LETTER_MARK = '\u06DF'

SOLAR_LETTERS = {'ت', 'ث', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ل', 'ن'}

def quranic_g2p(text: str) -> List[str]:
    """
    Converts classical Uthmani Quranic Arabic text into a list of phonetic symbols (phonemes),
    applying standard Tajweed pronunciation rules:
    1. Solar/Lunar Laam (Al- prefix)
    2. Shaddah (doubling)
    3. Long vowels (Madd) vs Short Vowels
    4. Idgham (assimilation of silent Noon/Tanween)
    """
    if not text:
        return []

    # Clean pause markers to make processing reliable
    text = re.sub(r"[\u0615-\u061A\u06D6-\u06DC\u06E0-\u06E8\u06EA-\u06ED]", "", text)
    words = text.split()
    phonemes = []
    
    # State tracking for Idgham across word boundaries
    double_next_consonant = False

    for word_idx, word in enumerate(words):
        word_phonemes = []
        char_idx = 0
        n_chars = len(word)

        # Handle 'Al-' (ال) prefix if it starts the word
        # In Quran, Alif can have no vowel or helper vowel, followed by Laam
        if n_chars > 2 and word[0] == 'ا' and word[1] == 'ل':
            # 1. Add initial 'a' sound for Alif
            word_phonemes.append('a')
            
            # Find the first consonant after 'ل' (skip diacritics if any)
            next_consonant_idx = 2
            while next_consonant_idx < n_chars and word[next_consonant_idx] in DIACRITICS:
                next_consonant_idx += 1
            
            if next_consonant_idx < n_chars:
                next_consonant = word[next_consonant_idx]
                if next_consonant in SOLAR_LETTERS:
                    # Solar Laam: Laam is silent. In Uthmani Quranic text, the solar letter is
                    # written with a Shaddah, so the main loop will handle its doubling naturally.
                    pass
                else:
                    # Lunar Laam: Laam is pronounced as 'l'
                    word_phonemes.append('l')
            else:
                # Fallback if nothing follows 'ال'
                word_phonemes.append('l')
            
            # Advance past 'ا' and 'ل'
            char_idx = 2

        # Main letter-by-letter loop
        while char_idx < n_chars:
            char = word[char_idx]
            
            # Skip silent letters explicitly marked with Sifr
            if char_idx + 1 < n_chars and word[char_idx + 1] == SILENT_LETTER_MARK:
                char_idx += 2
                continue
                
            # Skip floating diacritics
            if char in DIACRITICS or char == SILENT_LETTER_MARK:
                char_idx += 1
                continue

            # Gather diacritics belonging to this consonant
            has_shaddah = False
            has_fatha = False
            has_damma = False
            has_kasra = False
            has_sukun = False
            has_tanween = False

            next_letter_idx = char_idx + 1
            while next_letter_idx < n_chars and word[next_letter_idx] in DIACRITICS:
                d = word[next_letter_idx]
                if d == SHADDAH:
                    has_shaddah = True
                elif d == FATHA or d == SUPERSCRIPT_ALIF:
                    has_fatha = True
                elif d == DAMMA:
                    has_damma = True
                elif d == KASRA:
                    has_kasra = True
                elif d == SUKUN:
                    has_sukun = True
                elif d in [FATHATAYN, DAMMATAYN, KASRATAYN]:
                    has_tanween = True
                next_letter_idx += 1

            # Check if this consonant is a Noon-Sukun or has Tanween for Idgham
            is_noon_sukun = (char == 'ن' and (has_sukun or (not has_fatha and not has_damma and not has_kasra and not has_tanween and next_letter_idx == n_chars)))
            
            if (is_noon_sukun or has_tanween) and word_idx + 1 < len(words):
                # Look at the first letter of the next word
                next_word = words[word_idx + 1]
                # If next word starts with Al- prefix, look past Al-
                next_word_target_char = next_word[0]
                if len(next_word) > 2 and next_word[0] == 'ا' and next_word[1] == 'ل':
                    # Find next consonant after Al-
                    target_idx = 2
                    while target_idx < len(next_word) and next_word[target_idx] in DIACRITICS:
                        target_idx += 1
                    if target_idx < len(next_word):
                        next_word_target_char = next_word[target_idx]

                if next_word_target_char in ['ي', 'ر', 'م', 'ل', 'و', 'ن']:
                    # Trigger Idgham: Skip the Noon sound, and set flag to double the next consonant
                    double_next_consonant = True
                    char_idx = next_letter_idx
                    continue

            # Convert character to phoneme
            p_letter = CHAR_TO_PHONEME.get(char, "")
            
            if p_letter:
                # Add the consonant phoneme
                word_phonemes.append(p_letter)
                
                # Apply doubling due to Shaddah OR Idgham
                if has_shaddah or double_next_consonant:
                    word_phonemes.append(p_letter)
                    double_next_consonant = False # Reset flag

                # Check for long vowel extensions (Madd)
                is_long_alif = has_fatha and next_letter_idx < n_chars and word[next_letter_idx] == 'ا'
                is_long_waw = has_damma and next_letter_idx < n_chars and word[next_letter_idx] == 'و'
                is_long_yaa = has_kasra and next_letter_idx < n_chars and word[next_letter_idx] == 'ي'
                
                if is_long_alif:
                    word_phonemes.append('aa')
                    next_letter_idx += 1 # Consume the silent long vowel letter
                elif is_long_waw:
                    word_phonemes.append('uu')
                    next_letter_idx += 1
                elif is_long_yaa:
                    word_phonemes.append('ii')
                    next_letter_idx += 1
                else:
                    # Standard short vowels
                    if has_fatha:
                        word_phonemes.append('a')
                    elif has_damma:
                        word_phonemes.append('u')
                    elif has_kasra:
                        word_phonemes.append('i')

            char_idx = next_letter_idx

        phonemes.extend(word_phonemes)

    return phonemes


if __name__ == "__main__":
    # Test cases to verify G2P conversions
    test_cases = [
        ("بِسْمِ", ["b", "i", "s", "m", "i"]),
        ("اللَّهِ", ["a", "l", "l", "a", "h", "i"]),
        ("مَن يَقُولُ", ["m", "a", "y", "y", "a", "q", "uu", "l", "u"]), # Idgham (Noon merges into Yaa)
        ("قَالُوا۟", ["q", "aa", "l", "uu"])                       # Silent Alif (Sifr Mustadeer ignored)
    ]
    
    print("Running Quranic G2P parser tests:")
    for arabic, expected in test_cases:
        phonemes_out = quranic_g2p(arabic)
        status = "PASS" if phonemes_out == expected else f"FAIL (Got: {phonemes_out})"
        print(f"Arabic   : {arabic}")
        print(f"Expected : {expected}")
        print(f"Phonemes : {phonemes_out} -> {status}\n")
