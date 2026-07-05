import re

# Arabic diacritics (Tashkeel) Unicode constants
TASHKEEL_RE = re.compile(r"[\u064B-\u0652\u0670]")  # Fatha, Damma, Kasra, Sukun, Shaddah, Tanween, Superscript Alif

# Quranic small pause and pronunciation markings
QURANIC_MARKS_RE = re.compile(
    r"[\u0615-\u061A"       # Small high signs (Sallah, Qallah, etc.)
    r"\u06D6-\u06DC"       # Small high letters (Seen, Laam-Alif, etc.)
    r"\u06DF-\u06E8"       # Small round zero, rectangular zero, etc.
    r"\u06EA-\u06ED]"      # Empty center dot, low Seen, etc.
)

def normalize_arabic(text: str) -> str:
    """
    Standardizes Arabic text for string matching by:
    1. Removing all diacritics (Harakaat/Tashkeel).
    2. Removing Quranic small pause/marking symbols.
    3. Normalizing variations of Alif (أ, إ, آ) to plain Alif (ا).
    4. Normalizing Alif Maqsura (ى) to Yaa (ي).
    5. Normalizing Taa Marbuta (ة) to Haa (ه).
    6. Collapsing extra whitespaces.
    """
    if not text:
        return ""

    # Remove Quranic markings and pause markers first
    text = QURANIC_MARKS_RE.sub("", text)

    # Remove diacritics (Tashkeel)
    text = TASHKEEL_RE.sub("", text)

    # Normalize Alifs
    text = re.sub(r"[أإآ]", "ا", text)

    # Normalize Alif Maqsura (ى) to Yaa (ي)
    text = re.sub(r"ى", "ي", text)

    # Normalize Taa Marbuta (ة) to Haa (ه)
    text = re.sub(r"ة", "ه", text)

    # Collapse multiple whitespaces and trim edges
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    # Small test cases to verify normalizer functionality
    test_cases = [
        ("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ", "بسم الله الرحمن الرحيم"),
        ("مَن يَقُولُ", "من يقول"),
        ("وَالسَّمَاءِ ذَاتِ الْبُرُوجِ", "والسماء ذات البروج"),
        ("عَمَّ يَتَسَاءَلُونَ", "عم يتساءلون")
    ]

    print("Running normalizer test cases:")
    for original, expected in test_cases:
        normalized = normalize_arabic(original)
        status = "PASS" if normalized == expected else f"FAIL (Got: {normalized})"
        print(f"Original  : {original}")
        print(f"Normalized: {normalized} -> {status}\n")
