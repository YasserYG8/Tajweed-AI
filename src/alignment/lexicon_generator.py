import os
import sys
from typing import List, Set
try:
    from src.text_utils.quran_g2p import quranic_g2p
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.text_utils.quran_g2p import quranic_g2p

# Ensure stdout supports UTF-8 on Windows
if __name__ == "__main__" and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def generate_lexicon_from_text(text: str, output_path: str) -> None:
    """
    Parses an input Quranic text, extracts all unique words, computes
    their phonetic symbols using our G2P engine, and writes them
    to a Montreal Forced Aligner (MFA) compatible lexicon dictionary file.
    
    Format:
    word <tab> p1 p2 p3
    """
    if not text:
        print("Empty text provided. Skipping lexicon generation.")
        return

    # Clean punctuation and split by spaces
    words = text.split()
    
    # Track unique words to avoid duplicate lexicon entries
    unique_words: Set[str] = set(words)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    print(f"Generating lexicon for {len(unique_words)} unique words...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for word in sorted(unique_words):
            # Compute phonemes using our G2P engine
            phonemes = quranic_g2p(word)
            if phonemes:
                # Format: word <tab> space-separated phonemes
                phonemes_str = " ".join(phonemes)
                f.write(f"{word}\t{phonemes_str}\n")
                
    print(f"Lexicon generated successfully at: {output_path}")


if __name__ == "__main__":
    # Test generation with Surah Al-Fatihah, Ayah 1
    sample_text = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"
    output_dict = "data/output/test_lexicon.txt"
    
    print("Testing lexicon generator...")
    generate_lexicon_from_text(sample_text, output_dict)
    
    # Read back and print the generated lexicon
    if os.path.exists(output_dict):
        with open(output_dict, "r", encoding="utf-8") as f:
            print("\nGenerated Lexicon Contents:")
            print("="*40)
            print(f.read().strip())
            print("="*40)
