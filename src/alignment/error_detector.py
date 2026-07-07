import sys
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional

# Set system console to support UTF-8 prints on Windows tests
if __name__ == "__main__" and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import normalizer from sibling package
try:
    from src.text_utils.normalizer import normalize_arabic
    from src.text_utils.quran_g2p import quranic_g2p
except ImportError:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.text_utils.normalizer import normalize_arabic
    from src.text_utils.quran_g2p import quranic_g2p


def get_dynamic_threshold(word: str) -> float:
    """
    Computes a length-normalized similarity threshold.
    Short words get a softer bar (floor at 0.5), long words get a stricter bar.
    """
    base_word = normalize_arabic(word)
    length = len(base_word)
    if length <= 0:
        return 0.60
    return max(0.50, 1.0 - (2.0 / length))


def check_phonetic_similarity(w1: str, w2: str, threshold: Optional[float] = None) -> bool:
    """
    Computes phonetic similarity ratio using our G2P engine, falling back to
    normalized character sequence matching. Tolerates minor ASR spelling errors.
    """
    if threshold is None:
        threshold = get_dynamic_threshold(w1)

    # 1. Check normalized character similarity (highly robust to vowel noise)
    norm1 = normalize_arabic(w1)
    norm2 = normalize_arabic(w2)
    char_ratio = SequenceMatcher(None, norm1, norm2).ratio()
    if char_ratio >= threshold:
        return True

    # 2. Check G2P phoneme similarity
    p1 = quranic_g2p(w1)
    p2 = quranic_g2p(w2)
    if p1 and p2:
        phonetic_ratio = SequenceMatcher(None, p1, p2).ratio()
        if phonetic_ratio >= threshold:
            return True
            
    return False


def detect_word_errors(
    ground_truth_text: str,
    spoken_words: List[str],
    spoken_timestamps: List[Dict[str, float]]
) -> Dict[str, Any]:
    """
    Compares spoken text with ground-truth Quranic text to find word errors
    (omissions, insertions, substitutions) using sequence diff matching.
    
    Args:
        ground_truth_text (str): The correct Uthmani Quranic text.
        spoken_words (List[str]): List of words transcribed from the audio (ASR output).
        spoken_timestamps (List[Dict[str, float]]): List of word timestamps (dict with 'start' and 'end' keys).
        
    Returns:
        Dict[str, Any]: A dictionary containing aligned words and detected error logs.
    """
    # 1. Tokenize ground truth
    gt_words_original = ground_truth_text.split()
    
    # 2. Normalize both sets of words for string comparison (strip diacritics)
    gt_words_normalized = [normalize_arabic(w) for w in gt_words_original]
    spoken_words_normalized = [normalize_arabic(w) for w in spoken_words]
    
    # 3. Perform Sequence Matching
    matcher = SequenceMatcher(None, gt_words_normalized, spoken_words_normalized)
    opcodes = matcher.get_opcodes()
    
    # Prepare outputs
    alignment = []
    errors = []
    
    # Track word indexes
    # opcodes yield tuples of: (tag, i1, i2, j1, j2)
    # i1, i2 is slice for ground truth words
    # j1, j2 is slice for spoken words
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # Spoken words matched expected words in sequence
            for idx in range(i1, i2):
                spoken_idx = j1 + (idx - i1)
                t_info = spoken_timestamps[spoken_idx]
                alignment.append({
                    "word": gt_words_original[idx],
                    "start": t_info["start"],
                    "end": t_info["end"],
                    "status": "correct"
                })
                
        elif tag == 'delete':
            # Word was expected in ground truth, but missing in spoken audio (Omission)
            for idx in range(i1, i2):
                alignment.append({
                    "word": gt_words_original[idx],
                    "start": None,
                    "end": None,
                    "status": "missing"
                })
                errors.append({
                    "type": "missing_word",
                    "word": gt_words_original[idx],
                    "expected_index": idx
                })
                
        elif tag == 'insert':
            # Word was spoken in audio, but not present in ground truth (Insertion)
            for idx in range(j1, j2):
                t_info = spoken_timestamps[idx]
                errors.append({
                    "type": "extra_word",
                    "word": spoken_words[idx],
                    "timestamp_start": t_info["start"],
                    "timestamp_end": t_info["end"]
                })
                
        elif tag == 'replace':
            # Substituted text block. We match words index-by-index:
            # - Pairs up to min(N, M) are logged as wrong_word (substitutions)
            # - Remaining expected words are logged as missing_word (omissions)
            # - Remaining spoken words are logged as extra_word (insertions)
            N = i2 - i1
            M = j2 - j1
            max_len = max(N, M)
            
            for k in range(max_len):
                if k < N and k < M:
                    # Individual word substitution
                    gt_word = gt_words_original[i1 + k]
                    spoken_word = spoken_words[j1 + k]
                    t_info = spoken_timestamps[j1 + k]
                    
                    # Run phonetic similarity check (tolerate ASR spelling mistakes if pronunciation is close)
                    if check_phonetic_similarity(gt_word, spoken_word, threshold=None):
                        alignment.append({
                            "word": gt_word,
                            "start": t_info["start"],
                            "end": t_info["end"],
                            "status": "correct"
                        })
                    else:
                        alignment.append({
                            "word": gt_word,
                            "start": None,
                            "end": None,
                            "status": "substitution"
                        })
                        errors.append({
                            "type": "wrong_word",
                            "expected": gt_word,
                            "detected": spoken_word,
                            "timestamp_start": t_info["start"],
                            "timestamp_end": t_info["end"]
                        })
                elif k < N:
                    # Omission (expected word was skipped because spoken block is shorter)
                    gt_word = gt_words_original[i1 + k]
                    alignment.append({
                        "word": gt_word,
                        "start": None,
                        "end": None,
                        "status": "missing"
                    })
                    errors.append({
                        "type": "missing_word",
                        "word": gt_word,
                        "expected_index": i1 + k
                    })
                else:
                    # Insertion (extra word spoken because spoken block is longer)
                    spoken_word = spoken_words[j1 + k]
                    t_info = spoken_timestamps[j1 + k]
                    errors.append({
                        "type": "extra_word",
                        "word": spoken_word,
                        "timestamp_start": t_info["start"],
                        "timestamp_end": t_info["end"]
                    })
                
    return {
        "text": ground_truth_text,
        "alignment": alignment,
        "errors": errors
    }


if __name__ == "__main__":
    # Test dataset pipeline comparison logic
    print("Running error detection engine test cases...\n")
    
    # Case A: Student skips a word ("الرحمن")
    gt = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    spoken = ["بسم", "الله", "الرحيم"]
    timestamps = [
        {"start": 0.10, "end": 0.45},
        {"start": 0.50, "end": 0.90},
        {"start": 1.00, "end": 1.45}
    ]
    
    result = detect_word_errors(gt, spoken, timestamps)
    print("TEST CASE A: Missing Word ('الرحمن' skipped)")
    print(f"Ground Truth: {result['text']}")
    print(f"Alignment   : {result['alignment']}")
    print(f"Errors      : {result['errors']}\n")
    
    # Case B: Student substitutes a word ("الرحيم" with "القدوس")
    spoken_b = ["بسم", "الله", "الرحمن", "القدوس"]
    timestamps_b = [
        {"start": 0.10, "end": 0.45},
        {"start": 0.50, "end": 0.90},
        {"start": 0.95, "end": 1.35},
        {"start": 1.40, "end": 1.85}
    ]
    result_b = detect_word_errors(gt, spoken_b, timestamps_b)
    print("TEST CASE B: Substituted Word ('الرحيم' -> 'القدوس')")
    print(f"Alignment   : {result_b['alignment']}")
    print(f"Errors      : {result_b['errors']}\n")
