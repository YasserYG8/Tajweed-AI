import os
import json
import sys
import argparse
import logging
from typing import Dict, Any

# Ensure stdout supports UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TajweedPipeline")

# Import system modules
try:
    from src.preprocessing.audio import preprocess_audio
    from src.alignment.aligner import TajweedAligner
    from src.alignment.error_detector import detect_word_errors
except ImportError:
    # Fallback to local paths if run as a script directly
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.preprocessing.audio import preprocess_audio
    from src.alignment.aligner import TajweedAligner
    from src.alignment.error_detector import detect_word_errors


def run_pipeline(
    audio_path: str,
    ground_truth_text: str,
    output_json_path: str,
    surah: int = 1,
    ayah: int = 1,
    reciter_id: str = "student_01",
    reciter_type: str = "student",
    use_ml: bool = False,
    force_align: bool = False
) -> Dict[str, Any]:
    """
    Executes the entire Phase 1 Word-Level MVP Pipeline:
    1. Preprocesses the raw audio (resampling, trimming, normalization).
    2. Aligns audio words to generate timestamps.
    3. Compares spoken text against ground-truth and flags errors.
    4. Combines results and saves them to a formatted JSON dataset entry.
    """
    logger.info(f"Starting pipeline processing for Surah {surah}, Ayah {ayah}...")
    
    # Define temporary file for processed audio
    dir_name = os.path.dirname(audio_path)
    base_name = os.path.basename(audio_path)
    processed_audio_path = os.path.join(dir_name, "processed_" + base_name)

    # Step 1: Preprocess Audio
    logger.info("Executing Step 1: Audio Preprocessing...")
    clean_audio = preprocess_audio(audio_path, processed_audio_path)
    logger.info(f"Clean audio saved to: {clean_audio}")

    # Step 2: Alignment (Transcribe + Word boundaries)
    logger.info("Executing Step 2: Speech-to-Text & Forced Alignment...")
    aligner = TajweedAligner(use_ml=use_ml)
    spoken_words, spoken_timestamps = aligner.align(clean_audio, ground_truth_text, force_align=force_align)
    logger.info(f"Aligned {len(spoken_words)} spoken words.")

    # Step 3: Error Detection
    logger.info("Executing Step 3: Levenshtein Word-Level Error Detection...")
    error_results = detect_word_errors(ground_truth_text, spoken_words, spoken_timestamps)
    
    # Step 4: Schema Generator & Export
    logger.info("Executing Step 4: Creating Dataset Entry JSON Schema...")
    dataset_entry = {
        "audio_id": os.path.basename(audio_path),
        "surah": surah,
        "ayah": ayah,
        "reciter_id": reciter_id,
        "reciter_type": reciter_type,
        "text": ground_truth_text,
        "alignment": error_results["alignment"],
        "errors": error_results["errors"],
        "teacher_verified": False
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_entry, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Successfully exported dataset entry to: {output_json_path}")
    return dataset_entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tajweed AI Phase 1 Pipeline CLI")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--text", type=str, required=True, help="Ground-truth Quranic text")
    parser.add_argument("--output", type=str, required=True, help="Destination JSON path")
    parser.add_argument("--surah", type=int, default=1, help="Surah index")
    parser.add_argument("--ayah", type=int, default=1, help="Ayah index")
    parser.add_argument("--reciter", type=str, default="student_01", help="Reciter Identifier")
    parser.add_argument("--reciter_type", type=str, choices=["student", "teacher"], default="student", help="Reciter role")
    parser.add_argument("--use_ml", action="store_true", help="Enable local ML models (Wav2Vec2)")
    parser.add_argument("--force_align", action="store_true", help="Bypass ASR spelling decoding and align directly to ground truth")
    
    args = parser.parse_args()
    
    try:
        run_pipeline(
            audio_path=args.audio,
            ground_truth_text=args.text,
            output_json_path=args.output,
            surah=args.surah,
            ayah=args.ayah,
            reciter_id=args.reciter,
            reciter_type=args.reciter_type,
            use_ml=args.use_ml,
            force_align=args.force_align
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)
