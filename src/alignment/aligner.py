import os
import random
import logging
from typing import List, Dict, Any, Tuple

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TajweedAligner")

class TajweedAligner:
    def __init__(self, use_ml: bool = False, model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"):
        """
        Initialize the Aligner.
        
        Args:
            use_ml (bool): If True, attempts to load local ML models (Wav2Vec2/HuggingFace).
                           If False, operates in simulation mode for rapid testing.
            model_name (str): The HF model repository path for Arabic speech alignment.
        """
        self.use_ml = use_ml
        self.model_name = model_name
        self.model = None
        self.processor = None

        if self.use_ml:
            try:
                import torch
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
                logger.info(f"Loading pre-trained Wav2Vec2 model (trying local cache first): {self.model_name}...")
                try:
                    # Attempt to load 100% offline from local cache to avoid HTTP requests
                    self.processor = Wav2Vec2Processor.from_pretrained(self.model_name, local_files_only=True)
                    self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name, local_files_only=True)
                except Exception:
                    # Fallback to online check if not cached locally
                    logger.info("Model not found in local cache. Checking Hugging Face hub...")
                    self.processor = Wav2Vec2Processor.from_pretrained(self.model_name, local_files_only=False)
                    self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name, local_files_only=False)
                logger.info("ML model loaded successfully.")
            except ImportError:
                logger.warning(
                    "ML libraries (torch/transformers) not found. "
                    "Aligner will fallback to simulation (Mock) mode."
                )
                self.use_ml = False

    def align(self, audio_path: str, ground_truth_text: str) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Aligns an audio file with ground truth words, yielding timestamps.
        
        Args:
            audio_path (str): Path to the normalized WAV audio.
            ground_truth_text (str): Expected canonical Quran text.
            
        Returns:
            Tuple[List[str], List[Dict[str, float]]]: Spoken words and their word boundaries.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.use_ml:
            return self._align_ml(audio_path, ground_truth_text)
        else:
            return self._align_simulation(audio_path, ground_truth_text)

    def _align_ml(self, audio_path: str, ground_truth_text: str) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Real ML alignment using HuggingFace Wav2Vec2 CTC Segmentation / Alignment.
        """
        import librosa
        import torch
        
        # Load audio (forced to 16kHz)
        speech, sr = librosa.load(audio_path, sr=16000)
        
        # Process input values
        input_values = self.processor(speech, sampling_rate=16000, return_tensors="pt").input_values
        
        # Perform inference
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        # Standard argmax to get predicted token IDs
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        # Break transcription into words
        spoken_words = transcription.split()
        
        # In a full forced-aligner (like MFA or CTC alignment), you backtrack the token sequence
        # to find frame indices. For this initial wrapper, we map approximate linear intervals 
        # from predicted logits duration for simplicity.
        duration = len(speech) / 16000.0
        num_words = len(spoken_words)
        
        spoken_timestamps = []
        if num_words > 0:
            step = duration / num_words
            for idx, word in enumerate(spoken_words):
                spoken_timestamps.append({
                    "start": round(idx * step, 2),
                    "end": round((idx + 1) * step, 2)
                })
                
        return spoken_words, spoken_timestamps

    def _align_simulation(self, audio_path: str, ground_truth_text: str) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Simulation Mode: Parses the ground-truth text, simulates minor reading errors 
        (like omissions or substitutions), and spreads out timestamps evenly across the audio length.
        """
        # Get audio duration
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
        except (ImportError, Exception):
            # Fallback if librosa is not installed or audio cannot be read
            duration = 4.0
            
        gt_words = ground_truth_text.split()
        
        # Simulate a 10% chance of an error (skipping the third word if length > 3)
        spoken_words = []
        skip_idx = 2 if len(gt_words) > 3 and random.random() < 0.3 else -1
        
        for idx, word in enumerate(gt_words):
            if idx == skip_idx:
                logger.info(f"[Simulation] Simulating student omitting word: '{word}'")
                continue
            spoken_words.append(word)
            
        # Distribute timestamps linearly across audio duration
        num_words = len(spoken_words)
        spoken_timestamps = []
        
        if num_words > 0:
            step = duration / num_words
            for idx in range(num_words):
                # Add tiny random jitter to make timestamps look realistic
                jitter_start = random.uniform(-0.02, 0.02)
                jitter_end = random.uniform(-0.02, 0.02)
                
                start_val = max(0.0, idx * step + jitter_start)
                end_val = min(duration, (idx + 1) * step + jitter_end)
                
                spoken_timestamps.append({
                    "start": round(start_val, 2),
                    "end": round(end_val, 2)
                })
                
        return spoken_words, spoken_timestamps


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    # Test script running in simulation mode
    print("Testing Aligner in Simulation Mode...")
    
    # We will pass a dummy duration test
    # Create a mock wav file if none exists, or just catch file not found
    aligner = TajweedAligner(use_ml=False)
    
    gt_text = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    
    try:
        # This will fail unless we have a file, so let's catch it or use a dummy duration
        words, timestamps = aligner.align("dummy_audio.wav", gt_text)
        print(f"Spoken Words      : {words}")
        print(f"Spoken Timestamps : {timestamps}")
    except FileNotFoundError:
        # If file is not present, we will test the internal simulator directly
        print("Note: dummy_audio.wav not found. Simulating alignment with fallback 4.0s duration:")
        words, timestamps = aligner._align_simulation("dummy_audio.wav", gt_text)
        print(f"Spoken Words      : {words}")
        print(f"Spoken Timestamps : {timestamps}")
