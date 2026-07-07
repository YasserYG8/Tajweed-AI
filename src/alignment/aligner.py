import os
import random
import logging
from typing import List, Dict, Any, Tuple

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TajweedAligner")

class TajweedAligner:
    def __init__(self, use_ml: bool = False, model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic", use_whisper: bool = False):
        """
        Initialize the Aligner.
        
        Args:
            use_ml (bool): If True, attempts to load local ML models (Wav2Vec2/HuggingFace).
                           If False, operates in simulation mode for rapid testing.
            model_name (str): The HF model repository path for Arabic speech alignment.
            use_whisper (bool): If True, uses Whisper for speech transcription instead of Wav2Vec2.
        """
        self.use_ml = use_ml
        self.model_name = model_name
        self.use_whisper = use_whisper
        self.model = None
        self.processor = None
        self.whisper_model_name = "tarteel-ai/whisper-base-ar-quran"
        self.whisper_model = None
        self.whisper_processor = None

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

    def align(self, audio_path: str, ground_truth_text: str, force_align: bool = False) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Aligns an audio file with ground truth words, yielding timestamps.
        
        Args:
            audio_path (str): Path to the normalized WAV audio.
            ground_truth_text (str): Expected canonical Quran text.
            force_align (bool): If True, bypasses ASR spelling decoding and aligns directly to ground truth.
            
        Returns:
            Tuple[List[str], List[Dict[str, float]]]: Spoken words and their word boundaries.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.use_ml:
            return self._align_ml(audio_path, ground_truth_text, force_align=force_align)
        else:
            return self._align_simulation(audio_path, ground_truth_text)

    def _align_ml(self, audio_path: str, ground_truth_text: str, force_align: bool = False) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Real ML alignment using HuggingFace Wav2Vec2 CTC Segmentation / Viterbi Forced Alignment.
        """
        import librosa
        import torch
        import numpy as np
        
        try:
            from src.text_utils.normalizer import normalize_arabic
        except ImportError:
            import sys
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            from src.text_utils.normalizer import normalize_arabic

        # 1. Load audio (forced to 16kHz)
        speech, sr = librosa.load(audio_path, sr=16000)
        duration = len(speech) / 16000.0
        
        # 2. Perform ML Inference to get character logits
        input_values = self.processor(speech, sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        # Convert logits to log-probabilities
        logits_np = logits[0].cpu().numpy()
        max_logits = np.max(logits_np, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_np - max_logits)
        log_probs = (logits_np - max_logits) - np.log(np.sum(exp_logits, axis=-1, keepdims=True) + 1e-9)
        
        frames = log_probs.shape[0]
        frame_duration = duration / frames

        # 3. Transcribe or use ground-truth directly
        # Split-Path Architecture: 
        # - Wav2Vec2 CTC logits are always used for forced-aligning the text to audio frames (Viterbi trellis).
        # - The text to align can come from three sources:
        #   1. Ground truth text (when force_align=True)
        #   2. Whisper transcription (when use_whisper=True) -> Whisper acts as the STT engine, Wav2Vec2 acts as the aligner.
        #   3. Wav2Vec2 transcription (when use_whisper=False) -> Wav2Vec2 acts as both STT and aligner.
        if force_align:
            transcription = ground_truth_text
        else:
            if self.use_whisper:
                # Load Whisper model on demand if not already loaded/disabled
                if self.whisper_model is None:
                    try:
                        from transformers import WhisperForConditionalGeneration, WhisperProcessor
                        logger.info(f"Loading Whisper model (trying local cache first): {self.whisper_model_name}...")
                        try:
                            self.whisper_processor = WhisperProcessor.from_pretrained(self.whisper_model_name, local_files_only=True)
                            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(self.whisper_model_name, local_files_only=True)
                        except Exception:
                            logger.info("Whisper model not found in local cache. Fetching from Hugging Face hub...")
                            self.whisper_processor = WhisperProcessor.from_pretrained(self.whisper_model_name, local_files_only=False)
                            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(self.whisper_model_name, local_files_only=False)
                    except Exception as e:
                        logger.error(f"Failed to load Whisper model: {str(e)}. Falling back to Wav2Vec2 transcription.")
                        self.whisper_model = False

                if self.whisper_model:
                    try:
                        # Whisper expects 16kHz audio array directly as input
                        input_features = self.whisper_processor(speech, sampling_rate=16000, return_tensors="pt").input_features
                        with torch.no_grad():
                            generated_ids = self.whisper_model.generate(input_features)
                        transcription = self.whisper_processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                        
                        # Clean Whisper special tokens and non-Arabic metadata characters
                        import re
                        transcription = re.sub(r"<\|.*?\|>", "", transcription)
                        transcription = re.sub(r"[a-zA-Z]", "", transcription)
                        transcription = re.sub(r"\s+", " ", transcription).strip()
                        
                        logger.info(f"Whisper Hybrid Transcription: {transcription}")
                    except Exception as e:
                        logger.error(f"Whisper transcription failed: {str(e)}. Falling back to Wav2Vec2 transcription.")
                        predicted_ids = torch.argmax(logits, dim=-1)
                        transcription = self.processor.batch_decode(predicted_ids)[0]
                else:
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = self.processor.batch_decode(predicted_ids)[0]
            else:
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
        
        words_list = transcription.split()
        if not words_list:
            return [], []
            
        norm_text = normalize_arabic(transcription)
        target_text = norm_text.replace(" ", "|")
        
        vocab = self.processor.tokenizer.get_vocab()
        blank_id = self.processor.tokenizer.pad_token_id
        if blank_id is None:
            blank_id = 0
            
        target_ids = []
        char_to_word_map = []
        curr_word_idx = 0
        
        for char in target_text:
            if char == '|':
                target_ids.append(vocab.get('|', 4))
                char_to_word_map.append(-1)
                curr_word_idx += 1
            else:
                target_ids.append(vocab.get(char, vocab.get('<unk>', 3)))
                char_to_word_map.append(curr_word_idx)
                
        # Interleave blank tokens to support true CTC alignment path transitions
        # T_ctc = [blank, char0, blank, char1, blank, ...]
        ctc_ids = []
        ctc_word_map = []
        for idx in range(len(target_ids)):
            ctc_ids.append(blank_id)
            ctc_word_map.append(-1)
            ctc_ids.append(target_ids[idx])
            ctc_word_map.append(char_to_word_map[idx])
        ctc_ids.append(blank_id)
        ctc_word_map.append(-1)
        
        L_ctc = len(ctc_ids)
        
        # Fallback to linear segmentation if text is longer than audio frames
        if L_ctc > frames or L_ctc == 0:
            logger.warning("Target text CTC token length exceeds audio frames. Falling back to linear alignment.")
            spoken_timestamps = []
            if len(words_list) > 0:
                step = duration / len(words_list)
                for idx in range(len(words_list)):
                    spoken_timestamps.append({
                        "start": round(idx * step, 2),
                        "end": round((idx + 1) * step, 2)
                    })
            return words_list, spoken_timestamps

        # 4. Run Viterbi CTC Alignment Trellis (interleaved blanks version)
        trellis = np.full((frames, L_ctc), -np.inf)
        backpointers = np.zeros((frames, L_ctc), dtype=int)
        
        # Init first frame: can start at blank (idx 0) or first char (idx 1)
        trellis[0, 0] = log_probs[0, blank_id]
        if L_ctc > 1:
            trellis[0, 1] = log_probs[0, ctc_ids[1]]
            
        for t in range(1, frames):
            for s in range(L_ctc):
                # Standard CTC state transitions:
                stay = trellis[t-1, s]
                prev = trellis[t-1, s-1] if s > 0 else -np.inf
                skip = -np.inf
                # Can skip blank if current token is not blank and is different from two steps back
                if s > 1 and ctc_ids[s] != blank_id and ctc_ids[s] != ctc_ids[s-2]:
                    skip = trellis[t-1, s-2]
                    
                candidates = [stay, prev, skip]
                best_idx = int(np.argmax(candidates))
                best_prob = candidates[best_idx]
                
                trellis[t, s] = log_probs[t, ctc_ids[s]] + best_prob
                
                if best_idx == 0:
                    backpointers[t, s] = s
                elif best_idx == 1:
                    backpointers[t, s] = s - 1
                else:
                    backpointers[t, s] = s - 2

        # 5. Backtrace optimal path
        path = []
        curr_s = L_ctc - 1
        # Final frame can end at last blank (L_ctc-1) or last char (L_ctc-2)
        if L_ctc > 1 and trellis[frames-1, L_ctc-2] > trellis[frames-1, L_ctc-1]:
            curr_s = L_ctc - 2
            
        for t in range(frames - 1, -1, -1):
            path.append(curr_s)
            curr_s = backpointers[t, curr_s]
            
        path.reverse()
        
        # 6. Group frame timestamps into words using the character-to-word map
        word_frames = {w_idx: [] for w_idx in range(len(words_list))}
        
        for t, s_idx in enumerate(path):
            w_idx = ctc_word_map[s_idx]
            if w_idx != -1:
                word_frames[w_idx].append(t)
                
        spoken_timestamps = []
        for w_idx in range(len(words_list)):
            frames_for_word = word_frames[w_idx]
            if frames_for_word:
                start_time = round(min(frames_for_word) * frame_duration, 2)
                end_time = round((max(frames_for_word) + 1) * frame_duration, 2)
                spoken_timestamps.append({
                    "start": start_time,
                    "end": end_time
                })
            else:
                spoken_timestamps.append({
                    "start": None,
                    "end": None
                })
                
        return words_list, spoken_timestamps

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
