import os
import sys
import argparse
import logging

# Ensure stdout supports UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LocalSpeechTester")

# Setup import paths
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.preprocessing.audio import preprocess_audio


def test_raw_transcription(
    audio_path: str, 
    model_name: str = "rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final",
    output_path: str = "data/output/raw_transcript.txt"
):
    """
    Loads your custom audio file, preprocesses it, and transcribes it locally
    using the downloaded Wav2Vec2 model, printing and saving the raw Arabic transcription.
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'")
        return

    # Step 1: Preprocess the custom audio to 16kHz mono
    temp_wav_path = os.path.join(os.path.dirname(audio_path), "temp_processed_test.wav")
    print(f"1. Preprocessing audio: {audio_path}...")
    clean_audio = preprocess_audio(audio_path, temp_wav_path)
    
    # Step 2: Load model locally (completely offline)
    print(f"\n2. Loading model from local cache: {model_name}...")
    try:
        import torch
        import librosa
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    except ImportError:
        print("Error: Missing required packages. Run: pip install torch transformers librosa soundfile")
        return

    try:
        # Load offline (local_files_only=True ensures it makes 0 internet checks)
        processor = Wav2Vec2Processor.from_pretrained(model_name, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(model_name, local_files_only=True)
    except Exception as e:
        print("\nCould not find model locally. Retrying with internet connection to download...")
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name, local_files_only=False)
            model = Wav2Vec2ForCTC.from_pretrained(model_name, local_files_only=False)
        except Exception as online_err:
            print(f"Failed to load model: {str(online_err)}")
            return

    # Step 3: Run Inference on the clean audio
    print("\n3. Transcribing audio...")
    speech, sr = librosa.load(clean_audio, sr=16000)
    
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
        
    predicted_ids = torch.argmax(logits, dim=-1)
    raw_transcript = processor.batch_decode(predicted_ids)[0]
    
    print("\n" + "="*40)
    print("🔊 RAW TRANSCRIPTION RESULTS:")
    print("="*40)
    print(f"File       : {audio_path}")
    print(f"Transcript : {raw_transcript if raw_transcript else '[Silence/No speech detected]'}")
    print("="*40)

    # Save transcription to file in UTF-8
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(raw_transcript)
        print(f"\n[Saved raw transcript text to file: {output_path}]")

    # Clean up temp file
    if os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test local ASR model with custom Arabic audio")
    parser.add_argument("--audio", type=str, required=True, help="Path to your recording (e.g. data/raw/my_voice.wav)")
    parser.add_argument("--output", type=str, default="data/output/raw_transcript.txt", help="Path to save the text transcript")
    args = parser.parse_args()
    
    test_raw_transcription(args.audio, output_path=args.output)
