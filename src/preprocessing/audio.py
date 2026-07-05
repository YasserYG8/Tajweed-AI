import os
import numpy as np
import librosa
import soundfile as sf

def preprocess_audio(
    input_path: str, 
    output_path: str, 
    target_sr: int = 16000, 
    silence_threshold_db: float = -30.0
) -> str:
    """
    Standardizes input audio file for speech processing:
    1. Loads the audio (supports wav, mp3, m4a, etc.).
    2. Resamples it to 16,000 Hz.
    3. Converts multi-channel (stereo) to mono.
    4. Trims leading and trailing silence.
    5. Normalizes amplitude peaks to -1.0 to 1.0.
    6. Saves as a standard WAV file.
    
    Args:
        input_path (str): Path to raw audio file.
        output_path (str): Destination path for the processed WAV file.
        target_sr (int): Sample rate to convert to (default 16kHz).
        silence_threshold_db (float): Threshold in dB for trimming silence.
        
    Returns:
        str: Absolute path to the saved WAV file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file not found: {input_path}")
        
    # Create directory for output if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1 & 2. Load audio and resample to target sample rate (sr=target_sr forces conversion)
    # mono=True converts stereo to mono automatically
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)

    # 3. Trim leading and trailing silences
    # librosa.effects.trim returns the trimmed signal and the start/end frame indices
    y_trimmed, index = librosa.effects.trim(y, top_db=-silence_threshold_db)

    # If the file became completely empty after trimming, default back to original y
    if len(y_trimmed) == 0:
        y_trimmed = y

    # 4. Peak Normalization (adjust scale to max out at 1.0 or -1.0)
    max_peak = np.max(np.abs(y_trimmed))
    if max_peak > 0:
        y_normalized = y_trimmed / max_peak
    else:
        y_normalized = y_trimmed

    # 5. Save as a standard WAV file
    sf.write(output_path, y_normalized, target_sr, format='WAV', subtype='PCM_16')

    return os.path.abspath(output_path)


if __name__ == "__main__":
    # Small self-test execution
    print("Audio preprocessing library loaded successfully.")
