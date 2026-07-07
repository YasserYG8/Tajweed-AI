import os
import shutil
import logging

logger = logging.getLogger("AudioPrep")

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

    try:
        import numpy as np
        import librosa
        import soundfile as sf
    except ImportError:
        logger.warning(
            "DSP libraries (librosa/soundfile/numpy) not found. "
            "Skipping preprocessing and copying raw file directly."
        )
        shutil.copy(input_path, output_path)
        return os.path.abspath(output_path)

    # 1 & 2. Load audio and resample to target sample rate (sr=target_sr forces conversion)
    # mono=True converts stereo to mono automatically
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)

    # 3. Trim leading and trailing silences
    # librosa.effects.trim returns the trimmed signal and the start/end frame indices
    y_trimmed, index = librosa.effects.trim(y, top_db=-silence_threshold_db)

    # If the file became completely empty after trimming, default back to original y
    if len(y_trimmed) == 0:
        y_trimmed = y

    # Apply zero-phase Butterworth Bandpass filter (80Hz to 7500Hz) to eliminate mic hiss and AC hum
    try:
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * target_sr
        low = 80.0 / nyq
        high = 7500.0 / nyq
        b, a = butter(4, [low, high], btype='band')
        y_filtered = filtfilt(b, a, y_trimmed)
    except Exception as e:
        logger.warning(f"Failed to apply digital bandpass filter: {str(e)}. Using raw trimmed audio.")
        y_filtered = y_trimmed

    # Apply adaptive rolling RMS Noise Gate to silence background reverb without gating quiet speech
    try:
        window_size = 480  # 30ms window at 16kHz
        padded = np.pad(y_filtered, window_size // 2, mode='reflect')
        squared = padded ** 2
        window = np.ones(window_size) / window_size
        local_mean_squared = np.convolve(squared, window, mode='valid')
        local_mean_squared = local_mean_squared[:len(y_filtered)]
        local_rms = np.sqrt(np.maximum(local_mean_squared, 1e-9))
        
        max_rms = np.max(local_rms)
        mean_rms = np.mean(local_rms)
        
        if max_rms > 0:
            # Threshold scales with mean energy, clamped between 0.001 (floor) and 0.008 (ceiling)
            adaptive_threshold = np.clip(0.015 * mean_rms, 0.001, 0.008)
            y_gated = np.where(local_rms < adaptive_threshold, 0.0, y_filtered)
            
            max_peak = np.max(np.abs(y_gated))
            y_normalized = y_gated / max_peak if max_peak > 0 else y_gated
        else:
            y_normalized = y_filtered
    except Exception as e:
        logger.warning(f"Failed to apply adaptive RMS noise gate: {str(e)}. Using bandpass filtered audio directly.")
        # Fallback to standard peak normalization
        max_peak = np.max(np.abs(y_filtered))
        y_normalized = y_filtered / max_peak if max_peak > 0 else y_filtered

    # 5. Save as a standard WAV file
    sf.write(output_path, y_normalized, target_sr, format='WAV', subtype='PCM_16')

    return os.path.abspath(output_path)


if __name__ == "__main__":
    # Small self-test execution
    print("Audio preprocessing library loaded successfully.")
