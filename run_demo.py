import os
import wave
import struct
import json
import sys

# Ensure stdout supports UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure directories exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/output", exist_ok=True)

# Path definitions
dummy_audio_path = "data/raw/test_student.wav"
output_json_path = "data/output/alignment_result.json"

# 1. Generate a 3-second dummy WAV file programmatically (sine wave / silence)
print(f"1. Generating dummy wave file at: {dummy_audio_path}...")
sample_rate = 16000
duration = 3.0  # seconds
num_samples = int(sample_rate * duration)

# Open WAV file for writing
with wave.open(dummy_audio_path, 'wb') as wav_file:
    # Set parameters: 1 channel (mono), 2 bytes per sample (16-bit), sample rate
    wav_file.setparams((1, 2, sample_rate, num_samples, 'NONE', 'not compressed'))
    
    # Write a simple quiet silent signal
    for _ in range(num_samples):
        # 16-bit PCM silence is 0
        data = struct.pack('<h', 0)
        wav_file.writeframesraw(data)

print("Dummy wave file generated successfully.")

# 2. Run the pipeline
print("\n2. Executing the Tajweed AI Alignment Pipeline...")
from src.pipeline import run_pipeline

gt_text = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"

result = run_pipeline(
    audio_path=dummy_audio_path,
    ground_truth_text=gt_text,
    output_json_path=output_json_path,
    surah=1,
    ayah=1,
    reciter_id="student_demo",
    reciter_type="student",
    use_ml=False  # Operates in simulation mode for local running
)

# 3. Read and print the resulting JSON
print("\n3. Generated JSON Schema Result:")
print(json.dumps(result, ensure_ascii=False, indent=2))
print("\nDemo completed successfully! Everything runs end-to-end.")
