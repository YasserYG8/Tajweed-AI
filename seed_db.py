import os
import json
import sys

# Setup imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.server.database import init_db, save_pipeline_output

def seed():
    # Make sure DB is initialized
    init_db()
    
    seeds = [
        {
            "json_path": "data/output/my_voice_alignment_fatihah.json",
            "audio_filename": "my_voice.wav"
        },
        {
            "json_path": "data/output/al_kawthar_alignment.json",
            "audio_filename": "Surat-Al-Kauthar-Mishary-Rashed-Alafasy.wav"
        },
        {
            "json_path": "data/output/test_voice_alignment.json",
            "audio_filename": "test_voice.wav"
        },
        {
            "json_path": "data/output/ikhlas_alignment.json",
            "audio_filename": "surah_ikhlas.wav"
        },
        {
            "json_path": "data/output/falaq_alignment.json",
            "audio_filename": "surah_falaq.wav"
        },
        {
            "json_path": "data/output/falaq_2_alignment.json",
            "audio_filename": "surah_falaq_2.wav"
        }
    ]
    
    for s in seeds:
        path = s["json_path"]
        audio = s["audio_filename"]
        
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue
            
        print(f"Loading alignment data from: {path}...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        print(f"Seeding '{audio}' into SQLite database...")
        save_pipeline_output(audio, data)
        
    print("Database seeding process completed!")

if __name__ == "__main__":
    seed()
