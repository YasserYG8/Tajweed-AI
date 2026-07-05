# Tajweed AI — Dataset & Alignment System

Tajweed AI is a machine learning data engine designed to automatically transform raw Quranic recitation audio and Uthmani ground-truth text into highly aligned, phoneme-level verified datasets. 

Instead of building a consumer app, this system serves as the core pipeline to ingest audio, map it with microsecond boundaries, analyze pronunciation errors, and output teacher-verified labels suitable for training downstream Tajweed neural networks.

---

## 🧠 System Architecture

The core pipeline utilizes a **Hybrid DSP + Data-Driven Evaluation Strategy**:
1.  **Data-Driven Models (AI)**: Segment raw audio and locate the exact timestamps of individual words and phonemes using WhisperX and the Montreal Forced Aligner (MFA).
2.  **DSP Rules (Mathematics)**: Evaluate specific physical Tajweed rules (Madd elongation duration, Ghunnah nasal resonance, Qalqalah transients) on the sliced segments.

For complete details, see:
*   [system_architecture.md](file:///C:/Users/Lenovo/Desktop/projects/Tajweed%20AI/system_architecture.md) — System Design & Schema Specs.

---

## 🗺️ Roadmap & Phases

The project is structured into three progressive engineering phases:

### 🚀 [Phase 1: Word-Level MVP](mvp_phase1_architecture.md)
*   **Goal**: Detect simple reading omissions, insertions, and substitutions.
*   **Engine**: Resampling/noise preprocessing, WhisperX alignment, Arabic character normalizer, and Levenshtein Diff error mapping.
*   *Details*: See [mvp_phase1_architecture.md](mvp_phase1_architecture.md)

### 🚀 [Phase 2: Phoneme Alignment & Human-in-the-Loop UI](phase2_architecture.md)
*   **Goal**: Enable letter-level alignments and build the teacher verification interface.
*   **Engine**: Classical Quran G2P (Grapheme-to-Phoneme) converter, Montreal Forced Aligner (MFA) integration, FastAPI endpoints, and wavesurfer.js-based interactive audio-waveform review dashboard.
*   *Details*: See [phase2_architecture.md](phase2_architecture.md)

### 🚀 [Phase 3: Tajweed Rule Engine & Advanced DSP](phase3_architecture.md)
*   **Goal**: Quantify Tajweed-specific rules on aligned phonemes.
*   **Engine**: Dynamic tempo calculation (speaking rate-aware Madd counts), Ghunnah spectral energy ratio bands, Qalqalah transient detectors, and DVC/Hugging Face dataset packagers.
*   *Details*: See [phase3_architecture.md](phase3_architecture.md)

---

## 📁 Suggested Directory Structure

```
tajweed-ai/
│
├── data/
│   ├── raw/                  # Ingested student & teacher audios
│   ├── processed/            # 16kHz mono standardized WAVs
│   └── output/               # Exported JSON schemas & datasets
│
├── src/
│   ├── preprocessing/        # Audio conversions & normalization
│   ├── alignment/            # WhisperX / MFA wrappers
│   ├── text_utils/           # Arabic diacritic removal & Quranic G2P
│   ├── rules/                # DSP Tajweed evaluation engine
│   └── server/               # FastAPI backend & database models
│
├── templates/                # Frontend UI source files (React/Vite)
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🛠️ Getting Started

### 1. Prerequisites
Ensure you have the following installed:
*   Python 3.10+
*   FFmpeg (required for audio conversions)

### 2. Setup Virtual Environment
```bash
python -m venv venv
# On Windows powershell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
