# 🛠️ Tajweed AI — Architecture Improvement Plan

Fixes for false-negative word detection (correct words showing as `missing` / `wrong_word`) and a path toward higher accuracy.

---

## 🎯 Root Causes Identified

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1 | Static amplitude noise gate (1.5% of global peak) | `src/preprocessing/audio.py` | Silences legitimate quiet segments → CTC sees silence → word marked `missing` |
| 2 | Default ASR model is general MSA Arabic, not Quran-tuned | `src/alignment/aligner.py` | Tajweed acoustics (madd, ghunnah, qalqalah) produce low-confidence logits → CTC blank dominates |
| 3 | Flat similarity threshold (0.60) regardless of word length | `src/alignment/error_detector.py` | Short words (لَمْ, مِن, هُم) fail disproportionately on single-character ASR slips |
| 4 | Whisper toggle implies interchangeability with CTC alignment | `src/alignment/aligner.py` | Whisper is autoregressive seq2seq — doesn't produce clean per-frame CTC logits for the Viterbi trellis |

---

## ✅ Solution 1: Replace the Static Noise Gate with VAD

**Problem:** Gating relative to a single global peak means one loud moment (qalqalah, mic bump) inflates the reference, causing quieter-but-correct speech to fall below threshold and get zeroed out.

**Fix — pick one:**

- **Best:** Silero VAD — per-frame speech probability, robust to volume swings
- **Lightweight alternative:** WebRTC VAD
- **Minimum viable fix:** Replace global-peak gating with a rolling RMS/energy window (adaptive, not fixed-reference)

```python
# Conceptual replacement
import torch
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
(get_speech_timestamps, _, read_audio, *_) = utils

speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=16000)
# Only silence frames OUTSIDE detected speech regions
```

---

## ✅ Solution 2: Swap to a Quran-Tuned Acoustic Model

**Problem:** `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` was trained on conversational/MSA Arabic. Quranic recitation is acoustically distinct — deliberate pacing, elongated madd, nasalization, qalqalah bounce — patterns the model has never learned.

**Fix:**

- Default the CTC backbone to a wav2vec2 model fine-tuned on Quranic/tajweed audio (check tarteel-ai's HF org for a wav2vec2 variant, and community tajweed-tagged fine-tunes)
- Keep the general Arabic model only as a fallback path, not primary

---

## ✅ Solution 3: Length-Normalize the Similarity Threshold

**Problem:** A flat 0.60 cutoff penalizes short words far more than long ones for the same absolute edit distance.

**Fix:**

```python
def get_dynamic_threshold(word: str) -> float:
    length = len(word)
    # Softer bar for short words, floor at 0.5
    return max(0.5, 1 - (2 / length))
```

**Bonus improvement:** Pair the phonetic/character similarity score with the ASR's own confidence (average log-prob across that word's frames). A word that looks different in text but was transcribed with high confidence is a stronger "correct" signal than string similarity alone.

---

## ✅ Solution 4: Separate the Whisper and CTC Alignment Paths

**Problem:** The diagram implies both `Wav2Vec2` and `WhisperClean` feed the same Viterbi trellis. Whisper doesn't emit clean per-frame CTC logits, so this either silently falls back to Wav2Vec2 or produces broken/null timestamps.

**Fix — two distinct lanes:**

| Path | Transcription | Alignment method |
|------|---------------|-------------------|
| **CTC (default)** | Wav2Vec2 (Quran fine-tuned) | Existing Viterbi trellis — keep as-is |
| **Whisper (optional)** | `tarteel-ai/whisper-base-ar-quran` | Needs a dedicated forced aligner (e.g. `ctc-forced-aligner`, or WhisperX-style cross-attention alignment) — **not** the same trellis code |

Don't let the `use_whisper` toggle imply interchangeable alignment logic downstream.

---

## ✅ Solution 5: Recovery Step Before Marking a Word "Missing"

**Problem:** Once a segment is gated to silence, there's no second chance for that word — it's permanently `missing`.

**Fix:** Before finalizing a `missing` classification, check if the aligned segment is anomalously short relative to expected phoneme count (e.g. under 80ms for a 4+ letter word). If so, re-run local alignment on that stretch using the **raw, ungated** audio before confirming the word is actually missing.

```python
MIN_MS_PER_CHAR = 20  # tune against your eval set

def needs_recovery_check(word: str, start, end) -> bool:
    if start is None or end is None:
        return True
    duration_ms = (end - start) * 1000
    expected_min = len(word) * MIN_MS_PER_CHAR
    return duration_ms < expected_min
```

This alone should catch most noise-gate false negatives even before VAD is fully tuned.

---

## 📋 Priority Order

1. **Silero VAD swap** — cheapest fix, likely biggest immediate impact
2. **Quran-tuned acoustic model** as default CTC backbone
3. **Length-normalized similarity threshold**
4. **Separate Whisper/CTC alignment code paths**
5. **Build a labeled eval set** — 20–30 clips of known-correct recitation, so thresholds are tuned against real numbers instead of guesswork

---

## 📊 Realistic Accuracy Target

With home-mic audio and classical Arabic phonology, **95–97% is a realistic target** without significantly more DSP/model investment. Treat remaining edge cases — near-synonym confusion, ASR hallucination, very quiet ayah endings — as an acceptable manual-review tail rather than something to fully automate away toward 99%.

---

## 🧪 Suggested Eval Set Structure

To actually validate these fixes instead of eyeballing thresholds:

```
eval_set/
├── clips/
│   ├── correct_01.wav       # known-correct recitation
│   ├── correct_02.wav
│   ├── substitution_01.wav  # deliberate wrong word
│   ├── missing_01.wav       # deliberate omission
│   └── quiet_ending_01.wav  # soft/quiet articulation edge case
└── ground_truth.json        # expected word-by-word labels per clip
```

Run before/after each fix against this set and track:
- False negative rate (correct words marked wrong/missing)
- False positive rate (actual errors missed)
- Timestamp accuracy drift
