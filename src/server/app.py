import os
import sys
import shutil
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure stdout supports UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import database controllers and core pipeline
try:
    from src.server.database import init_db, save_pipeline_output, get_recitation, list_recitations, verify_recitation
    from src.pipeline import run_pipeline
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.server.database import init_db, save_pipeline_output, get_recitation, list_recitations, verify_recitation
    from src.pipeline import run_pipeline

# Initialize Fast API
app = FastAPI(title="Tajweed AI Annotation Server")

# Ensure directories exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/output", exist_ok=True)

# Initialize Database tables
init_db()

# Mount processed audio directory to serve raw WAVs to wavesurfer.js
app.mount("/audio_files", StaticFiles(directory="data/raw"), name="audio_files")


# Pydantic Schemas for validation
class AlignmentItem(BaseModel):
    word: str
    start: float = None
    end: float = None
    status: str

class ErrorItem(BaseModel):
    type: str
    word: str = None
    expected: str = None
    detected: str = None
    timestamp_start: float = None
    timestamp_end: float = None
    expected_index: int = None

class VerificationPayload(BaseModel):
    teacher_name: str
    alignment: List[AlignmentItem]
    errors: List[ErrorItem]


@app.get("/api/recitations")
def get_recitations_list():
    """Lists all processed audio files and metadata."""
    try:
        return list_recitations()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recitation/{audio_id}")
def get_recitation_detail(audio_id: str):
    """Retrieves full alignment and error logs for a specific recitation."""
    data = get_recitation(audio_id)
    if not data:
        raise HTTPException(status_code=404, detail="Recitation not found")
    return data


@app.post("/api/upload")
async def upload_audio_for_alignment(
    audio: UploadFile = File(...),
    text: str = Form(...),
    surah: int = Form(1),
    ayah: int = Form(1),
    reciter_id: str = Form("student_01"),
    reciter_type: str = Form("student"),
    use_ml: bool = Form(False),
    force_align: bool = Form(False),
    use_whisper: bool = Form(False)
):
    """
    Uploads an audio file, saves it to data/raw, executes the alignment pipeline,
    and inserts the parsed results into SQLite database.
    """
    # 1. Save uploaded file to raw data
    raw_path = os.path.join("data", "raw", audio.filename)
    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
        
    output_json = os.path.join("data", "output", f"{audio.filename}.json")
    
    # 2. Run Pipeline
    try:
        # Run pipeline generates clean WAV and outputs JSON schema
        pipeline_result = run_pipeline(
            audio_path=raw_path,
            ground_truth_text=text,
            output_json_path=output_json,
            surah=surah,
            ayah=ayah,
            reciter_id=reciter_id,
            reciter_type=reciter_type,
            use_ml=use_ml,
            force_align=force_align,
            use_whisper=use_whisper
        )
        
        # 3. Save to database
        save_pipeline_output(audio.filename, pipeline_result)
        
        return pipeline_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


@app.post("/api/verify/{audio_id}")
def submit_teacher_verification(audio_id: str, payload: VerificationPayload):
    """Saves teacher-corrected alignment logs and flags recitation as verified."""
    try:
        # Convert Pydantic list models to dict
        alignments = [item.dict() for item in payload.alignment]
        errors = [item.dict() for item in payload.errors]
        
        verify_recitation(
            audio_id=audio_id,
            teacher_name=payload.teacher_name,
            alignments=alignments,
            errors=errors
        )
        return {"status": "success", "message": f"Verified recitation: {audio_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def get_teacher_dashboard():
    """Serves the interactive HIL Teacher Dashboard (rendered statically)."""
    # We will serve a premium looking single page HTML/JS app directly from here
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Tajweed AI — Teacher Annotation Dashboard</title>
        <!-- Google Fonts -->
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
        <!-- Wavesurfer.js -->
        <script src="https://unpkg.com/wavesurfer.js@7"></script>
        <style>
            :root {
                --bg-primary: #0f172a;
                --bg-secondary: #1e293b;
                --accent-primary: #10b981;
                --accent-secondary: #059669;
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --border-color: #334155;
                --danger: #ef4444;
            }
            body {
                margin: 0;
                font-family: 'Inter', sans-serif;
                background-color: var(--bg-primary);
                color: var(--text-main);
                display: flex;
                height: 100vh;
                overflow: hidden;
            }
            /* Sidebar Layout */
            #sidebar {
                width: 320px;
                background-color: var(--bg-secondary);
                border-right: 1px solid var(--border-color);
                display: flex;
                flex-direction: column;
                padding: 20px;
            }
            h1 {
                font-family: 'Outfit', sans-serif;
                font-size: 24px;
                margin-top: 0;
                color: var(--accent-primary);
            }
            .list-title {
                font-size: 14px;
                text-transform: uppercase;
                color: var(--text-muted);
                margin: 20px 0 10px 0;
                font-weight: 600;
            }
            #audio-list {
                list-style: none;
                padding: 0;
                margin: 0;
                overflow-y: auto;
                flex-grow: 1;
            }
            #audio-list li {
                padding: 12px;
                border-radius: 8px;
                background-color: #0f172a50;
                border: 1px solid var(--border-color);
                margin-bottom: 10px;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            #audio-list li:hover {
                border-color: var(--accent-primary);
                background-color: #1e293b;
            }
            #audio-list li.active {
                background-color: #10b98120;
                border-color: var(--accent-primary);
            }
            .verified-badge {
                float: right;
                font-size: 10px;
                background-color: var(--accent-primary);
                color: var(--bg-primary);
                padding: 2px 6px;
                border-radius: 10px;
                font-weight: bold;
            }
            /* Main Content Layout */
            #main {
                flex-grow: 1;
                display: flex;
                flex-direction: column;
                padding: 30px;
                overflow-y: auto;
            }
            #waveform-container {
                background-color: var(--bg-secondary);
                border-radius: 12px;
                border: 1px solid var(--border-color);
                padding: 20px;
                margin-bottom: 25px;
                position: relative;
            }
            #waveform {
                width: 100%;
                height: 120px;
            }
            .controls {
                margin-top: 15px;
                display: flex;
                gap: 15px;
            }
            button {
                background-color: var(--accent-primary);
                color: var(--bg-primary);
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.2s;
            }
            button:hover {
                background-color: var(--accent-secondary);
            }
            button.secondary {
                background-color: transparent;
                border: 1px solid var(--border-color);
                color: var(--text-main);
            }
            button.secondary:hover {
                background-color: #ffffff10;
            }
            /* Quran Transcript display */
            #transcript-container {
                background-color: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 30px;
                direction: rtl;
                font-family: 'Inter', sans-serif;
                font-size: 28px;
                line-height: 1.8;
                text-align: center;
                margin-bottom: 25px;
            }
            .quran-word {
                display: inline-block;
                margin: 0 8px;
                padding: 4px 8px;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.2s ease;
                border: 1px solid transparent;
            }
            .quran-word:hover {
                background-color: #ffffff10;
                border-color: var(--text-muted);
            }
            .quran-word.correct {
                color: var(--text-main);
            }
            .quran-word.missing {
                color: var(--text-muted);
                text-decoration: line-through;
                border-color: var(--border-color);
            }
            .quran-word.substitution {
                color: var(--danger);
                border-bottom: 2px dashed var(--danger);
            }
            .quran-word.active-playing {
                background-color: #10b98140;
                border-color: var(--accent-primary);
                transform: scale(1.05);
            }
            /* Error Panel */
            #error-panel {
                background-color: var(--bg-secondary);
                border-radius: 12px;
                border: 1px solid var(--border-color);
                padding: 20px;
            }
            .error-card {
                background-color: #ef444415;
                border: 1px solid #ef444430;
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .error-card.correct-check {
                border-color: var(--accent-primary);
                background-color: #10b98115;
            }
            #verification-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            .ayah-marker {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 34px;
                height: 34px;
                border: 2px solid var(--accent-primary);
                border-radius: 50%;
                font-size: 16px;
                font-weight: bold;
                color: var(--accent-primary);
                margin: 0 15px;
                vertical-align: middle;
                font-family: 'Outfit', sans-serif;
                background-color: #10b98110;
            }
        </style>
    </head>
    <body>
        <div id="sidebar">
            <h1>Tajweed AI</h1>
            <span style="font-size:12px; color:var(--text-muted)">Teacher Verification Panel</span>
            
            <div class="list-title">Recitations</div>
            <ul id="audio-list">
                <!-- Dynamically loaded -->
            </ul>
        </div>
        
        <div id="main">
            <div id="verification-header">
                <h2 style="margin:0; font-family:'Outfit'">Verification Workspace</h2>
                <div style="display:flex; gap:10px;">
                    <input type="text" id="teacher-name" placeholder="Teacher Name" style="padding:10px; border-radius:6px; border:1px solid var(--border-color); background:var(--bg-secondary); color:#fff;">
                    <button onclick="commitVerification()">Verify & Save</button>
                </div>
            </div>
            
            <div id="waveform-container">
                <div id="waveform"></div>
                <div class="controls">
                    <button onclick="wavesurfer.playPause()">Play / Pause</button>
                    <span id="time-display" style="align-self:center; color:var(--text-muted)">00:00 / 00:00</span>
                </div>
            </div>
            
            <div id="transcript-container">
                <!-- Quranic words load here -->
            </div>
            
            <div id="error-panel">
                <h3 style="margin-top:0; font-family:'Outfit'">Detected Issues</h3>
                <div id="error-list">
                    <!-- Error list dynamic -->
                </div>
            </div>
        </div>

        <script>
            let currentAudioId = null;
            let activeRecord = null;
            let wavesurfer = null;

            // Load list of recitations on boot
            async function fetchRecitations() {
                const response = await fetch('/api/recitations');
                const list = await response.json();
                const listEl = document.getElementById('audio-list');
                listEl.innerHTML = '';
                
                list.forEach(item => {
                    const li = document.createElement('li');
                    li.onclick = () => loadRecitation(item.audio_id);
                    if (item.audio_id === currentAudioId) li.className = 'active';
                    
                    const badge = item.teacher_verified ? '<span class="verified-badge">Verified</span>' : '';
                    li.innerHTML = `${item.audio_id} <br/><span style="font-size:11px;color:var(--text-muted)">Surah ${item.surah} Ayah ${item.ayah}</span> ${badge}`;
                    listEl.appendChild(li);
                });
            }

            // Load specific recitation
            async function loadRecitation(audioId) {
                currentAudioId = audioId;
                
                // Active highlight in list
                document.querySelectorAll('#audio-list li').forEach(li => li.classList.remove('active'));
                
                const response = await fetch(`/api/recitation/${audioId}`);
                activeRecord = await response.json();
                
                renderTranscript();
                renderErrors();
                setupAudio(audioId);
                fetchRecitations(); // Refresh sidebar list selection class
            }

            function cleanWordForCompare(str) {
                if (!str) return "";
                return str.replace(/[\u064B-\u0652\u0670]/g, "") // remove Tashkeel
                          .replace(/[أإآ]/g, "ا")
                          .replace(/ى/g, "ي")
                          .replace(/ة/g, "ه")
                          .replace(/\s+/g, "").trim();
            }

            // Render Transcript words
            function renderTranscript() {
                const container = document.getElementById('transcript-container');
                container.innerHTML = '';
                
                const QURAN_VERSES = {
                    1: ["الرَّحِيمِ", "الْعَالَمِينَ", "الرَّحِيمِ", "الدِّينِ", "نَسْتَعِينُ", "الْمُسْتَقِيمَ", "الضَّالِّينَ"],
                    108: ["الْكَوْثَرَ", "وَانْحَرْ", "الْأَبْتَرُ"],
                    112: ["الرَّحِيمِ", "أَحَدٌ", "الصَّمَدُ", "يُولَدْ", "أَحَدٌ"],
                    113: ["الرَّحِيمِ", "الْفَلَقِ", "خَلَقَ", "وَقَبَ", "الْعُقَدِ", "حَسَدَ"]
                };
                
                const surahNum = activeRecord.surah;
                const verseEnds = QURAN_VERSES[surahNum] || [];
                let currentAyahIndex = 0;
                
                activeRecord.alignment.forEach((word, idx) => {
                    const span = document.createElement('span');
                    span.className = `quran-word ${word.status}`;
                    span.innerText = word.word;
                    
                    // Click to hear segment with 150ms start padding and 200ms end padding
                    if (word.start !== null && word.end !== null) {
                        span.onclick = () => {
                            const padStart = Math.max(0, word.start - 0.15);
                            const padEnd = word.end + 0.20;
                            wavesurfer.setTime(padStart);
                            wavesurfer.play();
                            setTimeout(() => {
                                wavesurfer.pause();
                            }, (padEnd - padStart) * 1000);
                        };
                    }
                    
                    span.setAttribute('data-index', idx);
                    container.appendChild(span);
                    
                    // Check if this word marks the end of an Ayah in the Quran text
                    if (currentAyahIndex < verseEnds.length) {
                        const wordClean = cleanWordForCompare(word.word);
                        const endWordClean = cleanWordForCompare(verseEnds[currentAyahIndex]);
                        
                        if (wordClean === endWordClean) {
                            const marker = document.createElement('span');
                            marker.className = 'ayah-marker';
                            marker.innerText = currentAyahIndex + 1;
                            container.appendChild(marker);
                            currentAyahIndex++;
                        }
                    }
                });
                
                // Fallback: If no markers were drawn (e.g. unknown Surah), draw a single one at the end
                if (currentAyahIndex === 0 && activeRecord.ayah) {
                    const marker = document.createElement('span');
                    marker.className = 'ayah-marker';
                    marker.innerText = activeRecord.ayah;
                    container.appendChild(marker);
                }
            }

            // Render Error list
            function renderErrors() {
                const container = document.getElementById('error-list');
                container.innerHTML = '';
                
                if (activeRecord.errors.length === 0) {
                    container.innerHTML = '<div style="color:var(--text-muted)">No errors detected in this recitation.</div>';
                    return;
                }
                
                activeRecord.errors.forEach((err, idx) => {
                    const div = document.createElement('div');
                    div.className = 'error-card';
                    
                    let text = '';
                    if (err.type === 'missing_word') {
                        text = `Omission: Word "${err.word}" was skipped.`;
                    } else if (err.type === 'wrong_word') {
                        text = `Substitution: Expected "${err.expected}", but spoke "${err.detected}".`;
                    } else if (err.type === 'extra_word') {
                        text = `Extra Word: Inserted "${err.word}" at ${err.timestamp_start}s.`;
                    }
                    
                    div.innerHTML = `
                        <span>${text}</span>
                        <div>
                            <button onclick="approveError(${idx})" style="background:#10b981; color:#fff; font-size:12px; padding:5px 10px;">Keep</button>
                            <button onclick="dismissError(${idx})" style="background:#ef4444; color:#fff; font-size:12px; padding:5px 10px;">Dismiss</button>
                        </div>
                    `;
                    container.appendChild(div);
                });
            }

            // Audio Player configuration
            function setupAudio(audioId) {
                if (wavesurfer) {
                    wavesurfer.destroy();
                }
                
                wavesurfer = WaveSurfer.create({
                    container: '#waveform',
                    waveColor: '#475569',
                    progressColor: '#10b981',
                    cursorColor: '#38bdf8',
                    url: `/audio_files/processed_${audioId}`
                });

                wavesurfer.on('timeupdate', (currentTime) => {
                    document.getElementById('time-display').innerText = 
                        formatTime(currentTime) + ' / ' + formatTime(wavesurfer.getDuration());
                    highlightActiveWord(currentTime);
                });
            }

            function highlightActiveWord(currentTime) {
                document.querySelectorAll('.quran-word').forEach(el => el.classList.remove('active-playing'));
                
                activeRecord.alignment.forEach((word, idx) => {
                    if (word.start !== null && currentTime >= word.start && currentTime <= word.end) {
                        const span = document.querySelector(`.quran-word[data-index="${idx}"]`);
                        if (span) span.classList.add('active-playing');
                    }
                });
            }

            function formatTime(seconds) {
                const mins = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
            }

            // Teacher interactive edits
            function dismissError(index) {
                const err = activeRecord.errors[index];
                
                // If wrong word was dismissed, revert alignment word status
                if (err.type === 'wrong_word') {
                    // Find all alignment entries mapped to this expected word and reset
                    activeRecord.alignment.forEach(align => {
                        if (align.word === err.expected || err.expected.includes(align.word)) {
                            align.status = 'correct';
                            align.start = err.timestamp_start;
                            align.end = err.timestamp_end;
                        }
                    });
                } else if (err.type === 'missing_word') {
                    // Set status to correct
                    const align = activeRecord.alignment[err.expected_index];
                    if (align) align.status = 'correct';
                }
                
                activeRecord.errors.splice(index, 1);
                renderTranscript();
                renderErrors();
            }

            function approveError(index) {
                // Keep error (visual confirmation checkmark style)
                const card = document.querySelectorAll('.error-card')[index];
                if (card) card.classList.add('correct-check');
            }

            // Save modifications back to DB
            async function commitVerification() {
                const name = document.getElementById('teacher-name').value;
                if (!name) {
                    alert('Please enter a Teacher Name to sign the verification.');
                    return;
                }
                
                const response = await fetch(`/api/verify/${currentAudioId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        teacher_name: name,
                        alignment: activeRecord.alignment,
                        errors: activeRecord.errors
                    })
                });
                
                const res = await response.json();
                if (res.status === 'success') {
                    alert('Recitation successfully verified and saved!');
                    fetchRecitations();
                } else {
                    alert('Save failed: ' + res.detail);
                }
            }

            // Boot load
            fetchRecitations();
        </script>
    </body>
    </html>
    """
    return html_content
