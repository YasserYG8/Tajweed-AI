import os
import sqlite3
import json
from typing import List, Dict, Any, Optional

DB_PATH = "data/output/tajweed_dataset.db"

def get_db_connection():
    """Establishes connection to the SQLite database with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initializes the SQLite tables for recitations, alignments, and errors."""
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Table for recitations metadata
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recitations (
        audio_id TEXT PRIMARY KEY,
        surah INTEGER,
        ayah INTEGER,
        reciter_id TEXT,
        reciter_type TEXT,
        text TEXT,
        teacher_verified BOOLEAN DEFAULT 0,
        verified_by TEXT,
        verified_at TEXT
    )
    """)
    
    # 2. Table for word alignments (word sequence with timestamps)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        audio_id TEXT,
        word_index INTEGER,
        word TEXT,
        start REAL,
        end REAL,
        status TEXT,
        FOREIGN KEY(audio_id) REFERENCES recitations(audio_id) ON DELETE CASCADE
    )
    """)
    
    # 3. Table for detected errors
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS errors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        audio_id TEXT,
        error_type TEXT,
        word TEXT,
        expected TEXT,
        detected TEXT,
        timestamp_start REAL,
        timestamp_end REAL,
        expected_index INTEGER,
        FOREIGN KEY(audio_id) REFERENCES recitations(audio_id) ON DELETE CASCADE
    )
    """)
    
    conn.commit()
    conn.close()


def save_pipeline_output(audio_id: str, pipeline_data: Dict[str, Any]) -> None:
    """
    Saves or updates a processed pipeline record inside the SQLite database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Insert or update recitation metadata
    cursor.execute("""
    INSERT INTO recitations (audio_id, surah, ayah, reciter_id, reciter_type, text, teacher_verified)
    VALUES (?, ?, ?, ?, ?, ?, 0)
    ON CONFLICT(audio_id) DO UPDATE SET
        surah=excluded.surah,
        ayah=excluded.ayah,
        reciter_id=excluded.reciter_id,
        reciter_type=excluded.reciter_type,
        text=excluded.text,
        teacher_verified=0
    """, (
        audio_id,
        pipeline_data.get("surah", 1),
        pipeline_data.get("ayah", 1),
        pipeline_data.get("reciter_id", "student_01"),
        pipeline_data.get("reciter_type", "student"),
        pipeline_data.get("text", "")
    ))
    
    # Clear previous alignment/errors associated with this audio_id
    cursor.execute("DELETE FROM alignments WHERE audio_id = ?", (audio_id,))
    cursor.execute("DELETE FROM errors WHERE audio_id = ?", (audio_id,))
    
    # Insert alignments
    alignments = pipeline_data.get("alignment", [])
    for idx, align in enumerate(alignments):
        cursor.execute("""
        INSERT INTO alignments (audio_id, word_index, word, start, end, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            audio_id,
            idx,
            align.get("word", ""),
            align.get("start"),
            align.get("end"),
            align.get("status", "correct")
        ))
        
    # Insert errors
    errors = pipeline_data.get("errors", [])
    for err in errors:
        cursor.execute("""
        INSERT INTO errors (audio_id, error_type, word, expected, detected, timestamp_start, timestamp_end, expected_index)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            audio_id,
            err.get("type", ""),
            err.get("word"),
            err.get("expected"),
            err.get("detected"),
            err.get("timestamp_start"),
            err.get("timestamp_end"),
            err.get("expected_index")
        ))
        
    conn.commit()
    conn.close()


def get_recitation(audio_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a complete recitation, including its alignments and errors.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM recitations WHERE audio_id = ?", (audio_id,))
    recitation_row = cursor.fetchone()
    
    if not recitation_row:
        conn.close()
        return None
        
    recitation = dict(recitation_row)
    
    # Get alignments
    cursor.execute("SELECT word, start, end, status FROM alignments WHERE audio_id = ? ORDER BY word_index ASC", (audio_id,))
    recitation["alignment"] = [dict(row) for row in cursor.fetchall()]
    
    # Get errors
    cursor.execute("SELECT error_type as type, word, expected, detected, timestamp_start, timestamp_end, expected_index FROM errors WHERE audio_id = ?", (audio_id,))
    recitation["errors"] = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    # Convert SQL boolean (0/1) to Python bool
    recitation["teacher_verified"] = bool(recitation["teacher_verified"])
    
    return recitation


def list_recitations() -> List[Dict[str, Any]]:
    """Lists all recitations logged in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT audio_id, surah, ayah, reciter_id, teacher_verified FROM recitations")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # Convert booleans
    for r in rows:
        r["teacher_verified"] = bool(r["teacher_verified"])
    return rows


def verify_recitation(audio_id: str, teacher_name: str, alignments: List[Dict[str, Any]], errors: List[Dict[str, Any]]) -> None:
    """Updates a recitation as teacher-verified, modifying alignments and errors."""
    import datetime
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Update status
    cursor.execute("""
    UPDATE recitations 
    SET teacher_verified = 1, verified_by = ?, verified_at = ?
    WHERE audio_id = ?
    """, (teacher_name, datetime.datetime.utcnow().isoformat(), audio_id))
    
    # Re-save updated alignments and errors
    cursor.execute("DELETE FROM alignments WHERE audio_id = ?", (audio_id,))
    cursor.execute("DELETE FROM errors WHERE audio_id = ?", (audio_id,))
    
    for idx, align in enumerate(alignments):
        cursor.execute("""
        INSERT INTO alignments (audio_id, word_index, word, start, end, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            audio_id,
            idx,
            align.get("word", ""),
            align.get("start"),
            align.get("end"),
            align.get("status", "correct")
        ))
        
    for err in errors:
        cursor.execute("""
        INSERT INTO errors (audio_id, error_type, word, expected, detected, timestamp_start, timestamp_end, expected_index)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            audio_id,
            err.get("type", ""),
            err.get("word"),
            err.get("expected"),
            err.get("detected"),
            err.get("timestamp_start"),
            err.get("timestamp_end"),
            err.get("expected_index")
        ))
        
    conn.commit()
    conn.close()


if __name__ == "__main__":
    print("Initializing Database...")
    init_db()
    print("Database initialized successfully at:", os.path.abspath(DB_PATH))
