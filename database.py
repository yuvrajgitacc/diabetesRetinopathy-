"""
OptiGemma — Database Layer (SQLite)
Stores patient records and scan history permanently.
"""
import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'optigemma.db')


def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Initialize the database tables."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT DEFAULT '',
            diabetes_duration INTEGER,
            sugar_level REAL,
            hba1c REAL,
            notes TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS scans (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            stage INTEGER NOT NULL,
            stage_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT DEFAULT '',
            color TEXT DEFAULT '',
            all_probabilities TEXT DEFAULT '{}',
            model_used TEXT DEFAULT '',
            heatmap_analysis TEXT DEFAULT '{}',
            vessel_stats TEXT DEFAULT '{}',
            report TEXT DEFAULT '{}',
            image_original TEXT DEFAULT '',
            image_heatmap TEXT DEFAULT '',
            image_vessels TEXT DEFAULT '',
            processing_time REAL DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        );

        CREATE INDEX IF NOT EXISTS idx_scans_patient ON scans(patient_id);
        CREATE INDEX IF NOT EXISTS idx_scans_created ON scans(created_at);
    """)
    conn.commit()
    conn.close()
    print("[DB] Database initialized at", DB_PATH)


def generate_patient_id():
    """Generate a unique patient ID like P-0001."""
    conn = get_db()
    row = conn.execute("SELECT COUNT(*) as cnt FROM patients").fetchone()
    conn.close()
    return f"P-{row['cnt'] + 1:04d}"


# === Patient CRUD ===

def create_patient(name, age=None, gender='', diabetes_duration=None,
                   sugar_level=None, hba1c=None, notes=''):
    """Create a new patient record."""
    pid = generate_patient_id()
    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO patients (id, name, age, gender, diabetes_duration,
               sugar_level, hba1c, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (pid, name.strip(), age, gender, diabetes_duration, sugar_level, hba1c, notes)
        )
        conn.commit()
        patient = conn.execute("SELECT * FROM patients WHERE id = ?", (pid,)).fetchone()
        return dict(patient)
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def get_patient(patient_id):
    """Get a single patient by ID."""
    conn = get_db()
    row = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    conn.close()
    if row:
        patient = dict(row)
        patient['scans'] = get_patient_scans(patient_id)
        return patient
    return None


def get_all_patients(search=''):
    """Get all patients, optionally filtered by search."""
    conn = get_db()
    if search:
        rows = conn.execute(
            "SELECT * FROM patients WHERE name LIKE ? OR id LIKE ? ORDER BY created_at DESC",
            (f'%{search}%', f'%{search}%')
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
    conn.close()

    patients = []
    for row in rows:
        p = dict(row)
        # Get latest scan info
        scan_conn = get_db()
        latest = scan_conn.execute(
            "SELECT stage, stage_name, confidence, created_at FROM scans WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1",
            (p['id'],)
        ).fetchone()
        scan_count = scan_conn.execute(
            "SELECT COUNT(*) as cnt FROM scans WHERE patient_id = ?", (p['id'],)
        ).fetchone()
        scan_conn.close()

        p['latest_scan'] = dict(latest) if latest else None
        p['scan_count'] = scan_count['cnt'] if scan_count else 0
        patients.append(p)

    return patients


def update_patient(patient_id, **kwargs):
    """Update patient fields."""
    conn = get_db()
    allowed = ['name', 'age', 'gender', 'diabetes_duration', 'sugar_level', 'hba1c', 'notes']
    updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if not updates:
        conn.close()
        return get_patient(patient_id)

    set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
    values = list(updates.values()) + [patient_id]
    conn.execute(f"UPDATE patients SET {set_clause}, updated_at = datetime('now') WHERE id = ?", values)
    conn.commit()
    conn.close()
    return get_patient(patient_id)


def delete_patient(patient_id):
    """Delete a patient and all their scans."""
    conn = get_db()
    conn.execute("DELETE FROM scans WHERE patient_id = ?", (patient_id,))
    conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    conn.commit()
    conn.close()


# === Scan CRUD ===

def save_scan(scan_id, patient_id, detection_result, heatmap_analysis,
              vessel_stats, report, image_paths, processing_time):
    """Save a completed scan to the database."""
    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO scans (id, patient_id, stage, stage_name, confidence,
               severity, color, all_probabilities, model_used, heatmap_analysis,
               vessel_stats, report, image_original, image_heatmap, image_vessels,
               processing_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                scan_id, patient_id,
                detection_result.get('stage', 0),
                detection_result.get('stage_name', 'Unknown'),
                detection_result.get('confidence', 0),
                detection_result.get('severity', ''),
                detection_result.get('color', ''),
                json.dumps(detection_result.get('all_probabilities', {})),
                detection_result.get('_model', 'unknown'),
                json.dumps(heatmap_analysis),
                json.dumps(vessel_stats),
                json.dumps(report),
                image_paths.get('original', ''),
                image_paths.get('heatmap', ''),
                image_paths.get('vessels', ''),
                processing_time
            )
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def get_patient_scans(patient_id):
    """Get all scans for a patient, newest first."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM scans WHERE patient_id = ? ORDER BY created_at DESC",
        (patient_id,)
    ).fetchall()
    conn.close()

    scans = []
    for row in rows:
        s = dict(row)
        # Parse JSON fields
        for field in ['all_probabilities', 'heatmap_analysis', 'vessel_stats', 'report']:
            try:
                s[field] = json.loads(s[field]) if s[field] else {}
            except (json.JSONDecodeError, TypeError):
                s[field] = {}
        scans.append(s)
    return scans


def get_scan(scan_id):
    """Get a single scan by ID."""
    conn = get_db()
    row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    conn.close()
    if row:
        s = dict(row)
        for field in ['all_probabilities', 'heatmap_analysis', 'vessel_stats', 'report']:
            try:
                s[field] = json.loads(s[field]) if s[field] else {}
            except (json.JSONDecodeError, TypeError):
                s[field] = {}
        return s
    return None


# === Dashboard Stats ===

def get_dashboard_stats():
    """Get overview statistics for the dashboard."""
    conn = get_db()
    total_patients = conn.execute("SELECT COUNT(*) as cnt FROM patients").fetchone()['cnt']
    total_scans = conn.execute("SELECT COUNT(*) as cnt FROM scans").fetchone()['cnt']

    # Stage distribution
    stage_dist = {}
    rows = conn.execute(
        "SELECT stage, stage_name, COUNT(*) as cnt FROM scans GROUP BY stage ORDER BY stage"
    ).fetchall()
    for row in rows:
        stage_dist[row['stage']] = {'name': row['stage_name'], 'count': row['cnt']}

    # Recent scans
    recent = conn.execute("""
        SELECT s.*, p.name as patient_name
        FROM scans s JOIN patients p ON s.patient_id = p.id
        ORDER BY s.created_at DESC LIMIT 5
    """).fetchall()

    recent_scans = []
    for row in recent:
        r = dict(row)
        try:
            r['report'] = json.loads(r['report']) if r['report'] else {}
        except:
            r['report'] = {}
        recent_scans.append(r)

    conn.close()

    return {
        'total_patients': total_patients,
        'total_scans': total_scans,
        'stage_distribution': stage_dist,
        'recent_scans': recent_scans,
    }


# Initialize on import
init_db()
