"""Anotaciones persistentes por experimento.

Una BD SQLite por experimento guarda tags de estado y notas de texto libre de
cada diseño. Módulo puro: solo necesita la ruta del experimento, no depende
de la configuración global ni de Dash.
"""
import contextlib
import os
import sqlite3

ANNOTATIONS_DB_NAME = '.binderlab_annotations.db'

DEFAULT_TAGS = [
    '⭐ candidato',
    '🧪 a clonar',
    '🔬 in vitro+',
    '❌ descartado',
]
# Tag reservado (cadena vacía) para la nota de texto libre, sin estado.
NOTE_TAG = ''


def _annotations_db_path(exp_dir):
    """Ruta del .db de anotaciones para un experimento dado, o None si no es válido."""
    if not exp_dir or not os.path.isdir(exp_dir):
        return None
    return os.path.join(exp_dir, ANNOTATIONS_DB_NAME)


def _annotations_connect(exp_dir, create=True):
    """Abre (y si hace falta crea) la BD de anotaciones del experimento."""
    db_path = _annotations_db_path(exp_dir)
    if not db_path or (not create and not os.path.exists(db_path)):
        return None
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                design_name TEXT NOT NULL,
                tag         TEXT NOT NULL DEFAULT '',
                note        TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP,
                PRIMARY KEY (design_name, tag)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tag ON annotations(tag)")
        conn.commit()
        return conn
    except sqlite3.Error as e:
        print(f"[BinderMonitor] Error abriendo anotaciones ({db_path}): {e}")
        return None


@contextlib.contextmanager
def _annotations_session(exp_dir, create=True):
    """Gestor de contexto que garantiza cierre y rollback si falla una escritura."""
    conn = _annotations_connect(exp_dir, create=create)
    if conn is None:
        yield None
        return
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_design_annotations(exp_dir, design_name):
    """Devuelve (tags, note) de un diseño concreto, o ([], '') si no hay datos."""
    if not design_name:
        return [], ''
    with _annotations_session(exp_dir, create=False) as conn:
        if conn is None:
            return [], ''
        try:
            rows = conn.execute(
                "SELECT tag, note FROM annotations WHERE design_name = ?",
                (design_name,),
            ).fetchall()
        except sqlite3.Error:
            rows = []
    tags, note = [], ''
    for tag, n in rows:
        if tag == NOTE_TAG:
            note = n or ''
        else:
            tags.append(tag)
    return tags, note


def set_design_tags(exp_dir, design_name, tags):
    """Reemplaza los tags de estado de un diseño preservando su nota libre."""
    if not design_name:
        return False
    clean_tags = list(dict.fromkeys(t for t in (tags or []) if t and t != NOTE_TAG))
    with _annotations_session(exp_dir, create=True) as conn:
        if conn is None:
            return False
        try:
            conn.execute(
                "DELETE FROM annotations WHERE design_name = ? AND tag != ?",
                (design_name, NOTE_TAG),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO annotations "
                "(design_name, tag, note, updated_at) "
                "VALUES (?, ?, NULL, CURRENT_TIMESTAMP)",
                [(design_name, t) for t in clean_tags],
            )
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"[BinderMonitor] Error guardando tags: {e}")
            conn.rollback()
            return False


def set_design_note(exp_dir, design_name, note):
    """Guarda (o borra si está vacía) la nota de texto libre de un diseño."""
    if not design_name:
        return False
    with _annotations_session(exp_dir, create=True) as conn:
        if conn is None:
            return False
        try:
            if note and note.strip():
                conn.execute(
                    "INSERT OR REPLACE INTO annotations "
                    "(design_name, tag, note, updated_at) "
                    "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                    (design_name, NOTE_TAG, note),
                )
            else:
                conn.execute(
                    "DELETE FROM annotations WHERE design_name = ? AND tag = ?",
                    (design_name, NOTE_TAG),
                )
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"[BinderMonitor] Error guardando nota: {e}")
            conn.rollback()
            return False


def get_all_annotations(exp_dir):
    """Devuelve {design_name: {'tags': [...], 'note': str}} del experimento."""
    with _annotations_session(exp_dir, create=False) as conn:
        if conn is None:
            return {}
        try:
            rows = conn.execute(
                "SELECT design_name, tag, note FROM annotations"
            ).fetchall()
        except sqlite3.Error:
            rows = []
    out = {}
    for design, tag, note in rows:
        entry = out.setdefault(design, {'tags': [], 'note': ''})
        if tag == NOTE_TAG:
            entry['note'] = note or ''
        else:
            entry['tags'].append(tag)
    return out
