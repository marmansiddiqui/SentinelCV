import sqlite3

from datetime import datetime

from app.config import DATABASE_PATH


def initialize_database():

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            camera_id TEXT,

            event_type TEXT,

            object_type TEXT,

            track_id INTEGER,

            confidence REAL,

            timestamp TEXT,

            snapshot TEXT

        )
    """)

    connection.commit()
    connection.close()


def save_event(
    camera_id,
    event_type,
    object_type,
    track_id,
    confidence,
    snapshot
):

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO events (
            camera_id,
            event_type,
            object_type,
            track_id,
            confidence,
            timestamp,
            snapshot
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        camera_id,
        event_type,
        object_type,
        track_id,
        confidence,
        datetime.now().isoformat(timespec="seconds"),
        str(snapshot)
    ))

    connection.commit()
    connection.close()