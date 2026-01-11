import sqlite3
from pathlib import Path

class PodcastDatabase:
    def __init__(self, db_path: str = "podcasts.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS downloaded_episodes (
                    id TEXT PRIMARY KEY,
                    podcast_id TEXT,
                    title TEXT,
                    pub_date TEXT,
                    url TEXT,
                    file_path TEXT
                )
            """)
            conn.commit()

    def is_downloaded(self, episode_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM downloaded_episodes WHERE id = ?", (episode_id,))
            return cursor.fetchone() is not None

    def mark_as_downloaded(self, episode_id: str, podcast_id: str, title: str, pub_date: str, url: str, file_path: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO downloaded_episodes (id, podcast_id, title, pub_date, url, file_path)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (episode_id, podcast_id, title, pub_date, url, file_path))
            conn.commit()
