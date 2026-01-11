from pathlib import Path

import feedparser
import httpx
from tqdm import tqdm

from .database import PodcastDatabase


class PodcastDownloader:
    def __init__(self, db: PodcastDatabase, download_dir: str = "downloads"):
        self.db = db
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download_podcast_episodes(
        self, podcast_id: str, rss_url: str, podcast_name: str, max_episodes: int = None
    ):
        """Parse RSS and download episodes in descending order of publication date."""
        print(f"Fetching episodes for: {podcast_name}")
        feed = feedparser.parse(rss_url)

        # Entries are usually already in descending order in RSS feeds, but let's be sure.
        entries = sorted(
            feed.entries, key=lambda x: x.get("published_parsed", (0,)), reverse=True
        )

        if max_episodes:
            entries = entries[:max_episodes]

        for entry in tqdm(entries, desc=f"Downloading {podcast_name}", unit="ep"):
            episode_id = entry.get("id") or entry.get("link")
            if not episode_id:
                continue

            if self.db.is_downloaded(episode_id):
                continue

            # Find audio enclosure
            audio_url = None
            for link in entry.get("links", []):
                if "audio" in link.get("type", ""):
                    audio_url = link.get("href")
                    break

            if not audio_url and "enclosures" in entry:
                for enc in entry.enclosures:
                    if "audio" in enc.get("type", ""):
                        audio_url = enc.get("href")
                        break

            if not audio_url:
                continue

            # Download the file
            filename = self._get_safe_filename(podcast_name, entry.title, audio_url)
            file_path = self.download_dir / filename

            try:
                self._download_file(audio_url, file_path)

                # Mark as downloaded in DB
                pub_date = entry.get("published", "")
                self.db.mark_as_downloaded(
                    episode_id=episode_id,
                    podcast_id=podcast_id,
                    title=entry.title,
                    pub_date=pub_date,
                    url=audio_url,
                    file_path=str(file_path),
                )
            except Exception as e:
                print(f"Failed to download {entry.title}: {e}")

    def _get_safe_filename(self, podcast_name: str, title: str, url: str) -> str:
        # Simple cleanup for filenames
        safe_name = "".join(
            c for c in f"{podcast_name}_{title}" if c.isalnum() or c in (" ", "_", "-")
        ).strip()
        safe_name = safe_name.replace(" ", "_")

        # Try to get extension from URL
        ext = ".mp3"
        if ".wav" in url.lower():
            ext = ".wav"
        elif ".m4a" in url.lower():
            ext = ".m4a"

        return f"{safe_name[:150]}{ext}"

    def _download_file(self, url: str, path: Path):
        with httpx.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            with open(path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
