from typing import Dict, List, Optional

import httpx


class ApplePodcastsAPI:
    def __init__(self, country: str = "us"):
        self.country = country
        self.base_url = "https://rss.marketingtools.apple.com/api/v2"
        self.lookup_url = "https://itunes.apple.com/lookup"

    def get_top_podcast_ids(self, limit: int = 50) -> List[str]:
        """Fetch top podcast IDs from Apple Charts."""
        url = f"{self.base_url}/{self.country}/podcasts/top/{limit}/podcasts.json"
        response = httpx.get(url, follow_redirects=True)
        response.raise_for_status()
        data = response.json()
        return [result["id"] for result in data["feed"]["results"]]

    def get_podcast_metadata(self, podcast_id: str) -> Optional[Dict]:
        """Fetch detailed metadata for a podcast, including its RSS feed URL."""
        params = {"id": podcast_id}
        response = httpx.get(self.lookup_url, params=params, follow_redirects=True)
        response.raise_for_status()
        data = response.json()
        if data["resultCount"] > 0:
            return data["results"][0]
        return None

    def get_top_podcasts_with_rss(self, limit: int = 50) -> List[Dict]:
        """Convenience method to get top podcasts with their RSS URLs."""
        ids = self.get_top_podcast_ids(limit)
        podcasts = []
        for pid in ids:
            meta = self.get_podcast_metadata(pid)
            if meta and meta.get("feedUrl"):
                podcasts.append(
                    {
                        "id": pid,
                        "name": meta.get("collectionName"),
                        "rss_url": meta.get("feedUrl"),
                        "artist": meta.get("artistName"),
                    }
                )
        return podcasts
