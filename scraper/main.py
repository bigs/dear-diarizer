import argparse
import sys
from .apple_podcasts import ApplePodcastsAPI
from .database import PodcastDatabase
from .downloader import PodcastDownloader

def main():
    parser = argparse.ArgumentParser(description="Download top podcasts from Apple Podcasts.")
    parser.add_argument("-n", "--num-podcasts", type=int, default=10, help="Number of top podcasts to fetch (default: 10)")
    parser.add_argument("-e", "--episodes-per-podcast", type=int, default=1, help="Number of latest episodes to download per podcast (default: 1)")
    parser.add_argument("-d", "--download-dir", type=str, default="downloads", help="Directory to save downloads (default: downloads)")
    parser.add_argument("--db-path", type=str, default="podcasts.db", help="Path to the deduplication database (default: podcasts.db)")
    parser.add_argument("--country", type=str, default="us", help="Country code for Apple Charts (default: us)")

    args = parser.parse_args()

    api = ApplePodcastsAPI(country=args.country)
    db = PodcastDatabase(db_path=args.db_path)
    downloader = PodcastDownloader(db=db, download_dir=args.download_dir)

    print(f"Fetching top {args.num_podcasts} podcasts...")
    try:
        podcasts = api.get_top_podcasts_with_rss(limit=args.num_podcasts)
    except Exception as e:
        print(f"Error fetching top podcasts: {e}")
        sys.exit(1)

    for pod in podcasts:
        try:
            downloader.download_podcast_episodes(
                podcast_id=pod["id"],
                rss_url=pod["rss_url"],
                podcast_name=pod["name"],
                max_episodes=args.episodes_per_podcast
            )
        except Exception as e:
            print(f"Error downloading episodes for {pod['name']}: {e}")

    print("Done!")

if __name__ == "__main__":
    main()
