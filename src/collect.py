import os
import sys
import argparse

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

sys.stdout.reconfigure(encoding="utf-8")  # so Urdu/emoji don't crash PowerShell

load_dotenv()
API_KEY = os.getenv("AIzaSyCQWpbOag1QDlbD_q0Xf4xxg29on_PMY8E")
if not API_KEY:
    raise SystemExit("YOUTUBE_API_KEY not found in .env")

youtube = build("youtube", "v3", developerKey=API_KEY)

# brand -> (search query, text that must appear in the channel name)
BRANDS = {
    "Khaadi":    ("Khaadi collection ad",          "khaadi"),
    "Gul Ahmed": ("Gul Ahmed collection ad",       "gul ahmed"),
    "Sapphire":  ("Sapphire Pakistan lawn collection", "sapphire"),
}


def find_videos(brand, n_videos):
    """Search YouTube, keep only videos from the brand's own channel."""
    query, channel_key = BRANDS[brand]
    resp = youtube.search().list(
        part="snippet", q=query, type="video", maxResults=50
    ).execute()

    rows = []
    for item in resp.get("items", []):
        sn = item["snippet"]
        if channel_key in sn["channelTitle"].lower():
            rows.append({
                "brand": brand,
                "video_id": item["id"]["videoId"],
                "video_title": sn["title"],
                "channel": sn["channelTitle"],
                "video_published": sn["publishedAt"],
                "video_url": "https://www.youtube.com/watch?v=" + item["id"]["videoId"],
            })
    return rows[:n_videos]


def get_comments(video_id, max_comments):
    """Top-level public comments only. Author names are NOT stored."""
    out, token = [], None
    while len(out) < max_comments:
        try:
            resp = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(out)),
                pageToken=token,
                order="time",
                textFormat="plainText",
            ).execute()
        except HttpError as e:
            print(f"  skipped {video_id}: {e.reason if hasattr(e, 'reason') else e}")
            break
        for item in resp.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            out.append({
                "video_id": video_id,
                "comment_text": s["textOriginal"],
                "comment_published": s["publishedAt"],
                "like_count": s["likeCount"],
            })
        token = resp.get("nextPageToken")
        if not token:
            break
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", required=True, choices=BRANDS.keys())
    ap.add_argument("--videos", type=int, default=5)
    ap.add_argument("--comments", type=int, default=300)
    args = ap.parse_args()

    videos = find_videos(args.brand, args.videos)
    print(f"Found {len(videos)} videos from official channel:")
    for v in videos:
        print(f"  [{v['channel']}] {v['video_title']}")

    rows = []
    for v in videos:
        for c in get_comments(v["video_id"], args.comments):
            rows.append({**v, **c})

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No comments collected. Try a different query.")
    df = df.drop_duplicates(subset=["video_id", "comment_text"])

    os.makedirs("data/raw", exist_ok=True)
    path = f"data/raw/comments_{args.brand.replace(' ', '_').lower()}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} comments -> {path}")