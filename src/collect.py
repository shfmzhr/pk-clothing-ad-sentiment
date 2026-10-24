import os
import sys
import argparse
import html
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

sys.stdout.reconfigure(encoding="utf-8")  # so Urdu/emoji don't crash PowerShell

from pathlib import Path; load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True, encoding="utf-8-sig")
API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise SystemExit("YOUTUBE_API_KEY not found in .env")

youtube = build("youtube", "v3", developerKey=API_KEY)

# brand -> (search query, text that must appear in the channel name)
BRANDS = {
    "Khaadi":    ("Khaadi collection ad",          "khaadi"),
    "Gul Ahmed": ("Gul Ahmed collection ad",       "gul ahmed"),
    "Sapphire":  ("Sapphire Pakistan lawn collection", "sapphire"),
    "Limelight": ("Limelight Pakistan collection", "limelight"),
    "Maria B":       ("Maria B collection",             "maria"),
    "Sana Safinaz":  ("Sana Safinaz collection",        "sana safinaz"),
    "Bonanza":       ("Bonanza Satrangi collection",    "bonanza"),
    "Nishat Linen":  ("Nishat Linen collection",        "nishat"),
}


def find_videos(brand, n_videos):
    """Search YouTube, keep the brand's own channel, rank by comment count."""
    query, channel_key = BRANDS[brand]
    found, token = {}, None
    for _ in range(2):  # 2 pages of search results
        resp = youtube.search().list(
            part="snippet", q=query, type="video",
            maxResults=50, pageToken=token
        ).execute()
        for item in resp.get("items", []):
            sn = item["snippet"]
            if channel_key in sn["channelTitle"].lower():
                vid = item["id"]["videoId"]
                found[vid] = {
                    "brand": brand,
                    "video_id": vid,
                    "video_title": html.unescape(sn["title"]),
                    "channel": html.unescape(sn["channelTitle"]),
                    "video_published": sn["publishedAt"],
                    "video_url": "https://www.youtube.com/watch?v=" + vid,
                }
        token = resp.get("nextPageToken")
        if not token:
            break
    if not found:
        return []

    stats = youtube.videos().list(
        part="statistics", id=",".join(list(found)[:50])
    ).execute()
    for item in stats.get("items", []):
        found[item["id"]]["comment_count"] = int(item["statistics"].get("commentCount", 0))

    rows = [v for v in found.values() if v.get("comment_count", 0) > 0]
    rows.sort(key=lambda v: v["comment_count"], reverse=True)
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
    ap.add_argument("--list", action="store_true", help="show videos and comment counts, then stop")
    args = ap.parse_args()

    videos = find_videos(args.brand, args.videos)
    if args.list:
        total = 0
        for v in videos:
            total += v["comment_count"]
            print(f"{v['comment_count']:>6}  {v['video_title'][:70]}")
        print(f"TOTAL comments available across {len(videos)} videos: {total}")
        raise SystemExit
    print(f"Found {len(videos)} videos from official channel:")
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

