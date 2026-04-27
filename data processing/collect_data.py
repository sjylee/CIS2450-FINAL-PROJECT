"""
=============================================================
  SONIC PARADOX — FAST Data Collection Script
  Uses ThreadPoolExecutor to run multiple Genius requests
  simultaneously instead of one at a time.
=============================================================
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

# ─────────────────────────────────────────────
#  1. PASTE YOUR API CREDENTIALS HERE
# ─────────────────────────────────────────────
SPOTIFY_CLIENT_ID     = "b8b40fde85944054b313cbd68aa73b26"
SPOTIFY_CLIENT_SECRET = "fcfd1b5be13441259ede7aab3e0a0eff"
GENIUS_ACCESS_TOKEN   = "bYCNZRzdOBdsfQCjSTD2EKXlvbvqwLybK5Y5IIEO25ivigUdCzyx9ckdUz6_jsdd"

# ─────────────────────────────────────────────
#  2. CONFIGURATION
# ─────────────────────────────────────────────
TARGET_ROWS      = 50000
YEARS            = range(1990, 2025)
OUTPUT_FILE      = "sonic_paradox_data.csv"
PARTIAL_FILE     = "sonic_paradox_partial.csv"
SAVE_EVERY       = 100
NUM_WORKERS      = 8
GENIUS_DELAY     = 0.3

# ─────────────────────────────────────────────
#  3. INITIALIZE SPOTIFY + SENTIMENT (shared)
# ─────────────────────────────────────────────
print("Initializing APIs...")

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    ),
    requests_timeout=10
)

analyzer = SentimentIntensityAnalyzer()
write_lock = threading.Lock()

# ─────────────────────────────────────────────
#  4. EACH WORKER GETS ITS OWN GENIUS CLIENT
# ─────────────────────────────────────────────
def make_genius_client():
    g = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)
    g.verbose = False
    g.remove_section_headers = True
    g.timeout = 8
    g.sleep_time = GENIUS_DELAY
    g.retries = 1
    return g

thread_local = threading.local()

def get_genius():
    if not hasattr(thread_local, "genius"):
        thread_local.genius = make_genius_client()
    return thread_local.genius

# ─────────────────────────────────────────────
#  5. HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_lyric_sentiment(lyrics_text):
    if not lyrics_text or len(lyrics_text.strip()) < 20:
        return None
    score = analyzer.polarity_scores(lyrics_text)['compound']
    return round((score + 1) / 2, 4)

def calculate_paradox_score(valence, lyric_sentiment):
    return round(abs(valence - lyric_sentiment), 4)

def load_existing_data():
    if os.path.exists(PARTIAL_FILE):
        df = pd.read_csv(PARTIAL_FILE)
        print(f"  Resuming: {len(df)} rows already saved.")
        return df.to_dict('records'), set(zip(df['track_name'], df['artist']))
    return [], set()

def save_progress(data):
    pd.DataFrame(data).to_csv(PARTIAL_FILE, index=False)

def estimate_time(done, total, elapsed_sec):
    if done == 0:
        return "calculating..."
    rate = done / elapsed_sec
    remaining = (total - done) / rate
    eta = datetime.now() + timedelta(seconds=remaining)
    mins = int(remaining // 60)
    secs = int(remaining % 60)
    return f"{mins}m {secs}s (ETA {eta.strftime('%H:%M:%S')})"

# ─────────────────────────────────────────────
#  6. CORE WORKER FUNCTION (fixed type hint)
# ─────────────────────────────────────────────

def process_track(track, features, year):
    track_name  = track['name']
    artist_name = track['artists'][0]['name']
    genius      = get_genius()

    lyrics_text = None
    genius_url  = None

    try:
        song = genius.search_song(track_name, artist_name)
        if song and song.lyrics:
            lyrics_text = song.lyrics
            genius_url  = song.url
    except Exception:
        pass

    lyric_sentiment = get_lyric_sentiment(lyrics_text) if lyrics_text else None
    paradox_score   = (
        calculate_paradox_score(features['valence'], lyric_sentiment)
        if lyric_sentiment is not None else None
    )

    return {
        'track_name'       : track_name,
        'artist'           : artist_name,
        'year'             : year,
        'spotify_id'       : track['id'],
        'genius_url'       : genius_url,
        'popularity'       : track['popularity'],
        'duration_ms'      : track['duration_ms'],
        'explicit'         : track['explicit'],
        'valence'          : features['valence'],
        'energy'           : features['energy'],
        'danceability'     : features['danceability'],
        'acousticness'     : features['acousticness'],
        'instrumentalness' : features['instrumentalness'],
        'speechiness'      : features['speechiness'],
        'liveness'         : features['liveness'],
        'loudness'         : features['loudness'],
        'tempo'            : features['tempo'],
        'key'              : features['key'],
        'mode'             : features['mode'],
        'has_lyrics'       : lyrics_text is not None,
        'lyric_sentiment'  : lyric_sentiment,
        'paradox_score'    : paradox_score,
    }

# ─────────────────────────────────────────────
#  7. SPOTIFY BATCH FETCHER
# ─────────────────────────────────────────────

def fetch_spotify_batch(year, offset):
    try:
        results = sp.search(
            q=f'year:{year}', type='track', limit=10, offset=offset
        )
        tracks = results['tracks']['items']
        if not tracks:
            return []

        ids      = [t['id'] for t in tracks]
        features = sp.audio_features(ids)

        pairs = []
        for i, track in enumerate(tracks):
            if features and features[i] is not None:
                pairs.append((track, features[i]))
        return pairs

    except Exception as e:
        print(f"  [Spotify error at year={year} offset={offset}]: {e}")
        time.sleep(5)
        return []

# ─────────────────────────────────────────────
#  8. MAIN COLLECTION LOOP
# ─────────────────────────────────────────────

def collect_data():
    data_list, already_collected = load_existing_data()
    start_time = time.time()
    rows_this_session = 0

    print(f"\n{'='*62}")
    print(f"  Target: {TARGET_ROWS} rows | Workers: {NUM_WORKERS} threads")
    print(f"  Already collected: {len(data_list)} rows")
    print(f"{'='*62}\n")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:

        for year in YEARS:
            if len(data_list) >= TARGET_ROWS:
                break

            for offset in range(0, 1000, 10):
                if len(data_list) >= TARGET_ROWS:
                    break

                batch = fetch_spotify_batch(year, offset)
                if not batch:
                    break

                new_batch = [
                    (track, feats) for track, feats in batch
                    if (track['name'], track['artists'][0]['name'])
                    not in already_collected
                ]
                if not new_batch:
                    continue

                slots_left = TARGET_ROWS - len(data_list)
                new_batch  = new_batch[:slots_left]

                futures = {
                    executor.submit(process_track, track, feats, year): (track, feats)
                    for track, feats in new_batch
                }

                for future in as_completed(futures):
                    try:
                        row = future.result()
                    except Exception as e:
                        print(f"  [Worker error]: {e}")
                        continue

                    if row is None:
                        continue

                    key = (row['track_name'], row['artist'])

                    with write_lock:
                        if key not in already_collected:
                            data_list.append(row)
                            already_collected.add(key)
                            rows_this_session += 1

                            elapsed   = time.time() - start_time
                            total     = len(data_list)
                            pct       = total / TARGET_ROWS * 100
                            eta_str   = estimate_time(rows_this_session, TARGET_ROWS, elapsed)
                            lyric_tag = "+" if row['has_lyrics'] else "-"

                            print(
                                f"  [{total:>5}/{TARGET_ROWS}] {pct:5.1f}% | "
                                f"{lyric_tag} | "
                                f"{row['artist'][:18]:<18} - {row['track_name'][:28]:<28} | "
                                f"ETA: {eta_str}"
                            )

                            if total % SAVE_EVERY == 0:
                                save_progress(data_list)
                                print(f"\n  >> Progress saved ({total} rows)\n")

    return data_list

# ─────────────────────────────────────────────
#  9. RUN & FINAL SAVE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("   SONIC PARADOX - Fast Concurrent Data Collector")
    print("=" * 62)

    t0        = time.time()
    collected = collect_data()
    elapsed   = time.time() - t0

    df = pd.DataFrame(collected)

    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print("\n" + "=" * 62)
    print(f"  Collection complete in {mins}m {secs}s")
    print(f"  Total rows:            {len(df)}")
    print(f"  Rows WITH lyrics:      {df['has_lyrics'].sum()}")
    print(f"  Rows WITHOUT lyrics:   {(~df['has_lyrics']).sum()}")
    if df['paradox_score'].notna().sum() > 0:
        avg_p = df['paradox_score'].mean()
        print(f"  Avg Paradox Score:     {avg_p:.4f}")
        print(f"\n  Top 5 Paradox Songs:")
        top5 = df.nlargest(5, 'paradox_score')[
            ['track_name', 'artist', 'valence', 'lyric_sentiment', 'paradox_score']
        ]
        print(top5.to_string(index=False))
    print("=" * 62)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Final dataset saved -> {OUTPUT_FILE}")

    if os.path.exists(PARTIAL_FILE):
        os.remove(PARTIAL_FILE)

    print("\nDone!\n")