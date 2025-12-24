import os
import pandas as pd
import yt_dlp
import re
import time

# -----------------------------
# Path setup (DO THIS PROPERLY)
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # data/audio
DATA_DIR = os.path.dirname(BASE_DIR)                      # data

LYRICS_DIR = os.path.join(DATA_DIR, "lyrics")
AUDIO_DIR = os.path.join(BASE_DIR, "english")

os.makedirs(AUDIO_DIR, exist_ok=True)

csv_path = os.path.join(LYRICS_DIR, "english_songs_subset_200.csv")
updated_csv_path = os.path.join(LYRICS_DIR, "updated_english_songs_subset_200.csv")

df = pd.read_csv(csv_path)

# -----------------------------
# yt-dlp configuration
# -----------------------------

ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': os.path.join(AUDIO_DIR, '%(id)s.%(ext)s'),
    'noplaylist': True,
    'quiet': False,
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '128',
    }],
}

# -----------------------------
# Helpers
# -----------------------------

def sanitize_filename(name):
    # Keep all letters (Unicode), numbers, spaces, dash, underscore, dot
    name = str(name).strip()
    # Replace only forbidden characters
    name = re.sub(r'[\/:*?"<>|]', '_', name)
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    return name


def download_and_rename(row):
    song_name = row['song']
    artist = row['artist']

    query = f"{song_name} {artist} audio"

    desired_filename = (
        f"{sanitize_filename(song_name)}_"
        f"{sanitize_filename(artist)}.mp3"
    )
    desired_path = os.path.join(AUDIO_DIR, desired_filename)

    if os.path.exists(desired_path):
        print(f"Already exists: {desired_filename}")
        return desired_path

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(f"ytsearch:1:{query}", download=True)

            if not info or 'entries' not in info or not info['entries']:
                raise Exception("No search results")

            temp_path = ydl.prepare_filename(info['entries'][0])
            time.sleep(1)

            mp3_path = temp_path.rsplit('.', 1)[0] + '.mp3'

            if os.path.exists(mp3_path):
                os.rename(mp3_path, desired_path)
                print(f"Downloaded: {desired_filename}")
                return desired_path

            raise Exception("MP3 not found after conversion")

        except Exception as e:
            print(f"Error: {song_name} — {artist} → {e}")
            return None

# -----------------------------
# Main loop
# -----------------------------

if 'audio_path' not in df.columns:
    df['audio_path'] = None

for idx, row in df.iterrows():
    path = download_and_rename(row)
    df.at[idx, 'audio_path'] = path

df.to_csv(updated_csv_path, index=False)
print(f"Updated CSV saved to:\n{updated_csv_path}")
