import json
import os
import yt_dlp
from pathlib import Path
from time import sleep

def download_asl_videos(json_file, output_dir="MS-ASL/videos"):
    # 1. Load the dataset
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 2. Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 3. Configure yt-dlp options
    # We use 'worstvideo' because ASL datasets usually don't need 4K 
    # and it saves significant disk space/bandwidth.
    ydl_opts = {
        'format': 'bestvideo[height<=360]+bestaudio/best[height<=360]',
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True, # Crucial: skips deleted/private videos
    }

    print(f"Starting download of {len(data)} potential videos...")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for entry in data:
            video_url = entry.get('url')
            label = entry.get('clean_text')
            
            # Check if we already have it
            video_id = video_url.split("v=")[-1]
            if os.path.exists(output_path / f"{video_id}.mp4"):
                continue
                
            try:
                print(f"Downloading '{label}': {video_url}")
                ydl.download([video_url])
                sleep(5) # sleep for 5s after each download to hopefully avoid rate limit
            except Exception as e:
                print(f"Failed to download {video_url}: {e}")

if __name__ == "__main__":
    # Point this to the file extracted by your previous script
    json_path = "../Get-Data/MS-ASL/MSASL_train.json" # this is absolute path for my laptop only
    print(json_path)
    download_asl_videos(json_path)