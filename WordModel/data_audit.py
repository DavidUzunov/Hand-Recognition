import json
import os
from collections import Counter
from pathlib import Path

def analyze_downloaded_data(json_file, video_dir):
    # 1. Load the JSON metadata
    with open(json_file, 'r') as f:
        metadata = json.load(f)
    
    # 2. Map YouTube IDs to their labels from the JSON
    url_to_label = {}
    for entry in metadata:
        video_id = entry['url'].split('v=')[-1]
        url_to_label[video_id] = entry['clean_text']

    # 3. Scan your downloaded folder for successful files
    downloaded_files = os.listdir(video_dir)
    found_labels = []

    for file_name in downloaded_files:
        # Extract ID (assuming filename is VIDEO_ID.mp4)
        video_id = Path(file_name).stem 
        if video_id in url_to_label:
            found_labels.append(url_to_label[video_id])

    # 4. Count frequencies
    counts = Counter(found_labels)
    
    print("--- DATASET AUDIT RESULTS ---")
    print(f"Total entries in JSON: {len(metadata)}")
    print(f"Total videos successfully downloaded: {len(downloaded_files)}")
    print(f"Unique signs available: {len(counts)}")
    print("\nTop 20 signs with the most usable videos:")
    
    for word, count in counts.most_common(20):
        print(f"{word.ljust(15)}: {count} videos")

    return counts

if __name__ == "__main__":
    # Update these paths to match your local setup
    JSON_PATH = "../Get-Data/MS-ASL/MSASL_train.json"
    VIDEO_PATH = "MS-ASL/videos"
    
    stats = analyze_downloaded_data(JSON_PATH, VIDEO_PATH)