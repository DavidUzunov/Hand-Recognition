import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

def trim_dataset(json_path, source_video_dir, target_dir, limit_per_class=20, total_limit=1000):
    # 1. Load the split JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 2. Group Video IDs by their Class ID (action[0])
    class_map = defaultdict(list)
    for video_id, info in data.items():
        class_id = info['action'][0]
        class_map[class_id].append(video_id)
    
    # 3. Sort classes by how many videos they have (most data first)
    sorted_classes = sorted(class_map.items(), key=lambda x: len(x[1]), reverse=True)
    
    # 4. Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    count = 0
    print(f"Trimming dataset to {total_limit} files...")

    for class_id, video_ids in sorted_classes:
        if count >= total_limit:
            break
            
        # Take up to 'limit_per_class' videos for this sign
        to_copy = video_ids[:limit_per_class]
        
        for vid in to_copy:
            if count >= total_limit: break
            
            source = Path(source_video_dir) / f"{vid}.mp4"
            destination = target_path / f"{vid}.mp4"
            
            if source.exists():
                shutil.copy(source, destination)
                count += 1
            
    print(f"Done! Collected {count} videos in {target_dir}")

if __name__ == "__main__":
    # Update these paths to your local setup
    JSON_FILE = "/home/david/Documents/projects/SBHacks-XII/Get-Data/WLASL/wlasl-complete/nslt_1000.json" # Or your split json
    SOURCE_DIR = "/home/david/Documents/projects/SBHacks-XII/Get-Data/WLASL/wlasl-complete/videos"
    TRIMMED_DIR = "WLASL_top1000"
    
    trim_dataset(JSON_FILE, SOURCE_DIR, TRIMMED_DIR)