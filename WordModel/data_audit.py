import json
import os
from pathlib import Path
from collections import Counter

def audit_trimmed_dataset(mapping_json_path, trimmed_video_dir):
    # 1. Load the WLASL mapping file
    with open(mapping_json_path, 'r') as f:
        mapping_data = json.load(f)
    
    # 2. Create a lookup dictionary: {video_id: gloss_word}
    id_to_gloss = {}
    for entry in mapping_data:
        gloss = entry['gloss']
        for instance in entry['instances']:
            vid_id = instance['video_id']
            id_to_gloss[vid_id] = gloss

    # 3. Scan your trimmed folder
    trimmed_path = Path(trimmed_video_dir)
    found_videos = list(trimmed_path.glob("*.mp4"))
    
    audit_results = []
    missing_metadata = []
    
    for vid_file in found_videos:
        vid_id = vid_file.stem
        if vid_id in id_to_gloss:
            audit_results.append(id_to_gloss[vid_id])
        else:
            missing_metadata.append(vid_id)

    # 4. Generate Report
    stats = Counter(audit_results)
    
    print("--- DATASET AUDIT REPORT ---")
    print(f"Total files in folder: {len(found_videos)}")
    print(f"Unique glosses (words) found: {len(stats)}")
    print(f"Videos with no mapping found: {len(missing_metadata)}")
    
    print("\nBreakdown of top 10 words in your trimmed set:")
    for word, count in stats.most_common(10):
        print(f" - {word.ljust(15)}: {count} videos")
    
    if missing_metadata:
        print(f"\nWarning: The following IDs were not found in the JSON: {missing_metadata[:5]}...")

    return stats

if __name__ == "__main__":
    # Update these to your actual paths
    JSON_MAP = "/home/david/Documents/projects/SBHacks-XII/Get-Data/WLASL/wlasl-complete/WLASL_v0.3.json"
    TRIMMED_FOLDER = "WLASL_top1000"
    
    audit_stats = audit_trimmed_dataset(JSON_MAP, TRIMMED_FOLDER)