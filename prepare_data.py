#!/usr/bin/env python3
"""
Data Preparation Script for ASL Recognition

This script helps prepare training data for the ASL recognition model.
It can convert video files or image sequences into the required .npy format.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import random


def extract_frames_from_video(video_path, num_frames=30, target_size=(224, 224)):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        target_size: Target resolution (height, width)
        
    Returns:
        Numpy array of shape (num_frames, 224, 224, 3) or None if extraction fails
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  Failed to open video: {video_path}")
            return None
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frame_idx = 0
        read_idx = 0
        
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in frame_indices:
                # Resize frame to target size
                frame = cv2.resize(frame, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        
        if len(frames) < num_frames:
            print(f"  Warning: Got {len(frames)}/{num_frames} frames from {video_path}")
            # Pad with last frame if needed
            if frames:
                last_frame = frames[-1]
                while len(frames) < num_frames:
                    frames.append(last_frame)
        
        # Convert to numpy array and normalize to [0, 1]
        seq = np.stack(frames, axis=0).astype(np.float32) / 255.0
        return seq
        
    except Exception as e:
        print(f"  Error extracting frames: {e}")
        return None


def load_image_sequence(image_dir, num_frames=30, target_size=(224, 224)):
    """
    Load a sequence of images from a directory.
    
    Args:
        image_dir: Directory containing sequential images
        num_frames: Number of frames to load
        target_size: Target resolution (height, width)
        
    Returns:
        Numpy array of shape (num_frames, 224, 224, 3) or None if loading fails
    """
    try:
        # Find all image files
        image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        
        if not image_files:
            print(f"  No image files found in {image_dir}")
            return None
        
        # Select frame indices
        frame_indices = np.linspace(0, len(image_files) - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            img_path = os.path.join(image_dir, image_files[idx])
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"  Failed to load {img_path}")
                continue
            
            # Resize and convert
            img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        
        if len(frames) < num_frames:
            print(f"  Warning: Got {len(frames)}/{num_frames} frames from {image_dir}")
            if frames:
                last_frame = frames[-1]
                while len(frames) < num_frames:
                    frames.append(last_frame)
        
        # Convert to numpy array and normalize
        seq = np.stack(frames, axis=0).astype(np.float32) / 255.0
        return seq
        
    except Exception as e:
        print(f"  Error loading images: {e}")
        return None


def prepare_video_dataset(
    input_dir,
    output_dir="data",
    num_frames=30,
    train_split=0.7,
    val_split=0.15,
):
    """
    Prepare dataset from video files.
    
    Expected input structure:
    input_dir/
    ├── A/
    │   ├── video1.mp4
    │   ├── video2.mp4
    │   └── ...
    ├── B/
    │   └── ...
    └── ...
    
    Args:
        input_dir: Directory containing class subdirectories with videos
        output_dir: Output data directory
        num_frames: Frames to extract per video
        train_split: Fraction for training set
        val_split: Fraction for validation set
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ["train", "val", "test"]:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_dir in sorted(input_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Find all video files
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']:
            video_files.extend(class_dir.glob(ext))
        
        video_files = sorted(video_files)
        
        if not video_files:
            print(f"  No video files found in {class_dir}")
            continue
        
        # Split into train/val/test
        random.shuffle(video_files)
        train_count = int(len(video_files) * train_split)
        val_count = int(len(video_files) * val_split)
        
        train_videos = video_files[:train_count]
        val_videos = video_files[train_count:train_count + val_count]
        test_videos = video_files[train_count + val_count:]
        
        # Process each split
        for split, videos in [("train", train_videos), ("val", val_videos), ("test", test_videos)]:
            split_dir = output_path / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  {split}: {len(videos)} videos")
            
            for i, video_file in enumerate(tqdm(videos, desc=f"  {split}")):
                seq = extract_frames_from_video(str(video_file), num_frames=num_frames)
                if seq is not None:
                    output_file = split_dir / f"gesture_{i:03d}.npy"
                    np.save(str(output_file), seq)


def prepare_image_dataset(
    input_dir,
    output_dir="data",
    num_frames=30,
    train_split=0.7,
    val_split=0.15,
):
    """
    Prepare dataset from image sequences.
    
    Expected input structure:
    input_dir/
    ├── A_sequence1/
    │   ├── frame_001.jpg
    │   ├── frame_002.jpg
    │   └── ...
    ├── A_sequence2/
    │   └── ...
    └── ...
    
    Class is extracted from directory name before first underscore.
    
    Args:
        input_dir: Directory containing sequence subdirectories
        output_dir: Output data directory
        num_frames: Frames to extract per sequence
        train_split: Fraction for training
        val_split: Fraction for validation
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ["train", "val", "test"]:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Find all sequence directories
    sequences = {}
    for seq_dir in sorted(input_path.iterdir()):
        if not seq_dir.is_dir():
            continue
        
        # Extract class name from directory name (before first underscore or digit)
        class_name = seq_dir.name.split('_')[0].upper()
        
        if class_name not in sequences:
            sequences[class_name] = []
        sequences[class_name].append(seq_dir)
    
    # Process each class
    for class_name, seq_dirs in sorted(sequences.items()):
        print(f"\nProcessing class: {class_name}")
        
        # Split into train/val/test
        random.shuffle(seq_dirs)
        train_count = int(len(seq_dirs) * train_split)
        val_count = int(len(seq_dirs) * val_split)
        
        train_seqs = seq_dirs[:train_count]
        val_seqs = seq_dirs[train_count:train_count + val_count]
        test_seqs = seq_dirs[train_count + val_count:]
        
        # Process each split
        for split, seqs in [("train", train_seqs), ("val", val_seqs), ("test", test_seqs)]:
            split_dir = output_path / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  {split}: {len(seqs)} sequences")
            
            for i, seq_dir in enumerate(tqdm(seqs, desc=f"  {split}")):
                seq = load_image_sequence(str(seq_dir), num_frames=num_frames)
                if seq is not None:
                    output_file = split_dir / f"gesture_{i:03d}.npy"
                    np.save(str(output_file), seq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ASL training data")
    parser.add_argument(
        "--input-videos",
        help="Directory containing video files organized by class",
    )
    parser.add_argument(
        "--input-images",
        help="Directory containing image sequences organized by class",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output data directory (default: data)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=30,
        help="Frames to extract per gesture (default: 30)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Training set fraction (default: 0.7)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation set fraction (default: 0.15)",
    )
    
    args = parser.parse_args()
    
    if args.input_videos:
        print("Preparing dataset from videos...")
        prepare_video_dataset(
            args.input_videos,
            args.output_dir,
            args.num_frames,
            args.train_split,
            args.val_split,
        )
    elif args.input_images:
        print("Preparing dataset from image sequences...")
        prepare_image_dataset(
            args.input_images,
            args.output_dir,
            args.num_frames,
            args.train_split,
            args.val_split,
        )
    else:
        print("Error: Specify either --input-videos or --input-images")
        parser.print_help()
