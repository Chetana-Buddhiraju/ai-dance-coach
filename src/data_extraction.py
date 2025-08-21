"""
Extract frames from all videos in input_dir and save to output_dir.
Usage:
    python src/data_extraction.py --input_dir /path/to/videos --output_dir /path/to/frames --frame_rate 2
"""
import os
import argparse
import cv2
from pathlib import Path

def extract_frames(video_path, output_dir, frame_rate=2):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(fps // frame_rate))
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1
    cap.release()
    print(f"Extracted {saved} frames to {output_dir}")

def batch_extract(input_dir, output_dir, frame_rate=2):
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(input_dir, fname)
            out_subdir = os.path.join(output_dir, Path(fname).stem)
            extract_frames(video_path, out_subdir, frame_rate=frame_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frame_rate", type=int, default=2)
    args = parser.parse_args()
    batch_extract(args.input_dir, args.output_dir, frame_rate=args.frame_rate)
