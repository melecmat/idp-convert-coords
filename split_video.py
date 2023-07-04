from convert_coords import read_videos
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description="Extract every second frame of videos.")
parser.add_argument("--video_dir", default="2022-10-06T16-34-42")
parser.add_argument("--video_names", nargs='+')
parser.add_argument("--dest", default="images")
args = parser.parse_args()


video_paths = [os.path.join(args.video_dir, video_name) for video_name in args.video_names]
print(video_paths)
if not os.path.exists(args.dest):
    os.makedirs(args.dest)

for frame_id, frame in enumerate(read_videos(video_paths)):
    cv2.imwrite(os.path.join(args.dest, f"image_{frame_id}.jpg"), frame)
