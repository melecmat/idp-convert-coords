from convert_coords import read_videos
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description="Extract every second frame of videos.")
parser.add_argument("--video_dir", default="2022-10-06T16-34-42")
parser.add_argument("--video_names", nargs='+')
args = parser.parse_args()


video_paths = [os.path.join(args.video_dir, video_name) for video_name in args.video_names]
print(video_paths)
image_folder = os.path.join(args.video_dir, "images")
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# for frame_id, frame in enumerate(read_videos(video_paths)):
#     cv2.imwrite(os.path.join(image_folder, f"image_{frame_id}.jpg"), frame)

def extract_frames(video_path, output_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    # Check if the video file was successfully opened
    if not video.isOpened():
        print("Error opening video file")
        return
    frame_count = 0
    # Read and save frames until the video ends
    while True:
        # Read a frame from the video
        ret, frame = video.read()
        # Break the loop if the video has ended
        if not ret:
            break
        # Save the frame as an image file
        frame_name = f"frame_{frame_count*2}.jpg"
        frame_path = output_path + '/' + frame_name
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    # Release the video file
    video.release()
    print(f"Extracted {frame_count} frames from the video")


for vid_path in video_paths:
    extract_frames(vid_path, image_folder)


