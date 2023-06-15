""" tools to convert coordinates from local GPS frame to image and the other way around """
import numpy as np
import math
from skimage import transform
import cv2
import os
import argparse

import json
import yaml

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def draw_points_on_image(image, points):
    # Load the image
    fig, ax = plt.subplots()
    ax.imshow(image)

    x_coords, y_coords = zip(*points)
    ax.plot(x_coords, y_coords, 'ro')
    height, width, _ = image.shape
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    plt.show()

def draw_rectangles_on_image(image, rectangles):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for rect in rectangles:
        print(rect)
        x_coords, y_coords = zip(*rect)
        x_coords = list(x_coords) + [x_coords[0]]
        y_coords = list(y_coords) + [y_coords[0]]
        ax.plot(x_coords, y_coords)
    height, width, _ = image.shape
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    plt.show()


def compute_transformation(points_frame1, points_frame2):
    """ Given lists of 2d points in two frames, compute the transform
    from frame 1 to frame 2. At least 3 points are needed """
    transformation = transform.estimate_transform(
        'affine', points_frame1, points_frame2)
    return transformation


def compute_rectangle_corners(center, dimensions, rotation_angle):
    half_width = dimensions[0] / 2
    half_height = dimensions[1] / 2

    cos_theta = math.cos(rotation_angle)
    sin_theta = math.sin(rotation_angle)

    top_left_corner = (
        center[0] - half_width * cos_theta + half_height * sin_theta,
        center[1] - half_width * sin_theta - half_height * cos_theta
    )

    top_right_corner = (
        center[0] + half_width * cos_theta + half_height * sin_theta,
        center[1] + half_width * sin_theta - half_height * cos_theta
    )

    bottom_right_corner = (
        center[0] + half_width * cos_theta - half_height * sin_theta,
        center[1] + half_width * sin_theta + half_height * cos_theta
    )

    bottom_left_corner = (
        center[0] - half_width * cos_theta - half_height * sin_theta,
        center[1] - half_width * sin_theta + half_height * cos_theta
    )

    return [top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner]


def track_sparse_points(video_path, initial_points, initial_frame=0):
    # TODO test if this works with sequences where the drone turns
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    points_to_track = initial_points.astype(np.float32)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03),
                     minEigThreshold=0.01)

    frame_count = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count < initial_frame:
            continue
        # we skip every second frame as we have annotations only for every second frame
        if frame_count % 2 == 1:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, points_to_track, None, **lk_params)
        status = status.flatten()

        good_new = next_points[status == 1]
        if len(good_new) < 3:
            print("Failed to track points, had,", len(good_new),
                  " < 3. We ended at frame", frame_count)
        points_to_track = good_new
        yield good_new, frame, status

        prev_gray = gray.copy()

    cap.release()


def transform_image_points(points_utm, points_picture, data, local_center=np.array([692009, 5338095])):
    # TODO add arg to save into JSON directly
    points_local_coord = points_utm - local_center
    # points picture from get_image_coords.py
    transformation = compute_transformation(points_local_coord, points_picture)

    annotations = data['annotations']
    pts_2_transform = np.array([ann['translation'][:-1] for ann in annotations])
    # get edges of a bounding box
    rotations = [ann['rotation'][-1] for ann in annotations]
    dimensions = [ann['dimension'][:-1] for ann in annotations]
    transformed_centers = transformation(pts_2_transform)
    corners = []
    for rotation, dimension, pt in zip(rotations, dimensions, pts_2_transform):
        corners.append(transformation(np.array(compute_rectangle_corners(pt, dimension, rotation))))

    return transformed_centers, corners


def example_single_frame():
    # order should be: big kanal, canal at the stop, small kanal
    # this will be constant always
    # TODO save this somewhere to a config file :)
    points_utm = np.array([[692294.614, 5338134.302], [692286.8445, 5338064.377], [692296.6575, 5338114.08]])
    local_center = np.array([692009, 5338095])
    points_local_coord = points_utm - local_center
    print("points local coord", points_local_coord)
    # points picture from get_image_coords.py
    points_picture = np.array([(2054.4821184304824, 951.5860976951102), (3811.5836308398125, 1260.5615193107983), (2568.2025837398946, 930.7936289401422)])
    #np.array([(2054.104856751003, 951.1460765903958), (3795.930533888976, 1252.5002568374548), (2572.029404848872, 925.3962370634737)])
    transformation = compute_transformation(points_local_coord, points_picture)

    print("testing transform", transformation(points_local_coord))


    # load points dynamically
    with open("./2022-10-06T16-34-42/annotations/2022-10-06T16-34-42_00000.json", "r") as f:
        data = json.load(f)
    annotations = data['annotations']
    pts_2_transform = np.array([ann['translation'][:-1] for ann in annotations])
    # get edges of a bounding box
    rotations = [ann['rotation'][-1] for ann in annotations]
    dimensions = [ann['dimension'][:-1] for ann in annotations]
    corners = []
    for rotation, dimension, pt in zip(rotations, dimensions, pts_2_transform):
        corners.append(transformation(np.array(compute_rectangle_corners(pt, dimension, rotation))))
    print(corners)
    #pts_2_transform = np.vstack((pts_2_transform, np.array(corners)))

    #print(pts_2_transform)
    transformed = transformation(pts_2_transform)
    print("transformed", transformed)
    image_path = "./2022-10-06T16-34-42/frame_0.jpeg"
    img = plt.imread(image_path)
    draw_points_on_image(img, transformed)
    draw_rectangles_on_image(img, corners)
    plt.show()


def run_video(video_dir, video_path, visualize=True, save=None):
    """
    Transform annotations for a video.

    If save is not None, it should be the folder where we save the modified json files
    """
    video_path = f'./{video_dir}/{video_path}'

    # TODO have all the jsons ready to take the points from them
    world2pic_file = f"{video_dir}/world2pic.yaml"
    with open(world2pic_file, "r") as f:
        world2pic = yaml.load(f, yaml.BaseLoader)
    # we use the groundtruth points as well as extra added points to estimate the transform
    points_utm = np.array(
        world2pic["measured_pts"]["world_utm"] + world2pic["extra_points"]["world_utm"],
        dtype=np.float64)
    points_picture = np.array(
        world2pic["measured_pts"]["pic"] + world2pic["extra_points"]["pic"],
        dtype=np.float64)
    annot_dir = f"{video_dir}/annotations"

    fig, ax = plt.subplots()
    for (pts, img, status), json_path in zip(
            track_sparse_points(video_path, points_picture),
            sorted([js_file for js_file in os.listdir(annot_dir)
                    if js_file.endswith(".json")])):
        with open(os.path.join(annot_dir, json_path), "r") as f:
            json_data = json.load(f)
        points_utm = points_utm[status == 1]
        centers, corners = transform_image_points(
            points_utm, pts, json_data)
        if save is not None:
            # TODO
            pass
        if visualize:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            x_coords, y_coords = zip(*pts)
            ax.plot(x_coords, y_coords, 'ro')
            for rect in corners:
                x_coords, y_coords = zip(*rect)
                x_coords = list(x_coords) + [x_coords[0]]
                y_coords = list(y_coords) + [y_coords[0]]
                ax.plot(x_coords, y_coords)

            height, width, _ = img.shape
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.axis('off')

            plt.pause(0.01)
            plt.cla()
            plt.draw()
    plt.close()


def print_global_coords(yaml_path):
    with open(yaml_path, "r") as f:
        world2pic = yaml.load(f, yaml.BaseLoader)
    trans = compute_transformation(np.array(world2pic["measured_pts"]["pic"], dtype=np.float64), np.array(world2pic["measured_pts"]["world_utm"], dtype=np.float64))
    print(trans(np.asarray(world2pic["extra_points"]["pic"], dtype=np.float64)).tolist())

# TODO polish and automate the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert coordinates between world and picture based on key points.")
    parser.add_argument("--video_path", default="2022-10-06T16-34-42/DJI_0777_cut.mp4")
    parser.add_argument("--compute_global_coords", action="store_true")
    args = parser.parse_args()
    if args.compute_global_coords:
        print_global_coords(f"{os.path.dirname(args.video_path)}/world2pic.yaml")
    else:
       run_video(os.path.dirname(args.video_path), os.path.basename(args.video_path))
