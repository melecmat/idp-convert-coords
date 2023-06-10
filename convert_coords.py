""" tools to convert coordinates from local GPS frame to image and the other way around """
import numpy as np
from skimage.transform import SimilarityTransform, ProjectiveTransform
from skimage import transform
# 1. load a description of the points from a yaml file or sth (first just put it manually)
# both pixel coords annotated and the GPS coords (which I need to convert to local frame first)

# 2. compute the transform between global / local

# 3. apply the transform to the points in the local frame

import json

import matplotlib.pyplot as plt

def draw_points_on_image(image_path, points):
    # Load the image
    image = plt.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(image)

    x_coords, y_coords = zip(*points)
    ax.plot(x_coords, y_coords, 'ro')
    height, width, _ = image.shape
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    plt.show()

def draw_rectangles_on_image(image_path, rectangles):
    # Load the image
    image = plt.imread(image_path)
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
    """ given lists of 2d points in two frames,
    compute the transform from frame 1 to frame 2 """
    transformation = transform.estimate_transform('affine', points_frame1, points_frame2)
    return transformation

import math

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


# TODO polish and automate the script
if __name__ == "__main__":
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
    with open("./2022-10-06T16-34-42/2022-10-06T16-34-42_00000.json", "r") as f:
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
    #draw_points_on_image("./2022-10-06T16-34-42/frame_0.jpeg", transformed)
    draw_rectangles_on_image("./2022-10-06T16-34-42/frame_0.jpeg", corners)

    # TODO do a loop, save into the json

