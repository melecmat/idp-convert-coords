import argparse
from builtins import breakpoint
import json
import os
import cv2


def convert2cocovid():
    pass

def get_area(bbox):
    return bbox[-1] * bbox[-2]


def main():
    parser = argparse.ArgumentParser(description="Extract every second frame of videos.")
    parser.add_argument("--dir", default="2022-10-06T16-34-42")
    args = parser.parse_args()

    # load global json
    with open(os.path.join(args.dir, "annotations_meta.json"), "r") as f:
            annotations_meta = json.load(f)
    coco_format = {}
    coco_format["categories"] = [{"id": id + 1, "name": category} for id, category
                                 in enumerate(annotations_meta["object_count"])]
    print(coco_format)

    coco_format["videos"] = [{"id": 1, "name": "dummy_video"}]
    image_names = os.listdir(os.path.join(args.dir, "images"))
    height, width, depth = cv2.imread(os.path.join(args.dir, "images", image_names[0])).shape
    coco_format["images"] = []
    for image_name in image_names:
        frame_id = int(image_name.split("_")[-1].strip(".jpg"))
        image_info = {
            "frame_id": frame_id,
            "file_name": image_name,
            "video_id": 1,
            "id": frame_id + 1,
            "width": width,
            "height": height
        }
        coco_format["images"].append(image_info)

    annotations_dir = os.path.join(args.dir, "annotations_idp")
    shitty_counter = 0
    coco_format ["annotations"] = []
    for annotation_file_name in os.listdir(annotations_dir):
        annotation_id =  int(annotation_file_name.split("_")[-1].strip(".json"))
        with open(os.path.join(annotations_dir, annotation_file_name), "r") as f:
            annotations = json.load(f)
        for annot in annotations["annotations"]:
            coco_annot = {
                "category_id": annot["category_id"],
                "bbox": annot["bbox"],
                "area": get_area(annot["bbox"]),
                "iscrowd": False,
                "visibility": 1.0,
                "occluded": False,
                "id": shitty_counter,
                "image_id": annotation_id // 2,
                "instance_id": annot["track_id"],
                "video_id": 1,
                "ignore": False  # if we ever want to ignore weird objects, this could be the way 
            }
            shitty_counter += 1
            coco_format["annotations"].append(coco_annot)
    with open(os.path.join(args.dir, "coco_annotations.json"), "w") as f:
        json.dump(coco_format, f)
    


if __name__ == "__main__":
    main()