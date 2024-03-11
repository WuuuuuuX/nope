import os
import json
import numpy as np
from PIL import Image
from src.lib3d.numpy import get_root_project

intrinsic = np.array([[525, 0.0, 256], [0.0, 525, 256], [0.0, 0.0, 1.0]])
bounding_box = np.array([128, 128, 384, 384])  # zoom-in twice given image size 512x512
train_categories = [
    "airplane",
    "bench",
    "cabinet",
    "car",
    "chair",
    "display",
    "lamp",
    "loudspeaker",
    "rifle",
    "sofa",
    "table",
    "telephone",
    "watercraft",  # "vessel" in the paper
]

test_categories = [
    "bottle",
    "bus",
    "clock",
    "dishwasher",
    "guitar",
    "mug",
    "pistol",
    "skateboard",
    "train",
    "washer",
]


def get_shapeNet_mapping():
    root_repo = get_root_project()
    path_shapenet_id2cat = os.path.join(root_repo, "src/utils/shapeNet_id2cat_v2.json")
    with open(path_shapenet_id2cat) as json_file:
        id2cat_mapping = json.load(json_file)
    # create inverse mapping
    cat2id_mapping = {}
    for key, value in id2cat_mapping.items():
        cat2id_mapping[value] = key
    return id2cat_mapping, cat2id_mapping


def open_image(path):
    img = Image.open(path).convert("RGB")
    img = img.crop(bounding_box)
    return img


def open_pose(path, img_name, view_id):
    poses = np.load(path)
    return poses[img_name][view_id]
