from pathlib import Path
from PIL import Image
import json
import numpy as np
from src.utils.shapeNet_utils import intrinsic
from src.lib3d.numpy import compute_cropping_from_obj_scale
from src.libVis.pil import draw_box


if __name__ == "__main__":
    root_dir = Path("/home/nguyen/Documents/datasets/nope_project/datasets/shapenet/")
    all_metaDatas = json.load(open(root_dir / "metaData_shapeNet.json"))

    save_dir = Path("./tmp")
    save_dir.mkdir(exist_ok=True, parents=True)

    for idx in range(len(all_metaDatas)):
        metaData = all_metaDatas[idx]
        obj_dir = root_dir / "test" / f"{idx:06d}"
        poses = np.load(obj_dir / "poses.npz")
        for view_id in range(5):
            im_path = obj_dir / f"{view_id:06d}_query.png"
            pose = poses["query"][view_id]
            obj_scale = np.linalg.norm(pose[:3, 3])
            # bbox2d = compute_cropping_from_obj_scale(obj_scale, intrinsic)
            print(obj_scale)
            bbox2d = np.array([128, 128, 384, 384])
            im = Image.open(im_path)
            im = draw_box(im, bbox2d, color="red", width=3)
            im.save(save_dir / f"{idx:06d}_{view_id:06d}_query.png")
        if idx == 100:
            break
