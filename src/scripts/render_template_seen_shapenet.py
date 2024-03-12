from pathlib import Path
import hydra
from tqdm import tqdm
import json
import time
from omegaconf import DictConfig
from functools import partial
import multiprocessing
from src.utils.logging import get_logger
import os
import glob
import numpy as np
from src.utils.inout import write_txt
from src.utils.shapeNet_utils import train_categories
from src.lib3d.utils import get_obj_poses_from_template_level

logger = get_logger(__name__)

"""For running with custom path for Blender: 
blenderproc run src/poses/blenderproc.py --blender-install-path 
"""


def call_blender(
    idx,
    root_dir,
    metaDatas,
    disable_output,
    tless_like,
    gpu_id,
    custom_blender_path,
):
    metaData = metaDatas[idx]
    cad_path = (
        root_dir
        / "models"
        / metaData["category"]
        / metaData["obj_id"]
        / "models/model_normalized.obj"
    )
    obj_dir = root_dir / "templates" / f"{metaData['index']:06d}"

    command = f"blenderproc run src/lib3d/blenderproc.py {cad_path} {obj_dir} {gpu_id}"
    if tless_like:
        command += " tless_like"
    else:
        command += " no_tless_like"
    if disable_output:
        command += " true"
    if custom_blender_path is not None:
        command += f" --custom-blender-path {custom_blender_path}"
    # disable output when running os.system
    if disable_output:
        command += " > /dev/null 2>&1"
    os.system(command)
    # count number of images
    num_imgs = len(glob.glob(os.path.join(obj_dir, "*.png")))
    return num_imgs == 642, command


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train",
)
def render(cfg: DictConfig) -> None:
    num_unseen_instances_per_cat = 50
    local_dir = Path(cfg.machine.root_dir) / "datasets/shapenet/"
    all_metaDatas = json.load(open(local_dir / "metaData_shapeNet.json"))
    logger.info(f"Loaded {len(all_metaDatas)} CAD models!")

    test_metaDatas = []
    counters = {cat: 0 for cat in train_categories}
    for idx in tqdm(range(len(all_metaDatas))):
        metaData = all_metaDatas[idx]
        if metaData["category_name"] in train_categories:
            if counters[metaData["category_name"]] >= num_unseen_instances_per_cat:
                continue
            counters[metaData["category_name"]] += 1
            metaData["index"] = idx
            test_metaDatas.append(metaData)
    logger.info(f"Loaded {len(test_metaDatas)} test CAD models!")

    template_root_dir = Path(cfg.machine.root_dir) / "datasets/shapenet/templates"
    template_root_dir.mkdir(parents=True, exist_ok=True)
    obj_template_poses = get_obj_poses_from_template_level(
        level=2, pose_distribution="all"
    )
    test_metaDatas = test_metaDatas[cfg.start_index : cfg.end_index]
    for idx in tqdm(range(len(test_metaDatas))):
        metaData = test_metaDatas[idx]
        diameter = metaData["diameter"]

        norm = np.linalg.norm(obj_template_poses[0, :3, 3])
        tmp = np.copy(obj_template_poses)
        tmp[:, :3, 3] *= 1 / norm * 1.2 * diameter

        obj_dir = template_root_dir / f"{metaData['index']:06d}"
        obj_dir.mkdir(parents=True, exist_ok=True)
        np.savez(obj_dir / "poses.npz", template_poses=tmp, diameter=diameter)

    pool = multiprocessing.Pool(processes=min(3, cfg.machine.num_workers))
    call_blender_with_index = partial(
        call_blender,
        root_dir=local_dir,
        metaDatas=test_metaDatas,
        disable_output=cfg.disable_output,
        tless_like=cfg.tless_like,
        gpu_id=cfg.gpu_ids,
        custom_blender_path=cfg.custom_blender_path,
    )
    # generate images
    start_time = time.time()
    mapped_values = list(
        tqdm(
            pool.imap_unordered(
                call_blender_with_index,
                range(len(test_metaDatas)),
            ),
            total=len(test_metaDatas),
        )
    )
    pool.close()

    # collect failed renderings to re-render
    list_fails = []
    success = 0
    for idx, value in enumerate(mapped_values):
        if not value[0]:
            list_fails.append(value[1])
        else:
            success += 1
    write_txt(
        local_dir / f"failed_template_seen_{cfg.start_index}to{cfg.end_index}.txt",
        list_fails,
    )
    finish_time = time.time()
    logger.info(
        f"Total time to render {success}/{len(mapped_values)} CAD models: {finish_time - start_time}"
    )


if __name__ == "__main__":
    render()
