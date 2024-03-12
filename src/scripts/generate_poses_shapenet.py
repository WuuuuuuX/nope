from pathlib import Path
import hydra
from tqdm import tqdm
import time
import json
import numpy as np
from omegaconf import DictConfig
from functools import partial
import multiprocessing
from src.utils.shapeNet_utils import (
    train_categories,
    test_categories,
    get_shapeNet_mapping,
)
from src.utils.logging import get_logger
from src.utils.trimesh_utils import get_obj_origin_and_diameter
from src.lib3d.numpy import look_at, spherical_to_cartesian, inverse_transform
from src.utils.inout import save_json

logger = get_logger(__name__)


def select_cad_within_category(
    idx_cat, root_models, list_cats, cat2id_mapping, max_num_cad=1000
):
    category = list_cats[idx_cat]
    if category in test_categories:
        max_num_cad = 100
    category_dir = root_models / cat2id_mapping[category]
    avail_dirs = category_dir.glob("*")
    avail_dirs = [x for x in avail_dirs if x.is_dir()]
    avail_dirs = sorted(avail_dirs, key=lambda x: x.name)
    metaDatas = []
    for obj_dir in tqdm(avail_dirs):
        obj_path = obj_dir / "models/model_normalized.obj"
        texture_dir = obj_dir / "models/images"
        material_path = obj_dir / "models/model_normalized.mtl"
        try:
            diameter, origin_bounds = get_obj_origin_and_diameter(str(obj_path))
        except:  # noqa: E722
            logger.warning(f"Error in getting diameter for {obj_path}")
            diameter = None
            origin_bounds = None
            continue
        having_texture = texture_dir.exists() or material_path.exists()
        if having_texture and diameter is not None:
            obj_id = str(obj_dir).split("/")[-1]
            metaData = {
                "category": cat2id_mapping[category],
                "category_name": category,
                "obj_id": obj_id,
                "diameter": diameter,
                "origin_bounds": origin_bounds.tolist(),
            }
            metaDatas.append(metaData)
            if len(metaDatas) >= max_num_cad:
                break
    if len(metaDatas) < max_num_cad:
        logger.warning(
            f"Only selected {len(metaDatas)} CAD models for {category} (less than {max_num_cad})"
        )
    else:
        logger.info(
            f"Succesfully selected {len(metaDatas)} ({max_num_cad}/{len(avail_dirs)}) CAD models for {category}"
        )
    return metaDatas


def generate_query_and_reference_poses(
    idx, save_paths, metaDatas, num_poses=5, radius=1.0
):
    """
    Generating camera query poses and reference poses
    """
    azimuths = np.random.uniform(0, 2 * np.pi, num_poses * 2)
    elevations = np.random.uniform(0, np.pi / 2, num_poses * 2)

    # convert to cartesian coordinates
    cam_locations = spherical_to_cartesian(azimuths, elevations, radius)
    poses = {}
    for name in ["query", "ref"]:
        poses[name] = np.zeros((num_poses, 4, 4))
        poses[name + "_norm1"] = np.zeros((num_poses, 4, 4))

    for idx_pose in range(num_poses):
        tmp = look_at(cam_locations[2 * idx_pose], np.zeros(3))
        poses["query_norm1"][idx_pose] = np.copy(inverse_transform(tmp))
        tmp[:3, 3] *= 0.6 * metaDatas[idx]["diameter"]
        poses["query"][idx_pose] = np.copy(inverse_transform(tmp))

        tmp = look_at(cam_locations[2 * idx_pose + 1], np.zeros(3))
        poses["ref_norm1"][idx_pose] = np.copy(inverse_transform(tmp))
        tmp[:3, 3] *= 0.6 * metaDatas[idx]["diameter"]
        poses["ref"][idx_pose] = np.copy(inverse_transform(tmp))

        norm = np.linalg.norm(poses["query_norm1"][idx_pose, :3, 3])
        if np.abs(norm - radius) > 0.1:
            logger.warning(f"Warning: location {norm} is bigger than radius {radius}")
    np.savez(save_paths[idx], **poses)
    return True


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train",
)
def select_cad_and_generate_poses(cfg: DictConfig) -> None:
    # Step 1: Selecting CAD models
    local_dir = Path(cfg.machine.root_dir) / "datasets/shapenet/models"
    save_metadata_path = local_dir.parent / "metaData_shapeNet.json"

    pool = multiprocessing.Pool(processes=cfg.machine.num_workers)
    if not save_metadata_path.exists():
        all_categories = train_categories + test_categories
        id2cat_mapping, cat2id_mapping = get_shapeNet_mapping()

        select_cad_within_category_with_index = partial(
            select_cad_within_category,
            root_models=local_dir,
            list_cats=all_categories,
            cat2id_mapping=cat2id_mapping,
        )

        # generate images
        start_time = time.time()
        metaDatas = list(
            tqdm(
                pool.imap_unordered(
                    select_cad_within_category_with_index,
                    range(len(all_categories)),
                ),
                total=len(all_categories),
            )
        )
        finish_time = time.time()
        logger.info(f"Total time to select CAD models: {finish_time - start_time}")

        metaData_shapeNet = []
        for metaData_cad in tqdm(metaDatas):
            metaData_shapeNet.extend(metaData_cad)
        logger.info(f"Total number of CAD models: {len(metaData_shapeNet)}")

        save_json(str(save_metadata_path), metaData_shapeNet)
        logger.info(f"Saved metadata to {save_metadata_path}")
    else:
        metaData_shapeNet = json.load(open(save_metadata_path))
        logger.info(f"Loaded metadata from {save_metadata_path}")
        logger.info(f"Total number of CAD models: {len(metaData_shapeNet)}")

    # Step 2: Generating object poses
    img_root_dir = local_dir.parent / "images"
    img_root_dir.mkdir(parents=True, exist_ok=True)
    obj_paths = []
    for idx in tqdm(range(len(metaData_shapeNet))):
        obj_dir = img_root_dir / f"{idx:06d}"
        obj_dir.mkdir(parents=True, exist_ok=True)
        obj_path = obj_dir / "poses.npz"
        obj_paths.append(obj_path)

    generate_query_and_reference_poses_with_index = partial(
        generate_query_and_reference_poses,
        save_paths=obj_paths,
        metaDatas=metaData_shapeNet,
    )

    # generate poses
    start_time = time.time()
    mapped_values = list(
        tqdm(
            pool.imap_unordered(
                generate_query_and_reference_poses_with_index,
                range(len(metaData_shapeNet)),
            ),
            total=len(metaData_shapeNet),
        )
    )
    pool.close()
    finish_time = time.time()
    logger.info(
        f"Total time to generate {sum(mapped_values)}/{len(metaData_shapeNet)} query pose {finish_time - start_time} s"
    )


if __name__ == "__main__":
    select_cad_and_generate_poses()
