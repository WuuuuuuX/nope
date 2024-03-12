from pathlib import Path
import hydra
from tqdm import tqdm
import time
import json
from omegaconf import DictConfig
from functools import partial
import multiprocessing
from src.utils.logging import get_logger
import os
import glob
from src.utils.inout import write_txt

logger = get_logger(__name__)

"""For running with custom path for Blender: 
blenderproc run src/poses/blenderproc.py --blender-install-path 
"""


def call_blender(
    idx,
    start_index,
    root_dir,
    metaDatas,
    disable_output,
    tless_like,
    gpu_id,
    custom_blender_path,
):
    metaData = metaDatas[idx + start_index]
    cad_path = (
        root_dir
        / "models"
        / metaData["category"]
        / metaData["obj_id"]
        / "models/model_normalized.obj"
    )
    obj_dir = root_dir / "images" / f"{idx+start_index:06d}"

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
    return num_imgs == 10, command


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train",
)
def render(cfg: DictConfig) -> None:
    # Step 1: Selecting CAD models
    local_dir = Path(cfg.machine.root_dir) / "datasets/shapenet/"
    metaData_shapeNet = json.load(open(local_dir / "metaData_shapeNet.json"))
    logger.info(f"Loaded {len(metaData_shapeNet)} CAD models!")

    pool = multiprocessing.Pool(processes=min(4, cfg.machine.num_workers))
    call_blender_with_index = partial(
        call_blender,
        start_index=cfg.start_index,
        root_dir=local_dir,
        metaDatas=metaData_shapeNet,
        disable_output=cfg.disable_output,
        tless_like=cfg.tless_like,
        gpu_id=cfg.gpu_ids,
        custom_blender_path=cfg.custom_blender_path,
    )
    metaData_shapeNet = metaData_shapeNet[cfg.start_index : cfg.end_index]
    # generate images
    start_time = time.time()
    mapped_values = list(
        tqdm(
            pool.imap_unordered(
                call_blender_with_index,
                range(len(metaData_shapeNet)),
            ),
            total=len(metaData_shapeNet),
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
    write_txt(local_dir / "failed_renderings.txt", list_fails)
    finish_time = time.time()
    logger.info(
        f"Total time to render {success}/{len(mapped_values)} CAD models: {finish_time - start_time}"
    )


if __name__ == "__main__":
    render()
