import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train",
)
def download(cfg: DictConfig) -> None:
    root_dir = Path(cfg.machine.root_dir)
    source_url = "https://huggingface.co/datasets/nv-nguyen/nope/resolve/main/"
    local_dir = root_dir / "datasets" / "shapenet"
    os.makedirs(local_dir, exist_ok=True)

    if cfg.only_sample:
        files = ["metaData_shapeNet.json", "image_samples.zip", "templates_seen.zip"]
    else:
        files = [
            "metaData_shapeNet.json",
            "images.zip",
            "templates_seen.zip",
            "templates_unseen_part1.zip",
            "templates_unseen_part2.zip",
        ]

    for file in files:
        # download file
        source_url_file = source_url + file
        target_file = local_dir / file
        download_cmd = f"wget {source_url_file} -O {target_file}"
        logger.info(f"Running {download_cmd}")
        os.system(download_cmd)

        # unzip files
        unzip_cmd = f"unzip {target_file} -d {local_dir}"
        logger.info(f"Running {unzip_cmd}")
        os.system(unzip_cmd)


if __name__ == "__main__":
    download()
