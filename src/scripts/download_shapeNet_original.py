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

    # make sure you read the license agreement before downloading ShapeNet
    source_url = "https://huggingface.co/datasets/ShapeNet/ShapeNetCore-archive/resolve/main/ShapeNetCore.v2.zip"
    zip_dir = root_dir / "datasets/zip"
    zip_dir.mkdir(parents=True, exist_ok=True)

    # download the zip file
    download_cmd = f"wget -O {zip_dir}/ShapeNetCore.v2.zip {source_url}"
    logger.info(f"Running {download_cmd}")
    os.system(download_cmd)

    # unzip the file (only models_orig/)
    unzip_cmd = f"unzip {zip_dir}/ShapeNetCore.v2.zip -d {root_dir}/datasets/shapenet/"
    logger.info(f"Running {unzip_cmd}")
    os.system(unzip_cmd)

    # rename from ShapeNetCore.v2 to models/
    rename_cmd = f"mv {root_dir}/datasets/shapenet/ShapeNetCore.v2/ {root_dir}/datasets/shapenet/models"
    logger.info(f"Running {rename_cmd}")
    os.system(rename_cmd)

if __name__ == "__main__":
    download()
