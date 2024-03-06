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
    source_url = "https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/tars/shapenetcorev2.zip"
    zip_dir = root_dir / "datasets/zip"
    zip_dir.mkdir(parents=True, exist_ok=True)

    # download the zip file
    download_cmd = f"wget -O {zip_dir}/shapenetcorev2.zip {source_url}"
    logger.info(f"Running {download_cmd}")
    os.system(download_cmd)

    # unzip the file (only models_orig/)
    unzip_cmd = f"unzip {zip_dir}/shapenetcorev2.zip 'shapenetcorev2/models_orig/*' -d {root_dir}/datasets/"
    logger.info(f"Running {unzip_cmd}")
    os.system(unzip_cmd)

    # rename the zip folders. To check the number of .obj: "find . -name "*.obj"|wc -l" -> 52472
    local_dir = root_dir / "datasets/shapenet/"
    local_dir.mkdir(parents=True, exist_ok=True)
    rename_cmd = f"mv {root_dir}/datasets/shapenetcorev2/models_orig {local_dir}/models"
    logger.info(f"Running {rename_cmd}")
    os.system(rename_cmd)

    # remove original folders
    remove_cmd = f"rm -r {root_dir}/datasets/shapenetcorev2"
    logger.info(f"Running {remove_cmd}")
    os.system(remove_cmd)


if __name__ == "__main__":
    download()
