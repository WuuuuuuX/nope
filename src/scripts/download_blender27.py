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

    source_url = "https://download.blender.org/release/Blender2.77/blender-2.77a-linux-glibc211-x86_64.tar.bz2"
    local_dir = root_dir / "blender"
    local_dir.mkdir(parents=True, exist_ok=True)

    # download command
    command = (
        f"curl {source_url} -o {local_dir}/blender-2.77a-linux-glibc211-x86_64.tar.bz2"
    )
    logger.info(f"Running command: {command}")
    os.system(command)

    # unzip command
    unzip_command = f"tar xf {local_dir}/blender-2.77a-linux-glibc211-x86_64.tar.bz2 -C {local_dir}"
    logger.info(f"Running command: {unzip_command}")
    os.system(unzip_command)


if __name__ == "__main__":
    download()
