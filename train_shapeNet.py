import logging
import os
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from src.utils.weight import load_checkpoint
import pytorch_lightning as pl
from src.utils.dataloader import concat_dataloader
from src.utils.logging import get_logger

pl.seed_everything(2022)
# set level logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def train(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # Delayed imports to get faster parsing
    from hydra.utils import instantiate

    logging.info("Initializing logger, callbacks and trainer")
    logger.info("Initializing logger, callbacks and trainer")
    cfg_trainer = cfg.machine.trainer
    if "WandbLogger" in cfg_trainer.logger._target_:
        os.environ["WANDB_API_KEY"] = cfg.user.wandb_api_key
        if cfg.machine.dryrun:
            os.environ["WANDB_MODE"] = "offline"
        logger.info(f"Wandb logger initialized at {cfg.save_dir}")
    elif "TensorBoardLogger" in cfg_trainer.logger._target_:
        tensorboard_dir = f"{cfg.save_dir}/{cfg_trainer.logger.name}"
        os.makedirs(tensorboard_dir, exist_ok=True)
        logger.info(f"Tensorboard logger initialized at {tensorboard_dir}")
    else:
        raise NotImplementedError("Only Wandb and Tensorboard loggers are supported")
    os.makedirs(cfg.save_dir, exist_ok=True)

    if cfg.machine.name == "slurm":
        cfg.machine.trainer.devices = int(os.environ["SLURM_GPUS_ON_NODE"])
        cfg.machine.trainer.num_nodes = int(os.environ["SLURM_NNODES"])
    trainer = instantiate(cfg.machine.trainer)
    logging.info("Trainer initialized")

    cfg.model.save_dir = Path(cfg.save_dir)
    model = instantiate(cfg.model)
    logging.info("Model initialized")

    pretrained_path = None  # cfg.model.u_net.encoder.pretrained_path
    if pretrained_path is not None and cfg.use_pretrained:
        logger.info(f"Loading pretrained ldm from {pretrained_path}")
        if "ldm" in cfg.model.u_net._target_:
            load_checkpoint(
                model.u_net,
                pretrained_path,
                checkpoint_key="state_dict",
                prefix="model.diffusion_model.",
            )
        else:
            load_checkpoint(model.u_net, pretrained_path)
            logging.info(f"{pretrained_path} loaded!")
    if "template" in cfg.model.u_net.encoder._target_:
        logger.info(f"Loading pretrained template encoder from {pretrained_path}")
        load_checkpoint(
            model.u_net.encoder,
            pretrained_path,
            checkpoint_key="state_dict",
            prefix="",
        )
    cfg.data.split = "training"
    train_dataloader = DataLoader(
        instantiate(cfg.data),
        batch_size=cfg.machine.batch_size,
        num_workers=cfg.machine.num_workers,
        shuffle=True,
    )
    train_dataloaders = concat_dataloader({"shapeNet": train_dataloader})

    cfg.data.split = "unseen_training"
    cfg.data.fast_evaluation = True
    val_dataloader = DataLoader(
        instantiate(cfg.data),
        batch_size=cfg.machine.batch_size,
        num_workers=cfg.machine.num_workers,
        shuffle=False,
    )
    val_dataloaders = concat_dataloader({"shapeNet": val_dataloader})
    logging.info("Fitting the model..")
    trainer.fit(
        model,
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloaders,
        ckpt_path=(
            cfg.model.checkpoint_path
            if cfg.model.checkpoint_path is not None and cfg.use_pretrained
            else None
        ),
    )
    logging.info("Fitting done")


if __name__ == "__main__":
    train()
