import pytorch_lightning as pl
import torch
from diffusers import AutoencoderKL
import os


class VAE_StableDiffusion(pl.LightningModule):
    def __init__(
        self,
        pretrained_path,
        latent_dim=4,
        name="vae",
        using_KL=False,
        **kwargs,
    ):
        super().__init__()
        if not os.path.exists(pretrained_path):
            self.encoder = AutoencoderKL.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="vae",
                torch_dtype=torch.float32,
            )
            self.encoder.save_pretrained(pretrained_path)
        self.encoder = AutoencoderKL.from_pretrained(pretrained_path)
        self.latent_dim = latent_dim
        self.name = name
        self.using_KL = using_KL
        if self.using_KL:
            self.encode_mode = None
        else:
            self.encode_mode = "mode"

    @torch.no_grad()
    def encode_image(self, image, mode=None):
        mode = self.encode_mode if mode is None else mode
        with torch.no_grad():
            if mode == "mode":
                latent = self.encoder.encode(image).latent_dist.mode() * 0.18215
            elif mode is None:
                latent = self.encoder.encode(
                    image
                ).latent_dist  # DiagonalGaussianDistribution instance
                latent.mean *= 0.18215
            else:
                raise NotImplementedError
        return latent

    @torch.no_grad()
    def decode_latent(self, latent):
        latent = latent / 0.18215
        with torch.no_grad():
            return self.encoder.decode(latent).sample


if __name__ == "__main__":
    from hydra.experimental import compose, initialize
    from hydra.utils import instantiate

    with initialize(config_path="../../../configs/"):
        cfg = compose(config_name="train.yaml")
    u_net = instantiate(cfg.model.u_net)
