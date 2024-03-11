import pytorch_lightning as pl
import torch
from diffusers import AutoencoderKL


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
        # self.encoder = AutoencoderKL.from_config(f"{pretrained_path}/config.json")
        # self.encoder.load_state_dict(
        #     torch.load(f"{pretrained_path}/diffusion_pytorch_model.bin")
        # )
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
    from diffusers import DiffusionPipeline

    encoder = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    save_dir = "/home/nguyen/Documents/datasets/nope_project/pretrained/stable-diffusion-v1-5_vae.pth"
    encoder.save_pretrained(save_dir)
    encoder_reloaded = AutoencoderKL.from_pretrained(save_dir)
    # torch.save(encoder.state_dict(), save_dir)
    # repo_id = "runwayml/stable-diffusion-v1-5"
    # pipe = DiffusionPipeline.from_pretrained(repo_id, safe_serialization=True)
    # pipe.save_pretrained(
    #     "/home/nguyen/Documents/datasets/nope_project//pretrained/stable-diffusion-v1-5_vae.pth"
    # )
