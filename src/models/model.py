from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import imageio
from einops import reduce
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import utils
from src.utils.visualization_utils import (
    put_image_to_grid,
)
from src.models.utils import unnormalize_to_zero_to_one
from src.models.loss import GeodesicError
from src.models.normal_kl_loss import DiagonalGaussianDistribution
from src.utils.logging import get_logger, log_image, log_video

logger = get_logger(__name__)


class PoseConditional(pl.LightningModule):
    def __init__(
        self,
        u_net,
        optim_config,
        testing_config,
        save_dir,
        **kwargs,
    ):
        super().__init__()
        self.u_net = u_net

        # define logger
        self.media_dir = save_dir / "media"
        self.media_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir = save_dir / "predictions"
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # define optimization scheme
        self.optim_config = optim_config
        self.optim_name = "AdamW"

        # define testing config
        self.testing_config = testing_config
        self.testing_category = "unseen"

        if optim_config.loss_type == "l1":
            self.loss = F.l1_loss
        elif optim_config.loss_type == "l2":
            self.loss = F.mse_loss
        self.metric = GeodesicError()

        # define cad_dir for vsd evaluation
        self.tless_cad_dir = None

    def warm_up_lr(self):
        for optim in self.trainer.optimizers:
            for pg in optim.param_groups:
                pg["lr"] = (
                    self.global_step
                    / float(self.optim_config.warm_up_steps)
                    * self.optim_config.lr
                )
            if self.global_step % 50 == 0:
                logger.info(f"Step={self.global_step}, lr warm up: lr={pg['lr']}")

    def configure_optimizers(self):
        self.u_net.encoder.requires_grad_(False)
        if self.optim_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                self.optim_config.lr,
                weight_decay=self.optim_config.weight_decay,
                momentum=0.9,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),  # self.u_net.get_trainable_params(),
                self.optim_config.lr,
                weight_decay=self.optim_config.weight_decay,
            )
        return optimizer
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[10, 30, 50, 100], gamma=0.5
        # )
        # return [optimizer], [lr_scheduler]

    def compute_loss(self, pred, gt):
        if self.loss is not None:
            loss = self.loss(pred, gt, reduction="none")
            loss = reduce(loss, "b ... -> b (...)", "mean")
            return loss.mean()
        else:
            pred = DiagonalGaussianDistribution(pred)
            loss = pred.kl(other=gt)
            return loss.mean()

    def forward(self, query, ref, relR):
        query_feat = self.u_net.encoder.encode_image(query)
        ref_feat = self.u_net.encoder.encode_image(ref, mode="mode")
        pred_query_feat = self.u_net(ref_feat, relR)
        loss = self.compute_loss(pred_query_feat, query_feat)
        return loss

    @torch.no_grad()
    def sample(self, ref, relR):
        reference_feat = self.u_net.encoder.encode_image(ref, mode="mode")
        pred_query_feat = self.u_net(reference_feat, relR)
        if hasattr(self.u_net.encoder, "decode_latent") and callable(
            self.u_net.encoder.decode_latent
        ):
            pred_rgb = self.u_net.encoder.decode_latent(pred_query_feat)
            pred_rgb = unnormalize_to_zero_to_one(pred_rgb)
        else:
            pred_rgb = None
        return pred_query_feat, pred_rgb

    def training_step_single_dataloader(self, batch, data_name):
        query = batch["query"]
        ref = batch["ref"]
        relR = batch["relR"]
        relR_inv = batch["relR_inv"]

        loss = self.forward(query=query, ref=ref, relR=relR)
        if self.optim_config.use_inv_deltaR:
            loss_inv = self.forward(query=ref, ref=query, relR=relR_inv)
            loss = (loss + loss_inv) / 2
        self.log(f"loss/train_{data_name}", loss)

        # visualize reconstruction under GT pose
        if self.global_step % 1000 == 0:
            _, pred_rgb = self.sample(ref=ref, relR=relR)
            if pred_rgb is not None:
                save_image_path = (
                    self.media_dir
                    / f"reconst_step{self.global_step}_rank{self.global_rank}.png"
                )
                vis_imgs = [
                    unnormalize_to_zero_to_one(ref),
                    unnormalize_to_zero_to_one(batch["query"]),
                    pred_rgb,
                ]
                vis_imgs, ncol = put_image_to_grid(vis_imgs)
                vis_imgs_resized = vis_imgs.clone()
                vis_imgs_resized = F.interpolate(
                    vis_imgs_resized, (64, 64), mode="bilinear", align_corners=False
                )
                utils.save_image(
                    vis_imgs_resized,
                    save_image_path,
                    nrow=ncol * 4,
                )
                log_image(
                    logger=self.logger,
                    name=f"reconstruction/train_{data_name}",
                    path=str(save_image_path),
                )
        return loss

    def training_step(self, batch, idx):
        loss_dict = {}
        loss_sum = 0
        for idx_dataloader, data_name in enumerate(batch.keys()):
            warm_up_steps = self.optim_config.warm_up_steps
            if self.trainer.global_step < warm_up_steps and idx_dataloader == 0:
                self.warm_up_lr()
            elif self.trainer.global_step == warm_up_steps and idx_dataloader == 0:
                logger.info(f"Finished warm up, setting lr to {self.optim_config.lr}")

            loss = self.training_step_single_dataloader(batch[data_name], data_name)
            loss_dict[data_name] = loss
            loss_sum += loss

        loss_avg = loss_sum / len(batch.keys())
        self.log("loss/train_avg", loss_avg)
        return loss_avg

    @torch.no_grad()
    def log_score(self, dict_scores, split_name):
        for key, value in dict_scores.items():
            self.log(
                f"{key}/{split_name}",
                value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )

    def generate_templates(self, ref, relRs, gt_templates, visualize=False):
        b, c, h, w = ref.shape
        num_templates = relRs.shape[1]
        # keep all predicted features of template for retrieval later
        if hasattr(self.u_net.encoder, "decode_latent") and callable(
            self.u_net.encoder.decode_latent
        ):
            pred = torch.zeros((b, num_templates, c, h, w), device=ref.device)
        else:
            pred = None
        pred_feat = torch.zeros(
            (b, num_templates, self.u_net.encoder.latent_dim, int(h / 8), int(w / 8)),
            device=ref.device,
        )
        frames = []
        for idx in tqdm(range(0, num_templates)):
            # get output of sample
            if visualize:
                vis_imgs = [
                    unnormalize_to_zero_to_one(ref),
                    unnormalize_to_zero_to_one(gt_templates[:, idx]),
                ]
            pred_feat_i, pred_rgb_i = self.sample(ref=ref, relR=relRs[:, idx, :])
            pred_feat[:, idx] = pred_feat_i
            if pred_rgb_i is not None:
                pred[:, idx] = pred_rgb_i
                if visualize:
                    vis_imgs.append(pred_rgb_i.to(torch.float16))
                    save_image_path = (
                        self.media_dir / f"template{idx}_rank{self.global_rank}.png"
                    )
                    vis_imgs, ncol = put_image_to_grid(vis_imgs)
                    vis_imgs_resized = vis_imgs.clone()
                    vis_imgs_resized = F.interpolate(
                        vis_imgs_resized, (64, 64), mode="bilinear", align_corners=False
                    )
                    utils.save_image(
                        vis_imgs_resized,
                        save_image_path,
                        nrow=ncol * 4,
                    )
                    frame = np.array(Image.open(save_image_path))
                    frames.append(frame)
        if visualize:
            # write video of denoising process with imageio ffmpeg
            vid_path = (
                self.media_dir
                / f"video_step{self.global_step}_rank{self.global_rank}.mp4"
            )
            imageio.mimwrite(vid_path, frames, fps=5, macro_block_size=8)
        else:
            vid_path = None
        return pred_feat, pred, vid_path

    def retrieval(self, query, template_feat):
        num_templates = template_feat.shape[1]
        if self.testing_config.similarity_metric == "l2":
            query_feat = self.u_net.encoder.encode_image(query, mode="mode")
            query_feat = query_feat.unsqueeze(1).repeat(1, num_templates, 1, 1, 1)

            distance = (query_feat - template_feat) ** 2
            distance = torch.norm(distance, dim=2)
            similarity = -distance.sum(axis=3).sum(axis=2)  # B x N

            # get top 5 nearest templates
            _, nearest_idx = similarity.topk(k=5, dim=1)  # B x 1
            return similarity, nearest_idx

    def eval_geodesic(self, batch, data_name, visualize=True, save_prediction=False):
        if not (
            hasattr(self.u_net.encoder, "decode_latent")
            and callable(self.u_net.encoder.decode_latent)
        ):
            visualize = False
            logger.info("Setting visualize=False!")
        # visualize same loss as training
        query = batch["query"]
        ref = batch["ref"]
        relR = batch["relR"]
        batch_size = query.shape[0]
        loss = self.forward(query=query, ref=ref, relR=relR)
        self.log(f"loss/val_{data_name}", loss)

        if visualize:
            # visualize reconstruction under GT pose
            save_image_path = (
                self.media_dir
                / f"reconst_step{self.global_step}_rank{self.global_rank}.png"
            )
            _, pred_rgb = self.sample(ref=ref, relR=relR)
            if pred_rgb is not None:
                vis_imgs = [
                    unnormalize_to_zero_to_one(ref),
                    unnormalize_to_zero_to_one(batch["query"]),
                    pred_rgb,
                ]
                vis_imgs, ncol = put_image_to_grid(vis_imgs)
                vis_imgs_resized = vis_imgs.clone()
                vis_imgs_resized = F.interpolate(
                    vis_imgs_resized, (64, 64), mode="bilinear", align_corners=False
                )
                utils.save_image(
                    vis_imgs_resized,
                    save_image_path,
                    nrow=ncol * 4,
                )
                log_image(
                    logger=self.logger,
                    name=f"reconstruction/val_{data_name}",
                    path=str(save_image_path),
                )
        # retrieval templates
        template_imgs = batch["template_imgs"]
        template_relRs = batch["template_relRs"]
        pred_feat, pred_rgb, vid_path = self.generate_templates(
            ref=ref,
            relRs=template_relRs,
            gt_templates=template_imgs,
            visualize=visualize,
        )
        if visualize and pred_rgb is not None:
            log_video(
                logger=self.logger,
                name=f"templates/val_{data_name}",
                path=str(vid_path),
            )
        similarity, nearest_idx = self.retrieval(query=query, template_feat=pred_feat)

        if visualize:
            # visualize prediction
            save_image_path = (
                self.media_dir
                / f"retrieved_step{self.global_step}_rank{self.global_rank}.png"
            )
            retrieved_template = template_imgs[
                torch.arange(0, batch_size, device=query.device), nearest_idx[:, 0]
            ]
            vis_imgs = [
                unnormalize_to_zero_to_one(ref),
                unnormalize_to_zero_to_one(batch["query"]),
                unnormalize_to_zero_to_one(retrieved_template),
            ]
            vis_imgs, ncol = put_image_to_grid(vis_imgs)
            vis_imgs_resized = vis_imgs.clone()
            vis_imgs_resized = F.interpolate(
                vis_imgs_resized, (64, 64), mode="bilinear", align_corners=False
            )
            utils.save_image(
                vis_imgs_resized,
                save_image_path,
                nrow=ncol * 4,
            )
            log_image(
                logger=self.logger,
                name=f"retrieval/val_{data_name}",
                path=str(save_image_path),
            )
        template_poses = batch["template_Rs"][0]
        error, acc = self.metric(
            predR=template_poses[nearest_idx],
            gtR=batch["queryR"],
            symmetry=batch["symmetry"].reshape(-1),
        )
        self.log_score(acc, split_name=f"val_{data_name}")

        # save predictions
        if save_prediction:
            save_path = (
                self.log_dir / f"pred_step{self.global_step}_rank{self.global_rank}"
            )
            vis_imgs = vis_imgs.cpu().numpy()
            query_pose = batch["queryR"].cpu().numpy()
            similarity = similarity.cpu().numpy()
            np.savez(
                save_path,
                vis_imgs=vis_imgs,
                query_pose=query_pose,
                similarity=similarity,
            )
            print(save_path)

    def validation_step(self, batch, idx):
        for idx_dataloader, data_name in enumerate(batch.keys()):
            self.eval_geodesic(batch[data_name], data_name)

    def test_step(self, batch, idx_batch):
        self.eval_geodesic(
            batch,
            self.testing_category,
            visualize=True,
            save_prediction=True,
        )
