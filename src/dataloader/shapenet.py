import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import torchvision
from einops import rearrange
import json
from src.utils.logging import get_logger
from src.lib3d.numpy import (
    get_obj_poses_from_template_level,
    load_index_level0_in_level2,
)
from src.lib3d.rotation_conversions import convert_rotation_representation
from src.utils.shapeNet_utils import train_categories, open_pose, open_image
from src.utils.inout import convert_list_to_dataframe
from pytorch_lightning import seed_everything

seed_everything(2023)
logger = get_logger(__name__)


class ShapeNet(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        fast_evaluation=False,  # use when training and see the validation loss
        img_size=256,
        num_views_per_instance=5,
        **kwargs,
    ):
        self.root_dir = Path(root_dir)
        self.split = split

        # implementation details
        self.img_size = img_size
        self.num_views_per_instance = num_views_per_instance
        self.rotation_representation = "rotation6d"
        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                torchvision.transforms.Resize(self.img_size, antialias=True),
                transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
            ]
        )
        self.load_metaData()

        # load template poses for running evaluation
        self.load_template_poses(fast_evaluation=fast_evaluation)

    def load_metaData(self):
        """
        There are three different splits:
        1. Training sets: ~1000 cads per category (with 13 categories in total)
        2. Unseen instances sets: 50 cads per category (with 13 categories in total)
        3. Unseen categories sets: 1000 cads per category (with 10 categories in total)
        """
        self.is_testing_split = False if self.split == "training" else True
        if self.split in ["training", "unseen_training"]:
            categories = train_categories
            num_cad_per_category = 1000 if self.split == "training" else 50
        else:
            categories = self.split
            num_cad_per_category = 100

        # keep only instances belong to correct split
        all_metaDatas = json.load(open(self.root_dir / "metaData_shapeNet.json"))

        # counter for number of instances for each category
        counters = {cat: 0 for cat in categories}

        self.metaDatas = []
        for obj_id, metaData in enumerate(all_metaDatas):
            cat = metaData["category_name"]
            if cat in categories:
                if counters[cat] >= num_cad_per_category:
                    continue
                counters[cat] += 1
                metaData["symmetry"] = 2 if cat in ["bottle"] else 0
                metaData["obj_id"] = obj_id
                for view_id in range(self.num_views_per_instance):
                    metaData_ = metaData.copy()
                    metaData_["view_id"] = int(view_id)
                    self.metaDatas.append(metaData_)

        self.metaDatas = convert_list_to_dataframe(self.metaDatas)
        if not self.is_testing_split:  # shuffle the training set
            self.metaDatas = self.metaDatas.sample(frac=1).reset_index(drop=True)
        num_cads = sum([counters[cat] for cat in categories])
        logger.info(
            f"Loaded {len(self.metaDatas)} images for {num_cads} CAD models from split {self.split}!"
        )

    def __len__(self):
        return len(self.metaDatas)

    def load_template_poses(self, fast_evaluation):
        # load poses
        level = 0 if fast_evaluation else 2
        (
            self.testing_indexes,
            self.testing_templates_poses,
        ) = get_obj_poses_from_template_level(
            level=level, pose_distribution="upper", return_index=True
        )
        # load indexes templates
        if fast_evaluation:
            self.testing_indexes = load_index_level0_in_level2("upper")

    def compute_relative_pose(self, query_pose, ref_pose):
        relative = query_pose[:3, :3] @ np.linalg.inv(ref_pose)[:3, :3]
        relative = torch.from_numpy(relative)
        relative = convert_rotation_representation(
            relative, self.rotation_representation
        )
        relative_inv = ref_pose[:3, :3] @ np.linalg.inv(query_pose)[:3, :3]
        relative_inv = torch.from_numpy(relative_inv)
        relative_inv = convert_rotation_representation(
            relative_inv, self.rotation_representation
        )
        return relative, relative_inv

    def __getitem__(self, index):
        obj_id = self.metaDatas["obj_id"].iloc[index]
        obj_dir = self.root_dir / "test" / f"{obj_id:06d}"
        view_id = int(self.metaDatas["view_id"].iloc[index])
        symmetry = int(self.metaDatas["symmetry"].iloc[index])
        ref_view_id = np.random.choice(self.num_views_per_instance)

        query = open_image(obj_dir / f"{view_id:06d}_query.png")
        query = self.img_transform(query)
        ref = open_image(obj_dir / f"{ref_view_id:06d}_ref.png")
        ref = self.img_transform(ref)

        query_pose = open_pose(obj_dir / "poses.npz", "query", view_id)
        ref_pose = open_pose(obj_dir / "poses.npz", "ref", ref_view_id)
        relR, relR_inv = self.compute_relative_pose(query_pose, ref_pose)

        if not self.is_testing_split:
            return {
                "query": query.permute(2, 0, 1).float(),
                "ref": ref.permute(2, 0, 1).float(),
                "relR": relR.float(),
                "relR_inv": relR_inv.float(),
            }
        else:
            template_data = {"template_imgs": [], "template_relRs": []}
            obj_template_dir = self.root_dir / "templates" / f"{obj_id:06d}"
            for idx, template_id in enumerate(self.testing_indexes):
                template = open_image(obj_template_dir / f"{template_id:06d}.png")
                template = self.img_transform(template)

                template_pose = self.testing_templates_poses[idx]
                relR, relR_inv = self.compute_relative_pose(query_pose, template_pose)
                template_data["template_imgs"].append(template.permute(2, 0, 1))
                template_data["template_relRs"].append(relR)
            template_data["template_imgs"] = torch.stack(
                template_data["template_imgs"]
            ).float()
            template_data["template_relRs"] = torch.stack(
                template_data["template_relRs"]
            ).float()
            return {
                "query": query.permute(2, 0, 1).float(),
                "queryR": torch.from_numpy(
                    query_pose[:3, :3]
                ).float(),  # for evaluation,
                "symmetry": torch.tensor(symmetry),
                "ref": ref.permute(2, 0, 1).float(),
                "relR": relR.float(),
                "relR_inv": relR_inv.float(),
                "template_Rs": torch.from_numpy(
                    self.testing_templates_poses[:, :3, :3]
                ).float(),
                **template_data,
            }


if __name__ == "__main__":
    import logging
    from torchvision.utils import save_image

    logging.basicConfig(level=logging.INFO)
    from hydra.experimental import compose, initialize

    save_dir = Path("./tmp")
    save_dir.mkdir(exist_ok=True, parents=True)
    with initialize(config_path="../../configs/"):
        cfg = compose(config_name="train.yaml")
    dataset = ShapeNet(cfg.data.root_dir, "unseen_training")
    for i in range(10):
        data = dataset[i]
        save_image(data["query"], save_dir / f"{i:06d}_query.png")
        save_image(data["ref"], save_dir / f"{i:06d}_ref.png")
        save_image(data["template_imgs"], save_dir / f"{i:06d}_gt_template.png")
        logger.info("-----------------")
