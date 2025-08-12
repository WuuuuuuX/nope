#!/usr/bin/env python3
"""
NOPE Pose Prediction Script
For predicting rotation between new reference and query images.
"""

import argparse
import logging
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
from einops import rearrange
import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.weight import load_checkpoint
from src.utils.logging import get_logger
from src.models.utils import unnormalize_to_zero_to_one
from src.lib3d.numpy import get_obj_poses_from_template_level
from src.lib3d.rotation_conversions import convert_rotation_representation

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class NOPEPredictor:
    """NOPE model predictor for arbitrary reference and query images."""
    
    def __init__(self, config_path="configs/train.yaml", checkpoint_path=None):
        """
        Initialize the NOPE predictor.
        
        Args:
            config_path: Path to the configuration file
            checkpoint_path: Path to the trained model checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = 256
        
        # Setup image transforms (same as training)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.Resize(self.img_size, antialias=True),
            transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
        ])
        
        # Load configuration
        self.cfg = self._load_config(config_path)
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Load template poses for pose retrieval
        self._load_template_poses()
        
    def _load_config(self, config_path):
        """Load configuration from yaml file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        # Load the config using hydra's compose API
        from hydra import compose, initialize_config_dir
        
        with initialize_config_dir(config_dir=str(config_path.parent.absolute())):
            cfg = compose(config_name=config_path.stem)
            
        return cfg
    
    def _load_model(self, checkpoint_path):
        """Load the trained NOPE model."""
        try:
            from hydra.utils import instantiate
            
            # Initialize model from config
            model = instantiate(self.cfg.model)
            model = model.to(self.device)
            model.eval()
            
            if checkpoint_path:
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                    
                model.load_state_dict(state_dict, strict=False)
                logger.info("Model loaded successfully!")
            else:
                logger.warning("No checkpoint provided, using randomly initialized model")
                
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_template_poses(self):
        """Load template poses for pose retrieval."""
        try:
            # Load template poses (using level 2 for full evaluation)
            level = 2
            (
                self.template_indexes,
                self.template_poses,
            ) = get_obj_poses_from_template_level(
                level=level, pose_distribution="upper", return_index=True
            )
            logger.info(f"Loaded {len(self.template_indexes)} template poses")
            
        except Exception as e:
            logger.warning(f"Could not load template poses: {e}")
            # Create dummy template poses for basic functionality
            self._create_dummy_templates()
    
    def _create_dummy_templates(self):
        """Create dummy template poses if real ones are not available."""
        logger.info("Creating dummy template poses for basic functionality")
        
        # Create a set of rotation matrices covering different viewpoints
        n_templates = 42  # Standard number used in NOPE
        angles = np.linspace(0, 2*np.pi, n_templates, endpoint=False)
        
        self.template_poses = []
        self.template_indexes = list(range(n_templates))
        
        for i, angle in enumerate(angles):
            # Create rotation around Y axis (vertical)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            
            # Create a 4x4 pose matrix
            pose = np.eye(4)
            pose[:3, :3] = R
            self.template_poses.append(pose)
            
        self.template_poses = np.array(self.template_poses)
    
    def load_image(self, image_path):
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            image_tensor = self.img_transform(image)
            
            # Convert from HWC to CHW format and add batch dimension
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).float()
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def compute_relative_rotation(self, query_R, ref_R):
        """
        Compute relative rotation between query and reference poses.
        
        Args:
            query_R: Query rotation matrix (3x3)
            ref_R: Reference rotation matrix (3x3)
            
        Returns:
            Relative rotation in rotation6d format
        """
        if isinstance(query_R, np.ndarray):
            query_R = torch.from_numpy(query_R)
        if isinstance(ref_R, np.ndarray):
            ref_R = torch.from_numpy(ref_R)
            
        # Compute relative rotation
        relative = query_R @ ref_R.T
        
        # Convert to rotation6d representation
        relative_6d = convert_rotation_representation(relative, "rotation6d")
        
        return relative_6d.float()
    
    def generate_templates(self, ref_image):
        """
        Generate template images for all template poses.
        
        Args:
            ref_image: Reference image tensor
            
        Returns:
            Template features for pose retrieval
        """
        logger.info("Generating template features...")
        
        batch_size = ref_image.shape[0]
        num_templates = len(self.template_poses)
        
        # Prepare tensor for template features
        template_features = []
        
        # Process templates in batches to avoid memory issues
        batch_size_templates = 10
        
        with torch.no_grad():
            for i in range(0, num_templates, batch_size_templates):
                batch_end = min(i + batch_size_templates, num_templates)
                batch_poses = self.template_poses[i:batch_end]
                
                # Convert poses to relative rotations w.r.t. reference (identity)
                ref_R = np.eye(3)  # Reference pose is identity
                relative_rotations = []
                
                for pose in batch_poses:
                    template_R = pose[:3, :3]
                    rel_R = self.compute_relative_rotation(template_R, ref_R)
                    relative_rotations.append(rel_R)
                
                # Stack relative rotations
                rel_R_batch = torch.stack(relative_rotations).to(self.device)
                
                # Generate features for this batch of templates
                for j, rel_R in enumerate(rel_R_batch):
                    rel_R_expanded = rel_R.unsqueeze(0)  # Add batch dimension
                    
                    # Generate template feature using model
                    template_feat, _ = self.model.sample(ref_image, rel_R_expanded)
                    template_features.append(template_feat)
        
        # Stack all template features
        template_features = torch.stack(template_features, dim=1)  # [batch, num_templates, ...]
        
        logger.info(f"Generated {num_templates} template features")
        return template_features
    
    def predict_pose(self, ref_image_path, query_image_path, save_results=True, output_dir="./results"):
        """
        Predict pose between reference and query images.
        
        Args:
            ref_image_path: Path to reference image
            query_image_path: Path to query image
            save_results: Whether to save visualization results
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing prediction results
        """
        logger.info(f"Predicting pose between {ref_image_path} and {query_image_path}")
        
        # Load images
        ref_image = self.load_image(ref_image_path)
        query_image = self.load_image(query_image_path)
        
        with torch.no_grad():
            # Generate template features
            template_features = self.generate_templates(ref_image)
            
            # Perform retrieval to find best matching template
            similarity, nearest_idx = self.model.retrieval(query_image, template_features)
            
            # Get the best matching template pose
            best_template_idx = nearest_idx[0, 0].item()  # Top-1 match
            best_template_pose = self.template_poses[best_template_idx]
            
            # Extract rotation matrix
            predicted_rotation = best_template_pose[:3, :3]
            
            # Get similarity scores
            similarity_scores = similarity[0].cpu().numpy()
            
        # Prepare results
        results = {
            'predicted_rotation': predicted_rotation,
            'best_template_idx': best_template_idx,
            'similarity_scores': similarity_scores,
            'top5_indices': nearest_idx[0, :5].cpu().numpy(),
            'ref_image_path': str(ref_image_path),
            'query_image_path': str(query_image_path)
        }
        
        if save_results:
            self._save_results(results, ref_image, query_image, output_dir)
            
        return results
    
    def _save_results(self, results, ref_image, query_image, output_dir):
        """Save prediction results and visualizations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save rotation matrix
        rotation_file = output_dir / "predicted_rotation.npy"
        np.save(rotation_file, results['predicted_rotation'])
        logger.info(f"Saved rotation matrix to {rotation_file}")
        
        # Create visualization
        self._create_visualization(results, ref_image, query_image, output_dir)
        
        # Save detailed results
        results_file = output_dir / "prediction_results.npz"
        np.savez(
            results_file,
            predicted_rotation=results['predicted_rotation'],
            similarity_scores=results['similarity_scores'],
            top5_indices=results['top5_indices'],
            best_template_idx=results['best_template_idx']
        )
        logger.info(f"Saved detailed results to {results_file}")
    
    def _create_visualization(self, results, ref_image, query_image, output_dir):
        """Create and save visualization of the prediction."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Convert images for visualization
            ref_vis = unnormalize_to_zero_to_one(ref_image[0]).permute(1, 2, 0).cpu().numpy()
            query_vis = unnormalize_to_zero_to_one(query_image[0]).permute(1, 2, 0).cpu().numpy()
            
            # Reference image
            axes[0].imshow(ref_vis)
            axes[0].set_title("Reference Image")
            axes[0].axis('off')
            
            # Query image
            axes[1].imshow(query_vis)
            axes[1].set_title("Query Image")
            axes[1].axis('off')
            
            # Similarity scores plot
            top5_scores = results['similarity_scores'][results['top5_indices']]
            axes[2].bar(range(5), top5_scores)
            axes[2].set_title("Top-5 Template Similarities")
            axes[2].set_xlabel("Template Rank")
            axes[2].set_ylabel("Similarity Score")
            
            plt.tight_layout()
            
            # Save visualization
            vis_file = output_dir / "prediction_visualization.png"
            plt.savefig(vis_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved visualization to {vis_file}")
            
        except Exception as e:
            logger.warning(f"Could not create visualization: {e}")


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description="NOPE Pose Prediction")
    parser.add_argument("--ref_image", type=str, required=True,
                       help="Path to reference image")
    parser.add_argument("--query_image", type=str, required=True,
                       help="Path to query image") 
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./prediction_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        logger.info("Initializing NOPE predictor...")
        predictor = NOPEPredictor(
            config_path=args.config,
            checkpoint_path=args.checkpoint
        )
        
        # Make prediction
        results = predictor.predict_pose(
            ref_image_path=args.ref_image,
            query_image_path=args.query_image,
            output_dir=args.output_dir
        )
        
        # Print results
        print("\n" + "="*50)
        print("POSE PREDICTION RESULTS")
        print("="*50)
        print(f"Reference Image: {args.ref_image}")
        print(f"Query Image: {args.query_image}")
        print(f"Best Template Index: {results['best_template_idx']}")
        print(f"Top-5 Template Indices: {results['top5_indices']}")
        print(f"Predicted Rotation Matrix:")
        print(results['predicted_rotation'])
        print(f"Results saved to: {args.output_dir}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()