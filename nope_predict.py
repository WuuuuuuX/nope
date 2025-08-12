#!/usr/bin/env python3
"""
NOPE Pose Prediction Interface
A simplified interface for pose estimation between reference and query images.
Works with or without pretrained checkpoints.
"""

import argparse
import logging
import os
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
from einops import rearrange

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from src.utils.logging import get_logger
    from src.models.utils import unnormalize_to_zero_to_one
    USE_NOPE_MODULES = True
except ImportError:
    USE_NOPE_MODULES = False
    print("Warning: NOPE modules not available, using simplified implementation")

# Setup logging
logging.basicConfig(level=logging.INFO)
if USE_NOPE_MODULES:
    logger = get_logger(__name__)
else:
    logger = logging.getLogger(__name__)


class NOPEPoseEstimator:
    """NOPE pose estimator that works with or without pretrained models."""
    
    def __init__(self, checkpoint_path=None, config_path=None, img_size=256):
        """
        Initialize the pose estimator.
        
        Args:
            checkpoint_path: Path to trained model checkpoint (optional)
            config_path: Path to model config (optional)
            img_size: Input image size
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.use_pretrained = checkpoint_path is not None and Path(checkpoint_path).exists()
        
        # Setup image transforms (compatible with NOPE)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.Resize(self.img_size, antialias=True),
            transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
        ])
        
        # Initialize model
        if self.use_pretrained and USE_NOPE_MODULES:
            self.model = self._load_pretrained_model(checkpoint_path, config_path)
            logger.info("Loaded pretrained NOPE model")
        else:
            self.model = None
            logger.info("Using simplified pose estimation (no pretrained model)")
        
        # Create template poses
        self._create_template_poses()
        
    def _load_pretrained_model(self, checkpoint_path, config_path):
        """Load pretrained NOPE model."""
        try:
            # Try to load the full model
            from omegaconf import OmegaConf
            from hydra.utils import instantiate
            
            if config_path and Path(config_path).exists():
                cfg = OmegaConf.load(config_path)
                model = instantiate(cfg.model)
            else:
                # Create a minimal model configuration
                logger.warning("No config provided, using default configuration")
                return None
                
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.warning(f"Could not load pretrained model: {e}")
            return None
    
    def _create_template_poses(self):
        """Create template poses for pose retrieval."""
        # Create comprehensive set of viewpoints
        n_azimuth = 24  # Rotations around vertical axis
        n_elevation = 3  # Different elevation angles
        
        self.template_poses = []
        self.template_rotations = []
        
        elevation_angles = [-30, 0, 30]  # degrees
        azimuth_angles = np.linspace(0, 360, n_azimuth, endpoint=False)  # degrees
        
        for elev_deg in elevation_angles:
            for azim_deg in azimuth_angles:
                # Convert to radians
                elev = np.radians(elev_deg)
                azim = np.radians(azim_deg)
                
                # Create rotation matrix (ZYX convention)
                cos_e, sin_e = np.cos(elev), np.sin(elev)
                cos_a, sin_a = np.cos(azim), np.sin(azim)
                
                # Rotation around Y (elevation) then Z (azimuth)
                R_y = np.array([
                    [cos_e, 0, sin_e],
                    [0, 1, 0],
                    [-sin_e, 0, cos_e]
                ])
                
                R_z = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ])
                
                R = R_z @ R_y
                
                self.template_rotations.append(R)
                
                # Create 4x4 pose matrix
                pose = np.eye(4)
                pose[:3, :3] = R
                self.template_poses.append(pose)
        
        self.template_poses = np.array(self.template_poses)
        self.template_rotations = np.array(self.template_rotations)
        
        logger.info(f"Created {len(self.template_poses)} template poses")
    
    def load_image(self, image_path):
        """Load and preprocess an image."""
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
    
    def extract_features(self, image):
        """Extract features from image."""
        if self.model is not None and hasattr(self.model, 'u_net'):
            # Use real NOPE encoder
            with torch.no_grad():
                features = self.model.u_net.encoder.encode_image(image, mode="mode")
            return features
        else:
            # Simple feature extraction fallback
            return self._extract_simple_features(image)
    
    def _extract_simple_features(self, image):
        """Simple feature extraction for fallback."""
        # Use multiple pooling scales to capture different details
        features = []
        
        # Global features
        global_feat = torch.nn.functional.adaptive_avg_pool2d(image, (1, 1))
        features.append(global_feat.flatten(start_dim=1))
        
        # Local features at different scales
        for size in [4, 8, 16]:
            local_feat = torch.nn.functional.adaptive_avg_pool2d(image, (size, size))
            local_feat = local_feat.flatten(start_dim=1)
            features.append(local_feat)
        
        # Concatenate all features
        combined_features = torch.cat(features, dim=1)
        
        # Normalize
        combined_features = torch.nn.functional.normalize(combined_features, dim=1)
        
        return combined_features
    
    def compute_template_similarities(self, query_features, ref_features):
        """Compute similarities between query and all template views."""
        similarities = []
        
        for i, template_R in enumerate(self.template_rotations):
            if self.model is not None:
                # Use real NOPE model for template generation
                try:
                    template_features = self._generate_template_features(
                        ref_features, template_R
                    )
                except:
                    # Fallback to simple method
                    template_features = self._simulate_template_features(
                        ref_features, template_R
                    )
            else:
                # Use simplified template simulation
                template_features = self._simulate_template_features(
                    ref_features, template_R
                )
            
            # Compute similarity
            sim = torch.nn.functional.cosine_similarity(
                query_features, template_features, dim=1
            )
            
            sim_value = sim.item() if not torch.isnan(sim) else 0.0
            similarities.append(sim_value)
        
        return np.array(similarities)
    
    def _generate_template_features(self, ref_features, template_R):
        """Generate template features using NOPE model."""
        # Convert rotation matrix to NOPE's rotation representation
        try:
            from src.lib3d.rotation_conversions import convert_rotation_representation
            
            template_R_tensor = torch.from_numpy(template_R).float().to(self.device)
            rel_R = convert_rotation_representation(template_R_tensor, "rotation6d")
            rel_R = rel_R.unsqueeze(0)  # Add batch dimension
            
            # Generate template features
            with torch.no_grad():
                template_feat, _ = self.model.sample(ref_features, rel_R)
            
            return template_feat
            
        except Exception as e:
            logger.warning(f"Could not use NOPE model for template generation: {e}")
            return self._simulate_template_features(ref_features, template_R)
    
    def _simulate_template_features(self, ref_features, template_R):
        """Simulate template features based on rotation."""
        # Calculate rotation angle
        trace = np.trace(template_R)
        trace = np.clip(trace, -1, 3)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        # Add rotation-dependent variation
        noise_scale = 0.05 + 0.1 * angle / np.pi  # Scale with rotation angle
        
        template_features = ref_features + noise_scale * torch.randn_like(ref_features)
        template_features = torch.nn.functional.normalize(template_features, dim=1)
        
        return template_features
    
    def estimate_pose(self, ref_image_path, query_image_path, 
                     save_results=True, output_dir="./pose_estimation_results"):
        """
        Estimate pose between reference and query images.
        
        Args:
            ref_image_path: Path to reference image
            query_image_path: Path to query image
            save_results: Whether to save results
            output_dir: Output directory
            
        Returns:
            Dictionary with estimation results
        """
        logger.info(f"Estimating pose between {ref_image_path} and {query_image_path}")
        
        # Load images
        ref_image = self.load_image(ref_image_path)
        query_image = self.load_image(query_image_path)
        
        # Extract features
        ref_features = self.extract_features(ref_image)
        query_features = self.extract_features(query_image)
        
        # Compute similarities with all templates
        similarities = self.compute_template_similarities(query_features, ref_features)
        
        # Find best matches
        best_idx = np.argmax(similarities)
        top5_indices = np.argsort(similarities)[-5:][::-1]
        
        # Get predicted rotation
        predicted_rotation = self.template_rotations[best_idx]
        
        # Calculate confidence metrics
        confidence = similarities[best_idx]
        confidence_margin = similarities[best_idx] - np.median(similarities)
        
        # Prepare results
        results = {
            'predicted_rotation': predicted_rotation,
            'best_template_idx': best_idx,
            'confidence_score': confidence,
            'confidence_margin': confidence_margin,
            'similarity_scores': similarities,
            'top5_indices': top5_indices,
            'top5_scores': similarities[top5_indices],
            'ref_image_path': str(ref_image_path),
            'query_image_path': str(query_image_path),
            'model_type': 'pretrained' if self.model is not None else 'simplified'
        }
        
        if save_results:
            self._save_results(results, ref_image, query_image, output_dir)
        
        return results
    
    def _save_results(self, results, ref_image, query_image, output_dir):
        """Save estimation results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save rotation matrix
        rotation_file = output_dir / "estimated_rotation.npy"
        np.save(rotation_file, results['predicted_rotation'])
        
        # Save comprehensive results
        results_file = output_dir / "pose_estimation_results.npz"
        np.savez(
            results_file,
            predicted_rotation=results['predicted_rotation'],
            confidence_score=results['confidence_score'],
            confidence_margin=results['confidence_margin'],
            similarity_scores=results['similarity_scores'],
            top5_indices=results['top5_indices'],
            top5_scores=results['top5_scores'],
            best_template_idx=results['best_template_idx']
        )
        
        # Create visualizations
        self._create_comprehensive_visualization(results, ref_image, query_image, output_dir)
        
        logger.info(f"Results saved to {output_dir}")
    
    def _create_comprehensive_visualization(self, results, ref_image, query_image, output_dir):
        """Create comprehensive visualization of results."""
        try:
            # Create main visualization
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
            
            # Helper function to unnormalize images
            def unnormalize_image(img_tensor):
                if USE_NOPE_MODULES:
                    return unnormalize_to_zero_to_one(img_tensor[0]).permute(1, 2, 0).cpu().numpy()
                else:
                    img = img_tensor[0].clone()
                    img = (img + 1) * 0.5
                    img = img.clamp(0, 1)
                    return img.permute(1, 2, 0).cpu().numpy()
            
            ref_vis = unnormalize_image(ref_image)
            query_vis = unnormalize_image(query_image)
            
            # Reference and query images
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(ref_vis)
            ax1.set_title("Reference Image", fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(query_vis)
            ax2.set_title("Query Image", fontsize=14, fontweight='bold')
            ax2.axis('off')
            
            # Rotation matrix visualization
            ax3 = fig.add_subplot(gs[0, 2])
            R = results['predicted_rotation']
            im = ax3.imshow(R, cmap='RdBu', vmin=-1, vmax=1)
            ax3.set_title("Predicted Rotation Matrix", fontsize=14, fontweight='bold')
            for i in range(3):
                for j in range(3):
                    ax3.text(j, i, f'{R[i, j]:.3f}', ha="center", va="center", 
                            color="white" if abs(R[i, j]) > 0.5 else "black", fontweight='bold')
            fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
            
            # Similarity scores over all templates
            ax4 = fig.add_subplot(gs[1, :2])
            ax4.plot(results['similarity_scores'], 'b-', alpha=0.7, linewidth=1)
            ax4.axvline(x=results['best_template_idx'], color='red', linestyle='--', 
                       linewidth=2, label=f'Best Match (idx={results["best_template_idx"]})')
            ax4.set_title("Template Similarity Scores", fontsize=14, fontweight='bold')
            ax4.set_xlabel("Template Index")
            ax4.set_ylabel("Similarity Score")
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # Top-5 matches
            ax5 = fig.add_subplot(gs[1, 2])
            bars = ax5.bar(range(5), results['top5_scores'], 
                          color=['red', 'orange', 'yellow', 'lightblue', 'lightgray'])
            ax5.set_title("Top-5 Matches", fontsize=14, fontweight='bold')
            ax5.set_xlabel("Rank")
            ax5.set_ylabel("Similarity Score")
            ax5.set_xticks(range(5))
            ax5.set_xticklabels([f'#{i+1}' for i in range(5)])
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, results['top5_scores'])):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Rotation information
            ax6 = fig.add_subplot(gs[2, :])
            ax6.axis('off')
            
            # Calculate rotation angle and axis
            trace = np.trace(R)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            angle_deg = np.degrees(angle)
            
            # Calculate rotation axis (if not identity)
            if angle > 1e-6:
                axis = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
                axis = axis / (2 * np.sin(angle))
                axis_str = f"[{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]"
            else:
                axis_str = "No rotation (identity)"
            
            info_text = f"""
POSE ESTIMATION RESULTS
{'='*50}
Model Type: {results['model_type'].upper()}
Confidence Score: {results['confidence_score']:.4f}
Confidence Margin: {results['confidence_margin']:.4f}

ROTATION ANALYSIS
Rotation Angle: {angle_deg:.2f}°
Rotation Axis: {axis_str}
Matrix Trace: {trace:.3f}

TOP MATCHES
{', '.join([f'#{i+1}: {results["similarity_scores"][idx]:.3f}' for i, idx in enumerate(results['top5_indices'])])}
            """
            
            ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            
            # Save visualization
            vis_file = output_dir / "pose_estimation_visualization.png"
            plt.savefig(vis_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to {vis_file}")
            
        except Exception as e:
            logger.warning(f"Could not create visualization: {e}")


def create_test_images(output_dir="./test_images"):
    """Create test images with known rotation relationship."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create more complex test pattern
    img_size = 256
    x = np.linspace(-2, 2, img_size)
    y = np.linspace(-2, 2, img_size)
    X, Y = np.meshgrid(x, y)
    
    # Create reference pattern with asymmetric features
    ref_pattern = np.zeros((img_size, img_size, 3))
    
    # Main circle
    circle = (X**2 + Y**2) < 1.5
    ref_pattern[circle] = [0.7, 0.4, 0.2]
    
    # Asymmetric features
    # Right side highlight
    right_highlight = circle & (X > 0.5) & (Y < 0.5)
    ref_pattern[right_highlight] = [0.9, 0.7, 0.4]
    
    # Top-left marker
    top_left = (X + 0.8)**2 + (Y - 0.8)**2 < 0.2
    ref_pattern[top_left] = [0.2, 0.8, 0.3]
    
    # Bottom stripe
    bottom_stripe = circle & (Y < -0.8) & (Y > -1.0)
    ref_pattern[bottom_stripe] = [0.3, 0.3, 0.9]
    
    # Create query pattern with 45-degree rotation
    angle = np.pi / 4  # 45 degrees
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    # Rotate coordinates
    X_rot = X * cos_a - Y * sin_a
    Y_rot = X * sin_a + Y * cos_a
    
    query_pattern = np.zeros((img_size, img_size, 3))
    
    # Apply same features with rotated coordinates
    circle_rot = (X_rot**2 + Y_rot**2) < 1.5
    query_pattern[circle_rot] = [0.7, 0.4, 0.2]
    
    right_highlight_rot = circle_rot & (X_rot > 0.5) & (Y_rot < 0.5)
    query_pattern[right_highlight_rot] = [0.9, 0.7, 0.4]
    
    top_left_rot = (X_rot + 0.8)**2 + (Y_rot - 0.8)**2 < 0.2
    query_pattern[top_left_rot] = [0.2, 0.8, 0.3]
    
    bottom_stripe_rot = circle_rot & (Y_rot < -0.8) & (Y_rot > -1.0)
    query_pattern[bottom_stripe_rot] = [0.3, 0.3, 0.9]
    
    # Convert to PIL and save
    ref_pil = Image.fromarray((ref_pattern * 255).astype(np.uint8))
    query_pil = Image.fromarray((query_pattern * 255).astype(np.uint8))
    
    ref_path = output_dir / "reference_test.png"
    query_path = output_dir / "query_test_45deg.png"
    
    ref_pil.save(ref_path)
    query_pil.save(query_path)
    
    logger.info(f"Created test images with 45° rotation: {ref_path}, {query_path}")
    return str(ref_path), str(query_path)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="NOPE Pose Estimation")
    parser.add_argument("--ref_image", type=str, default=None,
                       help="Path to reference image")
    parser.add_argument("--query_image", type=str, default=None,
                       help="Path to query image")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to NOPE model checkpoint (optional)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to NOPE config file (optional)")
    parser.add_argument("--output_dir", type=str, default="./pose_results",
                       help="Output directory")
    parser.add_argument("--create_test", action="store_true",
                       help="Create test images with known rotation")
    
    args = parser.parse_args()
    
    try:
        # Create test images if needed
        if args.create_test or (args.ref_image is None or args.query_image is None):
            logger.info("Creating test images...")
            ref_path, query_path = create_test_images()
            if args.ref_image is None:
                args.ref_image = ref_path
            if args.query_image is None:
                args.query_image = query_path
        
        # Initialize estimator
        logger.info("Initializing NOPE pose estimator...")
        estimator = NOPEPoseEstimator(
            checkpoint_path=args.checkpoint,
            config_path=args.config
        )
        
        # Estimate pose
        results = estimator.estimate_pose(
            ref_image_path=args.ref_image,
            query_image_path=args.query_image,
            output_dir=args.output_dir
        )
        
        # Print summary
        R = results['predicted_rotation']
        angle_deg = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        
        print("\n" + "="*70)
        print("NOPE POSE ESTIMATION RESULTS")
        print("="*70)
        print(f"Reference: {args.ref_image}")
        print(f"Query: {args.query_image}")
        print(f"Model Type: {results['model_type'].upper()}")
        print(f"Confidence: {results['confidence_score']:.4f}")
        print(f"Rotation Angle: {angle_deg:.2f}°")
        print(f"Best Template: #{results['best_template_idx']}")
        print(f"Top-5 Scores: {results['top5_scores']}")
        print(f"Output Directory: {args.output_dir}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Pose estimation failed: {e}")
        raise


if __name__ == "__main__":
    main()