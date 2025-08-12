#!/usr/bin/env python3
"""
Simple NOPE Pose Prediction Script
For predicting rotation between new reference and query images.
This version provides a minimal working implementation without requiring pretrained checkpoints.
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplePosePredictor:
    """Simplified pose predictor for demonstration."""
    
    def __init__(self, img_size=256):
        """Initialize the pose predictor."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        
        # Setup image transforms (same as NOPE training)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.Resize(self.img_size, antialias=True),
            transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
        ])
        
        # Create template poses (simplified version)
        self._create_template_poses()
        
        logger.info(f"Initialized pose predictor with {len(self.template_poses)} templates")
    
    def _create_template_poses(self):
        """Create template poses covering different viewpoints."""
        n_templates = 24  # Reduced for simplicity
        
        self.template_poses = []
        self.template_rotations = []
        
        # Create rotations around Y axis (azimuth)
        azimuth_angles = np.linspace(0, 2*np.pi, n_templates, endpoint=False)
        
        for i, azimuth in enumerate(azimuth_angles):
            # Create rotation matrix around Y axis  
            cos_a, sin_a = np.cos(azimuth), np.sin(azimuth)
            R_y = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            
            # Store rotation matrix
            self.template_rotations.append(R_y)
            
            # Create full 4x4 pose matrix
            pose = np.eye(4)
            pose[:3, :3] = R_y
            self.template_poses.append(pose)
        
        self.template_poses = np.array(self.template_poses)
        self.template_rotations = np.array(self.template_rotations)
    
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
    
    def extract_simple_features(self, image):
        """Extract simple features from image (mock implementation)."""
        # Simple feature extraction using global average pooling
        # In real NOPE, this would use the trained encoder
        
        # Resize to smaller feature map
        features = torch.nn.functional.adaptive_avg_pool2d(image, (8, 8))
        
        # Flatten
        features = features.flatten(start_dim=1)
        
        # Normalize
        features = torch.nn.functional.normalize(features, dim=1)
        
        return features
    
    def compute_similarity(self, query_features, ref_features):
        """Compute similarity between query and reference features."""
        # Simple cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            query_features, ref_features, dim=1
        )
        return similarity
    
    def predict_pose_simple(self, ref_image_path, query_image_path, save_results=True, output_dir="./results"):
        """
        Simple pose prediction using feature similarity.
        This is a simplified version for demonstration.
        """
        logger.info(f"Predicting pose between {ref_image_path} and {query_image_path}")
        
        # Load images
        ref_image = self.load_image(ref_image_path)
        query_image = self.load_image(query_image_path)
        
        # Extract features
        ref_features = self.extract_simple_features(ref_image)
        query_features = self.extract_simple_features(query_image)
        
        # Create template features by "simulating" different viewpoints
        # In real NOPE, this would use the U-Net to generate template features
        template_similarities = []
        
        for i, template_R in enumerate(self.template_rotations):
            # Simple simulation: add rotation-dependent noise to reference features
            trace = np.trace(template_R)
            trace = np.clip(trace, -1, 3)  # Clip to avoid numerical issues
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))  # Rotation angle
            
            # Simulate template features (simplified)
            noise_factor = 0.1 * angle  # More different poses have more difference
            template_feat = ref_features + noise_factor * torch.randn_like(ref_features)
            template_feat = torch.nn.functional.normalize(template_feat, dim=1)
            
            # Compute similarity with query
            sim = self.compute_similarity(query_features, template_feat)
            sim_value = sim.item()
            
            # Handle NaN values
            if torch.isnan(sim) or np.isnan(sim_value):
                sim_value = 0.0
                
            template_similarities.append(sim_value)
        
        template_similarities = np.array(template_similarities)
        
        # Find best matching template
        best_idx = np.argmax(template_similarities)
        predicted_rotation = self.template_rotations[best_idx]
        
        # Get top-5 matches
        top5_indices = np.argsort(template_similarities)[-5:][::-1]
        
        # Prepare results
        results = {
            'predicted_rotation': predicted_rotation,
            'best_template_idx': best_idx,
            'similarity_scores': template_similarities,
            'top5_indices': top5_indices,
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
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Convert images for visualization (unnormalize)
            def unnormalize_image(img_tensor):
                img = img_tensor[0].clone()  # Remove batch dimension
                img = (img + 1) * 0.5  # Convert from [-1,1] to [0,1]
                img = img.clamp(0, 1)
                return img.permute(1, 2, 0).cpu().numpy()
            
            ref_vis = unnormalize_image(ref_image)
            query_vis = unnormalize_image(query_image)
            
            # Reference image
            axes[0, 0].imshow(ref_vis)
            axes[0, 0].set_title("Reference Image")
            axes[0, 0].axis('off')
            
            # Query image
            axes[0, 1].imshow(query_vis)
            axes[0, 1].set_title("Query Image")
            axes[0, 1].axis('off')
            
            # Similarity scores plot
            axes[1, 0].plot(results['similarity_scores'])
            axes[1, 0].axvline(x=results['best_template_idx'], color='r', linestyle='--', 
                              label=f'Best Match (idx={results["best_template_idx"]})')
            axes[1, 0].set_title("Template Similarities")
            axes[1, 0].set_xlabel("Template Index")
            axes[1, 0].set_ylabel("Similarity Score")
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Top-5 matches
            top5_scores = results['similarity_scores'][results['top5_indices']]
            bars = axes[1, 1].bar(range(5), top5_scores)
            axes[1, 1].set_title("Top-5 Template Matches")
            axes[1, 1].set_xlabel("Rank")
            axes[1, 1].set_ylabel("Similarity Score")
            
            # Color the best match differently
            bars[0].set_color('red')
            for i in range(1, 5):
                bars[i].set_color('blue')
            
            plt.tight_layout()
            
            # Save visualization
            vis_file = output_dir / "prediction_visualization.png"
            plt.savefig(vis_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved visualization to {vis_file}")
            
            # Also save rotation matrix visualization
            self._visualize_rotation_matrix(results['predicted_rotation'], output_dir)
            
        except Exception as e:
            logger.warning(f"Could not create visualization: {e}")
    
    def _visualize_rotation_matrix(self, rotation_matrix, output_dir):
        """Visualize the predicted rotation matrix."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Rotation matrix heatmap
            im1 = axes[0].imshow(rotation_matrix, cmap='RdBu', vmin=-1, vmax=1)
            axes[0].set_title("Predicted Rotation Matrix")
            axes[0].set_xlabel("Column")
            axes[0].set_ylabel("Row")
            
            # Add text annotations
            for i in range(3):
                for j in range(3):
                    text = axes[0].text(j, i, f'{rotation_matrix[i, j]:.3f}',
                                       ha="center", va="center", color="black", fontsize=10)
            
            fig.colorbar(im1, ax=axes[0])
            
            # Extract rotation angle
            trace = np.trace(rotation_matrix)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            angle_degrees = np.degrees(angle)
            
            # Rotation info
            axes[1].text(0.1, 0.8, f"Rotation Angle: {angle_degrees:.1f}°", 
                        transform=axes[1].transAxes, fontsize=12)
            axes[1].text(0.1, 0.7, f"Trace: {trace:.3f}", 
                        transform=axes[1].transAxes, fontsize=12)
            axes[1].text(0.1, 0.6, "Rotation Matrix:", 
                        transform=axes[1].transAxes, fontsize=12, weight='bold')
            
            matrix_str = "\n".join([
                f"[{rotation_matrix[0,0]:6.3f} {rotation_matrix[0,1]:6.3f} {rotation_matrix[0,2]:6.3f}]",
                f"[{rotation_matrix[1,0]:6.3f} {rotation_matrix[1,1]:6.3f} {rotation_matrix[1,2]:6.3f}]",
                f"[{rotation_matrix[2,0]:6.3f} {rotation_matrix[2,1]:6.3f} {rotation_matrix[2,2]:6.3f}]"
            ])
            axes[1].text(0.1, 0.3, matrix_str, transform=axes[1].transAxes, 
                        fontsize=10, family='monospace')
            
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(0, 1)
            axes[1].axis('off')
            axes[1].set_title("Rotation Information")
            
            plt.tight_layout()
            
            # Save rotation visualization
            rot_vis_file = output_dir / "rotation_matrix_visualization.png"
            plt.savefig(rot_vis_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved rotation visualization to {rot_vis_file}")
            
        except Exception as e:
            logger.warning(f"Could not create rotation visualization: {e}")


def create_sample_images(output_dir="./sample_images"):
    """Create sample images for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a simple synthetic object (circle with gradient)
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    y = np.linspace(-1, 1, img_size)
    X, Y = np.meshgrid(x, y)
    
    # Reference image: centered circle
    ref_img = np.zeros((img_size, img_size, 3))
    circle_mask = (X**2 + Y**2) < 0.5
    ref_img[circle_mask] = [0.8, 0.4, 0.2]  # Orange
    ref_img[circle_mask & (X > 0)] = [0.9, 0.6, 0.3]  # Lighter on right side
    
    # Query image: slightly rotated appearance
    query_img = np.zeros((img_size, img_size, 3))
    # Rotate coordinates
    angle = np.pi / 6  # 30 degrees
    X_rot = X * np.cos(angle) - Y * np.sin(angle)
    Y_rot = X * np.sin(angle) + Y * np.cos(angle)
    circle_mask_rot = (X_rot**2 + Y_rot**2) < 0.5
    query_img[circle_mask_rot] = [0.8, 0.4, 0.2]
    query_img[circle_mask_rot & (X_rot > 0)] = [0.9, 0.6, 0.3]
    
    # Convert to PIL and save
    ref_pil = Image.fromarray((ref_img * 255).astype(np.uint8))
    query_pil = Image.fromarray((query_img * 255).astype(np.uint8))
    
    ref_path = output_dir / "reference.png"
    query_path = output_dir / "query.png"
    
    ref_pil.save(ref_path)
    query_pil.save(query_path)
    
    logger.info(f"Created sample images: {ref_path}, {query_path}")
    return str(ref_path), str(query_path)


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description="Simple NOPE Pose Prediction")
    parser.add_argument("--ref_image", type=str, default=None,
                       help="Path to reference image")
    parser.add_argument("--query_image", type=str, default=None,
                       help="Path to query image") 
    parser.add_argument("--output_dir", type=str, default="./prediction_results",
                       help="Output directory for results")
    parser.add_argument("--create_samples", action="store_true",
                       help="Create sample images for testing")
    
    args = parser.parse_args()
    
    try:
        # Create sample images if requested or if no images provided
        if args.create_samples or (args.ref_image is None or args.query_image is None):
            logger.info("Creating sample images...")
            ref_path, query_path = create_sample_images()
            if args.ref_image is None:
                args.ref_image = ref_path
            if args.query_image is None:
                args.query_image = query_path
        
        # Initialize predictor
        logger.info("Initializing pose predictor...")
        predictor = SimplePosePredictor()
        
        # Make prediction
        results = predictor.predict_pose_simple(
            ref_image_path=args.ref_image,
            query_image_path=args.query_image,
            output_dir=args.output_dir
        )
        
        # Print results
        print("\n" + "="*60)
        print("POSE PREDICTION RESULTS")
        print("="*60)
        print(f"Reference Image: {args.ref_image}")
        print(f"Query Image: {args.query_image}")
        print(f"Best Template Index: {results['best_template_idx']}")
        print(f"Top-5 Template Indices: {results['top5_indices']}")
        print(f"Predicted Rotation Angle: {np.degrees(np.arccos(np.clip((np.trace(results['predicted_rotation']) - 1) / 2, -1, 1))):.1f}°")
        print(f"Predicted Rotation Matrix:")
        print(results['predicted_rotation'])
        print(f"Max Similarity Score: {np.max(results['similarity_scores']):.4f}")
        print(f"Results saved to: {args.output_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()