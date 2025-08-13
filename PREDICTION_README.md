# NOPE Pose Prediction Scripts

This directory contains scripts for predicting pose (rotation) between reference and query images using the NOPE (Novel Object Pose Estimation) framework.

## Available Scripts

### 1. `nope_predict.py` - Main Prediction Script

The comprehensive script that supports both pretrained NOPE models and simplified fallback estimation.

#### Basic Usage
```bash
# Use with test images (automatically generated)
python nope_predict.py --create_test

# Use with your own images
python nope_predict.py --ref_image path/to/reference.jpg --query_image path/to/query.jpg

# Use with pretrained model (if available)
python nope_predict.py --ref_image ref.jpg --query_image query.jpg --checkpoint model.ckpt --config config.yaml
```

#### Arguments
- `--ref_image`: Path to reference image
- `--query_image`: Path to query image  
- `--checkpoint`: Path to pretrained NOPE model checkpoint (optional)
- `--config`: Path to NOPE model configuration file (optional)
- `--output_dir`: Output directory for results (default: ./pose_results)
- `--create_test`: Create test images with known 45° rotation

### 2. `simple_predict.py` - Simplified Version

A lightweight script that demonstrates the core pose estimation concepts without requiring the full NOPE framework.

#### Basic Usage
```bash
# Create sample images and run prediction
python simple_predict.py --create_samples

# Use with custom images
python simple_predict.py --ref_image ref.jpg --query_image query.jpg
```

## Output Files

Both scripts generate the following outputs in the specified output directory:

### Results Files
- `estimated_rotation.npy`: Predicted 3x3 rotation matrix (NumPy format)
- `pose_estimation_results.npz`: Comprehensive results including:
  - Predicted rotation matrix
  - Confidence scores
  - Template similarity scores
  - Top-5 matching templates

### Visualizations
- `pose_estimation_visualization.png`: Comprehensive visualization showing:
  - Reference and query images
  - Rotation matrix heatmap
  - Template similarity scores
  - Top-5 matches
  - Rotation analysis (angle, axis, confidence)

## Understanding the Results

### Rotation Matrix
The predicted rotation matrix R represents the rotation from reference to query pose:
- R is a 3×3 orthogonal matrix
- Rotation angle: `θ = arccos((trace(R) - 1) / 2)`
- For small rotations, the matrix is close to identity

### Confidence Metrics
- **Confidence Score**: Similarity score of the best matching template (0-1)
- **Confidence Margin**: Difference between best score and median score
- Higher values indicate more confident predictions

### Template Matching
The system uses multiple template poses covering different viewpoints:
- Templates span azimuth angles (0-360°) and elevation angles
- Best match indicates the most likely relative pose
- Top-5 matches show alternative hypotheses

## Example Results

For a 45° rotation between reference and query images:
```
Rotation Angle: ~45.0°
Confidence: 0.8-0.9 (high confidence)
Best Template: Index corresponding to ~45° azimuth rotation
```

## Requirements

### Basic Requirements (for simple_predict.py)
- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib
- PIL (Pillow)
- einops

### Full Requirements (for nope_predict.py with pretrained models)
All basic requirements plus:
- hydra-core
- omegaconf
- pytorch-lightning
- wandb
- The full NOPE codebase (src/ directory)

## Notes

1. **Without Pretrained Models**: The scripts use simplified feature extraction and template matching for demonstration purposes.

2. **With Pretrained Models**: When a valid checkpoint is provided, the scripts use the full NOPE architecture with trained encoders for more accurate pose estimation.

3. **Image Formats**: Supports common image formats (PNG, JPG, etc.). Images are automatically resized to 256×256 pixels.

4. **Coordinate System**: Rotations follow the standard computer vision convention (Y-axis up, Z-axis forward).

## Troubleshooting

### Import Errors
If you get import errors related to NOPE modules:
```bash
# Make sure you're in the NOPE repository root
cd /path/to/nope
python nope_predict.py ...
```

### Memory Issues
For large numbers of templates or high-resolution images:
- Reduce template count in the script
- Use CPU instead of GPU if CUDA memory is limited
- Process images at lower resolution

### Visualization Issues
If matplotlib figures don't display properly:
- Check that you have a display environment
- Results are always saved as PNG files regardless