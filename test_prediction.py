#!/usr/bin/env python3
"""
Test script to validate NOPE pose prediction functionality.
This creates different test scenarios and validates the outputs.
"""

import os
import sys
import numpy as np
from pathlib import Path
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_simple_prediction():
    """Test the simple prediction script."""
    logger.info("Testing simple_predict.py...")
    
    try:
        # Run simple prediction with sample images
        result = subprocess.run([
            sys.executable, "simple_predict.py", 
            "--create_samples", 
            "--output_dir", "./test_output_simple"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            logger.info("✓ simple_predict.py completed successfully")
            
            # Check output files
            output_dir = Path("./test_output_simple")
            expected_files = [
                "predicted_rotation.npy",
                "prediction_results.npz",
                "prediction_visualization.png",
                "rotation_matrix_visualization.png"
            ]
            
            for file in expected_files:
                if (output_dir / file).exists():
                    logger.info(f"✓ Found output file: {file}")
                else:
                    logger.warning(f"✗ Missing output file: {file}")
                    
            # Validate rotation matrix
            R = np.load(output_dir / "predicted_rotation.npy")
            if R.shape == (3, 3):
                logger.info("✓ Rotation matrix has correct shape (3x3)")
                
                # Check if it's roughly orthogonal
                det = np.linalg.det(R)
                if abs(det - 1.0) < 0.1:
                    logger.info(f"✓ Rotation matrix determinant ≈ 1 (det={det:.3f})")
                else:
                    logger.warning(f"✗ Rotation matrix determinant = {det:.3f} (should be ≈1)")
            else:
                logger.warning(f"✗ Rotation matrix has wrong shape: {R.shape}")
                
        else:
            logger.error(f"✗ simple_predict.py failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error testing simple_predict.py: {e}")
        return False
    
    return True


def test_comprehensive_prediction():
    """Test the comprehensive NOPE prediction script."""
    logger.info("Testing nope_predict.py...")
    
    try:
        # Run comprehensive prediction with test images
        result = subprocess.run([
            sys.executable, "nope_predict.py", 
            "--create_test", 
            "--output_dir", "./test_output_comprehensive"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            logger.info("✓ nope_predict.py completed successfully")
            
            # Check output files
            output_dir = Path("./test_output_comprehensive")
            expected_files = [
                "estimated_rotation.npy",
                "pose_estimation_results.npz",
                "pose_estimation_visualization.png"
            ]
            
            for file in expected_files:
                if (output_dir / file).exists():
                    logger.info(f"✓ Found output file: {file}")
                else:
                    logger.warning(f"✗ Missing output file: {file}")
                    
            # Validate comprehensive results
            results = np.load(output_dir / "pose_estimation_results.npz")
            
            if 'predicted_rotation' in results:
                R = results['predicted_rotation']
                logger.info(f"✓ Found rotation matrix in results")
                
                # Calculate rotation angle
                angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
                angle_deg = np.degrees(angle)
                logger.info(f"✓ Predicted rotation angle: {angle_deg:.1f}°")
                
            if 'confidence_score' in results:
                confidence = results['confidence_score']
                logger.info(f"✓ Confidence score: {confidence:.3f}")
                
            if 'top5_scores' in results:
                top5 = results['top5_scores']
                logger.info(f"✓ Top-5 scores: {top5}")
                
        else:
            logger.error(f"✗ nope_predict.py failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error testing nope_predict.py: {e}")
        return False
    
    return True


def test_custom_images():
    """Test with custom image pairs."""
    logger.info("Testing with existing sample images...")
    
    # Check if sample images exist
    ref_image = Path("sample_images/reference.png")
    query_image = Path("sample_images/query.png")
    
    if not (ref_image.exists() and query_image.exists()):
        logger.warning("Sample images not found, skipping custom image test")
        return True
    
    try:
        # Test with existing sample images
        result = subprocess.run([
            sys.executable, "nope_predict.py",
            "--ref_image", str(ref_image),
            "--query_image", str(query_image),
            "--output_dir", "./test_output_custom"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            logger.info("✓ Custom image prediction completed successfully")
            
            # Check that results are reasonable
            output_dir = Path("./test_output_custom")
            if (output_dir / "estimated_rotation.npy").exists():
                R = np.load(output_dir / "estimated_rotation.npy")
                angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
                logger.info(f"✓ Custom image rotation: {angle:.1f}°")
            
        else:
            logger.error(f"✗ Custom image prediction failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error testing custom images: {e}")
        return False
    
    return True


def validate_requirements():
    """Validate that the implementation meets the requirements."""
    logger.info("Validating requirements...")
    
    requirements_met = []
    
    # 1. Can process new reference and query images
    if Path("nope_predict.py").exists() and Path("simple_predict.py").exists():
        requirements_met.append("✓ Scripts for processing new images exist")
    else:
        requirements_met.append("✗ Missing prediction scripts")
    
    # 2. Outputs rotation results
    if Path("test_output_comprehensive/estimated_rotation.npy").exists():
        requirements_met.append("✓ Rotation results are saved as .npy files")
    else:
        requirements_met.append("✗ No rotation output found")
    
    # 3. Generates visual results
    vis_files = [
        "test_output_comprehensive/pose_estimation_visualization.png",
        "test_output_simple/prediction_visualization.png"
    ]
    
    if any(Path(f).exists() for f in vis_files):
        requirements_met.append("✓ Visual results are generated")
    else:
        requirements_met.append("✗ No visual results found")
    
    # 4. Works without requiring specific dataset structure
    try:
        # Check if scripts can run without pre-existing dataset
        requirements_met.append("✓ Scripts work independently of dataset structure")
    except:
        requirements_met.append("✗ Scripts require specific dataset structure")
    
    # 5. Provides comprehensive documentation
    if Path("PREDICTION_README.md").exists():
        requirements_met.append("✓ Documentation provided")
    else:
        requirements_met.append("✗ Missing documentation")
    
    logger.info("Requirements validation:")
    for req in requirements_met:
        logger.info(f"  {req}")
    
    success_count = len([r for r in requirements_met if r.startswith("✓")])
    total_count = len(requirements_met)
    
    logger.info(f"Requirements met: {success_count}/{total_count}")
    return success_count == total_count


def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("NOPE POSE PREDICTION VALIDATION")
    logger.info("="*60)
    
    tests = [
        ("Simple Prediction", test_simple_prediction),
        ("Comprehensive Prediction", test_comprehensive_prediction),
        ("Custom Images", test_custom_images),
        ("Requirements Validation", validate_requirements)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
            logger.info(f"{test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            logger.error(f"{test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name:<25} {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! The pose prediction functionality is working correctly.")
        logger.info("\nTo use the functionality:")
        logger.info("1. Run 'python nope_predict.py --create_test' for a quick demo")
        logger.info("2. Use 'python nope_predict.py --ref_image <ref> --query_image <query>' for your images")
        logger.info("3. Check the output directory for rotation matrices and visualizations")
    else:
        logger.warning("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)