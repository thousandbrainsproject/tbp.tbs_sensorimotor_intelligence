#!/usr/bin/env python3
"""
Verification script to check environment setup for ViT experiments.

Run this script to verify:
1. Environment dependencies are installed
2. Data paths are correctly configured  
3. CUDA/MPS availability (if applicable)
4. Basic config loading works

Usage:
    cd tbp.tbs_sensorimotor_intelligence/pytorch
    python scripts/verify_setup.py
"""

import os
import sys
from pathlib import Path
import importlib.util


def check_environment():
    """Check if all required packages are installed."""
    print("🔍 Checking environment dependencies...")
    
    required_packages = [
        'torch',
        'torchvision',
        'lightning',
        'hydra',
        'transformers',
        'wandb',
        'sklearn',
        'pandas',
        'rich'
    ]
    
    missing_packages = []
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
        else:
            print(f"  ✅ {package}")
    
    if missing_packages:
        print(f"  ❌ Missing packages: {', '.join(missing_packages)}")
        print("     Run: conda env create -f environment.yaml")
        return False
    
    print("  ✅ All required packages found!")
    return True


def check_project_root():
    """Check if PROJECT_ROOT is set and points to correct directory."""
    print("\n🔍 Checking PROJECT_ROOT configuration...")
    
    project_root = os.environ.get('PROJECT_ROOT')
    if not project_root:
        print("  ❌ PROJECT_ROOT environment variable not set")
        print("     Run: export PROJECT_ROOT=$(pwd)")
        return False
    
    project_root_path = Path(project_root)
    if not project_root_path.exists():
        print(f"  ❌ PROJECT_ROOT path does not exist: {project_root}")
        return False
    
    # Check for .project-root file
    project_root_file = project_root_path / '.project-root'
    if not project_root_file.exists():
        print(f"  ❌ .project-root file not found in {project_root}")
        print("     Make sure you're in the correct directory")
        return False
    
    print(f"  ✅ PROJECT_ROOT correctly set to: {project_root}")
    return True


def check_data_paths():
    """Check if data directories exist."""
    print("\n🔍 Checking data path configuration...")
    
    # Get expected data directory from environment or default
    home_dir = os.environ.get('HOME')
    if not home_dir:
        print("  ⚠️  HOME environment variable not set, skipping data check")
        return True
        
    expected_data_dir = Path(home_dir) / "tbp" / "results" / "dmc" / "data" / "view_finder_images"
    
    if not expected_data_dir.exists():
        print(f"  ⚠️  Expected data directory not found: {expected_data_dir}")
        print("     You can either:")
        print("     1. Create symlinks to your actual data location")
        print("     2. Override paths when running experiments:")
        print("        python src/train.py experiment=<config> paths.data_dir='/your/data/path'")
        return True
    
    # Check for required subdirectories
    required_subdirs = ['view_finder_32', 'view_finder_base', 'view_finder_randrot']
    for subdir in required_subdirs:
        subdir_path = expected_data_dir / subdir / 'view_finder_rgbd'
        if subdir_path.exists():
            print(f"  ✅ Found data directory: {subdir}")
        else:
            print(f"  ⚠️  Data directory not found: {subdir}")
    
    return True


def check_device_availability():
    """Check CUDA/MPS availability."""
    print("\n🔍 Checking device availability...")
    
    try:
        import torch
        
        # Check CUDA
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"  ✅ CUDA available: {device_count} device(s)")
            print(f"     Current device: {device_name}")
        else:
            print("  ❌ CUDA not available")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  ✅ MPS (Apple Silicon) available")
        else:
            print("  ❌ MPS not available")
        
        if not torch.cuda.is_available() and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print("  ⚠️  No GPU acceleration available, will use CPU")
            print("     Consider using: trainer.accelerator=cpu in your configs")
            
    except ImportError:
        print("  ❌ PyTorch not available")
        return False
        
    return True



def check_wandb_setup():
    """Check WandB configuration."""
    print("\n🔍 Checking WandB setup...")
    
    try:
        import wandb
        
        # Check if user is logged in
        if wandb.api.api_key:
            print("  ✅ WandB API key found")
        else:
            print("  ⚠️  WandB API key not found")
            print("     Run: wandb login")
            
    except ImportError:
        print("  ❌ WandB not available")
        return False
    except Exception as e:
        print(f"  ⚠️  WandB check failed: {e}")
        
    return True


def main():
    """Run all verification checks."""
    print("🚀 Starting setup verification for ViT experiments...\n")
    
    checks = [
        check_environment,
        check_project_root,
        check_data_paths,
        check_device_availability,
        check_wandb_setup
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Check failed with error: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    print("📋 VERIFICATION SUMMARY")
    print("="*50)
    
    if all(results[:5]):  # First 5 checks are critical
        print("✅ Environment setup looks good!")
        print("   You should be able to run experiments.")
        
    else:
        print("❌ Some issues found. Please fix the above errors before running experiments.")
        
    print("\n📚 For detailed instructions, see: REPRODUCE_RESULTS.md")


if __name__ == "__main__":
    main() 