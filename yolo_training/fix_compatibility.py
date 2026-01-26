# -*- coding: utf-8 -*-
"""
Quick script to fix PyTorch/Ultralytics compatibility issues
Run this if you get the '_utils' AttributeError
"""
import subprocess
import sys

print("=" * 60)
print("Fixing PyTorch/Ultralytics Compatibility")
print("=" * 60)

print("\nUpgrading packages...")
print("This may take a few minutes...\n")

try:
    # Upgrade PyTorch and related packages
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "--upgrade", 
        "torch", "torchvision", "torchaudio",
        "ultralytics"
    ])
    
    print("\n" + "=" * 60)
    print("✓ Packages upgraded successfully!")
    print("=" * 60)
    print("\nPlease restart your Jupyter kernel and try again.")
    
except subprocess.CalledProcessError as e:
    print("\n" + "=" * 60)
    print("✗ Error upgrading packages")
    print("=" * 60)
    print(f"Error: {e}")
    print("\nTry running manually:")
    print("  pip install --upgrade torch torchvision torchaudio ultralytics")
    sys.exit(1)

except Exception as e:
    print(f"\nUnexpected error: {e}")
    sys.exit(1)


