# -*- coding: utf-8 -*-
"""
Script to fix OpenCV GUI support issue
Run this if you get "The function is not implemented" error
"""
import subprocess
import sys

print("=" * 60)
print("Fixing OpenCV GUI Support")
print("=" * 60)

print("\nThis will reinstall OpenCV with GUI support...")
print("This may take a few minutes...\n")

try:
    # Uninstall existing OpenCV packages
    print("Step 1: Uninstalling existing OpenCV packages...")
    packages_to_remove = ['opencv-python', 'opencv-contrib-python', 'opencv-python-headless']
    
    for package in packages_to_remove:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "uninstall", 
                "-y", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  ✓ Removed {package}")
        except subprocess.CalledProcessError:
            # Package might not be installed, that's okay
            pass
    
    print("\nStep 2: Installing OpenCV with GUI support...")
    # Install opencv-python with GUI support
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "opencv-python"
    ], stdout=sys.stdout, stderr=sys.stderr)
    
    print("\n" + "=" * 60)
    print("✓ OpenCV reinstalled with GUI support!")
    print("=" * 60)
    print("\nYou can now run the test script:")
    print("  python yolo_training/test_kinect_yolo12.py")
    
except subprocess.CalledProcessError as e:
    print("\n" + "=" * 60)
    print("✗ Error during installation")
    print("=" * 60)
    print(f"Error code: {e.returncode}")
    print("\nTry running manually:")
    print("  pip uninstall opencv-python opencv-contrib-python opencv-python-headless")
    print("  pip install opencv-python")
    sys.exit(1)

except Exception as e:
    print(f"\nUnexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

