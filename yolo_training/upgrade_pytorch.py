# -*- coding: utf-8 -*-
"""
Script to upgrade PyTorch to a compatible version for Ultralytics
Run this to fix the 'torch._utils' AttributeError
"""
import subprocess
import sys

print("=" * 60)
print("Upgrading PyTorch for Ultralytics Compatibility")
print("=" * 60)

print("\nThis will upgrade PyTorch, torchvision, and torchaudio")
print("to versions compatible with Ultralytics 8.4.6\n")

try:
    # Upgrade to latest compatible versions
    print("Installing/upgrading PyTorch packages...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "--upgrade", 
        "torch", "torchvision", "torchaudio"
    ], stdout=sys.stdout, stderr=sys.stderr)
    
    print("\n" + "=" * 60)
    print("✓ Packages upgraded successfully!")
    print("=" * 60)
    print("\nIMPORTANT: Please restart your Jupyter kernel now!")
    print("1. Go to Kernel -> Restart Kernel in Jupyter")
    print("2. Re-run all cells from the beginning")
    print("\nThis is required for the changes to take effect.")
    
except subprocess.CalledProcessError as e:
    print("\n" + "=" * 60)
    print("✗ Error upgrading packages")
    print("=" * 60)
    print(f"Error code: {e.returncode}")
    print("\nTry running manually in terminal:")
    print("  pip install --upgrade torch torchvision torchaudio")
    sys.exit(1)

except Exception as e:
    print(f"\nUnexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


