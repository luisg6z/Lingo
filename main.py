"""
MagicboARd - Launcher.
Delegates to the real entry point in src/main.py so that `python main.py` still works.
"""
import os
import runpy
import sys

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    src_main = os.path.join(project_root, "src", "main.py")
    runpy.run_path(src_main, run_name="__main__")
