"""
=============================================================================
NYA AI Studio - 工業二分類即時監控與 HTTP 主站伺服器 (ClassifyTool)
=============================================================================
"""
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_DIR = os.path.join(CURRENT_DIR, "Detection")
if DETECTION_DIR not in sys.path:
    sys.path.insert(0, DETECTION_DIR)

from Detection.ClassifyTool import main

if __name__ == "__main__":
    main()
