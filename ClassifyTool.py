import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_DIR = os.path.join(CURRENT_DIR, "Detection")
if DETECTION_DIR not in sys.path:
    sys.path.insert(0, DETECTION_DIR)

from Detection.ClassifyTool import ClassifyTool

if __name__ == "__main__":
    tool = ClassifyTool()
    tool.run()
