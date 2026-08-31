import os
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(CURRENT_DIR, "..", "weight", "best.pt")

model = YOLO(model_path)
results = model.predict(r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\NYA_Project\dataset\val\OK\工位1_工位1_26-07-20-153712-176_0045_二分类检测_16.jpg", imgsz=512)
results[0].show()
