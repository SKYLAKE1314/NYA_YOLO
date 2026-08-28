import os
import cv2
import tempfile
from ultralytics import YOLO

# 載入訓練好的二分類模型權重檔 (.pt)
model = YOLO(r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\runs\classify\train-5\weights\best.pt")

# 進行推論
results = model.predict(r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\NYA_Project\dataset\val\OK\工位1_工位1_26-07-20-153712-176_0045_二分类检测_16.jpg", imgsz=512)

# 儲存繪製標籤圖並使用 Windows 內建「相片」軟體開啟
temp_path = os.path.join(tempfile.gettempdir(), "nya_classify_result.jpg")
cv2.imwrite(temp_path, results[0].plot())
os.startfile(temp_path)
