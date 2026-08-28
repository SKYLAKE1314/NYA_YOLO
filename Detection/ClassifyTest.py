import cv2
from ultralytics import YOLO

# 載入訓練好的二分類模型權重檔 (.pt)
model = YOLO(r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\runs\classify\train-5\weights\best.pt")

# 進行推論
results = model.predict(r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\NYA_Project\dataset\val\OK\工位1_工位1_26-07-20-153712-176_0045_二分类检测_16.jpg", imgsz=512)

# 顯示結果圖像視窗 (按任意鍵關閉視窗)
cv2.imshow("Binary Classification Result", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
