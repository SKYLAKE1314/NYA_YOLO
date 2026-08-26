"""
ClassifyTrain.py — 圖像二分類 / 多分類模型訓練腳本 (支援 ResNet 與 YOLO-cls)
支援 ResNet 各種規模版本 (ResNet-18 / ResNet-50 / ResNet-101) 與 YOLO 分類權重
"""

from ultralytics import YOLO

# 1. 選擇模型 (支援 ResNet 系列與 YOLO-cls 系列各尺寸版本)
#   - ResNet-18 (二分類/AOI工業檢測推薦，輕量高速): "yolo11-cls-resnet18.yaml"
#   - ResNet-50 (高精度分類): "yolov8-cls-resnet50.yaml"
#   - ResNet-101 (深層複雜特徵): "yolov8-cls-resnet101.yaml"
#   - YOLO11-cls (預訓練模型): "yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt"
#   - YOLOv8-cls (經典系列): "yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt"
model = YOLO("yolo11-cls-resnet18.yaml")

# 2. 開始訓練
# 資料集結構標準格式範例：
#   dataset/
#     ├── train/
#     │    ├── OK/
#     │    └── NG/
#     └── val/
#          ├── OK/
#          └── NG/
results = model.train(
    data=r"Datasets/classify",  # 分類資料集根目錄或 config
    epochs=100,
    imgsz=224,                  # 分類標準大小 (224 或 256)
    batch=32,
    device=0,                   # 0 為 GPU 加速，或 'cpu'
    workers=4,
    optimizer="auto",
    cos_lr=True,
    patience=20,
    save=True
)
