from ultralytics import YOLO

# Load a model
#model = YOLO("yolo11n.pt")
# model = YOLO(r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\runs\detect\train3\weights\best.pt")
# model = YOLO(r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\runs\segment\train3\weights\best.pt")
model = YOLO(r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\dist_nuitka\NyaYOLOStudio.dist\runs\segment\train-2\weights\best.pt")

# Export the model
model.export(format="onnx", dynamic=True) # 動態尺寸輸出
# 針對各個平臺加速 onnx openvino engine(TensorRT)