import os
import cv2
import tempfile
from ultralytics import YOLO

class BinaryClassifier:
    def __init__(self, model_path=r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\runs\classify\train-5\weights\best.pt", device=None):
        self.model = YOLO(model_path)
        self.device = device

    def predict(self, source, imgsz=512):
        kwargs = {"imgsz": imgsz, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model.predict(source, **kwargs)
        probs = results[0].probs
        label = results[0].names[int(probs.top1)]
        conf = float(probs.top1conf)
        return label, conf, results[0]

    def show(self, result):
        # 儲存判定標籤圖並使用 Windows 內建相片軟體開啟
        temp_img_path = os.path.join(tempfile.gettempdir(), "nya_classify_result.jpg")
        cv2.imwrite(temp_img_path, result.plot())
        os.startfile(temp_img_path)


if __name__ == "__main__":
    # 載入二分類模型
    classifier = BinaryClassifier(r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\runs\classify\train-5\weights\best.pt")

    # 進行推論
    img_path = r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\NYA_Project\dataset\val\OK\工位1_工位1_26-07-20-153712-176_0045_二分类检测_16.jpg"
    label, conf, res = classifier.predict(img_path)

    print(f"分類判定: [{label}] (信心度: {conf * 100:.2f}%)")

    # 調用 Windows 內建相片軟體開啟查看
    classifier.show(res)
