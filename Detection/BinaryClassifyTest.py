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


if __name__ == "__main__":
    # 載入二分類模型
    classifier = BinaryClassifier(r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\runs\classify\train-5\weights\best.pt")

    # 進行推論
    img_path = r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\NYA_Project\dataset\val\OK\工位1_工位1_26-07-20-153712-176_0045_二分类检测_16.jpg"
    label, conf, res = classifier.predict(img_path)

    print(f"分類判定: [{label}] (信心度: {conf * 100:.2f}%)")

    # 顯示結果圖
    res.show()
