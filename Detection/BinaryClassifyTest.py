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
    classifier = BinaryClassifier(r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\runs\classify\train-5\weights\best.pt")

    img_path = r"Z:\VisionTek\X6AA\ScreenShot_2026-08-28_144810_088.png"
    label, conf, res = classifier.predict(img_path)

    print(f"result: [{label}] (準度: {conf * 100:.2f}%)")
    res.show()
