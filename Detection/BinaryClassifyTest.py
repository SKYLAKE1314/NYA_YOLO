import os
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(CURRENT_DIR, "..", "weight", "best.pt")

class BinaryClassifier:
    def __init__(self, model_path=DEFAULT_MODEL, device=None):
        if model_path is None or not os.path.exists(model_path):
            weight_dir = os.path.join(CURRENT_DIR, "..", "weight")
            model_path = os.path.join(weight_dir, "best.pt")
            if not os.path.exists(model_path) and os.path.exists(weight_dir):
                for f in os.listdir(weight_dir):
                    if f.endswith(".pt"):
                        model_path = os.path.join(weight_dir, f)
                        break
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
    classifier = BinaryClassifier()

    img_path = r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\NYA_Project\dataset\val\OK\工位1_工位1_26-07-20-153712-176_0045_二分类检测_16.jpg"
    label, conf, res = classifier.predict(img_path)

    print(f"result: [{label}] (準度: {conf * 100:.2f}%)")
    res.show()
