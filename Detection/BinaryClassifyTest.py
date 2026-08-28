"""
=============================================================================
NYA AI Studio - 工業二分類 (OK / NG) 快速測試與推論類
提供單圖推論、批次檢測、良率統計、混淆矩陣評估與可視化標籤繪製
=============================================================================
"""

import os
import sys
import time
import glob
import cv2
import numpy as np
from ultralytics import YOLO

# 確定當前目錄與專案根目錄
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class BinaryClassifier:
    """
    工業二分類 (OK / NG) 測試與推論核心類
    
    支援功能:
      1. 單圖 / 多圖記憶體內快速推論 (predict / predict_single)
      2. 整個資料夾批次檢測與良率統計 (predict_batch / predict_folder)
      3. 驗證集良品/不良品資料夾評估與混淆矩陣 (evaluate)
      4. 影像可視化標記與儲存 (predict_and_visualize)
    """

    def __init__(
        self,
        model_path=None,
        conf_threshold=0.5,
        device=None,
        imgsz=512,
        ok_label="OK",
        ng_label="NG"
    ):
        """
        初始化二分類模型
        
        :param model_path: 模型權重路徑 (.pt)。若為 None 則自動搜尋最新訓練完成之權重。
        :param conf_threshold: 判定閾值 (預設 0.5)
        :param device: 運算裝置 ('0', 'cuda', 'cpu'，若 None 則自動偵測)
        :param imgsz: 輸入影像解析度 (預設 512，亦可為 224 或自訂尺寸)
        :param ok_label: 良品標籤名稱 (預設 "OK")
        :param ng_label: 不良品標籤名稱 (預設 "NG")
        """
        if model_path is None or not os.path.exists(str(model_path)):
            model_path = self._auto_find_best_weight()

        self.model_path = str(model_path)
        self.conf_threshold = float(conf_threshold)
        self.device = device
        self.imgsz = int(imgsz)
        self.ok_label = str(ok_label).upper()
        self.ng_label = str(ng_label).upper()

        print(f"[BinaryClassifier] 載入二分類模型: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # 讀取模型內部類別名稱字典
        self.names = getattr(self.model, "names", {0: "OK", 1: "NG"})
        if isinstance(self.names, (list, tuple)):
            self.names = {i: name for i, name in enumerate(self.names)}
        print(f"[BinaryClassifier] 類別對應表: {self.names}")

    def _auto_find_best_weight(self):
        """
        自動搜尋專案中最新訓練完成的分類 best.pt 權重
        """
        candidates = []
        
        # 1. 搜尋 runs/classify/*/weights/best.pt
        classify_runs = os.path.join(PROJECT_ROOT, "runs", "classify")
        if os.path.exists(classify_runs):
            for root, _, files in os.walk(classify_runs):
                if "best.pt" in files:
                    pt_path = os.path.join(root, "best.pt")
                    candidates.append((os.path.getmtime(pt_path), pt_path))

        # 2. 搜尋 weights/*.pt
        weights_dir = os.path.join(PROJECT_ROOT, "weights")
        if os.path.exists(weights_dir):
            for f in os.listdir(weights_dir):
                if f.endswith(".pt") and ("cls" in f.lower() or "resnet" in f.lower()):
                    pt_path = os.path.join(weights_dir, f)
                    candidates.append((os.path.getmtime(pt_path), pt_path))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_found = candidates[0][1]
            print(f"[BinaryClassifier] 自動選用最新訓練權重: {best_found}")
            return best_found

        default_fallback = os.path.join(PROJECT_ROOT, "runs", "classify", "train-5", "weights", "best.pt")
        return default_fallback

    def predict(self, source, imgsz=None, verbose=False):
        """
        對單一影像或輸入進行二分類推論
        
        :param source: 圖片路徑 (str)、OpenCV BGR 矩陣 (np.ndarray) 或 PIL Image
        :param imgsz: 推論尺寸 (預設使用初始化設定)
        :param verbose: 是否輸出詳細日誌
        :return: dict 包含 {
            'is_ok': bool,          # True: 良品 (OK), False: 不良品 (NG)
            'label': str,           # 'OK' 或 'NG'
            'confidence': float,    # 頂層信心度 (0.0 ~ 1.0)
            'top1_index': int,      # Top-1 類別索引
            'probs': dict,          # 各類別機率分佈 {'OK': 0.99, 'NG': 0.01}
            'speed_ms': float       # 推論耗時 (毫秒)
        }
        """
        sz = imgsz if imgsz is not None else self.imgsz
        kwargs = {"imgsz": sz, "verbose": verbose}
        if self.device is not None:
            kwargs["device"] = self.device

        t0 = time.time()
        results = self.model.predict(source, **kwargs)
        elapsed_ms = (time.time() - t0) * 1000.0

        res = results[0]
        probs_dict = {}

        if hasattr(res, "probs") and res.probs is not None:
            top1_idx = int(res.probs.top1)
            top1_conf = float(res.probs.top1conf)
            top1_name = str(self.names.get(top1_idx, top1_idx)).upper()

            # 建立各類機率字典
            raw_data = res.probs.data.cpu().numpy()
            for idx, prob in enumerate(raw_data):
                c_name = str(self.names.get(idx, idx)).upper()
                probs_dict[c_name] = float(prob)
        else:
            top1_idx = 0
            top1_conf = 1.0
            top1_name = self.ok_label
            probs_dict = {self.ok_label: 1.0, self.ng_label: 0.0}

        # 判定是否為 OK
        is_ok = (top1_name == self.ok_label)

        # 若良品信心度未達門檻，則視為 NG
        if is_ok and top1_conf < self.conf_threshold:
            is_ok = False
            top1_name = self.ng_label

        return {
            "is_ok": is_ok,
            "label": top1_name,
            "confidence": top1_conf,
            "top1_index": top1_idx,
            "probs": probs_dict,
            "speed_ms": elapsed_ms
        }

    def predict_single(self, source, imgsz=None):
        """ predict 的別名方法 """
        return self.predict(source, imgsz=imgsz)

    def predict_batch(self, images_or_folder, batch_size=16, imgsz=None):
        """
        批次對圖片列表或資料夾進行二分類檢測與良率統計
        
        :param images_or_folder: 圖片路徑列表 (list) 或 資料夾路徑 (str)
        :param batch_size: 批次大小
        :param imgsz: 推論尺寸
        :return: (list_of_results, summary_dict)
        """
        img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
        image_paths = []

        if isinstance(images_or_folder, str):
            if os.path.isdir(images_or_folder):
                for f in os.listdir(images_or_folder):
                    if f.lower().endswith(img_exts):
                        image_paths.append(os.path.join(images_or_folder, f))
            elif os.path.isfile(images_or_folder):
                image_paths = [images_or_folder]
        elif isinstance(images_or_folder, (list, tuple)):
            image_paths = list(images_or_folder)

        total = len(image_paths)
        if total == 0:
            print("[BinaryClassifier] ⚠️ 未找到任何待檢測影像！")
            return [], {}

        print(f"[BinaryClassifier] 開始批次檢測: 共 {total} 張圖片...")
        results_list = []
        ok_count = 0
        ng_count = 0
        total_time_ms = 0.0

        for path in image_paths:
            out = self.predict(path, imgsz=imgsz, verbose=False)
            out["path"] = path
            out["filename"] = os.path.basename(path)
            results_list.append(out)

            if out["is_ok"]:
                ok_count += 1
            else:
                ng_count += 1
            total_time_ms += out["speed_ms"]

        yield_rate = (ok_count / total * 100.0) if total > 0 else 0.0
        avg_time = (total_time_ms / total) if total > 0 else 0.0

        summary = {
            "total_images": total,
            "ok_count": ok_count,
            "ng_count": ng_count,
            "yield_rate_pct": yield_rate,
            "avg_time_ms": avg_time,
            "fps": (1000.0 / avg_time) if avg_time > 0 else 0.0
        }

        print("=" * 60)
        print("  📊【批次二分類檢測報告】")
        print(f"  總檢測數量 : {total} 張")
        print(f"  🟢 良品 (OK): {ok_count} 張 ({yield_rate:.2f}%)")
        print(f"  🔴 不良品(NG): {ng_count} 張 ({100.0 - yield_rate:.2f}%)")
        print(f"  ⚡ 平均耗時 : {avg_time:.2f} ms/張 (約 {summary['fps']:.1f} FPS)")
        print("=" * 60)

        return results_list, summary

    def evaluate(self, ok_folder, ng_folder, imgsz=None):
        """
        對標準驗證集（良品資料夾與不良品資料夾）進行指標評估與混淆矩陣計算
        
        :param ok_folder: 真實良品 (OK) 影像資料夾
        :param ng_folder: 真實不良品 (NG) 影像資料夾
        :return: dict 評估指標 (Accuracy, Precision, Recall, F1, TP, TN, FP, FN)
        """
        print("\n" + "=" * 60)
        print("  🧪【二分類模型效能評估 (Confusion Matrix)】")
        print(f"  良品資料夾 (Ground Truth OK): {ok_folder}")
        print(f"  不良資料夾 (Ground Truth NG): {ng_folder}")
        print("=" * 60)

        img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
        
        ok_files = [os.path.join(ok_folder, f) for f in os.listdir(ok_folder) if f.lower().endswith(img_exts)] if os.path.exists(ok_folder) else []
        ng_files = [os.path.join(ng_folder, f) for f in os.listdir(ng_folder) if f.lower().endswith(img_exts)] if os.path.exists(ng_folder) else []

        tp = 0 # 真實 OK 預測為 OK
        fn = 0 # 真實 OK 預測為 NG (過殺 / 誤判)
        tn = 0 # 真實 NG 預測為 NG
        fp = 0 # 真實 NG 預測為 OK (漏檢 / 漏判)

        # 評估 OK 集
        for p in ok_files:
            res = self.predict(p, imgsz=imgsz)
            if res["is_ok"]:
                tp += 1
            else:
                fn += 1

        # 評估 NG 集
        for p in ng_files:
            res = self.predict(p, imgsz=imgsz)
            if not res["is_ok"]:
                tn += 1
            else:
                fp += 1

        total_samples = len(ok_files) + len(ng_files)
        acc = ((tp + tn) / total_samples * 100.0) if total_samples > 0 else 0.0
        precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics = {
            "total": total_samples,
            "TP": tp, "FN": fn, "TN": tn, "FP": fp,
            "accuracy_pct": acc,
            "precision_pct": precision,
            "recall_pct": recall,
            "f1_score": f1
        }

        print(f"\n  [混淆矩陣統計]")
        print(f"  ┌───────────────────────┬──────────────┬──────────────┐")
        print(f"  │        真實 \\ 預測    │   預測 OK    │   預測 NG    │")
        print(f"  ├───────────────────────┼──────────────┼──────────────┤")
        print(f"  │  真實 OK (良品: {len(ok_files):>4}) │ TP: {tp:>8} │ FN: {fn:>8} │ (過殺率: {fn/len(ok_files)*100:.1f}%)" if ok_files else "  │  真實 OK               │ TP: 0        │ FN: 0        │")
        print(f"  │  真實 NG (不良: {len(ng_files):>4}) │ FP: {fp:>8} │ TN: {tn:>8} │ (漏檢率: {fp/len(ng_files)*100:.1f}%)" if ng_files else "  │  真實 NG               │ FP: 0        │ TN: 0        │")
        print(f"  └───────────────────────┴──────────────┴──────────────┘")
        print(f"  🎯 總體準確率 (Accuracy) : {acc:.2f}%")
        print(f"  📌 精確率 (Precision)    : {precision:.2f}%")
        print(f"  🔍 召回率 (Recall)       : {recall:.2f}%")
        print(f"  ⭐ F1-Score              : {f1:.2f}%")
        print("=" * 60 + "\n")

        return metrics

    def predict_and_visualize(self, image_input, save_path=None, show=False, imgsz=None):
        """
        執行推論並在影像左上角繪製綠色 [OK] 或紅色 [NG] 狀態 Badge
        
        :param image_input: 圖片路徑 (str) 或 OpenCV BGR 影像 (np.ndarray)
        :param save_path: 儲存路徑 (若為 None 則不儲存)
        :param show: 是否彈出視窗顯示
        :param imgsz: 推論尺寸
        :return: (annotated_bgr_img, pred_result_dict)
        """
        # 讀取影像
        if isinstance(image_input, str):
            data = np.fromfile(image_input, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            raise ValueError("image_input 必須為檔案路徑或 OpenCV numpy 陣列")

        pred = self.predict(img, imgsz=imgsz)
        is_ok = pred["is_ok"]
        label = pred["label"]
        conf = pred["confidence"]

        # 設定顏色 (BGR: 綠色良品, 紅色不良品)
        color = (0, 200, 0) if is_ok else (0, 0, 230)
        text = f"[{label}] {conf*100:.1f}% ({pred['speed_ms']:.1f}ms)"

        # 繪製半透明背景 Badge
        h, w = img.shape[:2]
        font_scale = max(0.6, min(w, h) / 600.0)
        thickness = max(1, int(font_scale * 2))
        (t_w, t_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        badge_x1, badge_y1 = 12, 12
        badge_x2, badge_y2 = badge_x1 + t_w + 16, badge_y1 + t_h + 16

        overlay = img.copy()
        cv2.rectangle(overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 2)

        # 繪製文字
        cv2.putText(
            img, text,
            (badge_x1 + 8, badge_y1 + t_h + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            ext = os.path.splitext(save_path)[1] or ".jpg"
            _, buf = cv2.imencode(ext, img)
            buf.tofile(save_path)
            print(f"[BinaryClassifier] 已保存判定影像至: {save_path}")

        if show:
            cv2.imshow("Binary Classifier Result", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img, pred


# 別名相容
BinaryClassificationTest = BinaryClassifier
BinaryClassifyTest = BinaryClassifier
BinaryClassifyDetector = BinaryClassifier


# ── 獨立執行示範 ──────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 啟動 NYA 工業二分類 (OK / NG) 測試類...")
    
    # 1. 建立分類器實例 (自動載入最新訓練的分類權重)
    classifier = BinaryClassifier(
        conf_threshold=0.5,
        imgsz=512
    )

    # 2. 測試驗證集整體評估
    val_ok_dir = os.path.join(PROJECT_ROOT, "NYA_Project", "dataset", "val", "OK")
    val_ng_dir = os.path.join(PROJECT_ROOT, "NYA_Project", "dataset", "val", "NG")

    if os.path.exists(val_ok_dir) and os.path.exists(val_ng_dir):
        # 進行完整混淆矩陣評估
        classifier.evaluate(ok_folder=val_ok_dir, ng_folder=val_ng_dir)

        # 進行單張可視化推論測試
        test_samples = [os.path.join(val_ok_dir, f) for f in os.listdir(val_ok_dir) if f.endswith(('.jpg', '.png'))]
        if test_samples:
            sample_img = test_samples[0]
            out_img, res = classifier.predict_and_visualize(
                sample_img,
                save_path=os.path.join(PROJECT_ROOT, "runs", "classify_test_preview.jpg")
            )
            print(f"單圖推論測試結果: {res}")
    else:
        print("未找到驗證集資料夾，請指定圖片路徑進行推論測試。")
