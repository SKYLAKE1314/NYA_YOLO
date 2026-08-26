import os
import sys
import shutil
import json
import time
import cv2
import yaml
import torch
import numpy as np
import psutil
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from ultralytics import YOLO

# 確定路徑包含專案根目錄
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from tools.ConvertToLables import (
    JSON2YOLO, XML2YOLO,
    auto_detect_classes, save_classes_list,
    RECURSIVE_SEARCH
)
from tools.seg.JSON2YOLOSeg import JSON2YOLOSeg
from ConfigCreator import create_config


# =========================================================
# Worker: 資料集轉換 & Auto Config & Seg Conversion
# =========================================================
class ConvertWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int)
    finished_signal = Signal(bool, str)

    def __init__(self, *args, **kwargs):
        super().__init__()
        # If single dictionary passed as first argument, unpack it
        if len(args) == 1 and isinstance(args[0], dict):
            kwargs.update(args[0])
            args = ()

        if args:
            self.task_type = args[0] if len(args) > 0 else "detect"
            self.anno_folder = args[1] if len(args) > 1 else ""
            self.image_folder = args[2] if len(args) > 2 else ""
            self.out_folder = args[3] if len(args) > 3 else ""
            self.use_auto = args[4] if len(args) > 4 else True
            raw_classes = args[5] if len(args) > 5 else []
            self.split_ratio = float(args[6]) if len(args) > 6 else 0.2
        else:
            self.task_type = kwargs.get("task_type", "detect")
            self.anno_folder = kwargs.get("anno_folder") or kwargs.get("anno_dir") or ""
            self.image_folder = kwargs.get("image_folder") or kwargs.get("image_dir") or ""
            self.out_folder = kwargs.get("out_folder") or kwargs.get("output_root") or kwargs.get("dataset_dir") or ""
            self.use_auto = kwargs.get("use_auto", kwargs.get("auto_class", True))
            raw_classes = kwargs.get("manual_classes", kwargs.get("class_str", []))
            self.split_ratio = float(kwargs.get("split_ratio", kwargs.get("val_ratio", 0.2)))

        if isinstance(raw_classes, str):
            self.manual_classes = [c.strip() for c in raw_classes.split(",") if c.strip()]
        elif isinstance(raw_classes, (list, tuple)):
            self.manual_classes = list(raw_classes)
        else:
            self.manual_classes = []

    def run(self):
        try:
            self.log_signal.emit(f"開始 [{self.task_type.upper()}] 資料集轉換作業...")
            if not self.anno_folder or not os.path.exists(self.anno_folder):
                self.finished_signal.emit(False, f"標註資料夾不存在: {self.anno_folder}")
                return

            # 若未設定影像資料夾，預設與標註資料夾相同
            if not self.image_folder or not os.path.exists(self.image_folder):
                self.image_folder = self.anno_folder

            # 決策 Classes
            if self.use_auto:
                classes = auto_detect_classes(self.anno_folder, self.anno_folder)
                self.log_signal.emit(f"自動檢測到的類別名單 ({len(classes)}): {classes}")
            else:
                classes = self.manual_classes
                self.log_signal.emit(f"手動指定的類別名單 ({len(classes)}): {classes}")

            if not classes:
                self.finished_signal.emit(False, "類別名單為空，請檢查標註檔案或手動設定類別！")
                return

            # 建立並清理目錄結構 train/images, train/labels, val/images, val/labels
            dataset_root = self.out_folder
            train_images = os.path.join(dataset_root, "train", "images")
            train_labels = os.path.join(dataset_root, "train", "labels")
            val_images = os.path.join(dataset_root, "val", "images")
            val_labels = os.path.join(dataset_root, "val", "labels")

            for p in [train_images, train_labels, val_images, val_labels]:
                if os.path.exists(p):
                    shutil.rmtree(p)
                os.makedirs(p, exist_ok=True)

            # 清理任何殘留的 .cache 快取檔
            for p in [dataset_root, train_labels, val_labels, train_images, val_images]:
                if os.path.exists(p):
                    for cf in os.listdir(p):
                        if cf.endswith(('.cache', '.cache.npy')):
                            try:
                                os.remove(os.path.join(p, cf))
                            except Exception:
                                pass

            classes_txt_path = os.path.join(dataset_root, "classes.txt")
            save_classes_list(classes, classes_txt_path)
            self.log_signal.emit(f"📁 classes.txt 已保存至: {classes_txt_path}")

            # 搜尋標註檔
            anno_files = [f for f in os.listdir(self.anno_folder) if f.lower().endswith(('.json', '.xml'))]
            if not anno_files:
                self.finished_signal.emit(False, "標註資料夾中未找到 .json 或 .xml 標註檔案！")
                return

            total = len(anno_files)
            count = 0

            if self.task_type == 'segment':
                seg_conv = JSON2YOLOSeg(classes=classes, output_dir=train_labels, image_folder=self.image_folder)
                for f in anno_files:
                    if f.lower().endswith('.json'):
                        seg_conv.convert(os.path.join(self.anno_folder, f))
                        base = os.path.splitext(f)[0]
                        self._copy_image(base, train_images)
                    count += 1
                    self.progress_signal.emit(int(count / total * 80))
            else: # detect
                json_conv = JSON2YOLO(classes, output_dir=train_labels, image_folder=self.image_folder)
                xml_conv = XML2YOLO(classes, output_dir=train_labels, image_folder=self.image_folder)
                for f in anno_files:
                    full_p = os.path.join(self.anno_folder, f)
                    if f.lower().endswith('.json'):
                        json_conv.convert(full_p)
                    elif f.lower().endswith('.xml'):
                        xml_conv.convert(full_p)
                    base = os.path.splitext(f)[0]
                    self._copy_image(base, train_images)
                    count += 1
                    self.progress_signal.emit(int(count / total * 80))

            # 自動劃分 Val 集並生成 config.yaml
            self.log_signal.emit("⚖ 劃分 Train / Val 資料集...")
            import random
            all_train_imgs = [f for f in os.listdir(train_images) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'))]
            random.shuffle(all_train_imgs)
            val_num = int(len(all_train_imgs) * self.split_ratio)
            val_select = all_train_imgs[:val_num]

            for img_f in val_select:
                base_f = os.path.splitext(img_f)[0]
                shutil.move(os.path.join(train_images, img_f), os.path.join(val_images, img_f))
                lbl_f = base_f + ".txt"
                src_lbl = os.path.join(train_labels, lbl_f)
                if os.path.exists(src_lbl):
                    shutil.move(src_lbl, os.path.join(val_labels, lbl_f))

            # 生成 config.yaml
            config_data = {
                'path': dataset_root.replace("\\", "/"),
                'train': 'train/images',
                'val': 'val/images',
                'nc': len(classes),
                'names': classes
            }
            yaml_path = os.path.join(dataset_root, "config.yaml")
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, sort_keys=False, allow_unicode=True)

            self.progress_signal.emit(100)
            self.log_signal.emit(f"✨ 轉換完成！Train 圖片: {len(all_train_imgs)-val_num}, Val 圖片: {val_num}")
            self.log_signal.emit(f"📄 已生成配置文件: {yaml_path}")
            self.finished_signal.emit(True, yaml_path)

        except Exception as e:
            self.log_signal.emit(f"❌ 轉換過程發生錯誤: {e}")
            self.finished_signal.emit(False, str(e))

    def _copy_image(self, base_name, dest_dir):
        exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
                ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP"]
        search_dirs = [self.image_folder, self.anno_folder]
        for sdir in search_dirs:
            if not sdir or not os.path.exists(sdir):
                continue
            for ext in exts:
                src = os.path.join(sdir, base_name + ext)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(dest_dir, base_name + ext.lower()))
                    return


# =========================================================
# Worker: 視覺化標註驗證 (DataCheck.py 可視化整合)
# =========================================================
# =========================================================
# Worker: 視覺化標註驗證 (DataCheck.py 可視化整合)
# =========================================================
class DataCheckWorker(QThread):
    log_signal = Signal(str)
    image_rendered_signal = Signal(str, str) # orig_path, output_path
    finished_signal = Signal(list, str)      # sample_items, verify_dir

    def __init__(self, config_or_dir):
        super().__init__()
        self.target_path = str(config_or_dir).strip()

    def run(self):
        try:
            self.log_signal.emit(f"🔍 載入路徑並開始畫框驗證: {self.target_path}")
            if not self.target_path or not os.path.exists(self.target_path):
                self.log_signal.emit(f"❌ 指定路徑不存在: {self.target_path}")
                self.finished_signal.emit([], "")
                return

            target = self.target_path
            root = target
            cfg = {}
            config_file = None

            if os.path.isfile(target):
                config_file = target
                root = os.path.dirname(target)
            elif os.path.isdir(target):
                root = target
                for cname in ["config.yaml", "data.yaml", "dataset.yaml"]:
                    cp = os.path.join(root, cname)
                    if os.path.exists(cp):
                        config_file = cp
                        break

            if config_file and os.path.exists(config_file):
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
                    if cfg.get("path"):
                        c_path = cfg.get("path")
                        if os.path.isabs(c_path) and os.path.exists(c_path):
                            root = c_path
                        elif os.path.exists(os.path.join(os.path.dirname(config_file), c_path)):
                            root = os.path.abspath(os.path.join(os.path.dirname(config_file), c_path))
                except Exception as e:
                    self.log_signal.emit(f"ℹ 讀取設定檔警告: {e}")

            # 解析類別清單
            names = cfg.get("names", [])
            if isinstance(names, dict):
                names = [names[i] for i in sorted(names.keys())]
            elif isinstance(names, str):
                names = [names]

            if not names:
                for c_txt in [
                    os.path.join(root, "classes.txt"),
                    os.path.join(root, "labels", "classes.txt"),
                    os.path.join(root, "train", "labels", "classes.txt"),
                    os.path.join(os.path.dirname(root), "labels", "classes.txt")
                ]:
                    if os.path.exists(c_txt):
                        try:
                            with open(c_txt, "r", encoding="utf-8") as f:
                                names = [line.strip() for line in f if line.strip()]
                            break
                        except Exception:
                            pass

            if not names:
                names = [f"cls_{i}" for i in range(100)]

            # 搜尋圖片與標註資料夾
            train_candidates = [
                (os.path.join(root, str(cfg.get("train", "train/images"))), os.path.join(root, "train", "labels")),
                (os.path.join(root, "train", "images"), os.path.join(root, "train", "labels")),
                (os.path.join(root, "images", "train"), os.path.join(root, "labels", "train")),
                (os.path.join(root, "images"), os.path.join(root, "labels")),
                (os.path.join(root, "raw_images"), os.path.join(root, "labels")),
                (root, os.path.join(root, "labels")),
                (root, root)
            ]

            img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.JPG', '.PNG', '.JPEG', '.BMP')
            train_dir = None
            lbl_dir = None

            for t_dir, l_dir in train_candidates:
                if os.path.exists(t_dir):
                    try:
                        has_imgs = any(f.endswith(img_exts) for f in os.listdir(t_dir) if os.path.isfile(os.path.join(t_dir, f)))
                        if has_imgs:
                            train_dir = t_dir
                            lbl_dir = l_dir if os.path.exists(l_dir) else t_dir.replace("images", "labels")
                            break
                    except Exception:
                        pass

            if not train_dir or not os.path.exists(train_dir):
                self.log_signal.emit(f"⚠ 在 {root} 中未找到含有圖片的資料夾")
                self.finished_signal.emit([], "")
                return

            verify_dir = os.path.join(root, "verify")
            os.makedirs(verify_dir, exist_ok=True)

            colors = [
                (255, 75, 75), (75, 220, 75), (75, 150, 255), (255, 180, 0),
                (200, 75, 255), (0, 220, 220), (255, 100, 180), (160, 220, 0)
            ]

            imgs = [f for f in os.listdir(train_dir) if f.endswith(img_exts) and os.path.isfile(os.path.join(train_dir, f))]
            self.log_signal.emit(f"找到 {len(imgs)} 張圖片，開始渲染 DataCheck 畫框預覽...")

            def safe_imread(p):
                try:
                    data = np.fromfile(p, dtype=np.uint8)
                    return cv2.imdecode(data, cv2.IMREAD_COLOR)
                except Exception:
                    return cv2.imread(p)

            def safe_imwrite(p, mat):
                try:
                    ext = os.path.splitext(p)[1] or ".jpg"
                    _, buf = cv2.imencode(ext, mat)
                    buf.tofile(p)
                    return True
                except Exception:
                    return cv2.imwrite(p, mat)

            rendered_samples = []

            for img_name in imgs[:30]:  # 最多渲染 30 張預覽
                img_path = os.path.join(train_dir, img_name)
                base_name = os.path.splitext(img_name)[0]
                lbl_path = os.path.join(lbl_dir, base_name + ".txt") if lbl_dir and os.path.exists(lbl_dir) else os.path.join(train_dir, base_name + ".txt")

                img = safe_imread(img_path)
                if img is None:
                    continue

                H, W = img.shape[:2]
                box_cnt = 0

                if os.path.exists(lbl_path):
                    try:
                        with open(lbl_path, "r", encoding="utf-8") as lf:
                            lines = lf.read().strip().splitlines()

                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_id = int(float(parts[0]))
                                color = colors[cls_id % len(colors)]
                                cls_name = names[cls_id] if cls_id < len(names) else str(cls_id)

                                if len(parts) > 5:
                                    # Polygon Segmentation (normalized points)
                                    pts_raw = list(map(float, parts[1:]))
                                    pts = np.array(pts_raw, dtype=np.float32).reshape(-1, 2)
                                    pts[:, 0] *= W
                                    pts[:, 1] *= H
                                    pts_int = pts.astype(np.int32)
                                    cv2.polylines(img, [pts_int], True, color, 2)
                                    if len(pts_int) > 0:
                                        cv2.putText(img, cls_name, (pts_int[0][0], max(pts_int[0][1] - 4, 15)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                                else:
                                    # Bounding Box (cx, cy, w, h)
                                    cx, cy, bw, bh = map(float, parts[1:5])
                                    x1 = int((cx - bw / 2.0) * W)
                                    y1 = int((cy - bh / 2.0) * H)
                                    x2 = int((cx + bw / 2.0) * W)
                                    y2 = int((cy + bh / 2.0) * H)
                                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(img, cls_name, (x1, max(y1 - 5, 15)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                                box_cnt += 1
                    except Exception as e:
                        self.log_signal.emit(f"⚠ 讀取標籤 {lbl_path} 錯誤: {e}")

                out_path = os.path.join(verify_dir, img_name)
                safe_imwrite(out_path, img)
                self.image_rendered_signal.emit(img_path, out_path)
                rendered_samples.append({
                    "img_path": out_path,
                    "orig_path": img_path,
                    "box_count": box_cnt,
                    "name": img_name
                })

            self.log_signal.emit(f"✨ 驗證完成！共生成 {len(rendered_samples)} 張畫框渲染圖，已保存至: {verify_dir}")
            self.finished_signal.emit(rendered_samples, verify_dir)

        except Exception as e:
            self.log_signal.emit(f"❌ 驗證畫框出錯: {e}")
            self.finished_signal.emit([], "")


# =========================================================
# Worker: YOLO 多任務訓練 (Detect, Segment, Classify)
# =========================================================
class TrainWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int)
    epoch_metrics_signal = Signal(dict) # epoch, loss, map, etc.
    finished_signal = Signal(bool, str)

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._is_running = True
        self._is_paused = False

    def stop(self):
        self._is_running = False

    def pause(self):
        self._is_paused = True
        
    def resume(self):
        self._is_paused = False

    def _clean_dataset_caches(self, data_config_path):
        """
        清理資料集目錄中殘留的過期快取檔 (train.cache, val.cache, *.cache.npy)
        """
        if not data_config_path or not os.path.exists(data_config_path):
            return

        try:
            with open(data_config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

            root = cfg.get("path") or os.path.dirname(data_config_path)
            train_rel = str(cfg.get("train", "train/images"))
            val_rel = str(cfg.get("val", "val/images"))

            train_img_dir = train_rel if os.path.isabs(train_rel) else os.path.join(root, train_rel)
            val_img_dir = val_rel if os.path.isabs(val_rel) else os.path.join(root, val_rel)

            train_lbl_dir = train_img_dir.replace("images", "labels")
            val_lbl_dir = val_img_dir.replace("images", "labels")

            for d in [root, train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
                if os.path.exists(d):
                    for cf in os.listdir(d):
                        if cf.endswith(('.cache', '.cache.npy')):
                            try:
                                os.remove(os.path.join(d, cf))
                            except Exception:
                                pass
        except Exception:
            pass

    def run(self):
        self.log_signal.emit("啟動 Ultralytics YOLO 訓練流程...")
        try:
            model_path = self.kwargs.pop("model_path")
            data_cfg_path = self.kwargs.get("data")
            self._clean_dataset_caches(data_cfg_path)

            self.log_signal.emit(f"載入模型結構/權重: {model_path}")
            model = YOLO(model_path)

            # 自訂 Ultralytics Callback 來捕捉訓練進度與處理暫停/取消
            def check_pause(trainer):
                while self._is_paused and self._is_running:
                    time.sleep(0.5)
                if not self._is_running:
                    trainer.stop = True

            def on_train_batch_end(trainer):
                check_pause(trainer)

            def on_train_epoch_end(trainer):
                check_pause(trainer)
                if not self._is_running:
                    return
                epoch = trainer.epoch + 1
                epochs = trainer.epochs
                pct = int((epoch / epochs) * 100)
                self.progress_signal.emit(pct)

                metrics = {"epoch": epoch, "total_epochs": epochs}
                try:
                    loss_dict = {}
                    # 1. 嘗試由 label_loss_items 取得 dict
                    if hasattr(trainer, "label_loss_items") and hasattr(trainer, "tloss") and trainer.tloss is not None:
                        try:
                            loss_dict = trainer.label_loss_items(trainer.tloss)
                        except Exception:
                            pass

                    # 2. 若 loss_items 本身就是 dict
                    if not loss_dict and hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                        if isinstance(trainer.loss_items, dict):
                            loss_dict = trainer.loss_items

                    if loss_dict and isinstance(loss_dict, dict):
                        vals = []
                        for k, v in loss_dict.items():
                            val = float(v.detach().cpu().item()) if hasattr(v, "detach") else (float(v.item()) if hasattr(v, "item") else float(v))
                            vals.append(val)
                            k_lower = str(k).lower()
                            if "box" in k_lower: metrics["box_loss"] = val
                            elif "cls" in k_lower or "class" in k_lower: metrics["cls_loss"] = val
                            elif "dfl" in k_lower or "seg" in k_lower or "pose" in k_lower: metrics["dfl_loss"] = val
                        
                        if "box_loss" not in metrics and len(vals) >= 1: metrics["box_loss"] = vals[0]
                        if "cls_loss" not in metrics and len(vals) >= 2: metrics["cls_loss"] = vals[1]
                        if "dfl_loss" not in metrics and len(vals) >= 3: metrics["dfl_loss"] = vals[2]
                    else:
                        # 3. 若為 tensor / list / ndarray 序列結構
                        loss_raw = getattr(trainer, "loss_items", None)
                        if loss_raw is None and hasattr(trainer, "tloss"):
                            loss_raw = trainer.tloss
                        
                        if loss_raw is not None:
                            if hasattr(loss_raw, "detach"):
                                loss_arr = loss_raw.detach().cpu().tolist()
                            elif hasattr(loss_raw, "tolist"):
                                loss_arr = loss_raw.tolist()
                            elif isinstance(loss_raw, (list, tuple)):
                                loss_arr = list(loss_raw)
                            else:
                                loss_arr = [float(loss_raw)]
                            
                            clean_arr = []
                            for x in loss_arr:
                                if isinstance(x, (int, float)):
                                    clean_arr.append(float(x))
                                elif hasattr(x, "item"):
                                    clean_arr.append(float(x.item()))
                                elif hasattr(x, "detach"):
                                    clean_arr.append(float(x.detach().cpu().item()))

                            if len(clean_arr) >= 1: metrics["box_loss"] = clean_arr[0]
                            if len(clean_arr) >= 2: metrics["cls_loss"] = clean_arr[1]
                            if len(clean_arr) >= 3: metrics["dfl_loss"] = clean_arr[2]
                except Exception as le:
                    self.log_signal.emit(f"[WARN] 解析 loss 發生異常: {le}")

                try:
                    if hasattr(trainer, "metrics") and trainer.metrics:
                        m = trainer.metrics
                        if isinstance(m, dict):
                            metrics["map50"]    = float(m.get("metrics/mAP50(B)",    m.get("metrics/mAP50(M)",    m.get("mAP50", 0))))
                            metrics["map50_95"] = float(m.get("metrics/mAP50-95(B)", m.get("metrics/mAP50-95(M)", m.get("mAP50-95", 0))))
                except Exception as me:
                    self.log_signal.emit(f"[WARN] 解析 metrics 發生異常: {me}")

                self.epoch_metrics_signal.emit(metrics)
                box_str = f"{metrics['box_loss']:.4f}" if "box_loss" in metrics else "N/A"
                self.log_signal.emit(f"Epoch [{epoch}/{epochs}] 進度: {pct}% | Box Loss: {box_str}")

            model.add_callback("on_train_batch_end", on_train_batch_end)
            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            # World Detection: 設定文字類別提示
            world_classes = self.kwargs.pop("world_classes", None)
            if world_classes:
                self.log_signal.emit(f"🌐 套用 World Detection 文字提示類別: {world_classes}")
                if hasattr(model, "set_classes"):
                    model.set_classes(world_classes)
                else:
                    self.log_signal.emit("[WARN] 此模型不支援 set_classes()，將以標準訓練模式繼續...")

            # 執行訓練
            results = model.train(**self.kwargs)

            self.progress_signal.emit(100)
            self.log_signal.emit("✅ 訓練任務順利完成！模型與結果已自動儲存。")
            self.finished_signal.emit(True, "訓練成功！")

        except Exception as e:
            self.log_signal.emit(f"❌ 訓練過程發生異常: {e}")
            self.finished_signal.emit(False, str(e))



# =========================================================
# Worker: 推理與實時目標追蹤 (Predict & Track)
# =========================================================
class InferenceWorker(QThread):
    frame_signal = Signal(QImage, str) # rendered_frame, status_text
    status_signal = Signal(str)        # status text for UI label
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, model_path, source, mode="predict", tracker="bytetrack.yaml", conf=0.25, iou=0.45, device="0", world_classes=None):
        super().__init__()
        self.model_path = model_path
        self.source = source
        self.mode = mode # 'predict', 'track', 或 'world'
        self.tracker = tracker
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.world_classes = world_classes
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        self.status_signal.emit("⏳ 正在啟動推理引擎與載入模型...")
        self.log_signal.emit(f"🎬 啟動 {self.mode.upper()} 推理/追蹤引擎...")
        try:
            model = YOLO(self.model_path)
            
            # World Detection / Text Detection 類別設定
            if self.world_classes and hasattr(model, "set_classes"):
                self.status_signal.emit(f"🌐 設定 World 類別提示 ({', '.join(self.world_classes)})...")
                self.log_signal.emit(f"🌐 套用 World/Text 檢測類別提示: {self.world_classes}")
                try:
                    model.set_classes(self.world_classes)
                except Exception as ex:
                    self.log_signal.emit(f"⚠️ 設定類別提示失敗 (若使用網絡預設需檢查 CLIP 快取/下載): {ex}")
                    self.status_signal.emit(f"⚠️ 設定類別提示異常: {ex}")

            self.status_signal.emit("📷 正在讀取測試媒體並執行推斷...")

            # 單圖或資料夾推斷
            src_lower = str(self.source).lower()
            if isinstance(self.source, str) and (src_lower.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')) or os.path.isdir(self.source)):

                if self.mode == "track":
                    results = list(model.track(source=self.source, tracker=self.tracker, conf=self.conf, iou=self.iou, device=self.device, stream=True))
                else:
                    results = list(model.predict(source=self.source, conf=self.conf, iou=self.iou, device=self.device, stream=True))

                for res in results:
                    if not self._is_running:
                        break
                    frame_bgr = res.plot()
                    qimg = self._cv_to_qimage(frame_bgr)
                    det_count = len(res.boxes) if res.boxes is not None else 0
                    info = f"✅ 檢測完成 | 檢測目標數: {det_count}"
                    if res.boxes is not None and res.boxes.id is not None:
                        info += f" | 追蹤ID數: {len(res.boxes.id)}"
                    self.frame_signal.emit(qimg, info)
                    self.status_signal.emit(info)
                    time.sleep(0.03)

            # 影片或相機串流
            else:
                cap_src = 0 if str(self.source) == "0" else self.source
                cap = cv2.VideoCapture(cap_src)
                if not cap.isOpened():
                    msg = f"❌ 無法開啟影像來源: {self.source}"
                    self.log_signal.emit(msg)
                    self.status_signal.emit(msg)
                    return

                fps_start_time = time.time()
                frame_count = 0

                while cap.isOpened() and self._is_running:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if self.mode == "track":
                        results = model.track(frame, tracker=self.tracker, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
                    else:
                        results = model.predict(frame, conf=self.conf, iou=self.iou, device=self.device, verbose=False)

                    res_frame = results[0].plot() if len(results) > 0 else frame
                    qimg = self._cv_to_qimage(res_frame)

                    frame_count += 1
                    elapsed = time.time() - fps_start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    det_count = len(results[0].boxes) if len(results) > 0 and results[0].boxes is not None else 0
                    info = f"FPS: {fps:.1f} | 檢測目標: {det_count}"

                    self.frame_signal.emit(qimg, info)
                    self.status_signal.emit(info)
                    time.sleep(0.01)

                cap.release()

            self.log_signal.emit("✨ 推理/追蹤流程結束")
            self.finished_signal.emit()

        except Exception as e:
            err_msg = f"❌ 推理過程出錯: {e}"
            self.log_signal.emit(err_msg)
            self.status_signal.emit(err_msg)
            self.finished_signal.emit()

    def _cv_to_qimage(self, cv_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return qimg.copy()


# =========================================================
# Worker: 模型導出 (ONNX, TensorRT, OpenVINO 等)
# =========================================================
class ExportWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, model_path, fmt="onnx", imgsz=640, half=False, dynamic=False, simplify=True, opset=12):
        super().__init__()
        self.model_path = model_path
        self.fmt = fmt
        self.imgsz = int(imgsz)
        self.half = half
        self.dynamic = dynamic
        self.simplify = simplify
        self.opset = int(opset)

    def run(self):
        os.environ["YOLO_AUTO_UPDATE"] = "0"
        os.environ["ULTRALYTICS_AUTOINSTALL"] = "0"
        self.log_signal.emit(f"🚀 開始導出模型 [{self.fmt.upper()}] 格式...")
        try:
            model = YOLO(self.model_path)
            try:
                export_path = model.export(
                    format=self.fmt,
                    imgsz=self.imgsz,
                    half=self.half,
                    dynamic=self.dynamic,
                    simplify=self.simplify,
                    opset=self.opset
                )
            except Exception as e:
                err_str = str(e).lower()
                if self.simplify and ("onnxslim" in err_str or "slim" in err_str or "simplify" in err_str):
                    self.log_signal.emit("ℹ onnxslim 精簡失敗，正在自動降級為標準 ONNX 導出...")
                    export_path = model.export(
                        format=self.fmt,
                        imgsz=self.imgsz,
                        half=self.half,
                        dynamic=self.dynamic,
                        simplify=False,
                        opset=self.opset
                    )
                else:
                    raise e

            self.log_signal.emit(f"✨ 模型導出成功: {export_path}")
            self.finished_signal.emit(True, str(export_path))
        except Exception as e:
            self.log_signal.emit(f"❌ 模型導出失敗: {e}")
            self.finished_signal.emit(False, str(e))


# =========================================================
# Worker: CUDA & GPU 硬體診斷 (cudatorch.py 整合)
# =========================================================
class CudaCheckWorker(QThread):
    info_signal = Signal(dict)

    def run(self):
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
        }
        self.info_signal.emit(info)


# =========================================================
# Worker: 非阻塞 Task Manager 效能監控
# =========================================================
class PerfMonitorThread(QThread):
    stats_signal = Signal(float, str, float, str, float, str, float, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.nvml_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nvml_handle = None

    def run(self):
        try:
            psutil.cpu_percent()
        except Exception:
            pass

        while self._running:
            time.sleep(1.0)
            if not self._running:
                break
            try:
                cpu_pct = psutil.cpu_percent()
                ram_mem = psutil.virtual_memory()
                ram_pct = ram_mem.percent
                ram_text = f"{ram_mem.used / (1024**3):.1f}/{ram_mem.total / (1024**3):.0f}G"

                gpu_pct, vram_pct, vram_text = 0.0, 0.0, "0/0G"

                if self.nvml_handle:
                    try:
                        import pynvml
                        util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                        gpu_pct = float(util.gpu)
                        vram_pct = float((mem_info.used / mem_info.total) * 100)
                        v_used_g = mem_info.used / (1024 ** 3)
                        v_total_g = mem_info.total / (1024 ** 3)
                        vram_text = f"{v_used_g:.1f}/{v_total_g:.0f}G"
                    except Exception:
                        pass
                elif torch.cuda.is_available():
                    try:
                        allocated = torch.cuda.memory_allocated(0)
                        total = torch.cuda.get_device_properties(0).total_memory
                        vram_pct = (allocated / total) * 100
                        vram_text = f"{allocated / (1024**3):.1f}/{total / (1024**3):.0f}G"
                    except Exception:
                        pass

                self.stats_signal.emit(
                    cpu_pct, f"{int(cpu_pct)}%",
                    ram_pct, ram_text,
                    gpu_pct, f"{int(gpu_pct)}%",
                    vram_pct, vram_text
                )
            except Exception:
                pass

    def stop(self):
        self._running = False


# =========================================================
# Worker: AI 模型批次自動標注 (Auto-Annotation)
# =========================================================
class AutoAnnotateWorker(QThread):
    progress_signal = Signal(int, int)          # current, total
    preview_signal = Signal(QImage, str, int)   # rendered_qimage, filename, detected_box_count
    log_signal = Signal(str)                    # log string
    status_signal = Signal(str)                 # status description
    finished_signal = Signal(bool, str, dict)   # success, message, result_dict

    def __init__(self, model_path, source_dir, project_root, project_name="NYA_AutoLabel",
                 conf=0.25, iou=0.45, imgsz=640, device="0", world_prompts=None,
                 task_type="detect", auto_split=True, split_ratio=0.2):
        super().__init__()
        self.model_path = model_path
        self.source_dir = source_dir
        self.project_root = project_root
        self.project_name = project_name or "NYA_AutoLabel"
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self.world_prompts = world_prompts
        self.task_type = task_type
        self.auto_split = auto_split
        self.split_ratio = float(split_ratio)
        self._is_running = True

    def stop(self):
        self._is_running = False

    def _cv_to_qimage(self, bgr_img):
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        return QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

    def run(self):
        try:
            self.status_signal.emit("⏳ 正在初始化 AI 自動標注引擎...")
            self.log_signal.emit("🤖 啟動 AI 批次自動標注引擎...")

            if not os.path.exists(self.source_dir):
                self.finished_signal.emit(False, "輸入圖像資料夾不存在！", {})
                return

            exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
            all_files = [f for f in sorted(os.listdir(self.source_dir)) if f.lower().endswith(exts)]
            if not all_files:
                self.finished_signal.emit(False, "輸入資料夾中未找到任何支援的圖像檔案！", {})
                return

            # 建立專案目錄結構
            target_proj_dir = os.path.join(self.project_root, self.project_name)
            raw_images_dir = os.path.join(target_proj_dir, "raw_images")
            labels_dir = os.path.join(target_proj_dir, "labels")
            os.makedirs(raw_images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            self.log_signal.emit(f"📁 專案根目錄: {target_proj_dir}")
            self.log_signal.emit(f"🖼 掃描到 {len(all_files)} 張待標註圖像")

            # 載入模型
            self.status_signal.emit(f"🧠 正在載入模型: {os.path.basename(self.model_path)}...")
            self.log_signal.emit(f"🧠 載入模型權重: {self.model_path}")
            model = YOLO(self.model_path)

            # 套用 World 檢測類別提示
            active_class_names = []
            if self.world_prompts and hasattr(model, "set_classes"):
                self.log_signal.emit(f"🌐 套用 World Detection 文字提示: {self.world_prompts}")
                try:
                    model.set_classes(self.world_prompts)
                    active_class_names = list(self.world_prompts)
                except Exception as ex:
                    self.log_signal.emit(f"⚠️ 設定 World 類別提示異常: {ex}")
            
            if not active_class_names:
                if hasattr(model, "names") and model.names:
                    if isinstance(model.names, dict):
                        active_class_names = [model.names[i] for i in sorted(model.names.keys())]
                    elif isinstance(model.names, (list, tuple)):
                        active_class_names = list(model.names)
                else:
                    active_class_names = ["object"]

            # 儲存 classes.txt
            classes_txt_path = os.path.join(labels_dir, "classes.txt")
            with open(classes_txt_path, "w", encoding="utf-8") as f:
                for c in active_class_names:
                    f.write(f"{c}\n")
            self.log_signal.emit(f"📝 標註類別名單已寫入: {classes_txt_path} ({len(active_class_names)} 類)")

            total = len(all_files)
            total_boxes_count = 0
            class_stats = {c: 0 for c in active_class_names}
            processed_count = 0

            self.log_signal.emit(f"⚡ 開始批次標注 (Conf: {self.conf}, IoU: {self.iou}, ImgSz: {self.imgsz}, Device: {self.device})...")

            for idx, fname in enumerate(all_files):
                if not self._is_running:
                    self.log_signal.emit("⏹ 自動標注已被使用者中止。")
                    self.finished_signal.emit(False, "標注已被使用者中途停止", {
                        "project_dir": target_proj_dir,
                        "processed": processed_count,
                        "total": total,
                        "total_boxes": total_boxes_count
                    })
                    return

                src_img_path = os.path.join(self.source_dir, fname)
                dst_img_path = os.path.join(raw_images_dir, fname)
                if src_img_path != dst_img_path:
                    shutil.copy2(src_img_path, dst_img_path)

                base_name = os.path.splitext(fname)[0]
                dst_label_path = os.path.join(labels_dir, f"{base_name}.txt")

                # 執行推論
                results = model.predict(
                    source=src_img_path,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False
                )

                label_lines = []
                img_boxes_count = 0
                plotted_bgr = None

                if results and len(results) > 0:
                    res = results[0]
                    plotted_bgr = res.plot()
                    if res.boxes is not None and len(res.boxes) > 0:
                        img_boxes_count = len(res.boxes)
                        total_boxes_count += img_boxes_count
                        for box in res.boxes:
                            cls_id = int(box.cls.item())
                            xywhn = box.xywhn[0].tolist() # [x_center, y_center, width, height]
                            x_c, y_c, w, h = xywhn[0], xywhn[1], xywhn[2], xywhn[3]
                            label_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                            
                            if 0 <= cls_id < len(active_class_names):
                                c_name = active_class_names[cls_id]
                                class_stats[c_name] = class_stats.get(c_name, 0) + 1

                # 寫入 label .txt
                with open(dst_label_path, "w", encoding="utf-8") as lf:
                    for line in label_lines:
                        lf.write(line + "\n")

                processed_count += 1
                self.progress_signal.emit(processed_count, total)

                # 發送預覽信號
                if plotted_bgr is not None:
                    qimg = self._cv_to_qimage(plotted_bgr)
                    self.preview_signal.emit(qimg, fname, img_boxes_count)

                self.status_signal.emit(f"標注進度: {processed_count}/{total} (當前圖片: {fname} 偵測到 {img_boxes_count} 個目標)")

            # 是否自動切分數據集
            dataset_config_path = None
            if self.auto_split and processed_count > 0:
                self.status_signal.emit("⚡ 正在自動拆分 Train/Val 數據集並生成 config.yaml...")
                self.log_signal.emit(f"⚡ 啟動資料集拆分 (Val 比例: {self.split_ratio})...")
                
                dataset_dir = os.path.join(target_proj_dir, "dataset")
                train_img_dir = os.path.join(dataset_dir, "train", "images")
                train_lbl_dir = os.path.join(dataset_dir, "train", "labels")
                val_img_dir = os.path.join(dataset_dir, "val", "images")
                val_lbl_dir = os.path.join(dataset_dir, "val", "labels")

                for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
                    if os.path.exists(d):
                        shutil.rmtree(d)
                    os.makedirs(d, exist_ok=True)

                import random
                random.seed(42)
                shuffled_files = list(all_files)
                random.shuffle(shuffled_files)

                num_val = max(1, int(len(shuffled_files) * self.split_ratio)) if len(shuffled_files) > 1 else 0
                val_files = set(shuffled_files[:num_val])
                train_files = set(shuffled_files[num_val:])

                for img_name in shuffled_files:
                    b_name = os.path.splitext(img_name)[0]
                    s_img = os.path.join(raw_images_dir, img_name)
                    s_lbl = os.path.join(labels_dir, f"{b_name}.txt")

                    if img_name in val_files:
                        d_img = os.path.join(val_img_dir, img_name)
                        d_lbl = os.path.join(val_lbl_dir, f"{b_name}.txt")
                    else:
                        d_img = os.path.join(train_img_dir, img_name)
                        d_lbl = os.path.join(train_lbl_dir, f"{b_name}.txt")

                    shutil.copy2(s_img, d_img)
                    if os.path.exists(s_lbl):
                        shutil.copy2(s_lbl, d_lbl)

                # 生成 config.yaml
                config_data = {
                    'path': dataset_dir.replace("\\", "/"),
                    'train': 'train/images',
                    'val': 'val/images',
                    'nc': len(active_class_names),
                    'names': active_class_names
                }
                dataset_config_path = os.path.join(dataset_dir, "config.yaml")
                with open(dataset_config_path, 'w', encoding='utf-8') as yf:
                    yaml.dump(config_data, yf, sort_keys=False, allow_unicode=True)

                self.log_signal.emit(f"✨ 數據集拆分完成！Train: {len(train_files)} 張, Val: {len(val_files)} 張")
                self.log_signal.emit(f"📄 config.yaml 已生成於: {dataset_config_path}")

            summary_dict = {
                "project_dir": target_proj_dir,
                "raw_images_dir": raw_images_dir,
                "labels_dir": labels_dir,
                "dataset_config_path": dataset_config_path,
                "processed": processed_count,
                "total": total,
                "total_boxes": total_boxes_count,
                "class_stats": class_stats,
                "classes": active_class_names
            }

            self.status_signal.emit(f"✅ 自動標注完成！共處理 {processed_count} 張圖像，生成 {total_boxes_count} 個標註框。")
            self.log_signal.emit(f"🎉 專案 [{self.project_name}] AI 自動標注圓滿完成！總計產生 {total_boxes_count} 個目標標籤。")
            self.finished_signal.emit(True, "自動標注完成！", summary_dict)

        except Exception as e:
            self.log_signal.emit(f"❌ 自動標注過程發生異常: {e}")
            self.finished_signal.emit(False, str(e), {})

