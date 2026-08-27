import os
import shutil
import random
import yaml

from tools.ConvertToLables import (
    JSON2YOLO, XML2YOLO, auto_detect_classes, save_classes_list
)


class HalconProjectManager:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.join(base_dir, "NYA_Project")
        self.raw_images_dir = os.path.join(self.project_root, "raw_images")
        self.labels_dir = os.path.join(self.project_root, "labels")
        self.dataset_dir = os.path.join(self.project_root, "dataset")

        self.train_images_dir = os.path.join(self.dataset_dir, "train", "images")
        self.train_labels_dir = os.path.join(self.dataset_dir, "train", "labels")
        self.val_images_dir = os.path.join(self.dataset_dir, "val", "images")
        self.val_labels_dir = os.path.join(self.dataset_dir, "val", "labels")

        self._ensure_dirs()

    def _ensure_dirs(self):
        for p in [
            self.project_root, self.raw_images_dir, self.labels_dir,
            self.dataset_dir, self.train_images_dir, self.train_labels_dir,
            self.val_images_dir, self.val_labels_dir
        ]:
            os.makedirs(p, exist_ok=True)

    def clear_project(self):
        """徹底清理專案目錄下的所有舊圖片、標註與資料集快取"""
        for d in [self.raw_images_dir, self.labels_dir, self.train_images_dir, self.train_labels_dir, self.val_images_dir, self.val_labels_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        if os.path.exists(self.dataset_dir):
            for f in os.listdir(self.dataset_dir):
                fp = os.path.join(self.dataset_dir, f)
                if os.path.isfile(fp):
                    try:
                        os.remove(fp)
                    except Exception:
                        pass

    def setup_project_from_folders(self, image_folder, label_folder=None, copy_files=True, clear_existing=False, log_func=print):
        """匯入圖像與可選的已有標註集 (支援 TXT / XML / JSON 自動轉 YOLO 格式)"""
        if clear_existing:
            self.clear_project()
        self._ensure_dirs()
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

        imported_images = []
        if os.path.exists(image_folder):
            for f in os.listdir(image_folder):
                if f.lower().endswith(exts):
                    src = os.path.join(image_folder, f)
                    dst = os.path.join(self.raw_images_dir, f)
                    imported_images.append(dst)
                    if copy_files and src != dst:
                        shutil.copy2(src, dst)

        # 匯入已有標註檔 (.txt / .xml / .json)
        if label_folder and os.path.exists(label_folder):
            label_files = os.listdir(label_folder)
            has_xml = any(f.lower().endswith('.xml') for f in label_files)
            has_json = any(f.lower().endswith('.json') for f in label_files)

            if has_xml or has_json:
                log_func("🔍 檢測到 XML/JSON 標註檔，啟動自動轉 YOLO 格式引擎...")
                classes = auto_detect_classes(label_folder, label_folder)
                if not classes:
                    classes = ["object"]
                
                classes_txt = os.path.join(self.labels_dir, "classes.txt")
                save_classes_list(classes, classes_txt)

                if has_xml:
                    xml_conv = XML2YOLO(classes, self.labels_dir, image_folder=self.raw_images_dir)
                    xml_conv.batch_convert(label_folder)
                    log_func(f"✅ XML 標註已自動轉為 YOLO .txt 並保存至 {self.labels_dir}")

                if has_json:
                    json_conv = JSON2YOLO(classes, self.labels_dir, image_folder=self.raw_images_dir)
                    json_conv.batch_convert(label_folder)
                    log_func(f"✅ JSON 標註已自動轉為 YOLO .txt 並保存至 {self.labels_dir}")
            else:
                for f in label_files:
                    if f.lower().endswith('.txt'):
                        src = os.path.join(label_folder, f)
                        dst = os.path.join(self.labels_dir, f)
                        if copy_files and src != dst:
                            shutil.copy2(src, dst)
                log_func(f"✅ 已有 TXT 標註集已成功匯入至 {self.labels_dir}")

        return self.raw_images_dir, self.labels_dir

    def split_and_build_dataset(self, val_ratio=0.2, class_names=None, task_type="detect", log_func=print):
        """一鍵拆分 Train / Val 並徹底清理舊檔案與生成 YOLO / ResNet config.yaml"""
        self._ensure_dirs()

        if not class_names:
            classes_txt = os.path.join(self.labels_dir, "classes.txt")
            if os.path.exists(classes_txt):
                with open(classes_txt, "r", encoding="utf-8") as f:
                    class_names = [line.strip() for line in f if line.strip()]

        if not class_names:
            class_names = ["OK", "NG"] if task_type == "classify" else ["object"]

        # 清理 dataset 根目錄及子目錄下的所有舊檔案與快取
        for d in [self.dataset_dir, self.train_images_dir, self.train_labels_dir, self.val_images_dir, self.val_labels_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)

        if os.path.exists(self.dataset_dir):
            for cf in os.listdir(self.dataset_dir):
                fp = os.path.join(self.dataset_dir, cf)
                if os.path.isfile(fp):
                    try: os.remove(fp)
                    except Exception: pass

        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        all_images = [f for f in os.listdir(self.raw_images_dir) if f.lower().endswith(exts)]
        
        if not all_images:
            raise ValueError("raw_images/ 資料夾中未找到圖片檔案！")

        random.seed(42)
        images_shuffled = list(all_images)
        random.shuffle(images_shuffled)

        num_val = int(len(images_shuffled) * val_ratio)
        val_set = set(images_shuffled[:num_val])
        train_set = set(images_shuffled[num_val:])

        # ── 分類模式：輸出至 train/<類別>/ 與 val/<類別>/ ────────
        if task_type == "classify":
            for cname in class_names:
                os.makedirs(os.path.join(self.dataset_dir, "train", cname), exist_ok=True)
                os.makedirs(os.path.join(self.dataset_dir, "val", cname), exist_ok=True)

            for img_name in images_shuffled:
                base_name = os.path.splitext(img_name)[0]
                src_img = os.path.join(self.raw_images_dir, img_name)
                src_label = os.path.join(self.labels_dir, f"{base_name}.txt")

                assigned_c = "OK" if "OK" in class_names else class_names[0]
                if os.path.exists(src_label) and os.path.getsize(src_label) > 0:
                    assigned_c = "NG" if "NG" in class_names else class_names[-1]

                target_subset = "val" if img_name in val_set else "train"
                dst_img = os.path.join(self.dataset_dir, target_subset, assigned_c, img_name)
                shutil.copy2(src_img, dst_img)

            config = {
                'path': self.dataset_dir.replace("\\", "/"),
                'train': 'train',
                'val': 'val',
                'nc': len(class_names),
                'names': class_names
            }
        else:
            # ── 檢測/分割模式：輸出至 train/images, train/labels ────
            for img_name in images_shuffled:
                base_name = os.path.splitext(img_name)[0]
                src_img = os.path.join(self.raw_images_dir, img_name)
                src_label = os.path.join(self.labels_dir, f"{base_name}.txt")

                if img_name in val_set:
                    dst_img = os.path.join(self.val_images_dir, img_name)
                    dst_label = os.path.join(self.val_labels_dir, f"{base_name}.txt")
                else:
                    dst_img = os.path.join(self.train_images_dir, img_name)
                    dst_label = os.path.join(self.train_labels_dir, f"{base_name}.txt")

                shutil.copy2(src_img, dst_img)
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)

            config = {
                'path': self.dataset_dir.replace("\\", "/"),
                'train': 'train/images',
                'val': 'val/images',
                'nc': len(class_names),
                'names': class_names
            }

        yaml_path = os.path.join(self.dataset_dir, "config.yaml")
        if os.path.exists(yaml_path):
            try:
                os.remove(yaml_path)
            except Exception:
                pass
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, sort_keys=False, allow_unicode=True)

        log_func(f"✨ [NYA 工作流] 拆分完成！訓練集: {len(train_set)} 張, 驗證集: {len(val_set)} 張")
        log_func(f"📄 config.yaml 已生成於: {yaml_path}")
        return {
            "config_path": yaml_path,
            "train_count": len(train_set),
            "val_count": len(val_set),
            "class_names": class_names
        }
