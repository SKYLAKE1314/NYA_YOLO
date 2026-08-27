"""
NyaUI — NYA AI Studio 主視窗與模組化調度器
採用高規解耦架構，調度 GoogleHeaderWidget, GoogleSidebarWidget 與獨立 Function Pages
"""

import os
import sys
import psutil
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QMessageBox, QDialog, QLabel
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor

# 模組路徑設定（相容 Nuitka 打包與開發模式）
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Nuitka 打包後 __file__ 指向 .dist 根目錄，資源檔在 UI/ 子目錄
_UI_SUBDIR = os.path.join(_BASE_DIR, "UI")
if os.path.isdir(_UI_SUBDIR) and os.path.exists(os.path.join(_UI_SUBDIR, "icon.ico")):
    CURRENT_DIR = _UI_SUBDIR          # Nuitka 打包模式
else:
    CURRENT_DIR = _BASE_DIR           # 開發模式（直接在 UI/ 目錄運行）
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

ICON_PATH = os.path.join(CURRENT_DIR, "icon.ico")

from styles import GoogleAccountTheme
from config_manager import load_ui_cache, save_ui_cache
from workers import (
    ConvertWorker, DataCheckWorker, TrainWorker,
    InferenceWorker, ExportWorker, CudaCheckWorker,
    PerfMonitorThread, AutoAnnotateWorker
)
from components import (
    GoogleHeaderWidget, GoogleSidebarWidget, show_environment_dialog
)
from pages import (
    HomePageWidget, DataPrepPageWidget, TrainConfigPageWidget,
    LiveTrainPageWidget, InferencePageWidget, ExportToolsPageWidget
)


def detect_system_dark_mode():
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        )
        value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        winreg.CloseKey(key)
        return value == 0
    except Exception:
        return False


class NyaUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NYA AI Studio")
        self.resize(1340, 850)

        _wp_path = os.path.join(CURRENT_DIR, "file_0000000031e8720681bd49398eace5bf.png")
        self._wallpaper = QPixmap(_wp_path) if os.path.exists(_wp_path) else QPixmap()

        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))

        self.theme_mode = "dark"
        self.dark_mode = True

        # 線程控制器
        self.train_worker = None
        self.infer_worker = None
        self.convert_worker = None
        self.datacheck_worker = None
        self.export_worker = None
        self.cuda_worker = None
        self.auto_annotate_worker = None

        self.init_ui()
        self.restore_all_settings_from_cache()
        self.apply_theme()
        self.auto_detect_hardware_acceleration()
        self.check_cuda_status()
        self.init_perf_monitor()

        QTimer.singleShot(600, lambda: show_environment_dialog(self, auto_on_startup=True))
        self.showMaximized()

    def init_ui(self):
        main_widget = QWidget()
        main_widget.setObjectName("MainContainer")
        self.setCentralWidget(main_widget)

        root_layout = QVBoxLayout(main_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. 頂部 Google Header 欄
        self.header = GoogleHeaderWidget(self)
        self.header.theme_changed.connect(self.cycle_theme_mode)
        self.header.compute_mode_changed.connect(self.set_compute_mode)
        self.header.switch_page_requested.connect(self.switch_page)
        root_layout.addWidget(self.header)

        # 2. 中間內容區 (導覽欄 + 堆疊頁面)
        body_layout = QHBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.sidebar = GoogleSidebarWidget(dark_mode=self.dark_mode, parent=self)
        self.sidebar.page_selected.connect(self.switch_page)
        body_layout.addWidget(self.sidebar, 0)

        self.stack = QStackedWidget()
        
        # 實例化各獨立頁面
        self.page_home = HomePageWidget(self)
        self.page_home.search_requested.connect(self.execute_home_search)

        self.page_dataprep = DataPrepPageWidget(self)
        self.page_dataprep.start_convert_requested.connect(self.start_convert)
        self.page_dataprep.start_datacheck_requested.connect(self.start_datacheck)
        self.page_dataprep.jump_to_train_requested.connect(self.on_halcon_jump_to_train)
        self.page_dataprep.start_auto_annotate_requested.connect(self.start_auto_annotate)
        self.page_dataprep.stop_auto_annotate_requested.connect(self.stop_auto_annotate)

        self.page_train_config = TrainConfigPageWidget(self)
        self.page_train_config.start_train_requested.connect(self.start_train)

        self.page_live_train = LiveTrainPageWidget(dark_mode=self.dark_mode, parent=self)
        self.page_live_train.pause_train_requested.connect(self.toggle_pause_train)
        self.page_live_train.stop_train_requested.connect(self.stop_train)

        self.page_inference = InferencePageWidget(self)
        self.page_inference.start_infer_requested.connect(self.start_inference)
        self.page_inference.stop_infer_requested.connect(self.stop_inference)

        self.page_export_tools = ExportToolsPageWidget(self)
        self.page_export_tools.start_export_requested.connect(self.start_export)
        self.page_export_tools.refresh_cuda_requested.connect(self.check_cuda_status)

        self.stack.addWidget(self.page_home)          # Page 0
        self.stack.addWidget(self.page_dataprep)      # Page 1
        self.stack.addWidget(self.page_train_config)  # Page 2
        self.stack.addWidget(self.page_live_train)    # Page 3
        self.stack.addWidget(self.page_inference)     # Page 4
        self.stack.addWidget(self.page_export_tools)  # Page 5

        body_layout.addWidget(self.stack, 1)
        root_layout.addLayout(body_layout)

    def switch_page(self, idx):
        self.sidebar.switch_page(idx)
        self.stack.setCurrentIndex(idx)

    def cycle_theme_mode(self, mode_str="cycle"):
        if self.theme_mode == "dark":
            self.theme_mode = "light"
        else:
            self.theme_mode = "dark"
        self.apply_theme()

    def apply_theme(self):
        if self.theme_mode == "light":
            self.dark_mode = False
            self.header.btn_theme.setText("☀ 昨日青空")
        else:
            self.theme_mode = "dark"
            self.dark_mode = True
            self.header.btn_theme.setText("🌙 半月星夢")

        qss = GoogleAccountTheme.get_style(self.dark_mode)
        self.setStyleSheet(qss)

        self.sidebar.set_dark_mode(self.dark_mode)
        self.page_live_train.set_dark_mode(self.dark_mode)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self.dark_mode:
            painter.fillRect(self.rect(), QColor("#0D0D10"))
            if not self._wallpaper.isNull():
                scaled = self._wallpaper.scaled(
                    self.size(),
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                painter.setOpacity(0.28)
                painter.drawPixmap(0, 0, scaled)
                painter.setOpacity(1.0)
        else:
            painter.fillRect(self.rect(), QColor("#F4F0FA"))
            if not self._wallpaper.isNull():
                scaled = self._wallpaper.scaled(
                    self.size(),
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                painter.setOpacity(0.25)
                painter.drawPixmap(0, 0, scaled)
                painter.setOpacity(1.0)
            painter.fillRect(self.rect(), QColor(244, 240, 250, 160))

        painter.end()
        super().paintEvent(event)

    def set_compute_mode(self, text_label, device_val, icon_name):
        short = text_label.split()[0]
        self.header.avatar_btn.setText(f" {short}")
        p = os.path.join(CURRENT_DIR, "icons", f"{icon_name}.png")
        if os.path.exists(p):
            self.header.avatar_btn.setIcon(QIcon(p))
        self.page_train_config.device_input.setText(device_val)
        self.append_log(f"⚙️ 系統運行模式已切換為: {text_label}")

    def auto_detect_hardware_acceleration(self):
        try:
            import torch
            if torch.cuda.is_available():
                dev_name = torch.cuda.get_device_name(0)
                self.set_compute_mode(f"CUDA  ({dev_name})", "0", "nvidia")
                self.append_log(f"[硬體加速計劃] 檢測到 NVIDIA GPU ({dev_name})，已自動設為預設算力加速裝置 (Device: 0)！")
                return
        except Exception:
            pass

        try:
            import openvino
            self.set_compute_mode("OpenVINO  (Intel)", "cpu", "openvino")
            self.append_log("[硬體加速計劃] 檢測到 Intel OpenVINO，已自動設為預設加速裝置！")
            return
        except Exception:
            pass

        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.set_compute_mode("MPS   (Apple)", "mps", "apple")
                self.append_log("[硬體加速計劃] 檢測到 Apple Silicon MPS，已自動設為預設加速裝置！")
                return
        except Exception:
            pass

        self.set_compute_mode("CPU   Mode", "cpu", "cpu")
        self.append_log("ℹ️ [硬體加速計劃] 當前使用 CPU 計算模式。")

    def on_halcon_jump_to_train(self, config_path):
        self.page_train_config.data_input.setText(config_path)
        self.switch_page(2)
        self.append_log(f"✨ [NYA 工作流] 已自動拆分資料集並將 config.yaml 帶入訓練配置: {config_path}")

    def execute_home_search(self, query):
        q = query.lower()
        if "yolo12" in q or "yolo26" in q or "yolo11" in q or ".pt" in q or ".yaml" in q:
            self.switch_page(2)
        elif "onnx" in q or "tensorrt" in q or "export" in q:
            self.switch_page(5)
        elif "data" in q or "check" in q or "xml" in q or "json" in q:
            self.switch_page(1)
        elif "cuda" in q or "gpu" in q:
            self.switch_page(5)
            self.check_cuda_status()
        else:
            self.switch_page(2)
        self.append_log(f"🔍 [搜尋引擎] 根據 [{query}] 自動切換至相應模組。")

    # --- 訓練控制 ---
    def start_train(self, kwargs):
        self.save_all_settings_to_cache()
        self.page_live_train.btn_pause_train.setEnabled(True)
        self.page_live_train.btn_stop_train.setEnabled(True)
        self.page_live_train.lbl_train_status.setText("訓練進行中...")
        self.switch_page(3)

        self.train_worker = TrainWorker(kwargs)
        self.train_worker.log_signal.connect(self.append_log)
        self.train_worker.progress_signal.connect(self.page_live_train.progress_bar.setValue)
        self.train_worker.epoch_metrics_signal.connect(self.page_live_train.update_metrics)
        self.train_worker.finished_signal.connect(self.on_train_finished)
        self.train_worker.start()

    def toggle_pause_train(self):
        if not self.train_worker:
            return
        if self.train_worker._is_paused:
            self.train_worker.resume()
            self.page_live_train.btn_pause_train.setText("⏸ 暫停訓練")
            self.page_live_train.lbl_train_status.setText("訓練進行中...")   
            self.append_log("▶ 繼續訓練...")
        else:
            self.train_worker.pause()
            self.page_live_train.btn_pause_train.setText("▶ 繼續訓練")
            self.page_live_train.lbl_train_status.setText("訓練已暫停 (Paused)")
            self.append_log("⏸ 訓練已暫停...")

    def stop_train(self):
        if self.train_worker:
            self.train_worker.stop()
            self.page_live_train.btn_pause_train.setEnabled(False)
            self.append_log("🛑 已發送訓練取消請求...")

    def on_train_finished(self, success, msg):
        self.page_train_config.btn_start_train.setEnabled(True)
        self.page_live_train.btn_pause_train.setEnabled(False)
        self.page_live_train.btn_stop_train.setEnabled(False)
        self.page_live_train.lbl_train_status.setText("訓練已完成" if success else "訓練被取消/出錯")

    # --- 推理控制 ---
    def start_inference(self):
        self.save_all_settings_to_cache()
        m_path = self.page_inference.infer_model_input.text().strip()
        src = self.page_inference.infer_source_input.text().strip()

        if not m_path:
            QMessageBox.warning(self, "提示", "請先選擇模型檔案路徑！")
            return
        if not src:
            QMessageBox.warning(self, "提示", "請先選擇測試圖片/影片檔案！")
            return

        mode_text = self.page_inference.infer_mode_combo.currentText()
        mode = "predict"
        if "track" in mode_text:
            mode = "track"
        elif "world" in mode_text or "text_det" in mode_text:
            mode = "world"

        tracker = self.page_inference.tracker_combo.currentText()
        world_classes = None
        if self.page_inference.infer_world_group.isVisible() or mode == "world":
            raw = self.page_inference.infer_world_prompts.text().strip()
            world_classes = [c.strip() for c in raw.split(",") if c.strip()] or ["text", "label"]

        self.page_inference.update_status("⏳ 正在啟動推理引擎...")

        self.infer_worker = InferenceWorker(
            m_path, src, mode, tracker,
            self.page_inference.conf_spin.value(),
            self.page_inference.iou_spin.value(),
            self.page_train_config.device_input.text().strip(),
            world_classes=world_classes
        )
        self.infer_worker.log_signal.connect(self.append_log)
        self.infer_worker.status_signal.connect(self.page_inference.update_status)
        self.infer_worker.frame_signal.connect(self.page_inference.update_canvas)
        self.infer_worker.start()

    def stop_inference(self):
        if self.infer_worker:
            self.infer_worker.stop()

    # --- 標註轉檔與驗證 ---
    def start_convert(self, kwargs):
        self.convert_worker = ConvertWorker(kwargs)
        self.convert_worker.log_signal.connect(self.page_dataprep.append_log)
        self.convert_worker.finished_signal.connect(self.on_convert_finished)
        self.page_dataprep.btn_start_convert.setEnabled(False)
        self.convert_worker.start()

    def on_convert_finished(self, success, result_msg):
        self.page_dataprep.btn_start_convert.setEnabled(True)
        if success:
            self.page_train_config.data_input.setText(result_msg)
            self.append_log(f"✨ [NYA 轉檔] 轉換完成！已自動將 config.yaml 帶入訓練配置: {result_msg}")
        else:
            self.append_log(f"❌ [NYA 轉檔失敗] {result_msg}")

    def start_datacheck(self):
        target_path = self.page_dataprep.dataset_input.text().strip()
        if not target_path or not os.path.exists(target_path):
            self.page_dataprep.append_log(f"❌ 請先指定有效的 Dataset 根目錄或 config.yaml 路徑: {target_path}")
            return
        self.datacheck_worker = DataCheckWorker(target_path)
        self.datacheck_worker.log_signal.connect(self.page_dataprep.append_log)
        self.datacheck_worker.finished_signal.connect(self.on_datacheck_finished)
        self.datacheck_worker.start()

    def on_datacheck_finished(self, sample_items, verify_dir=""):
        # 繪製 DataCheck 驗證圖
        grid_layout = self.page_dataprep.grid_layout
        while grid_layout.count():
            item = grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if isinstance(sample_items, str):
            verify_dir = sample_items
            sample_items = []
            if os.path.exists(verify_dir):
                for f in os.listdir(verify_dir):
                    if f.lower().endswith(('.jpg', '.png', '.bmp', '.webp', '.jpeg')):
                        sample_items.append({"img_path": os.path.join(verify_dir, f), "name": f})

        for i, item in enumerate(sample_items[:12]):
            lbl = QLabel()
            lbl.setFixedSize(160, 160)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("border: 1px solid rgba(128,128,128,0.3); border-radius: 8px; background: rgba(0,0,0,0.25);")
            pixmap = QPixmap(item["img_path"])
            if not pixmap.isNull():
                lbl.setPixmap(pixmap.scaled(156, 156, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            b_cnt = item.get("box_count", "")
            tip = f"{item.get('name', '')} ({b_cnt} 個標註)" if b_cnt != "" else item.get('name', '')
            lbl.setToolTip(tip)
            grid_layout.addWidget(lbl, i // 3, i % 3)

        if sample_items:
            self.append_log(f"✨ [DataCheck] 驗證網格已更新，顯示 {min(12, len(sample_items))} 張畫框預覽圖！")

    # --- AI 自動標注 ---
    def start_auto_annotate(self, data):
        self.auto_annotate_worker = AutoAnnotateWorker(
            model_path=data["model_path"],
            source_dir=data["source_dir"],
            project_root=data["project_root"],
            project_name=data["project_name"],
            conf=data["conf"],
            iou=data["iou"],
            imgsz=data["imgsz"],
            device=data["device"],
            world_prompts=data["world_prompts"],
            task_type=data["task_type"],
            auto_split=data["auto_split"],
            split_ratio=data["split_ratio"]
        )
        self.auto_annotate_worker.log_signal.connect(self.page_dataprep.append_auto_log)
        self.auto_annotate_worker.progress_signal.connect(self.page_dataprep.update_auto_progress)
        self.auto_annotate_worker.status_signal.connect(self.page_dataprep.update_auto_status)
        self.auto_annotate_worker.preview_signal.connect(self.page_dataprep.update_auto_preview)
        self.auto_annotate_worker.finished_signal.connect(self.page_dataprep.on_auto_annotate_finished)
        self.auto_annotate_worker.start()

    def stop_auto_annotate(self):
        if self.auto_annotate_worker:
            self.auto_annotate_worker.stop()

    # --- 導出與 CUDA 診斷 ---
    def start_export(self, data):
        self.export_worker = ExportWorker(
            data["model_path"], data["format"],
            self.page_train_config.imgsz_spin.value(),
            data["half"], data["dynamic"], data["simplify"]
        )
        self.export_worker.log_signal.connect(self.append_log)
        self.export_worker.start()

    def check_cuda_status(self):
        self.cuda_worker = CudaCheckWorker()
        self.cuda_worker.info_signal.connect(self.page_export_tools.update_cuda_info)
        self.cuda_worker.start()

    def append_log(self, text):
        self.page_live_train.append_log(text)

    # --- 快取存取 ---
    def save_all_settings_to_cache(self):
        cache = {
            "infer_model_path": self.page_inference.infer_model_input.text().strip(),
            "infer_source_path": self.page_inference.infer_source_input.text().strip(),
            "infer_mode_idx": self.page_inference.infer_mode_combo.currentIndex(),
            "tracker_idx": self.page_inference.tracker_combo.currentIndex(),
            "conf_val": self.page_inference.conf_spin.value(),
            "iou_val": self.page_inference.iou_spin.value(),
            "infer_world_prompts": self.page_inference.infer_world_prompts.text().strip(),
            "dataset_path": self.page_train_config.data_input.text().strip(),
            "task_type_idx": self.page_train_config.task_combo.currentIndex(),
            "world_prompts": self.page_train_config.world_prompts_input.text().strip(),
            "epochs": self.page_train_config.epochs_spin.value(),
            "batch_size": self.page_train_config.batch_spin.value(),
            "imgsz": self.page_train_config.imgsz_spin.value(),
            "device": self.page_train_config.device_input.text().strip(),
            "auto_proj_name": self.page_dataprep.auto_proj_name_input.text().strip(),
            "auto_proj_root": self.page_dataprep.auto_proj_root_input.text().strip(),
            "auto_source_dir": self.page_dataprep.auto_source_input.text().strip(),
            "auto_model_path": self.page_dataprep.auto_model_input.text().strip(),
            "auto_world_prompts": self.page_dataprep.auto_world_prompt_input.text().strip(),
            "auto_conf_val": self.page_dataprep.auto_conf_spin.value(),
            "auto_iou_val": self.page_dataprep.auto_iou_spin.value(),
            "theme_mode": self.theme_mode,
        }
        save_ui_cache(cache)

    def restore_all_settings_from_cache(self):
        cache = load_ui_cache()
        if not cache:
            return
        if "theme_mode" in cache and cache["theme_mode"] in ("dark", "light"):
            self.theme_mode = cache["theme_mode"]
        else:
            self.theme_mode = "dark"

        if "infer_model_path" in cache and cache["infer_model_path"]:
            self.page_inference.infer_model_input.setText(cache["infer_model_path"])
        if "infer_source_path" in cache and cache["infer_source_path"]:
            self.page_inference.infer_source_input.setText(cache["infer_source_path"])
        if "infer_mode_idx" in cache and 0 <= cache["infer_mode_idx"] < self.page_inference.infer_mode_combo.count():
            self.page_inference.infer_mode_combo.setCurrentIndex(cache["infer_mode_idx"])
        if "tracker_idx" in cache and 0 <= cache["tracker_idx"] < self.page_inference.tracker_combo.count():
            self.page_inference.tracker_combo.setCurrentIndex(cache["tracker_idx"])
        if "conf_val" in cache:
            self.page_inference.conf_spin.setValue(float(cache["conf_val"]))
        if "iou_val" in cache:
            self.page_inference.iou_spin.setValue(float(cache["iou_val"]))
        if "infer_world_prompts" in cache and cache["infer_world_prompts"]:
            self.page_inference.infer_world_prompts.setText(cache["infer_world_prompts"])

        if "dataset_path" in cache and cache["dataset_path"]:
            self.page_train_config.data_input.setText(cache["dataset_path"])
        if "task_type_idx" in cache:
            self.page_train_config.task_combo.setCurrentIndex(cache["task_type_idx"])
        if "world_prompts" in cache and cache["world_prompts"]:
            self.page_train_config.world_prompts_input.setText(cache["world_prompts"])
        if "epochs" in cache:
            self.page_train_config.epochs_spin.setValue(int(cache["epochs"]))
        if "batch_size" in cache:
            self.page_train_config.batch_spin.setValue(int(cache["batch_size"]))
        if "imgsz" in cache:
            self.page_train_config.imgsz_spin.setValue(int(cache["imgsz"]))
        if "device" in cache and cache["device"]:
            self.page_train_config.device_input.setText(cache["device"])

        if "auto_proj_name" in cache and cache["auto_proj_name"]:
            self.page_dataprep.auto_proj_name_input.setText(cache["auto_proj_name"])
        if "auto_proj_root" in cache and cache["auto_proj_root"]:
            self.page_dataprep.auto_proj_root_input.setText(cache["auto_proj_root"])
        if "auto_source_dir" in cache and cache["auto_source_dir"]:
            self.page_dataprep.auto_source_input.setText(cache["auto_source_dir"])
        if "auto_model_path" in cache and cache["auto_model_path"]:
            self.page_dataprep.auto_model_input.setText(cache["auto_model_path"])
        if "auto_world_prompts" in cache and cache["auto_world_prompts"]:
            self.page_dataprep.auto_world_prompt_input.setText(cache["auto_world_prompts"])
        if "auto_conf_val" in cache:
            self.page_dataprep.auto_conf_spin.setValue(float(cache["auto_conf_val"]))
        if "auto_iou_val" in cache:
            self.page_dataprep.auto_iou_spin.setValue(float(cache["auto_iou_val"]))

    def init_perf_monitor(self):
        self.perf_thread = PerfMonitorThread(self)
        self.perf_thread.stats_signal.connect(self.on_perf_stats_updated)
        self.perf_thread.start()

    def on_perf_stats_updated(self, cpu_pct, cpu_txt, ram_pct, ram_txt, gpu_pct, gpu_txt, vram_pct, vram_txt):
        self.sidebar.graph_cpu.add_data(cpu_pct, cpu_txt)
        self.sidebar.graph_ram.add_data(ram_pct, ram_txt)
        self.sidebar.graph_gpu.add_data(gpu_pct, gpu_txt)
        self.sidebar.graph_vram.add_data(vram_pct, vram_txt)

    def closeEvent(self, event):
        if hasattr(self, 'perf_thread') and self.perf_thread:
            self.perf_thread.stop()
            self.perf_thread.wait(500)
        self.save_all_settings_to_cache()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NyaUI()
    window.showMaximized()
    sys.exit(app.exec())