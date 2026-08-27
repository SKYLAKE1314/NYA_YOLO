"""
TrainConfigPageWidget — 模型與訓練超參數設定頁面模組
提供全任務 (Detect/Segment/Classify/World) 選擇、3-Tab 模型權重挑選 (Network/Local/Export) 與完整超參數配置
"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFrame, QFormLayout, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QTabWidget, QGroupBox, QScrollArea, QListView, QFileDialog
)
from PySide6.QtCore import Signal, Qt

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))


class TrainConfigPageWidget(QWidget):
    start_train_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background-color: transparent;")

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        card = QFrame()
        card.setObjectName("GoogleCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(20)

        header = QLabel("YOLO 全任務訓練超參數配置")
        header.setObjectName("GoogleCardTitle")
        card_layout.addWidget(header)

        # 1. 任務類型選擇
        task_row = QHBoxLayout()
        task_lbl = QLabel("任務類型:")
        task_lbl.setFixedWidth(110)
        self.task_combo = QComboBox()
        self.task_combo.setView(QListView())
        self.task_combo.addItems([
            "🎯 Detect  (目標檢測)",
            "✂️ Segment (實例分割)",
            "🏷 Classify (圖像二分類 / 多分類)",
            "🌐 World Detection (開放詞彙)"
        ])
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        task_row.addWidget(task_lbl)
        task_row.addWidget(self.task_combo, 1)
        card_layout.addLayout(task_row)

        # 2. 模型權重選擇 (三個 Tab: Network / Local / Export)
        model_group = QFrame()
        model_group_layout = QVBoxLayout(model_group)
        model_group_layout.setContentsMargins(0, 4, 0, 4)
        model_group_layout.setSpacing(8)

        model_tabs = QTabWidget()
        model_tabs.setMinimumHeight(150)

        # Tab A: Network
        tab_dl = QWidget()
        dl_layout = QVBoxLayout(tab_dl)
        dl_layout.setContentsMargins(8, 8, 8, 8)
        dl_layout.setSpacing(6)

        self.model_presets = {
            "🎯 Detect  (目標檢測)": [
                "yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt",
                "yolo26n.pt", "yolo26s.pt", "yolo26m.pt",
                "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
                "yolo12.yaml", "yolo26.yaml",
            ],
            "✂️ Segment (實例分割)": [
                "yolo12n-seg.pt", "yolo12s-seg.pt", "yolo12m-seg.pt",
                "yolo26n-seg.pt", "yolo26s-seg.pt",
                "yolo11n-seg.pt", "yolo11s-seg.pt",
                "yolo12-seg.yaml",
            ],
            "🏷 Classify (圖像二分類 / 多分類)": [
                # ResNet 系列 (二分類 / 缺陷分類首選)
                "ResNet-18 (yolo11-cls-resnet18.yaml)",
                "ResNet-50 (yolov8-cls-resnet50.yaml)",
                "ResNet-101 (yolov8-cls-resnet101.yaml)",
                # YOLO11-cls 系列
                "YOLO11n-cls (yolo11n-cls.pt)",
                "YOLO11s-cls (yolo11s-cls.pt)",
                "YOLO11m-cls (yolo11m-cls.pt)",
                "YOLO11l-cls (yolo11l-cls.pt)",
                # YOLOv8-cls 系列
                "YOLOv8n-cls (yolov8n-cls.pt)",
                "YOLOv8s-cls (yolov8s-cls.pt)",
                "YOLOv8m-cls (yolov8m-cls.pt)",
                # YOLO12-cls 系列
                "YOLO12n-cls (yolo12n-cls.pt)",
                "YOLO12s-cls (yolo12s-cls.pt)",
            ],
            "🌐 World Detection (開放詞彙)": [
                "yolov8s-world.pt", "yolov8m-world.pt", "yolov8l-world.pt",
                "yolov8s-worldv2.pt", "yolov8m-worldv2.pt", "yolov8l-worldv2.pt",
                "yolov8-world.yaml", "yolov8-worldv2.yaml",
            ],
        }

        self.model_dl_combo = QComboBox()
        self.model_dl_combo.setView(QListView())
        self.model_dl_combo.addItems(self.model_presets["🎯 Detect  (目標檢測)"])

        dl_hint = QLabel("💡 首次使用將自動從 Ultralytics Hub 下載預訓練權重")
        dl_hint.setStyleSheet("font-size: 11px; opacity: 0.7;")

        dl_layout.addWidget(self.model_dl_combo)
        dl_layout.addWidget(dl_hint)
        model_tabs.addTab(tab_dl, "☁ Network")

        # Tab B: Local
        tab_local = QWidget()
        loc_main = QVBoxLayout(tab_local)
        loc_main.setContentsMargins(8, 8, 8, 8)
        loc_main.setSpacing(6)

        loc_dir_row = QHBoxLayout()
        self.local_scan_dir = QLineEdit()
        _default_weights = os.path.join(PARENT_DIR, "weights")
        os.makedirs(_default_weights, exist_ok=True)
        self.local_scan_dir.setText(_default_weights)
        self.local_scan_dir.setPlaceholderText("掃描資料夾路徑...")
        btn_scan_dir = QPushButton("📂")
        btn_scan_dir.setFixedWidth(36)
        btn_scan_dir.clicked.connect(self._browse_local_scan_dir)
        btn_refresh = QPushButton("🔄")
        btn_refresh.setFixedWidth(36)
        btn_refresh.clicked.connect(self._refresh_local_models)
        loc_dir_row.addWidget(self.local_scan_dir)
        loc_dir_row.addWidget(btn_scan_dir)
        loc_dir_row.addWidget(btn_refresh)
        loc_main.addLayout(loc_dir_row)

        self.local_model_combo = QComboBox()
        self.local_model_combo.setView(QListView())
        self.local_model_combo.setPlaceholderText("（點擊 🔄 掃描後顯示找到的 .pt 文件）")
        loc_main.addWidget(self.local_model_combo)

        loc_hint = QLabel("💡 預設掃描 weights/ 與 runs/ 本地模型庫，亦可指定其他目錄")
        loc_hint.setStyleSheet("font-size: 11px;")
        loc_main.addWidget(loc_hint)
        model_tabs.addTab(tab_local, "📁 Local")

        # Tab C: Export
        tab_export = QWidget()
        exp_layout = QVBoxLayout(tab_export)
        exp_layout.setContentsMargins(8, 8, 8, 8)
        exp_layout.setSpacing(6)
        exp_hint = QLabel("選擇本機已匯出的模型文件（支援 .pt / .onnx / .engine / .yaml）")
        exp_hint.setStyleSheet("font-size: 11px;")
        exp_row = QHBoxLayout()
        self.model_export_input = QLineEdit()
        self.model_export_input.setPlaceholderText("瀏覽或貼入 .pt / .onnx / .engine 文件路徑...")
        btn_browse_exp = QPushButton("📂 瀏覽")
        btn_browse_exp.setFixedWidth(80)
        btn_browse_exp.clicked.connect(lambda: self._select_file(self.model_export_input, "選擇模型文件"))
        exp_row.addWidget(self.model_export_input)
        exp_row.addWidget(btn_browse_exp)
        exp_layout.addWidget(exp_hint)
        exp_layout.addLayout(exp_row)
        model_tabs.addTab(tab_export, "📤 Export")

        self.model_tabs_widget = model_tabs
        model_group_layout.addWidget(model_tabs)
        card_layout.addWidget(model_group)

        # 3. Dataset Config
        ds_form = QFormLayout()
        ds_form.setVerticalSpacing(12)
        ds_form.setContentsMargins(0, 5, 0, 5)
        self.data_input = QLineEdit(os.path.join(CURRENT_DIR, "NYA_Project", "dataset", "config.yaml"))
        btn_data_select = QPushButton("選擇 Dataset config.yaml")
        btn_data_select.clicked.connect(lambda: self._select_file(self.data_input, "選擇 config.yaml"))
        ds_form.addRow("Dataset config:", self.data_input)
        ds_form.addRow("", btn_data_select)
        card_layout.addLayout(ds_form)

        # 4. World Detection Group
        self.world_prompts_group = QGroupBox("🌐 World Detection — 文字類別提示 (Text Prompts)")
        wp_layout = QVBoxLayout(self.world_prompts_group)
        wp_layout.setContentsMargins(12, 12, 12, 12)
        wp_layout.setSpacing(8)
        wp_hint = QLabel("請輸入您要偵測的類別名稱，以英文逗號分隔（如：person, car, dog）")
        wp_hint.setStyleSheet("font-size: 11px;")
        self.world_prompts_input = QLineEdit()
        self.world_prompts_input.setPlaceholderText("person, car, dog, bicycle, ...")
        wp_layout.addWidget(wp_hint)
        wp_layout.addWidget(self.world_prompts_input)
        self.world_prompts_group.setVisible(False)
        card_layout.addWidget(self.world_prompts_group)

        self._refresh_local_models()

        # 5. 訓練參數模板 (Presets)
        preset_layout = QHBoxLayout()
        preset_layout.setContentsMargins(12, 12, 12, 0)
        preset_lbl = QLabel("預設參數模板:")
        preset_lbl.setStyleSheet("font-weight: bold; color: #D97706;")
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "一般目標檢測 (預設)",
            "World / Character Tracking (文字/小目標追蹤 - 穩定防崩潰)"
        ])
        self.preset_combo.currentIndexChanged.connect(self._apply_training_preset)
        preset_layout.addWidget(preset_lbl)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch()
        card_layout.addLayout(preset_layout)

        # 6. 超參數 Tabs
        tabs = QTabWidget()
        tabs.setMinimumHeight(380)

        # Tab 1: 基礎
        tab_basic = QWidget()
        f_basic = QFormLayout(tab_basic)
        f_basic.setContentsMargins(12, 12, 12, 12)
        f_basic.setVerticalSpacing(12)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(-1, 512)
        self.batch_spin.setValue(16)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setValue(640)

        self.device_input = QLineEdit("0")
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 64)
        self.workers_spin.setValue(4)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 1000)
        self.patience_spin.setValue(50)

        self.pretrained_cb = QCheckBox("Pretrained (預訓練權重)")
        self.pretrained_cb.setChecked(True)
        self.amp_cb = QCheckBox("AMP (自動混合精度)")
        self.amp_cb.setChecked(True)

        f_basic.addRow("Epochs (訓練輪次):", self.epochs_spin)
        f_basic.addRow("Batch Size (-1 為自動):", self.batch_spin)
        f_basic.addRow("Image Size (圖像尺寸):", self.imgsz_spin)
        f_basic.addRow("Device (0, 1 或 cpu):", self.device_input)
        f_basic.addRow("Workers (線程數):", self.workers_spin)
        f_basic.addRow("Patience (早停輪次):", self.patience_spin)
        f_basic.addRow(self.pretrained_cb)
        f_basic.addRow(self.amp_cb)
        tabs.addTab(tab_basic, "1. 基礎與硬體")

        # Tab 2: 優化器
        tab_opt = QWidget()
        f_opt = QFormLayout(tab_opt)
        f_opt.setContentsMargins(12, 12, 12, 12)
        f_opt.setVerticalSpacing(12)
        
        self.opt_combo = QComboBox()
        self.opt_combo.setView(QListView())
        self.opt_combo.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])

        self.lr0_spin = QDoubleSpinBox()
        self.lr0_spin.setDecimals(5)
        self.lr0_spin.setSingleStep(0.0001)
        self.lr0_spin.setValue(0.01)
        self.lrf_spin = QDoubleSpinBox()
        self.lrf_spin.setDecimals(5)
        self.lrf_spin.setSingleStep(0.01)
        self.lrf_spin.setValue(0.01)

        self.cos_lr_cb = QCheckBox("cos_lr (餘弦退火學習率)")
        self.cos_lr_cb.setChecked(True)

        f_opt.addRow("Optimizer (優化器):", self.opt_combo)
        f_opt.addRow("lr0 (初始學習率):", self.lr0_spin)
        f_opt.addRow("lrf (最終學習率比率):", self.lrf_spin)
        f_opt.addRow(self.cos_lr_cb)
        tabs.addTab(tab_opt, "2. 優化器與學習率")

        # Tab 3: 損失權重
        tab_loss = QWidget()
        f_loss = QFormLayout(tab_loss)
        f_loss.setContentsMargins(12, 12, 12, 12)
        f_loss.setVerticalSpacing(12)
        
        self.box_loss_spin = QDoubleSpinBox()
        self.box_loss_spin.setValue(7.5)
        self.cls_loss_spin = QDoubleSpinBox()
        self.cls_loss_spin.setValue(0.5)
        self.dfl_loss_spin = QDoubleSpinBox()
        self.dfl_loss_spin.setValue(1.5)

        f_loss.addRow("box Loss 權重:", self.box_loss_spin)
        f_loss.addRow("cls Loss 權重:", self.cls_loss_spin)
        f_loss.addRow("dfl Loss 權重:", self.dfl_loss_spin)
        tabs.addTab(tab_loss, "3. 損失權重")

        card_layout.addWidget(tabs)

        # 6. 控制按鈕
        btn_row = QHBoxLayout()
        self.btn_start_train = QPushButton("▶ 開始模型訓練")
        self.btn_start_train.setObjectName("GooglePrimaryButton")
        self.btn_start_train.clicked.connect(self._on_start_train)

        btn_row.addStretch()
        btn_row.addWidget(self.btn_start_train)
        card_layout.addLayout(btn_row)

        layout.addWidget(card)
        scroll.setWidget(page)
        root_layout.addWidget(scroll)

        # 初始自動掃描本地模型庫
        self._refresh_local_models()

    def get_selected_model_path(self):
        idx = self.model_tabs_widget.currentIndex()
        if idx == 0:
            raw = self.model_dl_combo.currentText().strip()
            # 支援如 "ResNet-18 (yolo11-cls-resnet18.yaml)" 提取括號內真實權重/設定檔名稱
            if "(" in raw and ")" in raw:
                import re
                m = re.search(r'\(([^)]+)\)', raw)
                if m and (m.group(1).endswith(('.yaml', '.pt', '.onnx', '.engine'))):
                    raw = m.group(1).strip()
            
            # 若 weights/ 目錄下已存在該權重/設定檔，優先回傳 weights/ 內的絕對路徑
            weights_dir = os.path.join(PARENT_DIR, "weights")
            candidate = os.path.join(weights_dir, raw)
            if os.path.exists(candidate):
                return candidate
            return raw
        elif idx == 1:
            sel = self.local_model_combo.currentText().strip()
            if sel and not sel.startswith("（"):
                return sel
            return self.local_scan_dir.text().strip()
        else:
            return self.model_export_input.text().strip()

    def _on_task_changed(self, idx):
        task_text = self.task_combo.currentText()
        if hasattr(self, 'world_prompts_group'):
            self.world_prompts_group.setVisible("World" in task_text)
        if hasattr(self, 'model_dl_combo') and task_text in self.model_presets:
            self.model_dl_combo.clear()
            self.model_dl_combo.addItems(self.model_presets[task_text])

        # 分類任務標準解析度預設為 224
        if hasattr(self, 'imgsz_spin'):
            if "Classify" in task_text and self.imgsz_spin.value() == 640:
                self.imgsz_spin.setValue(224)
            elif "Classify" not in task_text and self.imgsz_spin.value() == 224:
                self.imgsz_spin.setValue(640)

    def _browse_local_scan_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇掃描資料夾", self.local_scan_dir.text())
        if folder:
            self.local_scan_dir.setText(folder)
            self._refresh_local_models()

    def _refresh_local_models(self):
        if not hasattr(self, 'local_model_combo') or not hasattr(self, 'local_scan_dir'):
            return
        scan_root = self.local_scan_dir.text().strip()
        scan_targets = []
        if scan_root and os.path.isdir(scan_root):
            scan_targets.append(scan_root)
        
        weights_dir = os.path.join(PARENT_DIR, "weights")
        runs_dir = os.path.join(PARENT_DIR, "runs")
        for d in [weights_dir, runs_dir]:
            if os.path.isdir(d) and d not in scan_targets:
                scan_targets.append(d)

        found = []
        seen = set()
        for root_d in scan_targets:
            for dirpath, _, filenames in os.walk(root_d):
                for f in filenames:
                    if f.endswith(('.pt', '.yaml', '.onnx', '.engine')):
                        full_p = os.path.join(dirpath, f)
                        if full_p not in seen:
                            seen.add(full_p)
                            found.append(full_p)

        self.local_model_combo.clear()
        if found:
            self.local_model_combo.addItems(found)
        else:
            self.local_model_combo.addItem("（未找到 .pt/.yaml 模型檔案）")

    def _select_file(self, line_edit, title):
        file_path, _ = QFileDialog.getOpenFileName(self, title)
        if file_path:
            line_edit.setText(file_path)

    def _apply_training_preset(self, idx):
        if idx == 1:
            # World / Character Tracking
            self.epochs_spin.setValue(200)
            self.batch_spin.setValue(4)
            self.imgsz_spin.setValue(640)
            self.workers_spin.setValue(0)
            self.pretrained_cb.setChecked(True)
            self.amp_cb.setChecked(False)
            
            # Optimizer & LR
            idx_opt = self.opt_combo.findText("AdamW")
            if idx_opt >= 0: self.opt_combo.setCurrentIndex(idx_opt)
            self.lr0_spin.setValue(0.0005)
            self.lrf_spin.setValue(0.01)
            self.cos_lr_cb.setChecked(True)
            
            # Loss weights
            self.box_loss_spin.setValue(7.5)
            self.cls_loss_spin.setValue(0.5)
            self.dfl_loss_spin.setValue(1.5)
        else:
            # General Default
            self.epochs_spin.setValue(100)
            self.batch_spin.setValue(16)
            self.imgsz_spin.setValue(640)
            self.workers_spin.setValue(4)
            self.pretrained_cb.setChecked(True)
            self.amp_cb.setChecked(True)
            
            idx_opt = self.opt_combo.findText("auto")
            if idx_opt >= 0: self.opt_combo.setCurrentIndex(idx_opt)
            self.lr0_spin.setValue(0.01)
            self.lrf_spin.setValue(0.01)
            self.cos_lr_cb.setChecked(True)
            
            self.box_loss_spin.setValue(7.5)
            self.cls_loss_spin.setValue(0.5)
            self.dfl_loss_spin.setValue(1.5)

    def _on_start_train(self):
        kwargs = {
            "model_path": self.get_selected_model_path(),
            "data": self.data_input.text().strip(),
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "device": self.device_input.text().strip(),
            "workers": self.workers_spin.value(),
            "patience": self.patience_spin.value(),
            "pretrained": self.pretrained_cb.isChecked(),
            "amp": self.amp_cb.isChecked(),
            "optimizer": self.opt_combo.currentText(),
            "lr0": self.lr0_spin.value(),
            "lrf": self.lrf_spin.value(),
            "cos_lr": self.cos_lr_cb.isChecked(),
            "box": self.box_loss_spin.value(),
            "cls": self.cls_loss_spin.value(),
            "dfl": self.dfl_loss_spin.value(),
            "world_classes": [c.strip() for c in self.world_prompts_input.text().split(",") if c.strip()] if "World" in self.task_combo.currentText() else None
        }
        self.start_train_requested.emit(kwargs)
