"""
DataPrepPageWidget — 資料與標註格式轉換 & LabelImg 互動標註工具頁面模組
整合深度學習工具風格工作流：匯入圖像/標註集 -> 自動專案固定目錄 (NYA_Project/) -> 一鍵拆分與生成 config.yaml -> 一鍵直達模型訓練
"""

import os
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QFrame, QFormLayout, QComboBox, QCheckBox, QDoubleSpinBox, QTextEdit,
    QScrollArea, QGridLayout, QFileDialog, QTabWidget, QListWidget,
    QListWidgetItem, QInputDialog, QMessageBox, QSizePolicy, QProgressBar
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QKeySequence, QShortcut, QPixmap, QImage
from components.annotation_canvas import AnnotationCanvasWidget
from project_manager import HalconProjectManager

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

COCO8_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


class DataPrepPageWidget(QWidget):
    start_convert_requested = Signal(dict)
    start_datacheck_requested = Signal()
    jump_to_train_requested = Signal(str)
    start_auto_annotate_requested = Signal(dict)
    stop_auto_annotate_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_mgr = HalconProjectManager(PARENT_DIR)
        self.image_files = []
        self.current_img_idx = -1
        self.label_dir = self.project_mgr.labels_dir
        self.class_list = ["NG", "OK"]
        self.last_auto_result = None
        self.init_ui()

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)

        self.main_tabs = QTabWidget()
        self.main_tabs.setObjectName("GoogleTabWidget")

        # ── Tab 1: XML/JSON 轉換與 DataCheck ────────────────────────
        tab_convert = QWidget()
        conv_layout = QHBoxLayout(tab_convert)
        conv_layout.setContentsMargins(12, 12, 12, 12)

        left_card = QFrame()
        left_card.setObjectName("GoogleCard")

        card_outer_layout1 = QVBoxLayout(left_card)
        card_outer_layout1.setContentsMargins(0, 0, 0, 0)
        card_outer_layout1.setSpacing(0)

        scroll_area1 = QScrollArea()
        scroll_area1.setWidgetResizable(True)
        scroll_area1.setFrameShape(QFrame.NoFrame)
        scroll_area1.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        scroll_content1 = QWidget()
        scroll_content1.setObjectName("CardScrollContent")
        left_layout = QVBoxLayout(scroll_content1)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(6)

        header = QLabel("資料與標註格式轉換")
        header.setObjectName("GoogleCardTitle")
        left_layout.addWidget(header)

        form = QFormLayout()
        form.setVerticalSpacing(8)
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems([
            "⚖️ 二分類 (OK / NG 良品與不良品直接建構)",
            "🏷️ 多分類 (Multi-Class 圖像分類)",
            "🎯 目標檢測 (Detect - XML/JSON 轉檔)",
            "✂️ 實例分割 (Segment - JSON 多邊形轉檔)"
        ])
        self.task_type_combo.currentIndexChanged.connect(self._on_prep_task_changed)
        form.addRow("任務類型:", self.task_type_combo)
        left_layout.addLayout(form)

        # ── 群組 A: 二分類專用欄位 (選擇 OK / NG 資料夾) ──
        self.binary_cls_group = QWidget()
        bin_form = QFormLayout(self.binary_cls_group)
        bin_form.setContentsMargins(0, 0, 0, 0)
        bin_form.setVerticalSpacing(8)

        self.ok_input = QLineEdit()
        btn_ok = QPushButton("選擇 OK 良品資料夾 🟢")
        btn_ok.clicked.connect(lambda: self._select_folder(self.ok_input))
        bin_form.addRow("良品 (OK) 影像:", self.ok_input)
        bin_form.addRow("", btn_ok)

        self.ng_input = QLineEdit()
        btn_ng = QPushButton("選擇 NG 不良品資料夾 🔴")
        btn_ng.clicked.connect(lambda: self._select_folder(self.ng_input))
        bin_form.addRow("不良品 (NG) 影像:", self.ng_input)
        bin_form.addRow("", btn_ng)
        left_layout.addWidget(self.binary_cls_group)

        # ── 群組 B: 多分類專用欄位 (選擇分類總資料夾) ──
        self.multi_cls_group = QWidget()
        multi_form = QFormLayout(self.multi_cls_group)
        multi_form.setContentsMargins(0, 0, 0, 0)
        multi_form.setVerticalSpacing(8)

        self.multi_cls_input = QLineEdit()
        btn_multi = QPushButton("選擇分類資料夾 (內含各類別子目錄)")
        btn_multi.clicked.connect(lambda: self._select_folder(self.multi_cls_input))
        multi_form.addRow("分類總目錄:", self.multi_cls_input)
        multi_form.addRow("", btn_multi)
        left_layout.addWidget(self.multi_cls_group)

        # ── 群組 C: 檢測與分割專用欄位 (標註檔與影像) ──
        self.detect_seg_group = QWidget()
        det_form = QFormLayout(self.detect_seg_group)
        det_form.setContentsMargins(0, 0, 0, 0)
        det_form.setVerticalSpacing(8)

        self.anno_input = QLineEdit()
        btn_anno = QPushButton("選擇標註資料夾")
        btn_anno.clicked.connect(lambda: self._select_folder(self.anno_input))
        det_form.addRow("標註資料夾:", self.anno_input)
        det_form.addRow("", btn_anno)

        self.image_input = QLineEdit()
        btn_img = QPushButton("選擇影像資料夾")
        btn_img.clicked.connect(lambda: self._select_folder(self.image_input))
        det_form.addRow("影像資料夾:", self.image_input)
        det_form.addRow("", btn_img)

        self.auto_class_cb = QCheckBox("Auto Classes (自動提取標註檔類別名單)")
        self.auto_class_cb.setChecked(True)
        det_form.addRow(self.auto_class_cb)

        self.class_input = QLineEdit("NG")
        det_form.addRow("手動指定類別 (逗號分隔):", self.class_input)
        left_layout.addWidget(self.detect_seg_group)

        # ── 通用欄位: Dataset 根目錄與 Val 比例 ──
        common_form = QFormLayout()
        common_form.setVerticalSpacing(8)

        self.dataset_input = QLineEdit(self.project_mgr.dataset_dir)
        btn_dataset = QPushButton("選擇 Dataset 根目錄")
        btn_dataset.clicked.connect(lambda: self._select_folder(self.dataset_input))
        common_form.addRow("Dataset 根目錄:", self.dataset_input)
        common_form.addRow("", btn_dataset)

        self.split_ratio_spin = QDoubleSpinBox()
        self.split_ratio_spin.setRange(0.05, 0.5)
        self.split_ratio_spin.setValue(0.2)
        common_form.addRow("Val 驗證集比例:", self.split_ratio_spin)

        left_layout.addLayout(common_form)
        left_layout.addSpacing(10)

        self.btn_start_convert = QPushButton("🚀 一鍵建立二分類資料集並生成 Config")
        self.btn_start_convert.setObjectName("GoogleAmberButton")
        self.btn_start_convert.clicked.connect(self._on_convert_click)
        left_layout.addWidget(self.btn_start_convert)

        # 初始執行一次切換以設定可見度
        self._on_prep_task_changed(0)

        btn_datacheck = QPushButton("執行 DataCheck 數據集驗證")
        btn_datacheck.setObjectName("GoogleSecondaryButton")
        btn_datacheck.clicked.connect(lambda: self.start_datacheck_requested.emit())
        left_layout.addWidget(btn_datacheck)

        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("轉換日誌:"))
        self.convert_log = QTextEdit()
        self.convert_log.setObjectName("GoogleLogViewer")
        self.convert_log.setReadOnly(True)
        self.convert_log.setMaximumHeight(90)
        left_layout.addWidget(self.convert_log)

        left_layout.addStretch()
        
        scroll_area1.setWidget(scroll_content1)
        card_outer_layout1.addWidget(scroll_area1)

        conv_layout.addWidget(left_card, 1)

        right_card = QFrame()
        right_card.setObjectName("GoogleCard")
        right_layout = QVBoxLayout(right_card)

        v_header = QLabel("DataCheck 畫框預覽網格")
        v_header.setObjectName("GoogleCardTitle")
        right_layout.addWidget(v_header)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.scroll_area.setWidget(self.grid_widget)

        right_layout.addWidget(self.scroll_area)
        conv_layout.addWidget(right_card, 1)

        self.main_tabs.addTab(tab_convert, "🔄 XML/JSON 轉 YOLO & DataCheck")

        # ── Tab 2:  工作流與 LabelImg 標註工具 ────────────
        tab_labeler = QWidget()
        labeler_layout = QHBoxLayout(tab_labeler)
        labeler_layout.setContentsMargins(12, 12, 12, 12)

        # 左側控制器面板
        lbl_ctrl_card = QFrame()
        lbl_ctrl_card.setObjectName("GoogleCard")
        lbl_ctrl_card.setFixedWidth(350)

        card_outer_layout = QVBoxLayout(lbl_ctrl_card)
        card_outer_layout.setContentsMargins(0, 0, 0, 0)
        card_outer_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        scroll_content = QWidget()
        scroll_content.setObjectName("CardScrollContent")
        ctrl_layout = QVBoxLayout(scroll_content)
        ctrl_layout.setContentsMargins(12, 12, 12, 12)
        ctrl_layout.setSpacing(0)

        # 上方按鈕群組
        top_layout = QVBoxLayout()
        top_layout.setSpacing(6)

        c_title = QLabel("✨ NYA 一鍵工作流")
        c_title.setObjectName("GoogleCardTitle")
        ctrl_layout.addWidget(c_title)

        # NYA 工作流 3 步驟快捷按鈕列
        btn_step1_img = QPushButton("📁 步驟 1: 匯入圖像資料夾")
        btn_step1_img.clicked.connect(self._import_images_dir)

        btn_step1_lbl = QPushButton("🏷️ 匯入已有標註集 (可選)")
        btn_step1_lbl.setObjectName("GoogleSecondaryButton")
        btn_step1_lbl.clicked.connect(self._import_labels_dir)

        btn_step2_split = QPushButton("⚡ 步驟 2: 一鍵拆分並生成 config.yaml")
        btn_step2_split.setObjectName("GoogleSecondaryButton")
        btn_step2_split.clicked.connect(self._halcon_split_dataset)

        btn_step3_train = QPushButton("🚀 步驟 3: 立即開啟模型訓練 ➔")
        btn_step3_train.setObjectName("GoogleAmberButton")
        btn_step3_train.clicked.connect(self._halcon_jump_to_train)

        top_layout.addWidget(btn_step1_img)
        top_layout.addWidget(btn_step1_lbl)
        top_layout.addWidget(btn_step2_split)
        top_layout.addWidget(btn_step3_train)

        top_layout.addSpacing(10)
        top_layout.addWidget(QLabel("專案圖像清單 (NYA_Project/):"))
        ctrl_layout.addLayout(top_layout)

        self.img_list_widget = QListWidget()
        self.img_list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.img_list_widget.setMinimumHeight(80)
        self.img_list_widget.setMaximumHeight(150)
        self.img_list_widget.currentRowChanged.connect(self._on_image_selected)
        ctrl_layout.addWidget(self.img_list_widget, 0)

        # 下方控制群組
        bot_layout = QVBoxLayout()
        bot_layout.setSpacing(6)
        
        # 類別管理與 COCO8
        bot_layout.addSpacing(6)
        bot_layout.addWidget(QLabel("當前標註類別:"))

        cls_box = QHBoxLayout()
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.class_list)
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)

        btn_add_cls = QPushButton("➕")
        btn_add_cls.setFixedWidth(36)
        btn_add_cls.setToolTip("新增自訂類別")
        btn_add_cls.clicked.connect(self._add_custom_class)

        btn_coco8 = QPushButton("📦 COCO8 預設 (80類)")
        btn_coco8.setToolTip("載入標準 COCO 80 類別名單")
        btn_coco8.clicked.connect(self._load_coco8_classes)

        cls_box.addWidget(self.class_combo, 1)
        cls_box.addWidget(btn_add_cls)
        bot_layout.addLayout(cls_box)
        bot_layout.addWidget(btn_coco8)

        # Val 比例
        bot_layout.addSpacing(6)
        val_row = QHBoxLayout()
        val_row.addWidget(QLabel("Val 比例:"))
        self.labeler_val_spin = QDoubleSpinBox()
        self.labeler_val_spin.setRange(0.05, 0.5)
        self.labeler_val_spin.setValue(0.2)
        self.labeler_val_spin.setSingleStep(0.05)
        val_row.addWidget(self.labeler_val_spin)
        bot_layout.addLayout(val_row)

        # 操作按鈕
        bot_layout.addSpacing(8)
        nav_row = QHBoxLayout()
        self.btn_prev = QPushButton("⬅ 上一張 (A)")
        self.btn_prev.clicked.connect(self._prev_image)
        self.btn_next = QPushButton("➡ 下一張 (D)")
        self.btn_next.clicked.connect(self._next_image)
        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.btn_next)
        bot_layout.addLayout(nav_row)

        self.btn_save_anno = QPushButton("💾 儲存標註 (Ctrl+S)")
        self.btn_save_anno.clicked.connect(self._save_current_annotation)
        bot_layout.addWidget(self.btn_save_anno)

        btn_delete_selected = QPushButton("❌ 刪除選取框 (Delete)")
        btn_delete_selected.setObjectName("GoogleSecondaryButton")
        btn_delete_selected.clicked.connect(lambda: self.canvas.delete_selected_box())
        bot_layout.addWidget(btn_delete_selected)

        btn_clear = QPushButton("🗑 清除此圖所有畫框")
        btn_clear.setObjectName("GoogleSecondaryButton")
        btn_clear.clicked.connect(self._clear_canvas_boxes)
        bot_layout.addWidget(btn_clear)
        
        ctrl_layout.addSpacing(6)
        ctrl_layout.addLayout(bot_layout)
        ctrl_layout.addStretch()

        scroll_area.setWidget(scroll_content)
        card_outer_layout.addWidget(scroll_area)

        labeler_layout.addWidget(lbl_ctrl_card, 0)

        # 右側繪圖畫布
        canvas_card = QFrame()
        canvas_card.setObjectName("GoogleCard")
        canvas_layout = QVBoxLayout(canvas_card)

        self.lbl_status = QLabel("待命：點擊「步驟 1」匯入圖像資料夾開啟標註器...")
        self.lbl_status.setObjectName("GoogleCardTitle")
        canvas_layout.addWidget(self.lbl_status)

        self.canvas = AnnotationCanvasWidget()
        self.canvas.annotation_changed.connect(self._on_annotation_changed)
        canvas_layout.addWidget(self.canvas, 1)

        hint_bar = QLabel("💡 操作提示：【點擊框體】拖曳移動 | 【8個控制點】拖曳縮放 | 【Delete / Backspace】刪除選取框 | 【Esc】取消選取")
        hint_bar.setStyleSheet("font-size: 11px; opacity: 0.8; margin-top: 4px;")
        canvas_layout.addWidget(hint_bar)

        labeler_layout.addWidget(canvas_card, 1)

        self.main_tabs.addTab(tab_labeler, "🏷️ LabelImg 互動標註 & NYA 工作流")

        # ── Tab 3: AI 模型批次自動標注 (Auto-Annotation) ────────────
        tab_auto = QWidget()
        auto_layout = QHBoxLayout(tab_auto)
        auto_layout.setContentsMargins(12, 12, 12, 12)

        # 左側控制卡片
        auto_ctrl_card = QFrame()
        auto_ctrl_card.setObjectName("GoogleCard")
        auto_ctrl_card.setFixedWidth(380)

        card_outer_layout3 = QVBoxLayout(auto_ctrl_card)
        card_outer_layout3.setContentsMargins(0, 0, 0, 0)
        card_outer_layout3.setSpacing(0)

        scroll_area3 = QScrollArea()
        scroll_area3.setWidgetResizable(True)
        scroll_area3.setFrameShape(QFrame.NoFrame)
        scroll_area3.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        scroll_content3 = QWidget()
        scroll_content3.setObjectName("CardScrollContent")
        left_layout3 = QVBoxLayout(scroll_content3)
        left_layout3.setContentsMargins(16, 16, 16, 16)
        left_layout3.setSpacing(6)

        header3 = QLabel("AI 模型批次自動標注")
        header3.setObjectName("GoogleCardTitle")
        left_layout3.addWidget(header3)

        form3 = QFormLayout()
        form3.setVerticalSpacing(8)

        # 專案名稱與目錄
        self.auto_proj_name_input = QLineEdit("NYA_AutoLabel")
        form3.addRow("專案名稱:", self.auto_proj_name_input)

        self.auto_proj_root_input = QLineEdit(PARENT_DIR)
        btn_proj_root = QPushButton("選擇專案輸出根目錄")
        btn_proj_root.clicked.connect(lambda: self._select_folder(self.auto_proj_root_input))
        form3.addRow("輸出根目錄:", self.auto_proj_root_input)
        form3.addRow("", btn_proj_root)

        self.auto_source_input = QLineEdit()
        btn_source_dir = QPushButton("選擇待標注圖像資料夾")
        btn_source_dir.clicked.connect(lambda: self._select_folder(self.auto_source_input))
        form3.addRow("待標注圖像目錄:", self.auto_source_input)
        form3.addRow("", btn_source_dir)

        # 模型與推論參數
        self.auto_model_input = QLineEdit("yolov8s-worldv2.pt")
        btn_select_model = QPushButton("選擇模型檔案 (.pt)")
        btn_select_model.clicked.connect(lambda: self._select_file(self.auto_model_input, "選擇 YOLO 模型"))
        form3.addRow("推論模型路徑:", self.auto_model_input)
        form3.addRow("", btn_select_model)

        self.auto_task_combo = QComboBox()
        self.auto_task_combo.addItems(["detect (目標檢測)", "segment (實例分割)"])
        form3.addRow("任務模式:", self.auto_task_combo)

        self.auto_conf_spin = QDoubleSpinBox()
        self.auto_conf_spin.setRange(0.01, 1.0)
        self.auto_conf_spin.setSingleStep(0.05)
        self.auto_conf_spin.setValue(0.25)
        form3.addRow("Conf (置信度閾值):", self.auto_conf_spin)

        self.auto_iou_spin = QDoubleSpinBox()
        self.auto_iou_spin.setRange(0.01, 1.0)
        self.auto_iou_spin.setSingleStep(0.05)
        self.auto_iou_spin.setValue(0.45)
        form3.addRow("IoU (重疊抑制閾值):", self.auto_iou_spin)

        self.auto_imgsz_combo = QComboBox()
        self.auto_imgsz_combo.addItems(["640", "320", "480", "800", "1024", "1280"])
        form3.addRow("推論尺寸 (ImgSz):", self.auto_imgsz_combo)

        self.auto_device_input = QLineEdit("0")
        form3.addRow("運算裝置 (Device):", self.auto_device_input)

        # World 檢測提示詞
        self.auto_world_prompt_input = QLineEdit("defect, scratch, text, label")
        self.auto_world_prompt_input.setPlaceholderText("例如: defect, scratch 或留空使用模型自帶類別")
        form3.addRow("World 類別提示詞:", self.auto_world_prompt_input)

        # 拆分與生成
        self.auto_split_cb = QCheckBox("自動切分 Train/Val 並生成 config.yaml")
        self.auto_split_cb.setChecked(True)
        form3.addRow(self.auto_split_cb)

        self.auto_split_ratio_spin = QDoubleSpinBox()
        self.auto_split_ratio_spin.setRange(0.05, 0.5)
        self.auto_split_ratio_spin.setSingleStep(0.05)
        self.auto_split_ratio_spin.setValue(0.2)
        form3.addRow("Val 驗證集比例:", self.auto_split_ratio_spin)

        left_layout3.addLayout(form3)
        left_layout3.addSpacing(10)

        # 操作按鈕
        self.btn_start_auto = QPushButton("▶ 啟動 AI 批次自動標注")
        self.btn_start_auto.setObjectName("GoogleAmberButton")
        self.btn_start_auto.clicked.connect(self._on_auto_annotate_click)
        left_layout3.addWidget(self.btn_start_auto)

        self.btn_stop_auto = QPushButton("⏹ 停止標注")
        self.btn_stop_auto.setObjectName("GoogleSecondaryButton")
        self.btn_stop_auto.clicked.connect(self._on_stop_auto_annotate_click)
        self.btn_stop_auto.setEnabled(False)
        left_layout3.addWidget(self.btn_stop_auto)

        self.btn_load_to_labeler = QPushButton("載入至標註視窗復核 (Tab 2) ➔")
        self.btn_load_to_labeler.setObjectName("GoogleSecondaryButton")
        self.btn_load_to_labeler.clicked.connect(self._load_auto_annotated_to_labeler)
        left_layout3.addWidget(self.btn_load_to_labeler)

        self.btn_jump_to_train_auto = QPushButton("模型訓練 (Page 2) ➔")
        self.btn_jump_to_train_auto.setObjectName("GoogleAmberButton")
        self.btn_jump_to_train_auto.clicked.connect(self._jump_to_train_from_auto)
        left_layout3.addWidget(self.btn_jump_to_train_auto)

        left_layout3.addSpacing(8)
        self.auto_progress_bar = QProgressBar()
        self.auto_progress_bar.setRange(0, 100)
        self.auto_progress_bar.setValue(0)
        left_layout3.addWidget(self.auto_progress_bar)

        self.auto_stat_lbl = QLabel("準備好：設定參數後點擊【啟動 AI 批次自動標注】...")
        self.auto_stat_lbl.setStyleSheet("font-size: 11px; opacity: 0.9;")
        self.auto_stat_lbl.setWordWrap(True)
        left_layout3.addWidget(self.auto_stat_lbl)

        left_layout3.addSpacing(6)
        left_layout3.addWidget(QLabel("自動標注日誌:"))
        self.auto_log = QTextEdit()
        self.auto_log.setObjectName("GoogleLogViewer")
        self.auto_log.setReadOnly(True)
        self.auto_log.setMaximumHeight(90)
        left_layout3.addWidget(self.auto_log)

        left_layout3.addStretch()

        scroll_area3.setWidget(scroll_content3)
        card_outer_layout3.addWidget(scroll_area3)
        auto_layout.addWidget(auto_ctrl_card, 0)

        # 右側實時預覽卡片
        auto_right_card = QFrame()
        auto_right_card.setObjectName("GoogleCard")
        right_layout3 = QVBoxLayout(auto_right_card)

        r3_header = QLabel("自動標注即時預覽")
        r3_header.setObjectName("GoogleCardTitle")
        right_layout3.addWidget(r3_header)

        self.auto_preview_lbl = QLabel()
        self.auto_preview_lbl.setAlignment(Qt.AlignCenter)
        self.auto_preview_lbl.setStyleSheet("background-color: #121214; border-radius: 8px; border: 1px dashed #444746;")
        self.auto_preview_lbl.setText("即時推論影像與檢測框預覽畫面...")
        right_layout3.addWidget(self.auto_preview_lbl, 1)

        self.auto_summary_lbl = QLabel("📊 標注統計：尚未啟動標注任務")
        self.auto_summary_lbl.setStyleSheet("font-size: 12px; font-weight: 500; padding: 4px;")
        right_layout3.addWidget(self.auto_summary_lbl)

        auto_layout.addWidget(auto_right_card, 1)

        self.main_tabs.addTab(tab_auto, "AI 模型自動標注 (Auto-Annotation)")
        root_layout.addWidget(self.main_tabs)

        # 鍵盤快捷鍵 (A: 上一張, D: 下一張, Ctrl+S: 儲存)
        self.shortcut_prev = QShortcut(QKeySequence("A"), self)
        self.shortcut_prev.activated.connect(self._prev_image)

        self.shortcut_next = QShortcut(QKeySequence("D"), self)
        self.shortcut_next.activated.connect(self._next_image)

        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save.activated.connect(self._save_current_annotation)

        # 初始嘗試載入既有 NYA_Project
        self._refresh_project_file_list()

    # --- NYA 工作流 3 步驟邏輯 ---
    def _import_images_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇圖像資料夾 (Import Images)")
        if not folder:
            return
        raw_dir, _ = self.project_mgr.setup_project_from_folders(folder, copy_files=True, clear_existing=True)
        self._refresh_project_file_list()
        QMessageBox.information(self, "匯入成功", f"已成功匯入圖像並建立專案目錄！\n專案位置: {raw_dir}")

    def _import_labels_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇已有標註集資料夾 (Import Labels)")
        if not folder:
            return
        _, lbl_dir = self.project_mgr.setup_project_from_folders(self.project_mgr.raw_images_dir, folder, copy_files=True)
        self._refresh_project_file_list()
        QMessageBox.information(self, "標註集匯入成功", f"已將已有標註檔匯入至專案！\n標註位置: {lbl_dir}")

    def _refresh_project_file_list(self):
        raw_dir = self.project_mgr.raw_images_dir
        self.label_dir = self.project_mgr.labels_dir
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

        if os.path.exists(raw_dir):
            self.image_files = [
                os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
                if f.lower().endswith(exts)
            ]
            self.image_files.sort()

        self.img_list_widget.clear()
        for p in self.image_files:
            fname = os.path.basename(p)
            txt_name = os.path.splitext(fname)[0] + ".txt"
            txt_path = os.path.join(self.label_dir, txt_name)
            is_labeled = os.path.exists(txt_path) and os.path.getsize(txt_path) > 0

            item_text = f"✅ {fname}" if is_labeled else f"📄 {fname}"
            item = QListWidgetItem(item_text)
            self.img_list_widget.addItem(item)

        if self.image_files:
            self.img_list_widget.setCurrentRow(0)
            self.lbl_status.setText(f"專案共有 {len(self.image_files)} 張圖片 | 標註目錄: NYA_Project/labels/")

        # Auto-load classes.txt if it exists
        classes_txt_path = os.path.join(self.label_dir, "classes.txt")
        if os.path.exists(classes_txt_path):
            try:
                with open(classes_txt_path, "r", encoding="utf-8") as f:
                    loaded_classes = [line.strip() for line in f if line.strip()]
                if loaded_classes:
                    self.class_list = loaded_classes
                    self.class_combo.clear()
                    self.class_combo.addItems(self.class_list)
                    self.canvas.set_class_names(self.class_list)
            except Exception:
                pass

    def _halcon_split_dataset(self):
        self._save_current_annotation()
        try:
            res = self.project_mgr.split_and_build_dataset(
                val_ratio=self.labeler_val_spin.value(),
                class_names=self.class_list
            )
            cfg_path = res["config_path"]
            QMessageBox.information(
                self, "拆分與 config.yaml 生成成功",
                f"✨ [NYA 工作流] 資料集拆分成功！\n\n"
                f"• 訓練集 (Train): {res['train_count']} 張\n"
                f"• 驗證集 (Val): {res['val_count']} 張\n"
                f"• 標註類別數: {len(res['class_names'])} 個\n"
                f"• Config 檔案: {cfg_path}"
            )
            return cfg_path
        except Exception as e:
            QMessageBox.warning(self, "拆分錯誤", f"無法完成拆分: {e}")
            return None

    def _halcon_jump_to_train(self):
        cfg_path = self._halcon_split_dataset()
        if cfg_path:
            self.jump_to_train_requested.emit(cfg_path)

    # --- LabelImg 畫布互動事件 ---
    def _on_image_selected(self, row):
        if row < 0 or row >= len(self.image_files):
            return
        self.current_img_idx = row
        img_path = self.image_files[row]
        fname = os.path.basename(img_path)
        base_name = os.path.splitext(fname)[0]
        txt_path = os.path.join(self.label_dir, f"{base_name}.txt")

        self.canvas.load_image(img_path, txt_path, self.class_list)
        box_cnt = len(self.canvas.boxes)
        self.lbl_status.setText(f"[{row+1}/{len(self.image_files)}] {fname} — 已有 {box_cnt} 個標註框")

    def _on_class_changed(self, idx):
        if idx >= 0:
            self.canvas.set_current_class(idx)

    def _add_custom_class(self):
        text, ok = QInputDialog.getText(self, "新增標註類別", "請輸入類別名稱:")
        if ok and text.strip():
            cname = text.strip()
            if cname not in self.class_list:
                self.class_list.append(cname)
                self.class_combo.addItem(cname)
                self.class_combo.setCurrentIndex(len(self.class_list) - 1)
                self.canvas.set_class_names(self.class_list)

    def _load_coco8_classes(self):
        self.class_list = list(COCO8_CLASSES)
        self.class_combo.clear()
        self.class_combo.addItems(self.class_list)
        self.canvas.set_class_names(self.class_list)
        QMessageBox.information(self, "COCO8 載入成功", "已為標註器一鍵載入標準 COCO 80 類別名稱庫！")

    def _save_current_annotation(self):
        if self.current_img_idx < 0 or self.current_img_idx >= len(self.image_files):
            return
        img_path = self.image_files[self.current_img_idx]
        fname = os.path.basename(img_path)
        base_name = os.path.splitext(fname)[0]
        txt_path = os.path.join(self.label_dir, f"{base_name}.txt")

        self.canvas.save_yolo_labels(txt_path)

        classes_txt = os.path.join(self.label_dir, "classes.txt")
        with open(classes_txt, "w", encoding="utf-8") as f:
            for cname in self.class_list:
                f.write(f"{cname}\n")

        item = self.img_list_widget.item(self.current_img_idx)
        if item:
            item.setText(f"✅ {fname}")

        self.lbl_status.setText(f"✅ 已儲存標註至: NYA_Project/labels/{base_name}.txt ({len(self.canvas.boxes)} 個框)")

    def _prev_image(self):
        if self.current_img_idx > 0:
            self._save_current_annotation()
            self.img_list_widget.setCurrentRow(self.current_img_idx - 1)

    def _next_image(self):
        if self.current_img_idx < len(self.image_files) - 1:
            self._save_current_annotation()
            self.img_list_widget.setCurrentRow(self.current_img_idx + 1)

    def _clear_canvas_boxes(self):
        self.canvas.clear_boxes()
        self._save_current_annotation()

    def _on_annotation_changed(self):
        if self.current_img_idx >= 0 and self.current_img_idx < len(self.image_files):
            fname = os.path.basename(self.image_files[self.current_img_idx])
            box_cnt = len(self.canvas.boxes)
            self.lbl_status.setText(f"[{self.current_img_idx+1}/{len(self.image_files)}] {fname} — 已編輯 ({box_cnt} 個標註框, 按 Ctrl+S 儲存)")

    def _select_file(self, line_edit, title="選擇檔案", filter_str="YOLO 模型 (*.pt);;所有檔案 (*.*)"):
        file_path, _ = QFileDialog.getOpenFileName(self, title, "", filter_str)
        if file_path:
            line_edit.setText(file_path)

    def _select_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "選擇資料夾")
        if folder:
            line_edit.setText(folder)

    def _on_prep_task_changed(self, idx):
        if not hasattr(self, 'binary_cls_group') or not hasattr(self, 'multi_cls_group') or not hasattr(self, 'detect_seg_group'):
            return
        txt = self.task_type_combo.currentText()
        is_binary = "二分類" in txt or "binary" in txt
        is_multi = "多分類" in txt or "multi" in txt
        is_det_seg = not is_binary and not is_multi
        
        self.binary_cls_group.setVisible(is_binary)
        self.multi_cls_group.setVisible(is_multi)
        self.detect_seg_group.setVisible(is_det_seg)
        
        if hasattr(self, 'btn_start_convert'):
            if is_binary:
                self.btn_start_convert.setText("🚀 一鍵建立二分類資料集並生成 Config")
            elif is_multi:
                self.btn_start_convert.setText("🚀 一鍵建立多分類資料集並生成 Config")
            else:
                self.btn_start_convert.setText("開始標註轉換與生成 Config.yaml")

    def _on_convert_click(self):
        txt = self.task_type_combo.currentText()
        if "二分類" in txt or "binary" in txt:
            data = {
                "task_type": "binary_cls",
                "ok_dir": self.ok_input.text().strip(),
                "ng_dir": self.ng_input.text().strip(),
                "output_root": self.dataset_input.text().strip(),
                "val_ratio": self.split_ratio_spin.value()
            }
        elif "多分類" in txt or "multi" in txt:
            data = {
                "task_type": "multi_cls",
                "image_dir": self.multi_cls_input.text().strip(),
                "output_root": self.dataset_input.text().strip(),
                "val_ratio": self.split_ratio_spin.value()
            }
        else:
            data = {
                "task_type": "segment" if "segment" in txt or "實例分割" in txt else "detect",
                "anno_dir": self.anno_input.text().strip(),
                "image_dir": self.image_input.text().strip(),
                "output_root": self.dataset_input.text().strip(),
                "auto_class": self.auto_class_cb.isChecked(),
                "class_str": self.class_input.text().strip(),
                "val_ratio": self.split_ratio_spin.value()
            }
        self.start_convert_requested.emit(data)

    def append_log(self, text):
        self.convert_log.append(text)

    # ── Tab 3 AI 自動標注交互與槽函數 ────────────────────────
    def _on_auto_annotate_click(self):
        source_dir = self.auto_source_input.text().strip()
        model_path = self.auto_model_input.text().strip()
        proj_root = self.auto_proj_root_input.text().strip()
        proj_name = self.auto_proj_name_input.text().strip() or "NYA_AutoLabel"

        if not source_dir or not os.path.exists(source_dir):
            QMessageBox.warning(self, "路徑錯誤", "請指定有效的待標注圖像資料夾！")
            return

        if not model_path:
            QMessageBox.warning(self, "模型錯誤", "請指定有效的推論模型路徑 (.pt)！")
            return

        # 整理 World Prompts
        raw_prompts = self.auto_world_prompt_input.text().strip()
        world_prompts = [p.strip() for p in raw_prompts.split(",") if p.strip()] if raw_prompts else None

        data = {
            "model_path": model_path,
            "source_dir": source_dir,
            "project_root": proj_root,
            "project_name": proj_name,
            "conf": self.auto_conf_spin.value(),
            "iou": self.auto_iou_spin.value(),
            "imgsz": int(self.auto_imgsz_combo.currentText()),
            "device": self.auto_device_input.text().strip() or "0",
            "world_prompts": world_prompts,
            "task_type": "segment" if "segment" in self.auto_task_combo.currentText() else "detect",
            "auto_split": self.auto_split_cb.isChecked(),
            "split_ratio": self.auto_split_ratio_spin.value()
        }

        self.btn_start_auto.setEnabled(False)
        self.btn_stop_auto.setEnabled(True)
        self.auto_progress_bar.setValue(0)
        self.auto_log.clear()
        self.start_auto_annotate_requested.emit(data)

    def _on_stop_auto_annotate_click(self):
        self.stop_auto_annotate_requested.emit()
        self.auto_stat_lbl.setText("⏹ 已發送停止指令，正在安全終止...")

    def _load_auto_annotated_to_labeler(self):
        target_dir = None
        if self.last_auto_result and "project_dir" in self.last_auto_result:
            target_dir = self.last_auto_result["project_dir"]
        else:
            proj_root = self.auto_proj_root_input.text().strip()
            proj_name = self.auto_proj_name_input.text().strip() or "NYA_AutoLabel"
            candidate = os.path.join(proj_root, proj_name)
            if os.path.exists(candidate):
                target_dir = candidate

        if not target_dir or not os.path.exists(target_dir):
            QMessageBox.information(self, "提示", "尚未找到已完成的自動標注專案目錄，請先執行自動標注！")
            return

        raw_imgs = os.path.join(target_dir, "raw_images")
        lbls_dir = os.path.join(target_dir, "labels")

        # 覆寫或同步至目前標註器專案
        self.project_mgr.setup_project_from_folders(raw_imgs, lbls_dir, copy_files=True)
        self._refresh_project_file_list()
        self.main_tabs.setCurrentIndex(1)
        QMessageBox.information(
            self, "載入成功",
            f"已將專案 [{os.path.basename(target_dir)}] 的影像與標註成果載入至 LabelImg 互動標註器！\n您可立即使用滑鼠進行框體檢查、微調或增刪。"
        )

    def _jump_to_train_from_auto(self):
        config_path = None
        if self.last_auto_result and self.last_auto_result.get("dataset_config_path"):
            config_path = self.last_auto_result["dataset_config_path"]
        else:
            proj_root = self.auto_proj_root_input.text().strip()
            proj_name = self.auto_proj_name_input.text().strip() or "NYA_AutoLabel"
            candidate_cfg = os.path.join(proj_root, proj_name, "dataset", "config.yaml")
            if os.path.exists(candidate_cfg):
                config_path = candidate_cfg

        if not config_path or not os.path.exists(config_path):
            QMessageBox.warning(self, "未找到 Config", "尚未找到已拆分的 dataset/config.yaml！請確認自動標注時有勾選【自動切分 Train/Val】。")
            return

        self.jump_to_train_requested.emit(config_path)

    def append_auto_log(self, text):
        self.auto_log.append(text)

    def update_auto_progress(self, current, total):
        if total > 0:
            pct = int(current / total * 100)
            self.auto_progress_bar.setValue(pct)

    def update_auto_status(self, text):
        self.auto_stat_lbl.setText(text)

    def update_auto_preview(self, qimage, filename, box_count):
        if not qimage.isNull():
            pix = QPixmap.fromImage(qimage)
            target_size = self.auto_preview_lbl.size()
            if target_size.width() > 50 and target_size.height() > 50:
                scaled_pix = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.auto_preview_lbl.setPixmap(scaled_pix)
            else:
                self.auto_preview_lbl.setPixmap(pix)

    def on_auto_annotate_finished(self, success, message, summary_dict):
        self.btn_start_auto.setEnabled(True)
        self.btn_stop_auto.setEnabled(False)
        if success:
            self.last_auto_result = summary_dict
            processed = summary_dict.get("processed", 0)
            total_boxes = summary_dict.get("total_boxes", 0)
            class_stats = summary_dict.get("class_stats", {})
            stat_lines = [f"📊 標注完成！共處理 {processed} 張圖像，生成 {total_boxes} 個目標框。"]
            if class_stats:
                stat_str = ", ".join([f"{k}: {v}" for k, v in class_stats.items() if v > 0])
                if stat_str:
                    stat_lines.append(f"類別分佈: {stat_str}")
            self.auto_summary_lbl.setText("\n".join(stat_lines))
            QMessageBox.information(
                self, "自動標注完成",
                f"🎉 AI 自動標注任務圓滿完成！\n共處理: {processed} 張圖像\n生成標註: {total_boxes} 個目標\n專案位置: {summary_dict.get('project_dir', '')}"
            )
        else:
            QMessageBox.warning(self, "標注提示", f"自動標注中斷或失敗: {message}")

