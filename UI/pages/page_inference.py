
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QFrame, QFormLayout, QComboBox, QDoubleSpinBox, QGroupBox, QListView,
    QFileDialog, QMessageBox
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap 


class InferencePageWidget(QWidget):
    start_infer_requested = Signal()
    stop_infer_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        left_card = QFrame()
        left_card.setObjectName("GoogleCard")
        left_layout = QVBoxLayout(left_card)
        left_layout.addWidget(QLabel("推理與測試控制器"))

        form = QFormLayout()
        form.setVerticalSpacing(8)

        self.infer_model_input = QLineEdit(r"runs\detect\train\weights\best.pt")
        btn_infer_model = QPushButton("選擇模型 (.pt)")
        btn_infer_model.clicked.connect(lambda: self._select_file(self.infer_model_input, "選擇模型"))
        form.addRow("模型路徑:", self.infer_model_input)
        form.addRow("", btn_infer_model)

        self.infer_source_input = QLineEdit()
        self.infer_source_input.setPlaceholderText("選擇或拖入測試圖片 / 影片 / 資料夾路徑...")
        
        src_btn_row = QHBoxLayout()
        btn_infer_img = QPushButton("🖼 選擇測試圖片/影片")
        btn_infer_img.clicked.connect(lambda: self._select_image_file(self.infer_source_input))
        
        btn_infer_dir = QPushButton("📁 選擇圖片資料夾")
        btn_infer_dir.clicked.connect(lambda: self._select_folder(self.infer_source_input))
        
        src_btn_row.addWidget(btn_infer_img)
        src_btn_row.addWidget(btn_infer_dir)

        form.addRow("測試來源:", self.infer_source_input)
        form.addRow("", src_btn_row)

        self.infer_mode_combo = QComboBox()
        self.infer_mode_combo.setView(QListView())
        self.infer_mode_combo.addItems([
            "predict (標準 YOLO 推斷)",
            "track (目標追蹤)",
            "world (World / Open-Vocabulary 檢測)",
            "text_det (Text Detection 專用文字檢測)"
        ])
        self.infer_mode_combo.currentIndexChanged.connect(self._on_infer_mode_changed)
        form.addRow("測試模式:", self.infer_mode_combo)

        self.tracker_combo = QComboBox()
        self.tracker_combo.setView(QListView())
        self.tracker_combo.addItems(["bytetrack.yaml", "botsort.yaml"])
        form.addRow("追蹤器:", self.tracker_combo)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setValue(0.25)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setValue(0.45)
        form.addRow("Conf (置信度):", self.conf_spin)
        form.addRow("IoU (重疊閾值):", self.iou_spin)

        left_layout.addLayout(form)

        # World / Text Detection Group
        self.infer_world_group = QGroupBox("🌐 World / Text 檢測類別提示")
        w_layout = QVBoxLayout(self.infer_world_group)
        w_layout.setContentsMargins(8, 8, 8, 8)
        w_layout.setSpacing(6)
        w_lbl = QLabel("類別提示 (逗號分隔):")
        w_lbl.setStyleSheet("font-size: 11px;")
        self.infer_world_prompts = QLineEdit("text, label, logo, person")
        self.infer_world_prompts.setPlaceholderText("text, label, person, car...")
        w_layout.addWidget(w_lbl)
        w_layout.addWidget(self.infer_world_prompts)
        self.infer_world_group.setVisible(False)
        left_layout.addWidget(self.infer_world_group)

        left_layout.addSpacing(10)

        self.btn_start_infer = QPushButton("▶ 啟動推斷 / 測試")
        self.btn_start_infer.setObjectName("GoogleAmberButton")
        self.btn_start_infer.clicked.connect(lambda: self.start_infer_requested.emit())
        left_layout.addWidget(self.btn_start_infer)

        self.btn_stop_infer = QPushButton("⏹ 停止推斷")
        self.btn_stop_infer.setObjectName("GoogleSecondaryButton")
        self.btn_stop_infer.clicked.connect(lambda: self.stop_infer_requested.emit())
        left_layout.addWidget(self.btn_stop_infer)

        left_layout.addStretch()
        layout.addWidget(left_card, 1)

        right_card = QFrame()
        right_card.setObjectName("GoogleCard")
        right_layout = QVBoxLayout(right_card)

        self.infer_status_lbl = QLabel("畫面待命...")
        self.infer_status_lbl.setObjectName("GoogleCardTitle")
        right_layout.addWidget(self.infer_status_lbl)

        self.canvas_label = QLabel()
        self.canvas_label.setAlignment(Qt.AlignCenter)
        self.canvas_label.setStyleSheet("background-color: #000000; border-radius: 16px;")
        self.canvas_label.setMinimumSize(480, 360)
        right_layout.addWidget(self.canvas_label)

        layout.addWidget(right_card, 2)

    def _on_infer_mode_changed(self, idx):
        mode_text = self.infer_mode_combo.currentText()
        is_world = ("world" in mode_text or "text_det" in mode_text)
        self.infer_world_group.setVisible(is_world)
        if "text_det" in mode_text:
            self.infer_world_prompts.setText("text, label, OCR area")
            if not self.infer_model_input.text().strip() or "best.pt" in self.infer_model_input.text():
                self.infer_model_input.setText("yolov8s-world.pt")
        elif "world" in mode_text:
            if not self.infer_model_input.text().strip() or "best.pt" in self.infer_model_input.text():
                self.infer_model_input.setText("yolov8s-world.pt")

    def update_canvas(self, qimg, info_text):
        pixmap = QPixmap.fromImage(qimg).scaled(self.canvas_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.canvas_label.setPixmap(pixmap)
        self.infer_status_lbl.setText(info_text)

    def update_status(self, text):
        self.infer_status_lbl.setText(text)

    def _select_file(self, line_edit, title):
        file_path, _ = QFileDialog.getOpenFileName(self, title)
        if file_path:
            line_edit.setText(file_path)

    def _select_image_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇測試圖片 / 影片檔案", "",
            "圖片與影片檔 (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mkv *.webp);;所有檔案 (*)"
        )
        if file_path:
            line_edit.setText(file_path)

    def _select_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "選擇資料夾")
        if folder:
            line_edit.setText(folder)
