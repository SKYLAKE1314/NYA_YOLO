"""
GoogleHeaderWidget — 頂部導覽與系統設定列
包含品牌標題、主題模式切換 (系統/淺色/深色)、說明對話框與算力裝置切換選單
"""

import os
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QMenu
from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QIcon, QAction
from components.env_dialog import show_environment_dialog


class GoogleHeaderWidget(QWidget):
    theme_changed = Signal(str)         # 發送主題切換請求
    compute_mode_changed = Signal(str, str, str) # title, device_val, icon_name
    switch_page_requested = Signal(int) # 頁面切換請求

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("GoogleHeader")
        self.setFixedHeight(60)
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 0, 24, 0)

        # Logo Text
        logo_box = QHBoxLayout()
        logo_box.setSpacing(6)
        logo_title = QLabel("Nya")
        logo_title.setObjectName("GoogleLogoText")
        logo_sub = QLabel("YOLO Studio")
        logo_sub.setObjectName("GoogleLogoSubtext")
        logo_box.addWidget(logo_title)
        logo_box.addWidget(logo_sub)

        # 主題切換按鈕 (預設半月星夢)
        self.btn_theme = QPushButton("🌙 半月星夢")
        self.btn_theme.setObjectName("GoogleHeaderBtn")
        self.btn_theme.setToolTip("點擊切換主題 (🌙 半月星夢 / ☀ 昨日青空)")
        self.btn_theme.clicked.connect(self._on_theme_clicked)

        # 專案 GitHub 按鈕
        btn_project = QPushButton("專案")
        btn_project.setObjectName("GoogleHeaderBtn")
        btn_project.setToolTip("造訪 SKYLAKE1314/NYA_YOLO 開源專案 GitHub")
        btn_project.clicked.connect(self._open_project_url)

        # Compute Mode Button
        ICON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "icons")
        def _icon(name):
            p = os.path.join(ICON_DIR, f"{name}.png")
            return QIcon(p) if os.path.exists(p) else QIcon()

        self.avatar_btn = QPushButton(" CPU")
        self.avatar_btn.setObjectName("GoogleHeaderBtn")
        self.avatar_btn.setIcon(_icon("cpu"))
        self.avatar_btn.setIconSize(QSize(18, 18))

        mode_menu = QMenu(self)

        modes = [
            ("CUDA  (NVIDIA GPU)", "0",   "nvidia"),
            ("CPU   Mode",         "cpu", "cpu"),
            ("OpenVINO  (Intel)",  "cpu", "openvino"),
            ("TensorFlow",         "cpu", "tf"),
            ("MPS   (Apple)",      "mps", "apple"),
        ]

        for label, dev_val, icon_name in modes:
            action = QAction(_icon(icon_name), label, self)
            action.triggered.connect(lambda checked, t=label, d=dev_val, ic=icon_name: self.compute_mode_changed.emit(t, d, ic))
            mode_menu.addAction(action)

        mode_menu.addSeparator()
        diag_action = QAction("🔍 硬體診斷面板", self)
        diag_action.triggered.connect(lambda: self.switch_page_requested.emit(5))
        mode_menu.addAction(diag_action)

        env_diag_action = QAction("🛡️ 環境與依賴檢測", self)
        env_diag_action.triggered.connect(lambda: show_environment_dialog(self.window(), auto_on_startup=False))
        mode_menu.addAction(env_diag_action)

        self.avatar_btn.setMenu(mode_menu)
        self._icon_dir = ICON_DIR

        layout.addLayout(logo_box)
        layout.addStretch()
        layout.addWidget(self.btn_theme)
        layout.addWidget(btn_project)
        layout.addWidget(self.avatar_btn)

    def _on_theme_clicked(self):
        self.theme_changed.emit("cycle")

    def _open_project_url(self):
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl("https://github.com/SKYLAKE1314/NYA_YOLO"))

