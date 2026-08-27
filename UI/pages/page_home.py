from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFrame, QGridLayout, QScrollArea, QCompleter, QListView
)
from PySide6.QtCore import Qt, Signal


class HomePageWidget(QWidget):
    search_requested = Signal(str)

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
        layout.setContentsMargins(32, 14, 32, 14)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignHCenter)

        # 1. 中央頭像與標題
        avatar_box = QVBoxLayout()
        avatar_box.setAlignment(Qt.AlignCenter)
        avatar_box.setSpacing(2)

        avatar_frame = QLabel()
        avatar_frame.setFixedSize(60, 60)
        avatar_frame.setAlignment(Qt.AlignCenter)
        avatar_frame.setStyleSheet("""
            background-color: rgba(130, 80, 220, 0.15);
            border: 2px solid rgba(130, 80, 220, 0.5);
            border-radius: 30px;
            font-size: 26px;
        """)
        avatar_frame.setText("✨")

        user_title = QLabel("NYA AI Studio 智慧深度學習視覺平台")
        user_title.setStyleSheet("font-size: 20px; font-weight: bold; margin-top: 4px;")
        user_title.setAlignment(Qt.AlignCenter)

        user_subtitle = QLabel("YOLO / ResNet 深度整合 • 檢測 / 分割 / 分類 / 實時追蹤全功能")
        user_subtitle.setStyleSheet("font-size: 12px; opacity: 0.85;")
        user_subtitle.setAlignment(Qt.AlignCenter)

        avatar_box.addWidget(avatar_frame, 0, Qt.AlignCenter)
        avatar_box.addWidget(user_title)
        avatar_box.addWidget(user_subtitle)
        layout.addLayout(avatar_box)

        # 2. 大型 Google 搜尋列
        search_box = QHBoxLayout()
        search_box.setAlignment(Qt.AlignCenter)
        search_box.setSpacing(8)

        self.home_search_bar = QLineEdit()
        self.home_search_bar.setObjectName("GoogleSearchBar")
        self.home_search_bar.setPlaceholderText("🔍 搜尋模型 (yolo12/26)、追蹤器 (bytetrack)、格式 (onnx)...")
        self.home_search_bar.setFixedWidth(560)

        search_keywords = [
            "yolo12n.pt - 一鍵載入並開啟 YOLO12 訓練配置",
            "yolo26n.pt - 一鍵載入並開啟 YOLO26 訓練配置",
            "yolo26n-seg.pt - 一鍵載入並開啟 YOLO26 Segmentation 實例分割",
            "bytetrack.yaml - 自動設置 ByteTrack 實態追蹤器",
            "botsort.yaml - 自動設置 BoT-SORT 多目標追蹤器",
            "onnx - 開啟 ONNX 模型導出面板",
            "tensorrt - 開啟 TensorRT Engine 導出面板",
            "datacheck - 開啟 XML/JSON 轉 YOLO 並進行畫框驗證",
            "cuda - 執行 CUDA GPU 硬體系統健康診斷",
            "train - 直接前往模型訓練與超參數配置"
        ]
        completer = QCompleter(search_keywords, self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.home_search_bar.setCompleter(completer)

        self.home_search_bar.returnPressed.connect(self._on_search_submit)

        btn_search_exec = QPushButton("搜尋執行 ➔")
        btn_search_exec.setObjectName("GooglePrimaryButton")
        btn_search_exec.clicked.connect(self._on_search_submit)

        search_box.addWidget(self.home_search_bar)
        search_box.addWidget(btn_search_exec)
        layout.addLayout(search_box)

        # 3. Quick Chips
        chip_box = QHBoxLayout()
        chip_box.setAlignment(Qt.AlignCenter)
        chip_box.setSpacing(8)

        chips = [
            ("📦 yolo12n.pt", lambda: self.search_requested.emit("yolo12n.pt")),
            ("📦 yolo26n.pt", lambda: self.search_requested.emit("yolo26n.pt")),
            ("⚡ CUDA GPU 診斷", lambda: self.search_requested.emit("cuda")),
            ("🎥 ByteTrack 追蹤器", lambda: self.search_requested.emit("bytetrack")),
            ("📄 config.yaml 配置", lambda: self.search_requested.emit("config.yaml")),
            ("🚀 ONNX 導出", lambda: self.search_requested.emit("onnx"))
        ]

        for text, cb in chips:
            btn = QPushButton(text)
            btn.setObjectName("GoogleChip")
            btn.clicked.connect(cb)
            chip_box.addWidget(btn)

        layout.addLayout(chip_box)

        # 4. 操作提示與工作流卡片
        tiles_header = QLabel("💡 智慧工作流建議與操作提示")
        tiles_header.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 2px;")
        tiles_header.setAlignment(Qt.AlignLeft)
        layout.addWidget(tiles_header, 0, Qt.AlignHCenter)

        metro_grid = QGridLayout()
        metro_grid.setSpacing(10)

        recommendations = [
            ("🚀 建議：一鍵訓練 YOLO12 / YOLO26", "檢測到已具備 PyTorch & GPU 加速，建議優先配置 100 Epochs 與 AMP 混合精度開啟模型訓練。", lambda: self.search_requested.emit("yolo12n.pt"), "配置訓練 ➔"),
            ("🔍 建議：DataCheck 標註畫框驗證", "在開啟正式訓練前，建議使用 DataCheck 預覽畫框與 YOLO label polygon 數據是否完美對齊。", lambda: self.search_requested.emit("datacheck"), "畫框驗證 ➔"),
            ("🎥 建議：ByteTrack 實體串流追蹤", "支援檔名過濾手動選圖與 MP4 影片物件追蹤，適合即時物體辨識與軌跡繪製。", lambda: self.search_requested.emit("bytetrack"), "啟動追蹤 ➔"),
            ("⚡ 建議：ONNX / TensorRT 部署導出", "訓練完成的權重可一鍵轉為 ONNX 或 TensorRT Engine，以利嵌入式設備極速推斷部署。", lambda: self.search_requested.emit("onnx"), "模型導出 ➔")
        ]

        for i, (rtitle, rdesc, rcb, rbtn_text) in enumerate(recommendations):
            tile = QFrame()
            tile.setObjectName("MetroTileCard")
            tile.setMinimumWidth(320)

            t_layout = QVBoxLayout(tile)
            t_layout.setContentsMargins(14, 10, 14, 10)
            t_layout.setSpacing(4)

            t_title = QLabel(rtitle)
            t_title.setObjectName("MetroTileTitle")
            t_title.setWordWrap(True)

            t_desc = QLabel(rdesc)
            t_desc.setObjectName("MetroTileDesc")
            t_desc.setWordWrap(True)

            t_btn = QPushButton(rbtn_text)
            t_btn.setObjectName("MetroTileBtn")
            t_btn.clicked.connect(rcb)

            t_layout.addWidget(t_title)
            t_layout.addWidget(t_desc)
            t_layout.addWidget(t_btn, 0, Qt.AlignRight)

            row, col = i // 2, i % 2
            metro_grid.addWidget(tile, row, col)

        layout.addLayout(metro_grid)

        # 5. 授權宣告 (單行置中、清晰可讀)
        footer_lbl = QLabel("本程式碼基於 Ultralytics YOLO • 使用 MIT License 授權 • 散佈與再發行請遵循開源授權條款 • by SKYLAKE")
        footer_lbl.setObjectName("HomeFooterText")
        footer_lbl.setWordWrap(False)
        footer_lbl.setAlignment(Qt.AlignCenter)
        layout.addSpacing(4)
        layout.addWidget(footer_lbl, 0, Qt.AlignCenter)

        scroll.setWidget(page)
        root_layout.addWidget(scroll)

    def _on_search_submit(self):
        query = self.home_search_bar.text().strip()
        if query:
            self.search_requested.emit(query)
