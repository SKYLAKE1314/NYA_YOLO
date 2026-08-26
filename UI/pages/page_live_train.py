"""
LiveTrainPageWidget — 即時訓練動態頁面模組
提供進度條、暫停/取消按鈕、pyqtgraph 動態 Loss/mAP 折線圖及 Console 日誌輸出
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QFrame, QSplitter, QTextEdit
)
from PySide6.QtCore import Signal, Qt
import pyqtgraph as pg


class LiveTrainPageWidget(QWidget):
    pause_train_requested = Signal()
    stop_train_requested = Signal()

    def __init__(self, dark_mode=True, parent=None):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.epochs_data = []
        self.box_data, self.cls_data, self.dfl_data = [], [], []
        self.map50_data, self.map95_data = [], []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        prog_card = QFrame()
        prog_card.setObjectName("GoogleCard")
        prog_layout = QVBoxLayout(prog_card)

        header_box = QHBoxLayout()
        header_lbl = QLabel("訓練動態與監控儀表板")
        header_lbl.setObjectName("GoogleCardTitle")
        header_box.addWidget(header_lbl)

        self.btn_pause_train = QPushButton("⏸ 暫停訓練")
        self.btn_pause_train.setObjectName("GoogleSecondaryButton")
        self.btn_pause_train.setEnabled(False)
        self.btn_pause_train.clicked.connect(lambda: self.pause_train_requested.emit())

        self.btn_stop_train = QPushButton("🛑 停止訓練")
        self.btn_stop_train.setObjectName("GoogleSecondaryButton")
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.clicked.connect(lambda: self.stop_train_requested.emit())

        header_box.addStretch()
        header_box.addWidget(self.btn_pause_train)
        header_box.addWidget(self.btn_stop_train)
        prog_layout.addLayout(header_box)

        self.lbl_train_status = QLabel("準備好...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        prog_layout.addWidget(self.lbl_train_status)
        prog_layout.addWidget(self.progress_bar)
        layout.addWidget(prog_card)

        chart_splitter = QSplitter(Qt.Horizontal)

        self.plot_loss = pg.PlotWidget(title="<b>Loss 訓練損失動態</b>")
        self.plot_loss.showGrid(x=True, y=True)
        self.plot_loss.addLegend()
        self.curve_box = self.plot_loss.plot(pen=pg.mkPen('#D97706', width=2), name="box_loss")
        self.curve_cls = self.plot_loss.plot(pen=pg.mkPen('#1A73E8', width=2), name="cls_loss")
        self.curve_dfl = self.plot_loss.plot(pen=pg.mkPen('#34A853', width=2), name="dfl_loss")
        chart_splitter.addWidget(self.plot_loss)

        self.plot_map = pg.PlotWidget(title="<b>mAP 驗證精度動態</b>")
        self.plot_map.showGrid(x=True, y=True)
        self.plot_map.addLegend()
        self.curve_map50 = self.plot_map.plot(pen=pg.mkPen('#F9AB00', width=2), name="mAP50")
        self.curve_map95 = self.plot_map.plot(pen=pg.mkPen('#A142F4', width=2), name="mAP50-95")
        chart_splitter.addWidget(self.plot_map)

        layout.addWidget(chart_splitter, 2)

        log_card = QFrame()
        log_card.setObjectName("GoogleCard")
        log_layout = QVBoxLayout(log_card)
        log_layout.addWidget(QLabel("即時 Console 訓練日誌"))
        
        self.log_viewer = QTextEdit()
        self.log_viewer.setObjectName("GoogleLogViewer")
        self.log_viewer.setReadOnly(True)
        log_layout.addWidget(self.log_viewer)

        layout.addWidget(log_card, 1)
        self.update_plot_styles()

    def update_metrics(self, m):
        ep = m.get("epoch", 0)
        self.epochs_data.append(ep)
        if "box_loss" in m: self.box_data.append(m["box_loss"])
        if "cls_loss" in m: self.cls_data.append(m["cls_loss"])
        if "dfl_loss" in m: self.dfl_data.append(m["dfl_loss"])
        if "map50" in m: self.map50_data.append(m["map50"])
        if "map50_95" in m: self.map95_data.append(m["map50_95"])

        self.curve_box.setData(self.epochs_data[:len(self.box_data)], self.box_data)
        self.curve_cls.setData(self.epochs_data[:len(self.cls_data)], self.cls_data)
        self.curve_dfl.setData(self.epochs_data[:len(self.dfl_data)], self.dfl_data)
        self.curve_map50.setData(self.epochs_data[:len(self.map50_data)], self.map50_data)
        self.curve_map95.setData(self.epochs_data[:len(self.map95_data)], self.map95_data)

    def append_log(self, text):
        self.log_viewer.append(text)

    def set_dark_mode(self, dark_mode):
        self.dark_mode = dark_mode
        self.update_plot_styles()

    def update_plot_styles(self):
        txt_col = "#EDE8FF" if self.dark_mode else "#1C1B1F"
        sub_col = "#B8AEDD" if self.dark_mode else "#49454F"
        bg_col = "#1E1A2E" if self.dark_mode else "#F8F9FA"
        grid_col = (180, 170, 220, 40) if self.dark_mode else (0, 0, 0, 35)
        leg_bg = (28, 23, 44, 200) if self.dark_mode else (255, 255, 255, 220)

        if self.dark_mode:
            pen_box = pg.mkPen('#FFA726', width=2.5)
            pen_cls = pg.mkPen('#42A5F5', width=2.5)
            pen_dfl = pg.mkPen('#66BB6A', width=2.5)
            pen_m50 = pg.mkPen('#FFCA28', width=2.5)
            pen_m95 = pg.mkPen('#AB47BC', width=2.5)
        else:
            pen_box = pg.mkPen('#D97706', width=2.5)
            pen_cls = pg.mkPen('#1565C0', width=2.5)
            pen_dfl = pg.mkPen('#2E7D32', width=2.5)
            pen_m50 = pg.mkPen('#B45309', width=2.5)
            pen_m95 = pg.mkPen('#6B21A8', width=2.5)

        self.curve_box.setPen(pen_box)
        self.curve_cls.setPen(pen_cls)
        self.curve_dfl.setPen(pen_dfl)
        self.curve_map50.setPen(pen_m50)
        self.curve_map95.setPen(pen_m95)

        plots = [(self.plot_loss, "Loss 訓練損失動態"), (self.plot_map, "mAP 驗證精度動態")]
        for p, title_str in plots:
            p.setBackground(bg_col)
            p.setTitle(f"<span style='color: {txt_col}; font-size: 14px; font-weight: bold;'>{title_str}</span>")
            plot_item = p.getPlotItem()
            for ax_name in ['left', 'bottom']:
                ax = plot_item.getAxis(ax_name)
                ax.setPen(pg.mkPen(sub_col, width=1))
                ax.setTextPen(pg.mkPen(txt_col))
                ax.setLabel(color=txt_col)
            legend = plot_item.legend
            if legend:
                legend.setPen(pg.mkPen(sub_col, width=1))
                legend.setBrush(pg.mkBrush(*leg_bg))
                for sample, label in legend.items:
                    label.setText(label.text, color=txt_col)
