"""
EnvironmentDialog & Diagnostic Tools — 系統環境與依賴診斷模組
提供自動依賴檢查、MS Defender 盾牌圖示繪製與報告對話框
"""

import os
import sys
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QPainterPath


def get_resource_path(relative_path):
    """取得相容 PyInstaller 打包 (.exe) 的動態資源檔案路徑"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), relative_path)


def create_defender_shield_icon(size=24):
    """繪製 MS Defender 風格的安全盾牌圖示 (Windows Security Shield)"""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing)

    w, h = size, size
    path = QPainterPath()
    path.moveTo(w * 0.5, h * 0.08)
    path.cubicTo(w * 0.85, h * 0.08, w * 0.92, h * 0.25, w * 0.92, h * 0.45)
    path.cubicTo(w * 0.92, h * 0.72, w * 0.65, h * 0.92, w * 0.5, h * 0.96)
    path.cubicTo(w * 0.35, h * 0.92, w * 0.08, h * 0.72, w * 0.08, h * 0.45)
    path.cubicTo(w * 0.08, h * 0.25, w * 0.15, h * 0.08, w * 0.5, h * 0.08)
    path.closeSubpath()

    # 繪製經典 MS Defender 藍黑色盾牌
    shield_color = QColor("#0078D4")
    painter.setPen(QPen(shield_color, 2.0))
    painter.drawPath(path)

    # 左半部填滿
    left_rect = QPainterPath()
    left_rect.addRect(0, 0, w * 0.5, h)
    painter.fillPath(path.intersected(left_rect), shield_color)

    painter.end()
    return QIcon(pm), pm


def check_environment_status():
    """檢測本機 Python 環境、核心套件、CUDA 加速與 OpenAI CLIP 快取狀態"""
    status = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "frozen_exe": getattr(sys, 'frozen', False),
        "pytorch": {"installed": False, "version": None, "cuda": False, "device_name": None},
        "ultralytics": {"installed": False, "version": None},
        "opencv": {"installed": False, "version": None},
        "pyside6": {"installed": False, "version": None},
        "clip_cache": {"cached": False, "path": None, "size_mb": 0.0},
        "missing_pkgs": [],
        "warnings": []
    }

    # PyTorch & CUDA
    try:
        import torch
        status["pytorch"]["installed"] = True
        status["pytorch"]["version"] = torch.__version__
        status["pytorch"]["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            status["pytorch"]["device_name"] = torch.cuda.get_device_name(0)
        else:
            status["warnings"].append("[警告] 未檢測到 PyTorch CUDA GPU 加速 (將使用 CPU 進行訓練/推斷，速度較慢)")
    except ImportError:
        status["missing_pkgs"].append("torch")
        status["warnings"].append("[錯誤] 缺少 PyTorch 核心庫 (pip install torch torchvision)")

    # Ultralytics
    try:
        import ultralytics
        status["ultralytics"]["installed"] = True
        status["ultralytics"]["version"] = ultralytics.__version__
    except ImportError:
        status["missing_pkgs"].append("ultralytics")
        status["warnings"].append("[錯誤] 缺少 Ultralytics YOLO 模組 (pip install ultralytics)")

    # OpenCV
    try:
        import cv2
        status["opencv"]["installed"] = True
        status["opencv"]["version"] = cv2.__version__
    except ImportError:
        status["missing_pkgs"].append("opencv-python")
        status["warnings"].append("[錯誤] 缺少 OpenCV 影像處理庫 (pip install opencv-python)")

    # PySide6
    try:
        import PySide6
        status["pyside6"]["installed"] = True
        status["pyside6"]["version"] = PySide6.__version__
    except ImportError:
        status["missing_pkgs"].append("PySide6")

    # OpenAI CLIP Cache Check
    user_home = os.path.expanduser("~")
    clip_path_1 = os.path.join(user_home, ".cache", "clip", "ViT-B-32.pt")
    clip_path_2 = os.path.join(user_home, "AppData", "Roaming", "ultralytics", "weights", "clip", "ViT-B-32.pt")
    
    target_clip = None
    if os.path.exists(clip_path_1):
        target_clip = clip_path_1
    elif os.path.exists(clip_path_2):
        target_clip = clip_path_2

    if target_clip and os.path.exists(target_clip):
        sz_mb = os.path.getsize(target_clip) / (1024 * 1024)
        status["clip_cache"]["cached"] = (sz_mb > 300)
        status["clip_cache"]["path"] = target_clip
        status["clip_cache"]["size_mb"] = round(sz_mb, 1)
        if sz_mb < 300:
            status["warnings"].append(f"[警告] OpenAI CLIP 模型快取不完整 ({sz_mb:.1f} MB / 338 MB)，使用 World 模式前請刪除殘檔並重新下載。")
    else:
        status["warnings"].append("[提示] 尚未快取 OpenAI CLIP 權重 (ViT-B-32.pt)，首次使用 World Detection 時將自動下載。")

    return status


def show_environment_dialog(parent=None, auto_on_startup=False):
    status = check_environment_status()
    
    # 啟動時自動檢測：只有在有缺失核心套件或無 CUDA 的狀況下才彈窗提醒
    if auto_on_startup and not status["missing_pkgs"] and status["pytorch"]["cuda"]:
        return

    dlg = QDialog(parent)
    dlg.setWindowTitle("系統環境與依賴診斷報告")
    if parent is not None:
        parent._env_dialog = dlg
    
    shield_icon, shield_pixmap = create_defender_shield_icon(24)
    dlg.setWindowIcon(shield_icon)
    dlg.resize(560, 420)
    dlg.setAttribute(Qt.WA_DeleteOnClose, True)

    # 顯式設定對話框主題樣式 (深色主題)
    dlg.setStyleSheet("""
        QDialog {
            background-color: #1E1F20;
            color: #E3E3E3;
        }
        QLabel {
            color: #E3E3E3;
        }
        QTextEdit {
            background-color: #131314;
            color: #C4C7C5;
            border: 1px solid #444746;
            border-radius: 8px;
            font-family: 'Consolas', 'Segoe UI', monospace;
            font-size: 12px;
            padding: 10px;
        }
        QPushButton {
            background-color: #D97706;
            color: #FFFFFF;
            border: 2px solid #F59E0B;
            border-radius: 16px;
            padding: 8px 20px;
            font-weight: bold;
            font-size: 13px;
        }
        QPushButton:hover {
            background-color: #B45309;
            border: 2px solid #FBBF24;
        }
    """)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(14)

    header_row = QHBoxLayout()
    header_row.setSpacing(10)
    icon_lbl = QLabel()
    icon_lbl.setPixmap(shield_pixmap)
    
    title = QLabel("系統環境與依賴診斷報告")
    title.setStyleSheet("font-size: 16px; font-weight: bold; color: #E3E3E3;")
    header_row.addWidget(icon_lbl)
    header_row.addWidget(title)
    header_row.addStretch()
    layout.addLayout(header_row)

    txt = QTextEdit()
    txt.setReadOnly(True)

    lines = []
    lines.append(f"Python 版本: {status['python_version']}")
    lines.append(f"執行模式: {'預發行測試管道' if status['frozen_exe'] else '開發環境'}")
    lines.append("-" * 52)
    
    if status["pytorch"]["installed"]:
        cuda_str = f"已啟用 GPU ({status['pytorch']['device_name']})" if status["pytorch"]["cuda"] else "未啟用 (僅 CPU 模式)"
        lines.append(f"PyTorch 版本: {status['pytorch']['version']} | CUDA: {cuda_str}")
    else:
        lines.append("PyTorch: 未安裝")

    if status["ultralytics"]["installed"]:
        lines.append(f"Ultralytics YOLO: v{status['ultralytics']['version']} (已安裝)")
    else:
        lines.append("Ultralytics YOLO: 未安裝")

    if status["opencv"]["installed"]:
        lines.append(f"OpenCV: v{status['opencv']['version']} (已安裝)")
    else:
        lines.append("OpenCV: 未安裝")

    if status["clip_cache"]["cached"]:
        lines.append(f"OpenAI CLIP (ViT-B-32.pt): 快取完整 ({status['clip_cache']['size_mb']} MB)")
    elif status["clip_cache"]["path"]:
        lines.append(f"OpenAI CLIP: 殘缺檔案 ({status['clip_cache']['size_mb']} MB / 338 MB)")
    else:
        lines.append("OpenAI CLIP: 未快取 (首次使用 World Detection 時將自動下載)")

    lines.append("=" * 52)
    if status["warnings"]:
        lines.append("\n檢測到的提示與建議:")
        for w in status["warnings"]:
            lines.append(f"  {w}")

    if status["missing_pkgs"]:
        lines.append("\n推薦修復指令 (在終端機執行):")
        lines.append(f"  pip install {' '.join(status['missing_pkgs'])}")

    txt.setPlainText("\n".join(lines))
    layout.addWidget(txt)

    btn_row = QHBoxLayout()
    btn_close = QPushButton("關閉並繼續")
    btn_close.clicked.connect(dlg.accept)
    btn_row.addStretch()
    btn_row.addWidget(btn_close)
    layout.addLayout(btn_row)

    if auto_on_startup:
        dlg.show()
    else:
        dlg.exec()
