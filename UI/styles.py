"""
Gemini UI Design System - Modern AI Studio Theme
- Sleek, modern, cold-dark and clean-light themes
- Blue/Purple gradient accents, Material You aesthetic
- Subtle borders, semi-transparent backgrounds
"""

class GoogleAccountTheme:
    # 1. Light Mode (Clean, airy, modern)
    LIGHT_BG = "#F8F9FA"
    LIGHT_SIDEBAR_BG = "#F0F4F9"
    LIGHT_SURFACE = "#FFFFFF"
    LIGHT_ACTIVE_PILL = "#D3E3FD"
    LIGHT_TEXT_PRIMARY = "#1F1F1F"
    LIGHT_TEXT_SECONDARY = "#444746"
    LIGHT_PRIMARY = "#0B57D0"
    LIGHT_PRIMARY_HOVER = "#0842A0"
    LIGHT_BORDER = "#E0E2E0"
    LIGHT_CARD_BG = "#FFFFFF"
    LIGHT_LOG_BG = "#F0F4F9"
    
    # 2. Dark Mode (Gemini cold dark, modern)
    DARK_BG = "#000000"
    DARK_SIDEBAR_BG = "#131314"
    DARK_SURFACE = "#1E1F20"
    DARK_ACTIVE_PILL = "#004A77"
    DARK_TEXT_PRIMARY = "#E3E3E3"       
    DARK_TEXT_SECONDARY = "#C4C7C5"     
    DARK_PRIMARY = "#A8C7FA"
    DARK_PRIMARY_HOVER = "#D3E3FD"
    DARK_BORDER = "#444746"
    DARK_CARD_BG = "#1E1F20"
    DARK_LOG_BG = "#131314"

    @staticmethod
    def get_style(dark_mode=False):
        if dark_mode:
            bg           = "rgba(0,0,0,0)"              # 完全透明，讓 paintEvent 的壁紙底色顯示
            sidebar_bg   = "rgba(15,12,24,0.88)"        # 深紫黑，高不透明度確保文字清晰
            surface      = "rgba(22,18,36,0.92)"
            active_pill  = "rgba(100,60,180,0.70)"
            active_pill_text = "#D4BAFF"
            text_p       = "#EDE8FF"
            text_s       = "#B8AEDD"
            primary      = "#C4A8FA"
            primary_hover = "#DDD0FF"
            border       = "rgba(120,90,200,0.30)"
            card_bg      = "rgba(20,15,38,0.86)"        # 深紫玻璃卡片
            log_bg       = "rgba(12,9,22,0.90)"
            accent_bg    = "rgba(130,80,220,1.0)"
            accent_text  = "#FFFFFF"
            pop_bg       = "rgba(22,18,36,0.97)"
            item_hover   = "rgba(60,40,100,0.80)"
            input_bg     = "rgba(18,12,30,0.85)"
            tile_bg      = "rgba(20,15,38,0.82)"
            tile_border  = "rgba(120,90,200,0.25)"
            search_border = "rgba(140,100,220,0.50)"
            btn_text     = "#FFFFFF"
        else:
            bg           = "rgba(0,0,0,0)"
            sidebar_bg   = "rgba(255,255,255,0.92)"     # 明亮極簡亮白側邊欄
            surface      = "rgba(255,255,255,0.95)"     # 亮白頂欄
            active_pill  = "#E8DEF8"                    # Material 3 淡紫選中標籤
            active_pill_text = "#1D192B"                # 深色選中文字
            text_p       = "#1C1B1F"                    # 主要文字：清晰高對比深灰/黑
            text_s       = "#49454F"                    # 次要文字：清晰深灰
            primary      = "#6750A4"                    # 高對比主題紫
            primary_hover = "#4F378B"
            border       = "rgba(121,116,126,0.30)"     # 類IOS果冻邊框
            card_bg      = "rgba(255,255,255,0.92)"     # 亮白卡片背景
            log_bg       = "rgba(244,240,248,0.95)"
            accent_bg    = "#6750A4"
            accent_text  = "#FFFFFF"
            pop_bg       = "#FFFFFF"
            item_hover   = "rgba(232,222,248,0.60)"     #
            input_bg     = "#FFFFFF"
            tile_bg      = "rgba(255,255,255,0.92)"
            tile_border  = "rgba(121,116,126,0.25)"
            search_border = "rgba(121,116,126,0.50)"
            btn_text     = "#FFFFFF"

        return f"""
        * {{
            font-family: 'Google Sans', 'Inter', 'Segoe UI', 'Microsoft JhengHei', sans-serif;
            font-size: 13px;
            color: {text_p};
        }}

        QMainWindow, QWidget#MainContainer {{
            background-color: {bg};
        }}

        /* Google Header Bar */
        QWidget#GoogleHeader {{
            background-color: {surface};
            border-bottom: 1px solid {border};
        }}

        QLabel#GoogleLogoText {{
            font-size: 22px;
            font-weight: 600;
            color: {text_p};
            letter-spacing: -0.5px;
        }}

        QLabel#GoogleLogoSubtext {{
            font-size: 22px;
            font-weight: 300;
            color: {primary};
            letter-spacing: -0.5px;
        }}

        QPushButton#GoogleHeaderBtn {{
            background-color: {surface};
            border: 1px solid {border};
            border-radius: 16px;
            padding: 6px 14px;
            font-size: 13px;
            font-weight: 600;
            color: {primary};
        }}

        QPushButton#GoogleHeaderBtn:hover {{
            background-color: {item_hover};
            color: {primary_hover};
        }}

        /* Google Sidebar */
        QWidget#GoogleSidebar {{
            background-color: {sidebar_bg};
            border-right: none;
        }}

        QPushButton#GoogleNavItem {{
            background-color: transparent;
            color: {text_s};
            border: none;
            border-radius: 20px;
            padding: 10px 16px;
            text-align: left;
            font-size: 14px;
            font-weight: 500;
        }}

        QPushButton#GoogleNavItem:hover {{
            background-color: {item_hover};
            color: {text_p};
        }}

        QPushButton#GoogleNavItem:checked {{
            background-color: {active_pill};
            color: {active_pill_text};
            font-weight: 600;
        }}

        /* Search Bar */
        QLineEdit#GoogleSearchBar {{
            background-color: {input_bg};
            border: 1px solid {search_border};
            border-radius: 20px;
            padding: 8px 16px;
            font-size: 14px;
            color: {text_p};
        }}

        QLineEdit#GoogleSearchBar:focus {{
            border: 2px solid {primary};
            background-color: {surface};
        }}

        /* QCompleter */
        QCompleter QAbstractItemView {{
            background-color: {pop_bg};
            color: {text_p};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 6px;
            selection-background-color: {item_hover};
            selection-color: {text_p};
            outline: none;
        }}

        QCompleter QAbstractItemView::item {{
            min-height: 32px;
            padding: 6px 12px;
            color: {text_p};
            border-radius: 6px;
        }}

        QCompleter QAbstractItemView::item:selected {{
            background-color: {item_hover};
            color: {primary};
            font-weight: 600;
        }}

        /* Chip Tags */
        QPushButton#GoogleChip {{
            background-color: {surface};
            border: 1px solid {border};
            border-radius: 16px;
            padding: 6px 16px;
            font-size: 13px;
            font-weight: 500;
            color: {text_p};
        }}

        QPushButton#GoogleChip:hover {{
            background-color: {item_hover};
            border-color: {search_border};
        }}

        /* Cards */
        QFrame#GoogleCard {{
            background-color: {card_bg};
            border: 1px solid {border};
            border-radius: 16px;
            padding: 20px;
        }}

        QFrame#MetroTileCard {{
            background-color: {card_bg};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 12px 16px;
        }}

        QFrame#MetroTileCard:hover {{
            border: 1px solid {search_border};
            background-color: {item_hover};
        }}

        QLabel#MetroTileTitle {{
            font-size: 14px;
            font-weight: 600;
            color: {text_p};
            margin-bottom: 2px;
        }}

        QLabel#HomeFooterText {{
            font-size: 11px;
            color: {text_s};
            font-weight: 400;
        }}

        QLabel#MetroTileDesc, QLabel#GoogleCardSubtitle {{
            font-size: 13px;
            color: {text_s};
            line-height: 1.5;
        }}

        QPushButton#MetroTileBtn {{
            background-color: transparent;
            color: {primary};
            border: none;
            border-radius: 14px;
            padding: 6px 14px;
            font-size: 13px;
            font-weight: 600;
        }}

        QPushButton#MetroTileBtn:hover {{
            background-color: {active_pill};
            color: {active_pill_text};
        }}

        QLabel#GoogleCardTitle {{
            font-size: 18px;
            font-weight: 500;
            color: {text_p};
        }}

        /* QComboBox & QListView 下拉清單選單 */
        QComboBox {{
            background-color: {input_bg};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 4px 8px;
            color: {text_p};
            font-weight: 500;
        }}

        QComboBox:focus {{
            border: 2px solid {primary};
        }}

        QComboBox QAbstractItemView, QListView, QComboBox QListView {{
            background: {pop_bg};
            background-color: {pop_bg};
            color: {text_p};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 4px;
            selection-background-color: {active_pill};
            selection-color: {active_pill_text};
            outline: none;
        }}

        QComboBox QAbstractItemView::item, QListView::item {{
            min-height: 32px;
            padding: 6px 12px;
            color: {text_p};
            background-color: {pop_bg};
            border-radius: 4px;
        }}

        QComboBox QAbstractItemView::item:hover, QListView::item:hover,
        QComboBox QAbstractItemView::item:selected, QListView::item:selected {{
            background-color: {active_pill};
            color: {active_pill_text};
            font-weight: bold;
        }}

        /* Buttons */
        QPushButton {{
            background-color: {surface};
            color: {primary};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 6px 12px;
            font-weight: 500;
        }}

        QPushButton:hover {{
            background-color: {item_hover};
            color: {primary_hover};
        }}

        QPushButton#GooglePrimaryButton {{
            background-color: {primary};
            color: #FFFFFF;
            border: none;
            border-radius: 16px;
            padding: 8px 16px;
            font-weight: 600;
            font-size: 13px;
        }}

        QPushButton#GoogleAmberButton {{
            background-color: #E65100;
            color: #FFFFFF;
            border: none;
            border-radius: 16px;
            padding: 8px 16px;
            font-weight: bold;
            font-size: 13px;
        }}

        QPushButton#GooglePrimaryButton:hover {{
            background-color: {primary_hover};
        }}

        QPushButton#GoogleAmberButton:hover {{
            background-color: #BF360C;
        }}
        
        QPushButton#GooglePrimaryButton:disabled,
        QPushButton#GoogleAmberButton:disabled {{
            background-color: {border};
            color: {text_s};
            border: 1px solid {border};
        }}

        QPushButton#GoogleSecondaryButton {{
            background-color: {surface};
            color: {primary};
            border: 1px solid {border};
            padding: 6px 12px;
            font-weight: 600;
            font-size: 13px;
            border-radius: 12px;
        }}

        QPushButton#GoogleSecondaryButton:hover {{
            background-color: {item_hover};
            color: {primary_hover};
            border-color: {primary};
        }}

        /* Inputs & Controls */
        QLineEdit, QSpinBox, QDoubleSpinBox {{
            background-color: {input_bg};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 4px 8px;
            color: {text_p};
            selection-background-color: {primary};
            selection-color: {btn_text};
        }}

        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 2px solid {primary};
        }}

        /* Tabs */
        QTabWidget::pane {{
            border: 1px solid {border};
            border-radius: 12px;
            background-color: {card_bg};
        }}

        QTabBar::tab {{
            background-color: transparent;
            color: {text_s};
            border-radius: 12px;
            padding: 8px 16px;
            margin: 4px;
            font-weight: 500;
        }}

        QTabBar::tab:selected {{
            background-color: {active_pill};
            color: {active_pill_text};
            font-weight: 600;
        }}

        /* Progress Bar */
        QProgressBar {{
            border: none;
            border-radius: 4px;
            background-color: {border};
            text-align: center;
            color: {text_p};
            font-weight: 500;
            height: 8px;
        }}

        QProgressBar::chunk {{
            background-color: {primary};
            border-radius: 4px;
        }}

        /* Log Viewer */
        QTextEdit#GoogleLogViewer {{
            background-color: {log_bg};
            color: {text_s};
            border: 1px solid {border};
            border-radius: 12px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 12px;
            padding: 12px;
        }}

        QCheckBox, QRadioButton {{
            color: {text_p};
            spacing: 8px;
            font-weight: 400;
        }}

        /* List Widget */
        QListWidget {{
            background-color: {input_bg};
            color: {text_p};
            border: 1px solid {border};
            border-radius: 8px;
            font-size: 12px;
        }}

        QListWidget::item {{
            padding: 5px;
            color: {text_p};
        }}

        QListWidget::item:selected {{
            background-color: {active_pill};
            color: {active_pill_text};
            border-radius: 4px;
        }}

        /* GroupBox */
        QGroupBox {{
            color: {text_p};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 16px;
            margin-top: 12px;
            font-weight: 500;
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            padding: 0 8px;
            color: {primary};
        }}

        /* Dialog Overrides */
        QDialog {{
            background-color: {card_bg};
            color: {text_p};
        }}

        QMessageBox {{
            background-color: {card_bg};
            color: {text_p};
        }}

        QMessageBox QLabel {{
            color: {text_p};
        }}

        QInputDialog {{
            background-color: {card_bg};
            color: {text_p};
        }}

        /* ScrollArea */
        QScrollArea {{
            background-color: transparent;
            border: none;
        }}

        QWidget#CardScrollContent {{
            background-color: transparent;
        }}

        QScrollBar:vertical {{
            background: transparent;
            width: 8px;
            border-radius: 4px;
        }}

        QScrollBar::handle:vertical {{
            background: {border};
            border-radius: 4px;
            min-height: 20px;
        }}

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}

        /* Menu */
        QMenu {{
            background-color: {pop_bg};
            color: {text_p};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 4px;
        }}

        QMenu::item {{
            padding: 8px 28px 8px 8px;
            border-radius: 4px;
            color: {text_p};
        }}

        QMenu::item:selected {{
            background-color: {active_pill};
            color: {active_pill_text};
        }}

        QMenu::separator {{
            height: 1px;
            background: {border};
            margin: 4px 8px;
        }}
        """

GeminiWarmTheme = GoogleAccountTheme
WarmEyeCareTheme = GoogleAccountTheme
Win10MetroWarmTheme = GoogleAccountTheme
AndroidMaterialYouTheme = GoogleAccountTheme
AIToolWarmTheme = GoogleAccountTheme
