"""
HỆ THỐNG DESIGN TOKENS & STYLESHEET (DARK OLED SPORTS ANALYTICS PRO)
Thiết kế theo chuẩn ui-ux-pro-max: Tương phản >= 4.5:1, không dùng emoji,
số đo dạng tabular monospace, vùng chạm >= 44px.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThemeColors:
    # Nền OLED & Surface Elevation
    BG_OBSIDIAN: str = "#05080E"
    BG_VIEWPORT: str = "#000000"
    BG_SURFACE: str = "#0B111C"
    BG_CARD: str = "#111A2B"
    BG_ELEVATED: str = "#172338"
    BG_HOVER: str = "#1E2D45"

    # Đường viền
    BORDER_SUBTLE: str = "#1E293B"
    BORDER_STRONG: str = "#334155"
    BORDER_FOCUS: str = "#38BDF8"

    # Ngữ nghĩa trạng thái (Semantic)
    EMERALD: str = "#10B981"
    EMERALD_HOVER: str = "#059669"
    EMERALD_BG: str = "#07261D"
    EMERALD_TEXT: str = "#A7F3D0"

    BLUE_PRO: str = "#3B82F6"
    BLUE_PRO_HOVER: str = "#2563EB"
    BLUE_BG: str = "#0F1E3A"
    BLUE_TEXT: str = "#BFDBFE"

    CRIMSON: str = "#EF4444"
    CRIMSON_HOVER: str = "#DC2626"
    CRIMSON_BG: str = "#261015"
    CRIMSON_TEXT: str = "#FECACA"

    AMBER: str = "#F59E0B"
    AMBER_BG: str = "#261D0D"
    AMBER_TEXT: str = "#FDE68A"

    # Văn bản (>= 4.5:1 trên nền tối)
    TEXT_PRIMARY: str = "#F8FAFC"
    TEXT_SECONDARY: str = "#CBD5E1"
    TEXT_MUTED: str = "#94A3B8"
    TEXT_DIM: str = "#64748B"


COLORS = ThemeColors()
FONT_UI = '"Segoe UI", "Inter", sans-serif'
FONT_MONO = '"Consolas", "Cascadia Mono", monospace'


def build_app_stylesheet() -> str:
    c = COLORS
    return f"""
    QMainWindow, QWidget#RootCanvas {{
        background-color: {c.BG_OBSIDIAN};
        color: {c.TEXT_PRIMARY};
        font-family: {FONT_UI};
    }}

    /* === TOP HEADER BAR === */
    QFrame#TopBar {{
        background-color: {c.BG_SURFACE};
        border-bottom: 1px solid {c.BORDER_SUBTLE};
    }}
    QLabel#BrandTitle {{
        color: {c.TEXT_PRIMARY};
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 0.4px;
    }}
    QLabel#BrandSubtitle {{
        color: {c.TEXT_MUTED};
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.6px;
    }}
    QLabel#StatusPill {{
        background-color: {c.BG_CARD};
        color: {c.EMERALD};
        border: 1px solid {c.BORDER_STRONG};
        border-radius: 14px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 700;
    }}
    QLabel#TopControlLabel {{
        color: {c.TEXT_MUTED};
        font-size: 12px;
        font-weight: 600;
    }}

    /* === COMBOBOX & CHECKBOX === */
    QComboBox {{
        background-color: {c.BG_CARD};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER_STRONG};
        border-radius: 8px;
        padding: 6px 12px;
        min-height: 28px;
        font-size: 13px;
        font-weight: 600;
    }}
    QComboBox:hover {{
        border-color: {c.BORDER_FOCUS};
        background-color: {c.BG_ELEVATED};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c.BG_CARD};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER_STRONG};
        selection-background-color: {c.EMERALD_HOVER};
        selection-color: #FFFFFF;
        outline: 0;
    }}

    QCheckBox {{
        color: {c.TEXT_SECONDARY};
        font-size: 13px;
        font-weight: 600;
        spacing: 8px;
        padding: 6px 8px;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 5px;
        border: 1px solid {c.BORDER_STRONG};
        background-color: {c.BG_CARD};
    }}
    QCheckBox::indicator:checked {{
        background-color: {c.EMERALD};
        border-color: {c.EMERALD};
    }}

    /* === VIDEO STAGE === */
    QFrame#StageCard {{
        background-color: {c.BG_VIEWPORT};
        border: 1px solid {c.BORDER_SUBTLE};
        border-radius: 14px;
    }}

    /* === BOTTOM CONTROL DOCK === */
    QFrame#ControlDock {{
        background-color: {c.BG_SURFACE};
        border: 1px solid {c.BORDER_SUBTLE};
        border-radius: 14px;
    }}
    QLabel#LiveHintLabel {{
        color: {c.TEXT_SECONDARY};
        font-size: 14px;
        font-weight: 500;
    }}

    /* === BUTTONS (TOUCH TARGET >= 44px) === */
    QPushButton#PrimaryCTA {{
        background-color: {c.EMERALD};
        color: #032219;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        min-height: 24px;
        font-size: 15px;
        font-weight: 800;
        letter-spacing: 0.3px;
    }}
    QPushButton#PrimaryCTA:hover {{
        background-color: #34D399;
    }}
    QPushButton#PrimaryCTA:pressed {{
        background-color: {c.EMERALD_HOVER};
    }}
    QPushButton#PrimaryCTA:disabled {{
        background-color: {c.BG_ELEVATED};
        color: {c.TEXT_DIM};
    }}

    QPushButton#DangerCTA {{
        background-color: {c.CRIMSON};
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        padding: 12px 22px;
        min-height: 24px;
        font-size: 14px;
        font-weight: 800;
    }}
    QPushButton#DangerCTA:hover {{
        background-color: {c.CRIMSON_HOVER};
    }}

    QPushButton#SecondaryBtn {{
        background-color: {c.BG_ELEVATED};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER_STRONG};
        border-radius: 9px;
        padding: 9px 16px;
        min-height: 24px;
        font-size: 13px;
        font-weight: 700;
    }}
    QPushButton#SecondaryBtn:hover {{
        background-color: {c.BG_HOVER};
        border-color: {c.BORDER_FOCUS};
    }}

    QPushButton#SegmentBtn {{
        background-color: {c.BG_CARD};
        color: {c.TEXT_MUTED};
        border: 1px solid {c.BORDER_SUBTLE};
        border-radius: 9px;
        padding: 9px 16px;
        min-height: 24px;
        font-size: 13px;
        font-weight: 700;
    }}
    QPushButton#SegmentBtn[active="true"] {{
        background-color: {c.BLUE_PRO};
        color: #FFFFFF;
        border-color: {c.BLUE_PRO};
    }}
    QPushButton#SegmentBtn[active="emerald"] {{
        background-color: {c.EMERALD};
        color: #032219;
        border-color: {c.EMERALD};
    }}
    QPushButton#SegmentBtn:hover {{
        border-color: {c.BORDER_FOCUS};
    }}

    QPushButton#PhaseBtn {{
        background-color: {c.BG_CARD};
        color: {c.TEXT_SECONDARY};
        border: 1px solid {c.BORDER_STRONG};
        border-radius: 8px;
        padding: 7px 14px;
        min-height: 24px;
        font-size: 12px;
        font-weight: 700;
    }}
    QPushButton#PhaseBtn:hover {{
        background-color: {c.BG_ELEVATED};
        border-color: {c.BORDER_FOCUS};
        color: {c.TEXT_PRIMARY};
    }}

    /* === PROGRESS BAR === */
    QProgressBar {{
        background-color: {c.BG_CARD};
        border: 1px solid {c.BORDER_STRONG};
        border-radius: 8px;
        height: 22px;
        text-align: center;
        color: {c.TEXT_PRIMARY};
        font-family: {FONT_MONO};
        font-size: 12px;
        font-weight: 700;
    }}
    QProgressBar::chunk {{
        background-color: {c.EMERALD};
        border-radius: 7px;
    }}

    /* === RESULT PANEL === */
    QFrame#RightPanel {{
        background-color: {c.BG_SURFACE};
        border: 1px solid {c.BORDER_SUBTLE};
        border-radius: 14px;
    }}
    QLabel#SectionHeading {{
        color: {c.TEXT_MUTED};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.9px;
    }}
    QLabel#ViewBadge {{
        background-color: {c.BLUE_BG};
        color: {c.BLUE_TEXT};
        border: 1px solid {c.BLUE_PRO};
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 700;
    }}
    QLabel#MetricValue {{
        font-family: {FONT_MONO};
        font-size: 12px;
        font-weight: 700;
        color: {c.TEXT_SECONDARY};
    }}

    /* === SLIDER === */
    QSlider::groove:horizontal {{
        background: {c.BG_ELEVATED};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::sub-page:horizontal {{
        background: {c.BLUE_PRO};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {c.TEXT_PRIMARY};
        border: 2px solid {c.BLUE_PRO};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 9px;
    }}

    /* === SCROLLBAR === */
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {c.BORDER_STRONG};
        border-radius: 4px;
        min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {c.TEXT_DIM};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    """
