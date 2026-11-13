"""
HỆ THỐNG DESIGN TOKENS & STYLESHEET (MODERN WHITE BENTO GOLF STUDIO)
Phong cách sáng hiện đại, nền trắng tinh khôi, bo góc mềm mại (14-22px),
đổ bóng nhẹ (QGraphicsDropShadowEffect), icon vector SVG, tương phản >= 4.5:1.
"""

from dataclasses import dataclass
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QWidget


@dataclass(frozen=True)
class ThemeColors:
    # Nền sáng & Bề mặt thẻ trắng (Clean Light Bento)
    BG_CANVAS: str = "#F3F6FA"
    BG_CARD: str = "#FFFFFF"
    BG_SUBTLE: str = "#F8FAFC"
    BG_MUTED: str = "#EFF3F8"
    BG_HOVER: str = "#F1F5F9"
    BG_VIEWPORT: str = "#0F172A"

    # Đường viền tinh tế
    BORDER_LIGHT: str = "#E2E8F0"
    BORDER_MEDIUM: str = "#CBD5E1"
    BORDER_FOCUS: str = "#10B981"

    # Màu nhấn Golf Emerald & Pro Blue
    EMERALD: str = "#059669"
    EMERALD_BRIGHT: str = "#10B981"
    EMERALD_HOVER: str = "#047857"
    EMERALD_SOFT: str = "#ECFDF5"
    EMERALD_BORDER: str = "#A7F3D0"
    EMERALD_TEXT: str = "#065F46"

    BLUE_PRO: str = "#2563EB"
    BLUE_HOVER: str = "#1D4ED8"
    BLUE_SOFT: str = "#EFF6FF"
    BLUE_BORDER: str = "#BFDBFE"
    BLUE_TEXT: str = "#1E40AF"

    # Ngữ nghĩa lỗi & cảnh báo (Light Mode contrast >= 4.5:1)
    CRIMSON: str = "#DC2626"
    CRIMSON_HOVER: str = "#B91C1C"
    CRIMSON_SOFT: str = "#FEF2F2"
    CRIMSON_BORDER: str = "#FECACA"
    CRIMSON_TEXT: str = "#991B1B"

    AMBER: str = "#D97706"
    AMBER_BRIGHT: str = "#F59E0B"
    AMBER_SOFT: str = "#FFFBEB"
    AMBER_BORDER: str = "#FDE68A"
    AMBER_TEXT: str = "#92400E"

    # Văn bản
    TEXT_PRIMARY: str = "#0F172A"
    TEXT_SECONDARY: str = "#334155"
    TEXT_MUTED: str = "#64748B"
    TEXT_DIM: str = "#94A3B8"


COLORS = ThemeColors()
FONT_UI = '"Segoe UI", "Inter", sans-serif'
FONT_MONO = '"Consolas", "Cascadia Mono", monospace'


def apply_soft_shadow(widget: QWidget, blur: int = 24, y_offset: int = 6, alpha: int = 20) -> None:
    """Gắn bóng đổ mềm phong cách Apple/Bento cho các thẻ trắng nổi."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    shadow.setOffset(0, y_offset)
    shadow.setColor(QColor(15, 23, 42, alpha))
    widget.setGraphicsEffect(shadow)


def build_app_stylesheet() -> str:
    c = COLORS
    return f"""
    QMainWindow, QWidget#RootCanvas {{
        background-color: {c.BG_CANVAS};
        color: {c.TEXT_PRIMARY};
        font-family: {FONT_UI};
    }}

    /* === TOP FLOATING BAR === */
    QFrame#TopBar {{
        background-color: {c.BG_CARD};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 18px;
    }}
    QFrame#BrandIconBox {{
        background-color: {c.EMERALD_SOFT};
        border: 1px solid {c.EMERALD_BORDER};
        border-radius: 12px;
    }}
    QLabel#BrandTitle {{
        color: {c.TEXT_PRIMARY};
        font-size: 16px;
        font-weight: 800;
        letter-spacing: -0.2px;
    }}
    QLabel#BrandSubtitle {{
        color: {c.TEXT_MUTED};
        font-size: 11px;
        font-weight: 600;
    }}
    QLabel#StatusPill {{
        background-color: {c.EMERALD_SOFT};
        color: {c.EMERALD_TEXT};
        border: 1px solid {c.EMERALD_BORDER};
        border-radius: 14px;
        padding: 4px 12px;
        font-size: 11px;
        font-weight: 700;
    }}
    QLabel#TopControlLabel {{
        color: {c.TEXT_MUTED};
        font-size: 12px;
        font-weight: 600;
    }}

    /* === COMBOBOX & CHECKBOX (ROUNDED PILLS) === */
    QComboBox {{
        background-color: {c.BG_SUBTLE};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 11px;
        padding: 6px 14px;
        min-height: 26px;
        font-size: 13px;
        font-weight: 600;
    }}
    QComboBox:hover {{
        border-color: {c.BORDER_MEDIUM};
        background-color: {c.BG_HOVER};
    }}
    QComboBox:focus {{
        border-color: {c.EMERALD};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c.BG_CARD};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 10px;
        padding: 4px;
        selection-background-color: {c.EMERALD_SOFT};
        selection-color: {c.EMERALD_TEXT};
        outline: 0;
    }}

    QCheckBox {{
        color: {c.TEXT_SECONDARY};
        font-size: 13px;
        font-weight: 600;
        spacing: 8px;
        padding: 6px 12px;
        background-color: {c.BG_SUBTLE};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 11px;
    }}
    QCheckBox:hover {{
        background-color: {c.BG_HOVER};
        border-color: {c.BORDER_MEDIUM};
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 6px;
        border: 1.5px solid {c.BORDER_MEDIUM};
        background-color: {c.BG_CARD};
    }}
    QCheckBox::indicator:checked {{
        background-color: {c.EMERALD};
        border-color: {c.EMERALD};
    }}

    /* === FLOATING WHITE STAGE & BENTO CARDS === */
    QFrame#StageCard {{
        background-color: {c.BG_CARD};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 22px;
    }}
    QFrame#ControlDock {{
        background-color: {c.BG_CARD};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 18px;
    }}
    QFrame#RightPanel {{
        background-color: {c.BG_CARD};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 22px;
    }}
    QLabel#LiveHintLabel {{
        color: {c.TEXT_SECONDARY};
        font-size: 14px;
        font-weight: 500;
    }}

    /* === SEGMENTED CONTROL TRACK === */
    QFrame#SegmentTrack {{
        background-color: {c.BG_MUTED};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 14px;
    }}
    QPushButton#SegmentBtn {{
        background-color: transparent;
        color: {c.TEXT_MUTED};
        border: none;
        border-radius: 10px;
        padding: 8px 16px;
        min-height: 24px;
        font-size: 13px;
        font-weight: 700;
    }}
    QPushButton#SegmentBtn:hover {{
        color: {c.TEXT_PRIMARY};
    }}
    QPushButton#SegmentBtn[active="emerald"] {{
        background-color: {c.BG_CARD};
        color: {c.EMERALD_TEXT};
        border: 1px solid {c.BORDER_LIGHT};
    }}
    QPushButton#SegmentBtn[active="true"] {{
        background-color: {c.BG_CARD};
        color: {c.BLUE_TEXT};
        border: 1px solid {c.BORDER_LIGHT};
    }}

    /* === BUTTONS (ROUNDED PILLS, >= 44px) === */
    QPushButton#PrimaryCTA {{
        background-color: {c.EMERALD};
        color: #FFFFFF;
        border: none;
        border-radius: 14px;
        padding: 12px 24px;
        min-height: 24px;
        font-size: 14px;
        font-weight: 700;
    }}
    QPushButton#PrimaryCTA:hover {{
        background-color: {c.EMERALD_HOVER};
    }}
    QPushButton#PrimaryCTA:disabled {{
        background-color: {c.BG_MUTED};
        color: {c.TEXT_DIM};
    }}

    QPushButton#DangerCTA {{
        background-color: {c.CRIMSON};
        color: #FFFFFF;
        border: none;
        border-radius: 14px;
        padding: 12px 22px;
        min-height: 24px;
        font-size: 14px;
        font-weight: 700;
    }}
    QPushButton#DangerCTA:hover {{
        background-color: {c.CRIMSON_HOVER};
    }}

    QPushButton#SecondaryBtn {{
        background-color: {c.BG_SUBTLE};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 12px;
        padding: 9px 16px;
        min-height: 24px;
        font-size: 13px;
        font-weight: 700;
    }}
    QPushButton#SecondaryBtn:hover {{
        background-color: {c.BG_HOVER};
        border-color: {c.BORDER_MEDIUM};
    }}

    QPushButton#PhaseBtn {{
        background-color: {c.BG_SUBTLE};
        color: {c.TEXT_SECONDARY};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 12px;
        padding: 7px 14px;
        min-height: 22px;
        font-size: 12px;
        font-weight: 700;
    }}
    QPushButton#PhaseBtn:hover {{
        background-color: {c.EMERALD_SOFT};
        border-color: {c.EMERALD_BORDER};
        color: {c.EMERALD_TEXT};
    }}

    QPushButton#ExitBtn {{
        background-color: {c.BG_SUBTLE};
        color: {c.TEXT_MUTED};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 12px;
        min-width: 38px;
        max-width: 38px;
        min-height: 38px;
        max-height: 38px;
        padding: 0px;
    }}
    QPushButton#ExitBtn:hover {{
        background-color: {c.CRIMSON_SOFT};
        border-color: {c.CRIMSON_BORDER};
        color: {c.CRIMSON};
    }}

    /* === PROGRESS BAR === */
    QProgressBar {{
        background-color: {c.BG_MUTED};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 10px;
        height: 20px;
        text-align: center;
        color: {c.TEXT_PRIMARY};
        font-family: {FONT_MONO};
        font-size: 11px;
        font-weight: 700;
    }}
    QProgressBar::chunk {{
        background-color: {c.EMERALD_BRIGHT};
        border-radius: 9px;
    }}

    /* === BADGES & LABELS === */
    QLabel#SectionHeading {{
        color: {c.TEXT_MUTED};
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.7px;
    }}
    QLabel#ViewBadge {{
        background-color: {c.BLUE_SOFT};
        color: {c.BLUE_TEXT};
        border: 1px solid {c.BLUE_BORDER};
        border-radius: 12px;
        padding: 6px 14px;
        font-size: 12px;
        font-weight: 700;
    }}
    QLabel#ScorePill {{
        background-color: {c.BG_MUTED};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 10px;
        padding: 4px 10px;
        font-family: {FONT_MONO};
        font-size: 12px;
        font-weight: 800;
    }}
    QLabel#MetricValue {{
        font-family: {FONT_MONO};
        font-size: 12px;
        font-weight: 700;
        color: {c.TEXT_SECONDARY};
    }}

    /* === SLIDER === */
    QSlider::groove:horizontal {{
        background: {c.BG_MUTED};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::sub-page:horizontal {{
        background: {c.BLUE_PRO};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {c.BG_CARD};
        border: 2.5px solid {c.BLUE_PRO};
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
        width: 7px;
        margin: 2px 0;
    }}
    QScrollBar::handle:vertical {{
        background: {c.BORDER_MEDIUM};
        border-radius: 3px;
        min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {c.TEXT_MUTED};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    """
