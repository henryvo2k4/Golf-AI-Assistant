"""
DESIGN TOKENS & STYLESHEET - VAR GOLF STUDIO (MINIMAL WHITE BENTO KIOSK UI)
Designed for mouse-free / touch-free display: crisp white cards, soft drop shadows,
large readable typography from 2-3 meters away, and high contrast >= 4.5:1.
"""

from dataclasses import dataclass
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QWidget


@dataclass(frozen=True)
class ThemeColors:
    # Clean Light Bento Canvas & Surfaces
    BG_CANVAS: str = "#F3F6FA"
    BG_CARD: str = "#FFFFFF"
    BG_SUBTLE: str = "#F8FAFC"
    BG_MUTED: str = "#EFF3F8"
    BG_HOVER: str = "#F1F5F9"

    # Hairline Borders
    BORDER_LIGHT: str = "#E2E8F0"
    BORDER_MEDIUM: str = "#CBD5E1"

    # Golf Emerald & Pro Blue Accents
    EMERALD: str = "#059669"
    EMERALD_BRIGHT: str = "#10B981"
    EMERALD_HOVER: str = "#047857"
    EMERALD_SOFT: str = "#ECFDF5"
    EMERALD_BORDER: str = "#A7F3D0"
    EMERALD_TEXT: str = "#065F46"

    BLUE_PRO: str = "#2563EB"
    BLUE_SOFT: str = "#EFF6FF"
    BLUE_BORDER: str = "#BFDBFE"
    BLUE_TEXT: str = "#1E40AF"

    # Semantic Feedback
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

    # Typography
    TEXT_PRIMARY: str = "#0F172A"
    TEXT_SECONDARY: str = "#334155"
    TEXT_MUTED: str = "#64748B"
    TEXT_DIM: str = "#94A3B8"


COLORS = ThemeColors()
FONT_UI = '"Segoe UI", "Inter", sans-serif'
FONT_MONO = '"Consolas", "Cascadia Mono", monospace'


def apply_soft_shadow(widget: QWidget, blur: int = 24, y_offset: int = 6, alpha: int = 18) -> None:
    """Attach soft Apple-style elevation shadow to floating white cards."""
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

    /* === FLOATING GLASS PILLS OVER FULL-BLEED CAMERA === */
    QFrame#FloatingPill {{
        background-color: rgba(255, 255, 255, 235);
        border: 1px solid rgba(226, 232, 240, 230);
        border-radius: 18px;
    }}
    QFrame#AnalyzeModal {{
        background-color: rgba(255, 255, 255, 244);
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 22px;
    }}
    QLabel#FloatingMeta {{
        color: {c.TEXT_SECONDARY};
        font-size: 13px;
        font-weight: 700;
        background: transparent;
    }}

    /* === MINIMAL BADGES === */
    QFrame#TopBar {{
        background-color: {c.BG_CARD};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 18px;
    }}
    QFrame#BrandIconBox {{
        background-color: {c.EMERALD_SOFT};
        border: 1px solid {c.EMERALD_BORDER};
        border-radius: 11px;
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
    QLabel#MetaPill {{
        background-color: {c.BG_SUBTLE};
        color: {c.TEXT_SECONDARY};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 12px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 600;
    }}
    QLabel#ScorePill {{
        background-color: {c.BLUE_SOFT};
        color: {c.BLUE_TEXT};
        border: 1px solid {c.BLUE_BORDER};
        border-radius: 12px;
        padding: 4px 12px;
        font-family: {FONT_MONO};
        font-size: 12px;
        font-weight: 800;
    }}

    /* === FLOATING STAGE & SLIDE CARDS === */
    QFrame#StageCard, QFrame#SlideCard {{
        background-color: {c.BG_CARD};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 22px;
    }}
    QFrame#ControlDock {{
        background-color: {c.BG_CARD};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 18px;
    }}
    QLabel#LiveHintLabel {{
        color: {c.TEXT_SECONDARY};
        font-size: 15px;
        font-weight: 600;
    }}

    /* === BUTTONS === */
    QPushButton#PrimaryCTA {{
        background-color: {c.EMERALD};
        color: #FFFFFF;
        border: none;
        border-radius: 14px;
        padding: 12px 26px;
        min-height: 24px;
        font-size: 15px;
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
        padding: 12px 24px;
        min-height: 24px;
        font-size: 15px;
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
        padding: 10px 18px;
        min-height: 22px;
        font-size: 13px;
        font-weight: 700;
    }}
    QPushButton#SecondaryBtn:hover {{
        background-color: {c.BG_HOVER};
        border-color: {c.BORDER_MEDIUM};
    }}

    QPushButton#ExitBtn {{
        background-color: {c.BG_SUBTLE};
        color: {c.TEXT_MUTED};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 11px;
        min-width: 36px;
        max-width: 36px;
        min-height: 36px;
        max-height: 36px;
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

    QProgressBar#SlideTimerBar {{
        background-color: {c.BG_MUTED};
        border: none;
        border-radius: 3px;
        height: 5px;
        text-align: center;
    }}
    QProgressBar#SlideTimerBar::chunk {{
        background-color: {c.EMERALD_BRIGHT};
        border-radius: 3px;
    }}
    """
