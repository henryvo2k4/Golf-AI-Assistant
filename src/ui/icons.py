"""
BỘ ICON VECTOR SVG ĐỒNG BỘ (LUCIDE-STYLE 2PX STROKE)
Thay thế hoàn toàn emoji hệ thống theo quy tắc UI/UX Pro Max.
"""

from functools import lru_cache
from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_SVG_PATHS: dict[str, str] = {
    # Biểu tượng tâm ngắm / Golf VAR
    "target": (
        '<circle cx="12" cy="12" r="9"/>'
        '<circle cx="12" cy="12" r="5"/>'
        '<circle cx="12" cy="12" r="1.5" fill="{color}" stroke="none"/>'
        '<path d="M12 1v3M12 20v3M1 12h3M20 12h3"/>'
    ),
    # Nút quay / Bắt đầu phiên
    "play": '<polygon points="6,4 20,12 6,20" fill="{color}" stroke="{color}"/>',
    "pause": (
        '<rect x="6" y="4" width="4" height="16" rx="1" fill="{color}" stroke="{color}"/>'
        '<rect x="14" y="4" width="4" height="16" rx="1" fill="{color}" stroke="{color}"/>'
    ),
    "stop": '<rect x="5" y="5" width="14" height="14" rx="2" fill="{color}" stroke="{color}"/>',
    "x_close": '<path d="M18 6L6 18M6 6l12 12"/>',
    "camera": (
        '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>'
        '<circle cx="12" cy="13" r="4"/>'
    ),
    "video": (
        '<polygon points="23 7 16 12 23 17 23 7"/>'
        '<rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>'
    ),
    "image": (
        '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>'
        '<circle cx="8.5" cy="8.5" r="1.5" fill="{color}" stroke="none"/>'
        '<polyline points="21 15 16 10 5 21"/>'
    ),
    "rotate_ccw": (
        '<polyline points="1 4 1 10 7 10"/>'
        '<path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>'
    ),
    "alert_triangle": (
        '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/>'
        '<circle cx="12" cy="17" r="1" fill="{color}" stroke="none"/>'
    ),
    "check_circle": (
        '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
        '<polyline points="22 4 12 14.01 9 11.01"/>'
    ),
    "eye_off": (
        '<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/>'
        '<path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/>'
        '<line x1="1" y1="1" x2="23" y2="23"/>'
    ),
    "bulb": (
        '<path d="M9 18h6"/>'
        '<path d="M10 22h4"/>'
        '<path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"/>'
    ),
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "compass": (
        '<circle cx="12" cy="12" r="10"/>'
        '<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>'
    ),
    "chevron_right": '<polyline points="9 18 15 12 9 6"/>',
}


def build_svg_bytes(name: str, color: str = "#F8FAFC", stroke_width: float = 2.0) -> bytes:
    """Tạo chuỗi byte SVG hợp lệ với màu và độ dày nét đồng nhất."""
    body_template = _SVG_PATHS.get(name, _SVG_PATHS["target"])
    body = body_template.format(color=color)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
        f"{body}</svg>"
    )
    return svg.encode("utf-8")


@lru_cache(maxsize=128)
def svg_pixmap(name: str, color: str = "#F8FAFC", size: int = 20, stroke_width: float = 2.0) -> QPixmap:
    """Render SVG thành QPixmap sắc nét."""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    renderer = QSvgRenderer(QByteArray(build_svg_bytes(name, color=color, stroke_width=stroke_width)))
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    return pix


@lru_cache(maxsize=128)
def svg_icon(name: str, color: str = "#F8FAFC", size: int = 20, stroke_width: float = 2.0) -> QIcon:
    """Tạo QIcon từ SVG chuẩn vector."""
    return QIcon(svg_pixmap(name, color=color, size=size, stroke_width=stroke_width))
