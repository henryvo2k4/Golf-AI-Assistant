"""
MODULE VẼ ĐỒ HỌA CHÚ THÍCH TRỰC QUAN (PILLOW + OPENCV)
Hỗ trợ 100% tiếng Việt có dấu sắc nét, khoanh vùng tròn tại đúng khớp sai kèm đường gióng chuẩn.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from src.core.pose_detector import SKELETON_CONNECTIONS


def _load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Tải font chữ hỗ trợ tiếng Việt có dấu trên Windows."""
    candidates = [
        config.FONT_BOLD_PATH if bold else config.FONT_REGULAR_PATH,
        config.FALLBACK_FONT_PATH,
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_dashed_line(draw: ImageDraw.ImageDraw, p1, p2, fill, width=3, dash_len=10, gap_len=7):
    """Vẽ đường nét đứt (Reference Line) bằng Pillow."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = float(np.hypot(dx, dy))
    if dist < 1:
        return
    ux, uy = dx / dist, dy / dist
    curr = 0.0
    while curr < dist:
        end = min(curr + dash_len, dist)
        sx, sy = x1 + ux * curr, y1 + uy * curr
        ex, ey = x1 + ux * end, y1 + uy * end
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)
        curr += dash_len + gap_len


def _draw_callout_box(draw: ImageDraw.ImageDraw, img_w: int, img_h: int, anchor_pt, title: str, subtitle: str, bg_rgba, border_rgba):
    """
    Vẽ hộp chữ chú thích có viền bo góc + đường chỉ vào tâm vị trí lỗi,
    tự động kẹp vị trí để không bao giờ bị tràn ra ngoài mép ảnh.
    """
    font_title = _load_font(20, bold=True)
    font_sub = _load_font(16, bold=False)

    # Đo kích thước chữ
    t_bbox = draw.textbbox((0, 0), title, font=font_title)
    s_bbox = draw.textbbox((0, 0), subtitle, font=font_sub)
    tw = t_bbox[2] - t_bbox[0]
    th = t_bbox[3] - t_bbox[1]
    sw = s_bbox[2] - s_bbox[0]
    sh = s_bbox[3] - s_bbox[1]

    pad_x, pad_y = 14, 10
    box_w = max(tw, sw) + pad_x * 2
    box_h = th + sh + pad_y * 2 + 8

    ax, ay = float(anchor_pt[0]), float(anchor_pt[1])

    # Đặt hộp lệch sang bên phải (hoặc trái nếu gần mép phải) và hơi chếch lên trên
    if ax + 55 + box_w < img_w - 10:
        bx = ax + 50
    else:
        bx = ax - 50 - box_w

    by = ay - box_h * 0.5 - 20

    # Kẹp tọa độ an toàn trong khung hình
    bx = max(12, min(bx, img_w - box_w - 12))
    by = max(12, min(by, img_h - box_h - 12))

    # Vẽ đường nối từ tâm vòng tròn tới cạnh hộp chữ
    box_center = (bx + box_w * 0.5, by + box_h * 0.5)
    draw.line([(ax, ay), box_center], fill=border_rgba, width=3)

    # Vẽ hộp chữ nền bo góc
    draw.rounded_rectangle(
        [(bx, by), (bx + box_w, by + box_h)],
        radius=10,
        fill=bg_rgba,
        outline=border_rgba,
        width=3,
    )

    draw.text((bx + pad_x, by + pad_y), title, font=font_title, fill=(255, 255, 255, 255))
    draw.text((bx + pad_x, by + pad_y + th + 6), subtitle, font=font_sub, fill=(235, 240, 245, 255))


def draw_skeleton_overlay(frame_bgr, lm_dict: dict | None, arm_angle: float = 0.0, handedness: str = "right"):
    """
    Vẽ khung xương cơ thể và số đo góc tay dẫn hướng lên frame Video Slow-Motion.
    """
    if not lm_dict:
        return frame_bgr

    img = frame_bgr.copy()

    # Vẽ khung xương cơ bản toàn thân
    for k1, k2 in SKELETON_CONNECTIONS:
        if k1 in lm_dict and k2 in lm_dict:
            if lm_dict[k1][3] >= 0.4 and lm_dict[k2][3] >= 0.4:
                p1 = (int(lm_dict[k1][0]), int(lm_dict[k1][1]))
                p2 = (int(lm_dict[k2][0]), int(lm_dict[k2][1]))
                cv2.line(img, p1, p2, (230, 190, 60), 2, cv2.LINE_AA)

    # Tô nổi bật tay dẫn hướng (vai - khuỷu - cổ tay)
    side = "left" if handedness == "right" else "right"
    s_k, e_k, w_k = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
    if all(k in lm_dict and lm_dict[k][3] >= 0.4 for k in (s_k, e_k, w_k)):
        p1 = (int(lm_dict[s_k][0]), int(lm_dict[s_k][1]))
        p2 = (int(lm_dict[e_k][0]), int(lm_dict[e_k][1]))
        p3 = (int(lm_dict[w_k][0]), int(lm_dict[w_k][1]))
        color = (0, 0, 255) if (0 < arm_angle < config.LEAD_ARM_MIN_ANGLE) else (64, 230, 80)
        cv2.line(img, p1, p2, color, 4, cv2.LINE_AA)
        cv2.line(img, p2, p3, color, 4, cv2.LINE_AA)
        cv2.circle(img, p2, 7, color, -1, cv2.LINE_AA)

    # Ghi góc tay lên góc ảnh bằng Pillow cho đẹp
    if arm_angle > 0:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = _load_font(20, bold=True)
        is_bad = arm_angle < config.LEAD_ARM_MIN_ANGLE
        box_fill = (185, 28, 28, 210) if is_bad else (16, 140, 85, 210)
        text = f"Góc khuỷu tay: {arm_angle:.1f}° {'(GẬP)' if is_bad else '(THẲNG)'}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rounded_rectangle([(16, 16), (16 + tw + 24, 16 + th + 18)], radius=8, fill=box_fill)
        draw.text((28, 24), text, font=font, fill=(255, 255, 255, 255))
        combined = Image.alpha_composite(pil_img, overlay).convert("RGB")
        return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

    return img


def render_annotated_snapshot(frame_bgr, lm_dict: dict | None, criterion, phase_label: str = "") -> np.ndarray:
    """
    Xuất ra ảnh chụp phân tích tĩnh tại khoảnh khắc xảy ra lỗi (hoặc đạt chuẩn):
    - Vẽ khung xương mờ toàn thân.
    - Khoanh vùng tròn màu đỏ (hoặc xanh) rực vào bộ phận tương ứng của `criterion`.
    - Vẽ đường gióng kỹ thuật chuẩn PGA để so sánh trực quan.
    - Hiển thị hộp chú thích tiếng Việt có dấu chỉ thẳng vào chỗ khoanh vùng.
    """
    base_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_base = Image.fromarray(base_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil_base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = pil_base.size

    # 1. Vẽ khung xương nền nhẹ
    if lm_dict:
        for k1, k2 in SKELETON_CONNECTIONS:
            if k1 in lm_dict and k2 in lm_dict:
                if lm_dict[k1][3] >= 0.4 and lm_dict[k2][3] >= 0.4:
                    p1 = (lm_dict[k1][0], lm_dict[k1][1])
                    p2 = (lm_dict[k2][0], lm_dict[k2][1])
                    draw.line([p1, p2], fill=(180, 210, 255, 130), width=3)

    is_err = criterion.status == "error"
    is_pass = criterion.status == "passed"

    accent = (239, 68, 68, 255) if is_err else ((16, 185, 129, 255) if is_pass else (245, 158, 11, 255))
    fill_glow = (239, 68, 68, 55) if is_err else ((16, 185, 129, 45) if is_pass else (245, 158, 11, 45))
    box_bg = (153, 27, 27, 230) if is_err else ((6, 95, 70, 225) if is_pass else (146, 64, 14, 225))
    ref_green = (34, 197, 94, 230)

    ad = criterion.annotation_data or {}
    atype = criterion.annotation_type
    anchor = (w * 0.5, h * 0.45)

    if atype == "elbow_angle" and "elbow" in ad:
        sh, el, wr = ad["shoulder"], ad["elbow"], ad["wrist"]
        draw.line([sh, el], fill=accent, width=6)
        draw.line([el, wr], fill=accent, width=6)
        r = 34
        draw.ellipse([(el[0] - r, el[1] - r), (el[0] + r, el[1] + r)], fill=fill_glow, outline=accent, width=5)
        draw.ellipse([(el[0] - r - 8, el[1] - r - 8), (el[0] + r + 8, el[1] + r + 8)], outline=accent, width=2)
        anchor = el

    elif atype == "head_drift" and "nose" in ad:
        nose = ad["nose"]
        addr_y = ad["addr_nose_y"]
        # Đường ngang chuẩn mức đầu lúc Setup
        _draw_dashed_line(draw, (max(20, nose[0] - 120), addr_y), (min(w - 20, nose[0] + 120), addr_y), fill=ref_green, width=3)
        draw.line([(nose[0], addr_y), (nose[0], nose[1])], fill=accent, width=4)
        r = 38
        draw.ellipse([(nose[0] - r, nose[1] - r), (nose[0] + r, nose[1] + r)], fill=fill_glow, outline=accent, width=5)
        anchor = nose

    elif atype == "hip_sway" and "trail_hip" in ad:
        hip = ad["trail_hip"]
        limit_x = ad["limit_x"]
        # Đường thẳng đứng giới hạn bàn chân sau
        _draw_dashed_line(draw, (limit_x, max(20, hip[1] - 100)), (limit_x, min(h - 20, ad["trail_ankle"][1] + 20)), fill=ref_green, width=3)
        draw.line([(limit_x, hip[1]), (hip[0], hip[1])], fill=accent, width=4)
        r = 36
        draw.ellipse([(hip[0] - r, hip[1] - r), (hip[0] + r, hip[1] + r)], fill=fill_glow, outline=accent, width=5)
        anchor = hip

    elif atype == "spine_tilt" and "mid_hip" in ad:
        m_hip = ad["mid_hip"]
        m_sh = ad["mid_shoulder"]
        # Đường thẳng đứng chuẩn từ tâm hông
        _draw_dashed_line(draw, m_hip, (m_hip[0], max(20, m_sh[1] - 40)), fill=ref_green, width=3)
        draw.line([m_hip, m_sh], fill=accent, width=6)
        mid_torso = ((m_hip[0] + m_sh[0]) * 0.5, (m_hip[1] + m_sh[1]) * 0.5)
        r = 42
        draw.ellipse([(mid_torso[0] - r, mid_torso[1] - r), (mid_torso[0] + r, mid_torso[1] + r)], fill=fill_glow, outline=accent, width=5)
        anchor = mid_torso

    elif atype == "knee_angle" and "knee" in ad:
        hp, kn, ak = ad["hip"], ad["knee"], ad["ankle"]
        draw.line([hp, kn], fill=accent, width=6)
        draw.line([kn, ak], fill=accent, width=6)
        r = 32
        draw.ellipse([(kn[0] - r, kn[1] - r), (kn[0] + r, kn[1] + r)], fill=fill_glow, outline=accent, width=5)
        anchor = kn

    # 2. Vẽ Callout Box chỉ trực tiếp vào vị trí vừa khoanh tròn
    prefix = "LỖI: " if is_err else ("ĐẠT: " if is_pass else "LƯU Ý: ")
    title_txt = f"{prefix}{criterion.title}"
    sub_txt = f"Đo được: {criterion.measured_text}  |  Chuẩn: {criterion.standard_text}"
    _draw_callout_box(draw, w, h, anchor, title_txt, sub_txt, box_bg, accent)

    # 3. Tem nhãn thời điểm góc trên trái
    if phase_label:
        font_phase = _load_font(17, bold=True)
        pb = draw.textbbox((0, 0), phase_label, font=font_phase)
        pw, ph = pb[2] - pb[0], pb[3] - pb[1]
        draw.rounded_rectangle([(14, 14), (14 + pw + 24, 14 + ph + 14)], radius=8, fill=(15, 23, 42, 215), outline=(100, 116, 139, 200), width=2)
        draw.text((26, 20), phase_label, font=font_phase, fill=(248, 250, 252, 255))

    result_rgb = Image.alpha_composite(pil_base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(result_rgb), cv2.COLOR_RGB2BGR)
