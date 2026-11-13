"""
MODULE VẼ ĐỒ HỌA CHÚ THÍCH TRỰC QUAN (PILLOW + OPENCV)
- Hỗ trợ 100% tiếng Việt có dấu sắc nét (Không dùng ký tự emoji tránh lỗi ô vuông).
- Khoanh vùng tròn chính xác tại khớp lỗi kèm đường gióng chuẩn PGA & hộp chú thích.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from src.core.pose_detector import SKELETON_CONNECTIONS


def _load_font(size: int, bold: bool = True, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono and os.path.exists(config.FONT_MONO_PATH):
        candidates = [config.FONT_MONO_PATH, config.FONT_BOLD_PATH, config.FALLBACK_FONT_PATH]
    else:
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
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    dist = float(np.hypot(dx, dy))
    if dist < 1:
        return
    ux, uy = dx / dist, dy / dist
    curr = 0.0
    while curr < dist:
        end = min(curr + dash_len, dist)
        draw.line([(x1 + ux * curr, y1 + uy * curr), (x1 + ux * end, y1 + uy * end)], fill=fill, width=width)
        curr += dash_len + gap_len


def _draw_target_ring(draw: ImageDraw.ImageDraw, center, radius: int, accent, fill_glow):
    cx, cy = center
    draw.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)], fill=fill_glow, outline=accent, width=4)
    draw.ellipse([(cx - radius - 7, cy - radius - 7), (cx + radius + 7, cy + radius + 7)], outline=(accent[0], accent[1], accent[2], 140), width=2)
    draw.ellipse([(cx - 5, cy - 5), (cx + 5, cy + 5)], fill=accent)


def _draw_callout_box(draw: ImageDraw.ImageDraw, img_w: int, img_h: int, anchor_pt, title: str, subtitle: str, bg_rgba, border_rgba):
    font_title = _load_font(19, bold=True)
    font_sub = _load_font(15, bold=False)

    t_bbox = draw.textbbox((0, 0), title, font=font_title)
    s_bbox = draw.textbbox((0, 0), subtitle, font=font_sub)
    tw, th = t_bbox[2] - t_bbox[0], t_bbox[3] - t_bbox[1]
    sw, sh = s_bbox[2] - s_bbox[0], s_bbox[3] - s_bbox[1]

    pad_x, pad_y = 14, 10
    box_w = max(tw, sw) + pad_x * 2
    box_h = th + sh + pad_y * 2 + 8

    ax, ay = float(anchor_pt[0]), float(anchor_pt[1])
    bx = (ax + 52) if (ax + 58 + box_w < img_w - 12) else (ax - 52 - box_w)
    by = ay - box_h * 0.5 - 24

    bx = max(14, min(bx, img_w - box_w - 14))
    by = max(56, min(by, img_h - box_h - 14))

    box_center = (bx + box_w * 0.5, by + box_h * 0.5)
    draw.line([(ax, ay), box_center], fill=border_rgba, width=3)
    draw.rounded_rectangle([(bx, by), (bx + box_w, by + box_h)], radius=10, fill=bg_rgba, outline=border_rgba, width=2)

    draw.text((bx + pad_x, by + pad_y), title, font=font_title, fill=(248, 250, 252, 255))
    draw.text((bx + pad_x, by + pad_y + th + 6), subtitle, font=font_sub, fill=(226, 232, 240, 255))


def draw_skeleton_overlay(frame_bgr, lm_dict: dict | None, arm_angle: float = 0.0, handedness: str = "right"):
    """Vẽ khung xương cơ thể và số đo góc 3D cho Video Slow-Motion."""
    if not lm_dict:
        return frame_bgr

    img = frame_bgr.copy()

    for k1, k2 in SKELETON_CONNECTIONS:
        if k1 in lm_dict and k2 in lm_dict:
            if lm_dict[k1][3] >= 0.4 and lm_dict[k2][3] >= 0.4:
                p1 = (int(lm_dict[k1][0]), int(lm_dict[k1][1]))
                p2 = (int(lm_dict[k2][0]), int(lm_dict[k2][1]))
                cv2.line(img, p1, p2, (210, 170, 50), 2, cv2.LINE_AA)

    side = "left" if handedness == "right" else "right"
    s_k, e_k, w_k = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
    if all(k in lm_dict and lm_dict[k][3] >= 0.4 for k in (s_k, e_k, w_k)):
        p1 = (int(lm_dict[s_k][0]), int(lm_dict[s_k][1]))
        p2 = (int(lm_dict[e_k][0]), int(lm_dict[e_k][1]))
        p3 = (int(lm_dict[w_k][0]), int(lm_dict[w_k][1]))
        color = (68, 68, 239) if (0 < arm_angle < config.LEAD_ARM_MIN_ANGLE_3D) else (129, 185, 16)
        cv2.line(img, p1, p2, color, 4, cv2.LINE_AA)
        cv2.line(img, p2, p3, color, 4, cv2.LINE_AA)
        cv2.circle(img, p2, 7, color, -1, cv2.LINE_AA)

    if arm_angle > 0:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = _load_font(18, bold=True)
        is_bad = arm_angle < config.LEAD_ARM_MIN_ANGLE_3D
        box_fill = (153, 27, 27, 220) if is_bad else (6, 95, 70, 220)
        border = (239, 68, 68, 255) if is_bad else (16, 185, 129, 255)
        text = f"Góc tay 3D: {arm_angle:.1f}° | {'GẬP TAY' if is_bad else 'THẲNG CHUẨN'}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rounded_rectangle([(14, 14), (14 + tw + 26, 14 + th + 16)], radius=8, fill=box_fill, outline=border, width=2)
        draw.text((27, 21), text, font=font, fill=(248, 250, 252, 255))
        combined = Image.alpha_composite(pil_img, overlay).convert("RGB")
        return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

    return img


def render_annotated_snapshot(frame_bgr, lm_dict: dict | None, criterion, phase_label: str = "") -> np.ndarray:
    """
    Vẽ ảnh phân tích tĩnh khoanh vùng vị trí sai / đạt chuẩn kèm đường gióng và chú thích sắc nét.
    """
    base_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_base = Image.fromarray(base_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil_base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = pil_base.size

    if lm_dict:
        for k1, k2 in SKELETON_CONNECTIONS:
            if k1 in lm_dict and k2 in lm_dict:
                if lm_dict[k1][3] >= 0.4 and lm_dict[k2][3] >= 0.4:
                    draw.line(
                        [(lm_dict[k1][0], lm_dict[k1][1]), (lm_dict[k2][0], lm_dict[k2][1])],
                        fill=(148, 163, 184, 140),
                        width=3,
                    )

    is_err = criterion.status == "error"
    is_pass = criterion.status == "passed"

    accent = (239, 68, 68, 255) if is_err else ((16, 185, 129, 255) if is_pass else (245, 158, 11, 255))
    fill_glow = (239, 68, 68, 60) if is_err else ((16, 185, 129, 50) if is_pass else (245, 158, 11, 50))
    box_bg = (127, 29, 29, 232) if is_err else ((6, 78, 59, 230) if is_pass else (120, 53, 15, 230))
    ref_green = (34, 197, 94, 235)

    ad = criterion.annotation_data or {}
    atype = criterion.annotation_type
    anchor = (w * 0.5, h * 0.45)

    if atype == "elbow_angle" and "elbow" in ad:
        sh, el, wr = ad["shoulder"], ad["elbow"], ad["wrist"]
        draw.line([sh, el], fill=accent, width=6)
        draw.line([el, wr], fill=accent, width=6)
        _draw_target_ring(draw, el, 34, accent, fill_glow)
        anchor = el

    elif atype == "head_drift" and "nose" in ad:
        nose = ad["nose"]
        addr_pt = ad.get("addr_nose", (nose[0], ad["addr_nose_y"]))
        _draw_dashed_line(draw, (max(20, addr_pt[0] - 110), addr_pt[1]), (min(w - 20, addr_pt[0] + 110), addr_pt[1]), fill=ref_green, width=3)
        _draw_dashed_line(draw, (addr_pt[0], max(20, addr_pt[1] - 50)), (addr_pt[0], min(h - 20, addr_pt[1] + 50)), fill=ref_green, width=2)
        draw.line([addr_pt, nose], fill=accent, width=4)
        _draw_target_ring(draw, nose, 38, accent, fill_glow)
        anchor = nose

    elif atype == "shoulder_turn" and "left_shoulder" in ad:
        l_sh, r_sh, ctr = ad["left_shoulder"], ad["right_shoulder"], ad["center"]
        draw.line([l_sh, r_sh], fill=accent, width=6)
        _draw_target_ring(draw, ctr, 38, accent, fill_glow)
        anchor = ctr

    elif atype == "hip_sway" and "trail_hip" in ad:
        hip = ad["trail_hip"]
        limit_x = ad["limit_x"]
        _draw_dashed_line(draw, (limit_x, max(20, hip[1] - 110)), (limit_x, min(h - 20, ad["trail_ankle"][1] + 15)), fill=ref_green, width=3)
        draw.line([(limit_x, hip[1]), (hip[0], hip[1])], fill=accent, width=4)
        _draw_target_ring(draw, hip, 34, accent, fill_glow)
        anchor = hip

    elif atype == "spine_tilt" and "mid_hip" in ad:
        m_hip, m_sh = ad["mid_hip"], ad["mid_shoulder"]
        _draw_dashed_line(draw, m_hip, (m_hip[0], max(20, m_sh[1] - 45)), fill=ref_green, width=3)
        draw.line([m_hip, m_sh], fill=accent, width=6)
        mid_torso = ((m_hip[0] + m_sh[0]) * 0.5, (m_hip[1] + m_sh[1]) * 0.5)
        _draw_target_ring(draw, mid_torso, 40, accent, fill_glow)
        anchor = mid_torso

    elif atype == "knee_angle" and "knee" in ad:
        hp, kn, ak = ad["hip"], ad["knee"], ad["ankle"]
        draw.line([hp, kn], fill=accent, width=6)
        draw.line([kn, ak], fill=accent, width=6)
        _draw_target_ring(draw, kn, 32, accent, fill_glow)
        anchor = kn

    prefix = "LỖI: " if is_err else ("ĐẠT: " if is_pass else "CHÚ Ý: ")
    _draw_callout_box(
        draw, w, h, anchor,
        f"{prefix}{criterion.title}",
        f"Đo được: {criterion.measured_text}  |  Chuẩn: {criterion.standard_text}",
        box_bg, accent,
    )

    if phase_label:
        font_phase = _load_font(15, bold=True)
        pb = draw.textbbox((0, 0), phase_label, font=font_phase)
        pw, ph = pb[2] - pb[0], pb[3] - pb[1]
        draw.rounded_rectangle([(16, 16), (16 + pw + 26, 16 + ph + 16)], radius=14, fill=(255, 255, 255, 235), outline=(226, 232, 240, 240), width=2)
        draw.text((29, 23), phase_label, font=font_phase, fill=(15, 23, 42, 255))

    result_rgb = Image.alpha_composite(pil_base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(result_rgb), cv2.COLOR_RGB2BGR)
