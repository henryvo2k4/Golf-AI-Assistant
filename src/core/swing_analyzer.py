"""
MODULE BỘ NÃO PHÂN TÍCH CÚ SWING TOÀN DIỆN (BATCH BIOMECHANICAL ANALYZER)
Phân tích chuỗi video 10 giây, tự động định vị các pha (Address -> Top -> Impact)
và kiểm tra 5 lỗi kỹ thuật Golf kinh điển chuẩn PGA.
"""

from dataclasses import dataclass, field
import numpy as np

import config
from src.core.pose_detector import GolfPoseDetector
from src.utils.geometry import (
    calculate_angle_2d,
    calculate_spine_away_tilt_deg,
    distance_2d,
    midpoint_2d,
)


@dataclass
class FaultCriterion:
    code: str
    title: str
    status: str              # "passed" | "error" | "not_visible"
    measured_text: str
    standard_text: str
    frame_idx: int
    description: str
    correction_tip: str
    annotation_type: str
    annotation_data: dict = field(default_factory=dict)


@dataclass
class SwingAnalysisResult:
    person_detected: bool
    total_frames: int
    fps: float
    address_idx: int
    top_idx: int
    impact_idx: int
    score_passed: int
    score_evaluated: int
    criteria: list[FaultCriterion]
    landmarks_per_frame: list
    arm_angles_per_frame: list[float]


class BatchSwingAnalyzer:
    """Phân tích toàn bộ video cú đánh vừa ghi để trích xuất chính xác khung hình và các lỗi."""

    def __init__(self, handedness: str = None):
        self.handedness = handedness or config.GOLFER_HANDEDNESS
        self.detector = GolfPoseDetector()

    def _side_keys(self):
        """Trả về tên khớp tay dẫn hướng (lead) và tay sau (trail) theo tay thuận."""
        if self.handedness == "right":
            return {
                "lead_shoulder": "left_shoulder",
                "lead_elbow": "left_elbow",
                "lead_wrist": "left_wrist",
                "lead_hip": "left_hip",
                "lead_knee": "left_knee",
                "lead_ankle": "left_ankle",
                "trail_shoulder": "right_shoulder",
                "trail_hip": "right_hip",
                "trail_knee": "right_knee",
                "trail_ankle": "right_ankle",
            }
        return {
            "lead_shoulder": "right_shoulder",
            "lead_elbow": "right_elbow",
            "lead_wrist": "right_wrist",
            "lead_hip": "right_hip",
            "lead_knee": "right_knee",
            "lead_ankle": "right_ankle",
            "trail_shoulder": "left_shoulder",
            "trail_hip": "left_hip",
            "trail_knee": "left_knee",
            "trail_ankle": "left_ankle",
        }

    @staticmethod
    def _is_visible(lm_dict: dict | None, keys: list[str], threshold: float = 0.42) -> bool:
        if lm_dict is None:
            return False
        return all((k in lm_dict and lm_dict[k][3] >= threshold) for k in keys)

    def analyze_clip(self, frames_bgr: list, timestamps_ms: list, progress_callback=None) -> SwingAnalysisResult:
        """
        Chạy AI MediaPipe PoseLandmarker trên toàn bộ clip vừa quay và tổng hợp 5 lỗi.
        """
        total_frames = len(frames_bgr)
        if total_frames < 5:
            return SwingAnalysisResult(
                person_detected=False, total_frames=total_frames, fps=30.0,
                address_idx=0, top_idx=0, impact_idx=0,
                score_passed=0, score_evaluated=0, criteria=[],
                landmarks_per_frame=[], arm_angles_per_frame=[]
            )

        # Tính FPS thực tế của clip vừa quay
        duration_sec = max((timestamps_ms[-1] - timestamps_ms[0]) / 1000.0, 0.1)
        actual_fps = total_frames / duration_sec

        # 1. Trích xuất khung xương cho toàn bộ frame
        landmarks_list = self.detector.detect_video_sequence(
            frames_bgr, timestamps_ms, progress_callback=progress_callback
        )

        sk = self._side_keys()

        # Tính trước góc tay dẫn hướng trên mỗi frame (để dùng cho Slow-motion replay)
        arm_angles = []
        valid_indices = []
        wrist_y_values = []

        for idx, lm in enumerate(landmarks_list):
            if self._is_visible(lm, [sk["lead_shoulder"], sk["lead_elbow"], sk["lead_wrist"]]):
                angle = calculate_angle_2d(
                    lm[sk["lead_shoulder"]], lm[sk["lead_elbow"]], lm[sk["lead_wrist"]]
                )
                arm_angles.append(angle)
                valid_indices.append(idx)
                wrist_y_values.append(lm[sk["lead_wrist"]][1])
            else:
                arm_angles.append(0.0)

        if len(valid_indices) < 8:
            return SwingAnalysisResult(
                person_detected=False, total_frames=total_frames, fps=actual_fps,
                address_idx=0, top_idx=0, impact_idx=0,
                score_passed=0, score_evaluated=0, criteria=[],
                landmarks_per_frame=landmarks_list, arm_angles_per_frame=arm_angles
            )

        # 2. Tìm ĐỈNH SWING (Top of Swing) = Khung hình cổ tay lên cao nhất (y nhỏ nhất)
        # Làm mượt nhẹ 3 điểm để chống nhiễu 1 frame lẻ
        smoothed_y = []
        for i in range(len(wrist_y_values)):
            window = wrist_y_values[max(0, i - 1): min(len(wrist_y_values), i + 2)]
            smoothed_y.append(float(np.median(window)))

        best_valid_pos = int(np.argmin(smoothed_y))
        top_idx = valid_indices[best_valid_pos]

        # 3. Tìm tư thế CHUẨN BỊ (Address): Khung hình ổn định trước khi bắt đầu vung gậy
        # Quét ngược từ top_idx về trước để tìm vùng cổ tay ở vị trí thấp yên ổn
        pre_top_valid = [i for i in valid_indices if i < top_idx]
        if len(pre_top_valid) >= 4:
            pre_wrist_ys = [landmarks_list[i][sk["lead_wrist"]][1] for i in pre_top_valid]
            # Vị trí nghỉ chuẩn bị là phần trăm thứ 80 (cổ tay ở thấp nhất trước khi giơ lên)
            resting_y = float(np.percentile(pre_wrist_ys, 80))
            # Lấy khung hình đầu tiên gần vị trí nghỉ trước pha backswing (tính từ giữa đoạn chờ)
            candidates = [idx for idx, wy in zip(pre_top_valid, pre_wrist_ys) if wy >= resting_y - 15.0]
            address_idx = candidates[len(candidates) // 2] if candidates else pre_top_valid[0]
        else:
            address_idx = valid_indices[0]

        # 4. Tìm thời điểm CHẠM BÓNG (Impact): Cổ tay hạ thấp nhanh nhất sau đỉnh swing
        post_top_window_frames = int(actual_fps * 1.5)
        post_top_valid = [i for i in valid_indices if top_idx < i <= top_idx + post_top_window_frames]
        if post_top_valid:
            impact_idx = max(post_top_valid, key=lambda i: landmarks_list[i][sk["lead_wrist"]][1])
        else:
            impact_idx = min(total_frames - 1, top_idx + 1)

        # ==========================
        # ĐÁNH GIÁ 5 TIÊU CHÍ LỖI GOLF
        # ==========================
        criteria: list[FaultCriterion] = []
        eval_range = [i for i in valid_indices if address_idx <= i <= max(impact_idx, top_idx)]
        if not eval_range:
            eval_range = valid_indices

        swing_peak_to_impact = [i for i in valid_indices if max(address_idx, top_idx - 2) <= i <= max(impact_idx, top_idx)]
        if not swing_peak_to_impact:
            swing_peak_to_impact = [top_idx]

        # --- TIÊU CHÍ 1: GẬP KHUỶU TAY DẪN HƯỚNG (LEAD ELBOW ANGLE) ---
        worst_arm_idx = min(swing_peak_to_impact, key=lambda i: arm_angles[i])
        min_arm_angle = arm_angles[worst_arm_idx]
        lm_arm = landmarks_list[worst_arm_idx]

        if min_arm_angle < config.LEAD_ARM_MIN_ANGLE:
            criteria.append(FaultCriterion(
                code="bent_lead_arm",
                title="Gập khuỷu tay dẫn hướng (Chicken Wing)",
                status="error",
                measured_text=f"{min_arm_angle:.1f}°",
                standard_text=f"≥ {config.LEAD_ARM_MIN_ANGLE:.0f}°",
                frame_idx=worst_arm_idx,
                description=(
                    f"Ở đỉnh Swing / vung xuống, khuỷu tay {'trái' if self.handedness == 'right' else 'phải'} "
                    f"bị co gập chỉ còn {min_arm_angle:.1f}° (chuẩn PGA phải giữ duỗi thẳng ≥ {config.LEAD_ARM_MIN_ANGLE:.0f}°). "
                    "Lỗi này làm thu hẹp cung vung gậy, giảm lực đánh và dễ gây đánh Slice."
                ),
                correction_tip="Cảm giác đẩy nhẹ cổ tay ra xa ngực khi đưa gậy lên đỉnh, giữ thẳng cánh tay dẫn hướng như một bán kính quay cố định.",
                annotation_type="elbow_angle",
                annotation_data={
                    "shoulder": lm_arm[sk["lead_shoulder"]][:2],
                    "elbow": lm_arm[sk["lead_elbow"]][:2],
                    "wrist": lm_arm[sk["lead_wrist"]][:2],
                    "angle": min_arm_angle,
                    "min_angle": config.LEAD_ARM_MIN_ANGLE,
                }
            ))
        else:
            criteria.append(FaultCriterion(
                code="bent_lead_arm",
                title="Độ thẳng cánh tay dẫn hướng",
                status="passed",
                measured_text=f"{min_arm_angle:.1f}°",
                standard_text=f"≥ {config.LEAD_ARM_MIN_ANGLE:.0f}°",
                frame_idx=top_idx,
                description=f"Tuyệt vời! Tay dẫn hướng được giữ thẳng rất tốt ({min_arm_angle:.1f}°), tạo cung swing rộng và ổn định.",
                correction_tip="Tiếp tục duy trì độ duỗi tay tự nhiên này.",
                annotation_type="elbow_angle",
                annotation_data={
                    "shoulder": landmarks_list[top_idx][sk["lead_shoulder"]][:2],
                    "elbow": landmarks_list[top_idx][sk["lead_elbow"]][:2],
                    "wrist": landmarks_list[top_idx][sk["lead_wrist"]][:2],
                    "angle": arm_angles[top_idx],
                    "min_angle": config.LEAD_ARM_MIN_ANGLE,
                }
            ))

        # --- TIÊU CHÍ 2: NHẤP NHÔ / NGẨNG ĐẦU SỚM (HEAD STABILITY) ---
        addr_lm = landmarks_list[address_idx]
        if self._is_visible(addr_lm, ["nose", "left_shoulder", "right_shoulder"]):
            addr_nose_y = addr_lm["nose"][1]
            # Ước lượng chiều cao cơ thể (nếu thấy mắt cá dùng chiều cao thật, không thì ước lượng theo bề ngang vai * 4.5)
            if self._is_visible(addr_lm, [sk["lead_ankle"], sk["trail_ankle"]]):
                ankle_mid_y = (addr_lm[sk["lead_ankle"]][1] + addr_lm[sk["trail_ankle"]][1]) * 0.5
                body_height = max(abs(ankle_mid_y - addr_nose_y), 120.0)
            else:
                shoulder_w = distance_2d(addr_lm["left_shoulder"], addr_lm["right_shoulder"])
                body_height = max(shoulder_w * 4.5, 180.0)

            head_frames = [i for i in eval_range if self._is_visible(landmarks_list[i], ["nose"])]
            if head_frames:
                worst_head_idx = max(head_frames, key=lambda i: abs(landmarks_list[i]["nose"][1] - addr_nose_y))
                dy = landmarks_list[worst_head_idx]["nose"][1] - addr_nose_y
                drift_ratio = abs(dy) / body_height
                direction_txt = "ngẩng đầu lên sớm" if dy < 0 else "nhấp nhô / gập đầu xuống"
                nose_pt = landmarks_list[worst_head_idx]["nose"][:2]

                if drift_ratio > config.HEAD_DRIFT_MAX_RATIO:
                    criteria.append(FaultCriterion(
                        code="head_movement",
                        title=f"Lỗi giữ trục đầu ({direction_txt})",
                        status="error",
                        measured_text=f"Lệch {drift_ratio*100:.1f}%",
                        standard_text=f"≤ {config.HEAD_DRIFT_MAX_RATIO*100:.0f}%",
                        frame_idx=worst_head_idx,
                        description=(
                            f"Đầu của bạn bị {direction_txt} khá nhiều ({drift_ratio*100:.1f}% chiều cao cơ thể) "
                            "so với vị trí chuẩn bị ban đầu. Điều này làm thay đổi đáy cung swing, rất dễ đánh cạp đất (fat) hoặc đánh mỏng đầu bóng (thin)."
                        ),
                        correction_tip="Cố định tâm mắt nhìn vào 1 điểm trên bóng đến sau khi mặt gậy chạm bóng mới xoay cổ nhìn theo bóng bay.",
                        annotation_type="head_drift",
                        annotation_data={
                            "nose": nose_pt,
                            "addr_nose_y": addr_nose_y,
                            "drift_pct": drift_ratio * 100.0,
                            "direction": direction_txt,
                        }
                    ))
                else:
                    criteria.append(FaultCriterion(
                        code="head_movement",
                        title="Độ ổn định đầu (Head Stability)",
                        status="passed",
                        measured_text=f"Lệch {drift_ratio*100:.1f}%",
                        standard_text=f"≤ {config.HEAD_DRIFT_MAX_RATIO*100:.0f}%",
                        frame_idx=top_idx,
                        description=f"Đầu giữ rất vững vàng (biên độ lệch chỉ {drift_ratio*100:.1f}%), giúp điểm chạm bóng cực kỳ ổn định.",
                        correction_tip="Rất tốt, tiếp tục giữ trục đầu ổn định.",
                        annotation_type="head_drift",
                        annotation_data={
                            "nose": landmarks_list[top_idx]["nose"][:2],
                            "addr_nose_y": addr_nose_y,
                            "drift_pct": drift_ratio * 100.0,
                            "direction": "ổn định",
                        }
                    ))
        else:
            criteria.append(FaultCriterion(
                code="head_movement",
                title="Độ ổn định đầu (Head Stability)",
                status="not_visible",
                measured_text="N/A",
                standard_text=f"≤ {config.HEAD_DRIFT_MAX_RATIO*100:.0f}%",
                frame_idx=top_idx,
                description="Không nhìn rõ vùng đầu trong khung hình chuẩn bị.",
                correction_tip="Điều chỉnh góc webcam thấy trọn đầu và thân người.",
                annotation_type="none",
            ))

        # --- TIÊU CHÍ 3: NGƯỢC TRỤC CỘT SỐNG (REVERSE SPINE ANGLE) ---
        req_spine = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        if self._is_visible(addr_lm, req_spine) and self._is_visible(landmarks_list[top_idx], req_spine):
            # Hướng mục tiêu là phía bên hông dẫn hướng (lead_hip - trail_hip)
            target_dx = addr_lm[sk["lead_hip"]][0] - addr_lm[sk["trail_hip"]][0]
            target_dir_sign = 1.0 if target_dx >= 0 else -1.0

            lm_top = landmarks_list[top_idx]
            mid_hip_top = midpoint_2d(lm_top["left_hip"], lm_top["right_hip"])
            mid_sh_top = midpoint_2d(lm_top["left_shoulder"], lm_top["right_shoulder"])
            away_tilt = calculate_spine_away_tilt_deg(mid_hip_top, mid_sh_top, target_dir_sign)

            if away_tilt < -config.REVERSE_SPINE_MAX_DEG:
                criteria.append(FaultCriterion(
                    code="reverse_spine",
                    title="Lỗi ngược trục cột sống (Reverse Spine)",
                    status="error",
                    measured_text=f"Nghiêng ngược {abs(away_tilt):.1f}°",
                    standard_text="Nghiêng ra sau ≥ 0°",
                    frame_idx=top_idx,
                    description=(
                        f"Tại đỉnh Swing, phần thân trên của bạn bị đổ ngược {abs(away_tilt):.1f}° về phía mục tiêu "
                        "(thay vì xoay và hơi ngả nhẹ ra chân sau). Đây là nguyên nhân số 1 gây đau lưng dưới và đánh cắt bóng (Over-the-top)."
                    ),
                    correction_tip="Khi xoay vai đưa gậy lên, hãy cảm giác trọng lượng dồn nhẹ vào mặt trong đùi chân sau và giữ ngực xoay chứ không đổ chúi về phía mục tiêu.",
                    annotation_type="spine_tilt",
                    annotation_data={
                        "mid_hip": mid_hip_top,
                        "mid_shoulder": mid_sh_top,
                        "away_tilt": away_tilt,
                    }
                ))
            else:
                criteria.append(FaultCriterion(
                    code="reverse_spine",
                    title="Trục cột sống ở Đỉnh Swing",
                    status="passed",
                    measured_text=f"{away_tilt:+.1f}°",
                    standard_text="≥ -4° (Không đổ ngược)",
                    frame_idx=top_idx,
                    description=f"Trục cột sống xoay đúng kỹ thuật ở đỉnh swing ({away_tilt:+.1f}°), tạo đòn bẩy lực tự nhiên mà không hại lưng.",
                    correction_tip="Tiếp tục duy trì trục cột sống cân bằng này.",
                    annotation_type="spine_tilt",
                    annotation_data={
                        "mid_hip": mid_hip_top,
                        "mid_shoulder": mid_sh_top,
                        "away_tilt": away_tilt,
                    }
                ))
        else:
            criteria.append(FaultCriterion(
                code="reverse_spine",
                title="Trục cột sống ở Đỉnh Swing",
                status="not_visible",
                measured_text="Chưa thấy hông",
                standard_text="Thấy cả vai & hông",
                frame_idx=top_idx,
                description="Camera chưa thấy đầy đủ cả phần vai và phần hông để đo trục cột sống.",
                correction_tip="Lùi ra xa camera khoảng 2m - 2.5m để lấy toàn thân người chơi.",
                annotation_type="none",
            ))

        # --- TIÊU CHÍ 4: TRƯỢT / LẮC HÔNG NGANG (PELVIC SWAY) ---
        req_sway = [sk["trail_hip"], sk["trail_ankle"], sk["lead_ankle"]]
        backswing_frames = [i for i in valid_indices if address_idx <= i <= top_idx and self._is_visible(landmarks_list[i], req_sway)]
        if self._is_visible(addr_lm, req_sway) and backswing_frames:
            stance_w = max(abs(addr_lm[sk["lead_ankle"]][0] - addr_lm[sk["trail_ankle"]][0]), 40.0)
            # Chiều trượt ra xa mục tiêu (hướng về chân sau)
            away_sign = 1.0 if (addr_lm[sk["trail_ankle"]][0] - addr_lm[sk["lead_ankle"]][0]) >= 0 else -1.0
            trail_foot_limit_x = addr_lm[sk["trail_ankle"]][0]

            def sway_amount(idx):
                hip_x = landmarks_list[idx][sk["trail_hip"]][0]
                return (hip_x - trail_foot_limit_x) * away_sign

            worst_sway_idx = max(backswing_frames, key=sway_amount)
            max_sway_px = sway_amount(worst_sway_idx)
            sway_ratio = max(0.0, max_sway_px / stance_w)
            lm_sway = landmarks_list[worst_sway_idx]

            if sway_ratio > config.HIP_SWAY_MAX_RATIO:
                criteria.append(FaultCriterion(
                    code="pelvic_sway",
                    title="Lỗi lắc / trượt hông ngang (Pelvic Sway)",
                    status="error",
                    measured_text=f"Trượt {sway_ratio*100:.1f}%",
                    standard_text=f"≤ {config.HIP_SWAY_MAX_RATIO*100:.0f}%",
                    frame_idx=worst_sway_idx,
                    description=(
                        f"Trong pha đưa gậy lên, hông sau bị trượt ngang vượt ra ngoài trục chân sau tới {sway_ratio*100:.1f}% bề rộng thế đứng "
                        "thay vì xoay hông tại chỗ. Điều này làm mất lực nén và khó đưa mặt gậy về vuông góc với bóng."
                    ),
                    correction_tip="Tưởng tượng hai bên hông nằm trong một ống trụ hẹp; khi backswing chỉ xoay túi quần sau ra phía sau chứ không đẩy trượt ngang.",
                    annotation_type="hip_sway",
                    annotation_data={
                        "trail_hip": lm_sway[sk["trail_hip"]][:2],
                        "trail_ankle": addr_lm[sk["trail_ankle"]][:2],
                        "limit_x": trail_foot_limit_x,
                        "sway_pct": sway_ratio * 100.0,
                    }
                ))
            else:
                criteria.append(FaultCriterion(
                    code="pelvic_sway",
                    title="Độ vững xoay hông (Không trượt ngang)",
                    status="passed",
                    measured_text=f"{sway_ratio*100:.1f}%",
                    standard_text=f"≤ {config.HIP_SWAY_MAX_RATIO*100:.0f}%",
                    frame_idx=top_idx,
                    description="Hông xoay gọn trong trục chân rất chắc chắn, không bị trượt ngang (Sway).",
                    correction_tip="Khả năng giữ trục hông xoay rất tốt!",
                    annotation_type="hip_sway",
                    annotation_data={
                        "trail_hip": landmarks_list[top_idx][sk["trail_hip"]][:2],
                        "trail_ankle": addr_lm[sk["trail_ankle"]][:2],
                        "limit_x": trail_foot_limit_x,
                        "sway_pct": sway_ratio * 100.0,
                    }
                ))
        else:
            criteria.append(FaultCriterion(
                code="pelvic_sway",
                title="Kiểm soát trượt hông (Pelvic Sway)",
                status="not_visible",
                measured_text="Chưa thấy chân",
                standard_text="Thấy hông & 2 chân",
                frame_idx=top_idx,
                description="Chưa quan sát được hông và hai bàn chân trong khung hình.",
                correction_tip="Đặt camera lấy toàn thân từ đầu tới chân để kiểm tra lỗi trượt hông.",
                annotation_type="none",
            ))

        # --- TIÊU CHÍ 5: ỔN ĐỊNH ĐẦU GỐI CHÂN TRỤ (KNEE STABILITY) ---
        req_knee = [sk["lead_hip"], sk["lead_knee"], sk["lead_ankle"]]
        knee_frames = [i for i in eval_range if self._is_visible(landmarks_list[i], req_knee)]
        if knee_frames:
            knee_angles = {
                i: calculate_angle_2d(landmarks_list[i][sk["lead_hip"]], landmarks_list[i][sk["lead_knee"]], landmarks_list[i][sk["lead_ankle"]])
                for i in knee_frames
            }
            min_k_idx = min(knee_frames, key=lambda i: knee_angles[i])
            max_k_idx = max(knee_frames, key=lambda i: knee_angles[i])
            min_k = knee_angles[min_k_idx]
            max_k = knee_angles[max_k_idx]

            if min_k < config.KNEE_MIN_ANGLE:
                lm_k = landmarks_list[min_k_idx]
                criteria.append(FaultCriterion(
                    code="knee_instability",
                    title="Lỗi chùng / sụp đầu gối chân trước",
                    status="error",
                    measured_text=f"{min_k:.1f}°",
                    standard_text=f"{config.KNEE_MIN_ANGLE:.0f}° - {config.KNEE_MAX_ANGLE:.0f}°",
                    frame_idx=min_k_idx,
                    description=f"Đầu gối chân dẫn hướng bị gập sụp xuống quá sâu ({min_k:.1f}° < {config.KNEE_MIN_ANGLE:.0f}°), làm hạ thấp trọng tâm đột ngột và mất ổn định thăng bằng.",
                    correction_tip="Giữ độ khuỵu nhẹ tự nhiên của đầu gối trong suốt quá trình xoay vai.",
                    annotation_type="knee_angle",
                    annotation_data={
                        "hip": lm_k[sk["lead_hip"]][:2],
                        "knee": lm_k[sk["lead_knee"]][:2],
                        "ankle": lm_k[sk["lead_ankle"]][:2],
                        "angle": min_k,
                    }
                ))
            elif max_k > config.KNEE_MAX_ANGLE and max_k_idx >= top_idx:
                lm_k = landmarks_list[max_k_idx]
                criteria.append(FaultCriterion(
                    code="knee_instability",
                    title="Lỗi bật thẳng chân quá sớm (Early Extension)",
                    status="error",
                    measured_text=f"{max_k:.1f}°",
                    standard_text=f"≤ {config.KNEE_MAX_ANGLE:.0f}°",
                    frame_idx=max_k_idx,
                    description=f"Chân trước bị duỗi thẳng cứng sớm ({max_k:.1f}°) làm người chơi bị nhổm dậy khỏi tư thế đánh bóng.",
                    correction_tip="Duy trì độ khuỵu nhẹ của chân cho tới khi gậy vung qua điểm chạm bóng.",
                    annotation_type="knee_angle",
                    annotation_data={
                        "hip": lm_k[sk["lead_hip"]][:2],
                        "knee": lm_k[sk["lead_knee"]][:2],
                        "ankle": lm_k[sk["lead_ankle"]][:2],
                        "angle": max_k,
                    }
                ))
            else:
                lm_k = landmarks_list[impact_idx] if self._is_visible(landmarks_list[impact_idx], req_knee) else landmarks_list[min_k_idx]
                k_val = knee_angles.get(impact_idx, min_k)
                criteria.append(FaultCriterion(
                    code="knee_instability",
                    title="Độ ổn định chân trụ & đầu gối",
                    status="passed",
                    measured_text=f"{k_val:.1f}°",
                    standard_text=f"{config.KNEE_MIN_ANGLE:.0f}° - {config.KNEE_MAX_ANGLE:.0f}°",
                    frame_idx=impact_idx if impact_idx in knee_angles else min_k_idx,
                    description="Đầu gối giữ trụ rất chắc chắn, không bị chùng gối hay bật thẳng sớm.",
                    correction_tip="Tiếp tục giữ phần thân dưới vững vàng như hiện tại.",
                    annotation_type="knee_angle",
                    annotation_data={
                        "hip": lm_k[sk["lead_hip"]][:2],
                        "knee": lm_k[sk["lead_knee"]][:2],
                        "ankle": lm_k[sk["lead_ankle"]][:2],
                        "angle": k_val,
                    }
                ))
        else:
            criteria.append(FaultCriterion(
                code="knee_instability",
                title="Độ ổn định chân trụ & đầu gối",
                status="not_visible",
                measured_text="Chưa thấy đầu gối",
                standard_text="Thấy hông, gối & mắt cá",
                frame_idx=top_idx,
                description="Chưa thấy đầy đủ đầu gối và mắt cá chân trong khung hình.",
                correction_tip="Đặt camera lấy toàn thân để kiểm tra chi tiết chân trụ.",
                annotation_type="none",
            ))

        # Sắp xếp lỗi (error) lên đầu, rồi tới đạt (passed), rồi tới chưa thấy (not_visible)
        order_map = {"error": 0, "passed": 1, "not_visible": 2}
        criteria.sort(key=lambda c: order_map.get(c.status, 3))

        evaluated = [c for c in criteria if c.status in ("error", "passed")]
        passed = [c for c in criteria if c.status == "passed"]

        return SwingAnalysisResult(
            person_detected=True,
            total_frames=total_frames,
            fps=actual_fps,
            address_idx=address_idx,
            top_idx=top_idx,
            impact_idx=impact_idx,
            score_passed=len(passed),
            score_evaluated=len(evaluated),
            criteria=criteria,
            landmarks_per_frame=landmarks_list,
            arm_angles_per_frame=arm_angles,
        )
