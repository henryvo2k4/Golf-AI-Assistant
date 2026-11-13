"""
MODULE BỘ NÃO PHÂN TÍCH CÚ SWING CHUẨN PGA / TPI (2D + 3D WORLD METRIC)
- Làm mượt tín hiệu 5 khung hình khử rung khớp xương.
- Tự động tách pha thông minh (Setup -> Top -> Impact -> Finish) dựa trên động học 2 tay cầm gậy.
- Đo góc 3D thực (pose_world_landmarks) bất biến theo góc nghiêng camera.
- Tự động nhận diện góc quay Chính diện (Face-On) hay Nhìn ngang (Down-the-Line).
"""

from dataclasses import dataclass, field
import numpy as np

import config
from src.core.pose_detector import GolfPoseDetector
from src.utils.geometry import (
    calculate_angle_3d,
    calculate_shoulder_turn_xz_deg,
    calculate_spine_away_tilt_deg,
    calculate_spine_forward_flexion_deg,
    distance_2d,
    midpoint_2d,
    smooth_landmarks_sequence,
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
    camera_view: str         # "face_on" | "down_the_line"
    total_frames: int
    fps: float
    address_idx: int
    top_idx: int
    impact_idx: int
    finish_idx: int
    score_passed: int
    score_evaluated: int
    criteria: list[FaultCriterion]
    landmarks_per_frame: list
    arm_angles_per_frame: list[float]


class BatchSwingAnalyzer:
    """Phân tích toàn bộ video cú đánh 10s bằng dữ liệu 2D + 3D World Metric."""

    def __init__(self, handedness: str = None):
        self.handedness = handedness or config.GOLFER_HANDEDNESS
        self.detector = GolfPoseDetector()

    def _side_keys(self):
        if self.handedness == "right":
            return {
                "lead_shoulder": "left_shoulder",
                "lead_elbow": "left_elbow",
                "lead_wrist": "left_wrist",
                "lead_hip": "left_hip",
                "lead_knee": "left_knee",
                "lead_ankle": "left_ankle",
                "trail_shoulder": "right_shoulder",
                "trail_wrist": "right_wrist",
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
            "trail_wrist": "left_wrist",
            "trail_hip": "left_hip",
            "trail_knee": "left_knee",
            "trail_ankle": "left_ankle",
        }

    @staticmethod
    def _is_visible(lm_dict: dict | None, keys: list[str], threshold: float = 0.42) -> bool:
        if lm_dict is None:
            return False
        return all((k in lm_dict and lm_dict[k][3] >= threshold) for k in keys)

    def _segment_swing_phases(self, lm_2d: list, sk: dict, fps: float) -> tuple[int, int, int, int, list[int]]:
        """
        Tự động định vị chính xác 4 pha của cú Swing thực sự trong chuỗi 10 giây:
        - Address (Chuẩn bị yên tĩnh trước khi vung)
        - Top of Swing (Đỉnh vung gậy cao nhất trước cú Downswing nhanh)
        - Impact (Chạm bóng - Cổ tay vung xuống đáy cung sau đỉnh)
        - Finish (Kết thúc vung gậy theo đà)
        """
        n = len(lm_2d)
        valid_indices = [
            i for i, lm in enumerate(lm_2d)
            if self._is_visible(lm, [sk["lead_shoulder"], sk["lead_elbow"], sk["lead_wrist"]])
        ]
        if len(valid_indices) < 6:
            return 0, 0, 0, 0, valid_indices

        # Tính tọa độ điểm cầm gậy (Grip point) và trạng thái 2 tay chập nhau
        wrist_y = np.zeros(n, dtype=np.float64)
        grip_together = np.zeros(n, dtype=bool)
        shoulder_widths = []

        for i in valid_indices:
            lm = lm_2d[i]
            lw = lm[sk["lead_wrist"]][:2]
            if self._is_visible(lm, [sk["trail_wrist"], sk["lead_shoulder"], sk["trail_shoulder"]], 0.35):
                rw = lm[sk["trail_wrist"]][:2]
                sw = max(distance_2d(lm[sk["lead_shoulder"]], lm[sk["trail_shoulder"]]), 25.0)
                shoulder_widths.append(sw)
                grip_dist = distance_2d(lw, rw)
                grip_together[i] = grip_dist <= 1.35 * sw
                wrist_y[i] = (lw[1] + rw[1]) * 0.5
            else:
                wrist_y[i] = lw[1]

        # Chấm điểm từng điểm ứng viên làm ĐỈNH SWING (Top of Swing)
        # Một đỉnh swing thật phải: (1) cổ tay ở vị trí cao, (2) 2 tay cầm gậy (hoặc gần nhau),
        # (3) sau đó 0.15s - 1.3s có pha vung gậy xuống sâu (Downswing drop)
        min_drop_window = max(2, int(0.12 * fps))
        max_drop_window = max(5, int(1.35 * fps))
        best_top_idx = valid_indices[0]
        best_score = -1e9

        y_min_global = min(wrist_y[i] for i in valid_indices)
        y_max_global = max(wrist_y[i] for i in valid_indices)
        y_span = max(y_max_global - y_min_global, 30.0)

        for i in valid_indices:
            # Chiều cao tương đối (1.0 = cao nhất, 0.0 = thấp nhất)
            rel_height = (y_max_global - wrist_y[i]) / y_span

            # Độ rơi xuống (Downswing amplitude) ngay sau i
            post_slice = [j for j in valid_indices if i + min_drop_window <= j <= i + max_drop_window]
            if post_slice:
                max_post_y = max(wrist_y[j] for j in post_slice)
                down_drop = max(0.0, (max_post_y - wrist_y[i]) / y_span)
            else:
                down_drop = 0.0

            # Độ nâng lên trước i (Backswing lift)
            pre_slice = [j for j in valid_indices if max(0, i - int(2.0 * fps)) <= j < i]
            if pre_slice:
                max_pre_y = max(wrist_y[j] for j in pre_slice)
                up_lift = max(0.0, (max_pre_y - wrist_y[i]) / y_span)
            else:
                up_lift = 0.0

            grip_bonus = 0.45 if grip_together[i] else 0.0
            score = (rel_height * 1.2) + (down_drop * 1.5) + (up_lift * 1.0) + grip_bonus

            if score > best_score:
                best_score = score
                best_top_idx = i

        top_idx = best_top_idx

        # Tìm IMPACT: Đáy thấp nhất của cổ tay ngay sau Top of Swing
        post_window = [j for j in valid_indices if top_idx < j <= top_idx + max_drop_window]
        if post_window:
            impact_idx = max(post_window, key=lambda j: wrist_y[j])
        else:
            impact_idx = min(n - 1, top_idx + 2)

        # Tìm FINISH: Điểm cổ tay vung lên cao nhất theo đà sau Impact
        finish_window = [j for j in valid_indices if impact_idx < j <= impact_idx + int(1.6 * fps)]
        if finish_window:
            finish_idx = min(finish_window, key=lambda j: wrist_y[j])
        else:
            finish_idx = min(n - 1, impact_idx + 1)

        # Tìm ADDRESS (Setup): Vùng đứng yên tĩnh ngay trước khi bắt đầu nâng tay lên đỉnh
        lookback = max(4, int(3.0 * fps))
        pre_top = [j for j in valid_indices if max(0, top_idx - lookback) <= j < top_idx]
        if len(pre_top) >= 4:
            pre_ys = [wrist_y[j] for j in pre_top]
            low_thresh = float(np.percentile(pre_ys, 70))
            quiet_candidates = [j for j in pre_top if wrist_y[j] >= low_thresh - 10.0]
            if quiet_candidates:
                # Chọn điểm có tốc độ di chuyển cổ tay nhỏ nhất (đứng yên chuẩn bị đánh)
                def motion_at(idx):
                    prev_i = max(0, idx - 1)
                    next_i = min(n - 1, idx + 1)
                    return abs(wrist_y[next_i] - wrist_y[prev_i])

                address_idx = min(quiet_candidates, key=motion_at)
            else:
                address_idx = pre_top[0]
        else:
            address_idx = valid_indices[0]

        return address_idx, top_idx, impact_idx, finish_idx, valid_indices

    def _detect_camera_view(self, addr_lm_2d: dict) -> str:
        """
        Tự động nhận diện góc máy:
        - face_on: Camera đứng chính diện nhìn thấy độ rộng 2 vai.
        - down_the_line: Camera đặt phía sau nhìn dọc đường đánh (2 vai gần chồng lên nhau theo chiều X).
        """
        req = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        if not self._is_visible(addr_lm_2d, req, 0.35):
            return "face_on"

        sh_w = abs(addr_lm_2d["left_shoulder"][0] - addr_lm_2d["right_shoulder"][0])
        mid_sh = midpoint_2d(addr_lm_2d["left_shoulder"], addr_lm_2d["right_shoulder"])
        mid_hip = midpoint_2d(addr_lm_2d["left_hip"], addr_lm_2d["right_hip"])
        torso_h = max(distance_2d(mid_sh, mid_hip), 40.0)

        return "face_on" if (sh_w / torso_h) >= 0.32 else "down_the_line"

    def analyze_clip(self, frames_bgr: list, timestamps_ms: list, progress_callback=None) -> SwingAnalysisResult:
        """
        Chạy AI phân tích 2D + 3D toàn bộ clip 10s và xuất báo cáo chuẩn PGA / TPI.
        """
        total_frames = len(frames_bgr)
        if total_frames < 6:
            return SwingAnalysisResult(
                person_detected=False, camera_view="face_on", total_frames=total_frames, fps=30.0,
                address_idx=0, top_idx=0, impact_idx=0, finish_idx=0,
                score_passed=0, score_evaluated=0, criteria=[],
                landmarks_per_frame=[], arm_angles_per_frame=[],
            )

        duration_sec = max((timestamps_ms[-1] - timestamps_ms[0]) / 1000.0, 0.1)
        actual_fps = total_frames / duration_sec

        # 1. Trích xuất tọa độ 2D và 3D từ MediaPipe PoseLandmarker
        raw_2d, raw_3d = self.detector.detect_video_sequence(
            frames_bgr, timestamps_ms, progress_callback=progress_callback
        )

        # 2. Làm mượt chuỗi khung xương để chống rung giật khớp
        lm_2d = smooth_landmarks_sequence(raw_2d)
        lm_3d = smooth_landmarks_sequence(raw_3d)

        sk = self._side_keys()
        address_idx, top_idx, impact_idx, finish_idx, valid_indices = self._segment_swing_phases(
            lm_2d, sk, actual_fps
        )

        # Tính góc khuỷu tay 3D cho từng frame (dùng cho Video Slow-motion & Đánh giá)
        arm_angles_3d = []
        for idx in range(total_frames):
            d3 = lm_3d[idx]
            if self._is_visible(d3, [sk["lead_shoulder"], sk["lead_elbow"], sk["lead_wrist"]]):
                ang = calculate_angle_3d(
                    d3[sk["lead_shoulder"]], d3[sk["lead_elbow"]], d3[sk["lead_wrist"]]
                )
                arm_angles_3d.append(ang)
            else:
                arm_angles_3d.append(0.0)

        if len(valid_indices) < 8:
            return SwingAnalysisResult(
                person_detected=False, camera_view="face_on", total_frames=total_frames, fps=actual_fps,
                address_idx=0, top_idx=0, impact_idx=0, finish_idx=0,
                score_passed=0, score_evaluated=0, criteria=[],
                landmarks_per_frame=lm_2d, arm_angles_per_frame=arm_angles_3d,
            )

        addr_2d = lm_2d[address_idx]
        addr_3d = lm_3d[address_idx]
        camera_view = self._detect_camera_view(addr_2d)

        criteria: list[FaultCriterion] = []
        eval_range = [i for i in valid_indices if address_idx <= i <= max(impact_idx, top_idx)]
        if not eval_range:
            eval_range = valid_indices

        swing_peak_to_impact = [
            i for i in valid_indices if max(address_idx, top_idx - 2) <= i <= max(impact_idx, top_idx)
        ] or [top_idx]

        # ==========================================================
        # TIÊU CHÍ 1 (3D): ĐỘ THẲNG CÁNH TAY DẪN HƯỚNG (LEAD ELBOW)
        # ==========================================================
        worst_arm_idx = min(swing_peak_to_impact, key=lambda i: arm_angles_3d[i] if arm_angles_3d[i] > 0 else 999)
        min_arm_angle = arm_angles_3d[worst_arm_idx]
        lm_arm_2d = lm_2d[worst_arm_idx]

        if min_arm_angle < config.LEAD_ARM_MIN_ANGLE_3D:
            criteria.append(FaultCriterion(
                code="bent_lead_arm",
                title="Gập khuỷu tay dẫn hướng (Chicken Wing)",
                status="error",
                measured_text=f"{min_arm_angle:.1f}° (3D)",
                standard_text=f"≥ {config.LEAD_ARM_MIN_ANGLE_3D:.0f}°",
                frame_idx=worst_arm_idx,
                description=(
                    f"Đo theo không gian 3D tại Đỉnh Swing / Downswing, khuỷu tay "
                    f"{'trái' if self.handedness == 'right' else 'phải'} bị gập xuống còn {min_arm_angle:.1f}° "
                    f"(chuẩn PGA ≥ {config.LEAD_ARM_MIN_ANGLE_3D:.0f}°). Gập tay thu hẹp bán kính vung gậy, gây đánh Slice và hụt lực."
                ),
                correction_tip="Đẩy cổ tay ra xa lồng ngực khi đưa gậy lên đỉnh, cảm giác vươn rộng vòng cung swing.",
                annotation_type="elbow_angle",
                annotation_data={
                    "shoulder": lm_arm_2d[sk["lead_shoulder"]][:2],
                    "elbow": lm_arm_2d[sk["lead_elbow"]][:2],
                    "wrist": lm_arm_2d[sk["lead_wrist"]][:2],
                    "angle": min_arm_angle,
                },
            ))
        else:
            top_ang = arm_angles_3d[top_idx] if arm_angles_3d[top_idx] > 0 else min_arm_angle
            criteria.append(FaultCriterion(
                code="bent_lead_arm",
                title="Độ thẳng cánh tay dẫn hướng",
                status="passed",
                measured_text=f"{min_arm_angle:.1f}° (3D)",
                standard_text=f"≥ {config.LEAD_ARM_MIN_ANGLE_3D:.0f}°",
                frame_idx=top_idx,
                description=f"Tay dẫn hướng giữ duỗi thẳng chuẩn xác ({min_arm_angle:.1f}° đo 3D) xuyên suốt pha đỉnh swing tới chạm bóng.",
                correction_tip="Duy trì độ duỗi tự nhiên này mà không gồng cứng cơ vai.",
                annotation_type="elbow_angle",
                annotation_data={
                    "shoulder": lm_2d[top_idx][sk["lead_shoulder"]][:2],
                    "elbow": lm_2d[top_idx][sk["lead_elbow"]][:2],
                    "wrist": lm_2d[top_idx][sk["lead_wrist"]][:2],
                    "angle": top_ang,
                },
            ))

        # ==========================================================
        # TIÊU CHÍ 2 (2D): ỔN ĐỊNH TRỤC ĐẦU (HEAD LIFT / DIP / SWAY)
        # ==========================================================
        if self._is_visible(addr_2d, ["nose", "left_shoulder", "right_shoulder"]):
            addr_nose = addr_2d["nose"][:2]
            if self._is_visible(addr_2d, [sk["lead_ankle"], sk["trail_ankle"]]):
                ankle_mid_y = (addr_2d[sk["lead_ankle"]][1] + addr_2d[sk["trail_ankle"]][1]) * 0.5
                body_height = max(abs(ankle_mid_y - addr_nose[1]), 120.0)
            else:
                sh_w = distance_2d(addr_2d["left_shoulder"], addr_2d["right_shoulder"])
                body_height = max(sh_w * 4.5, 180.0)

            head_frames = [i for i in eval_range if self._is_visible(lm_2d[i], ["nose"])]
            if head_frames:
                worst_v_idx = max(head_frames, key=lambda i: abs(lm_2d[i]["nose"][1] - addr_nose[1]))
                dy = lm_2d[worst_v_idx]["nose"][1] - addr_nose[1]
                v_ratio = abs(dy) / body_height

                worst_h_idx = max(head_frames, key=lambda i: abs(lm_2d[i]["nose"][0] - addr_nose[0]))
                dx = lm_2d[worst_h_idx]["nose"][0] - addr_nose[0]
                h_ratio = abs(dx) / body_height

                is_v_err = v_ratio > config.HEAD_VERTICAL_DRIFT_MAX_RATIO
                is_h_err = h_ratio > config.HEAD_LATERAL_DRIFT_MAX_RATIO

                if is_v_err or is_h_err:
                    use_idx = worst_v_idx if (v_ratio / config.HEAD_VERTICAL_DRIFT_MAX_RATIO) >= (h_ratio / config.HEAD_LATERAL_DRIFT_MAX_RATIO) else worst_h_idx
                    nose_pt = lm_2d[use_idx]["nose"][:2]
                    reason = ("ngẩng đầu lên sớm" if dy < 0 else "tụt/gập đầu xuống") if is_v_err else "lắc đầu sang ngang"
                    max_pct = max(v_ratio, h_ratio) * 100.0
                    criteria.append(FaultCriterion(
                        code="head_movement",
                        title=f"Lỗi giữ trục đầu ({reason})",
                        status="error",
                        measured_text=f"Lệch {max_pct:.1f}%",
                        standard_text=f"≤ {config.HEAD_VERTICAL_DRIFT_MAX_RATIO*100:.0f}%",
                        frame_idx=use_idx,
                        description=(
                            f"Đầu bị {reason} với biên độ {max_pct:.1f}% chiều cao cơ thể so với lúc Setup. "
                            "Đầu nhấp nhô làm thay đổi điểm thấp nhất của đầu gậy, dễ đánh cạp đất (Fat) hoặc đánh mỏng (Thin)."
                        ),
                        correction_tip="Giữ cố định tầm mắt nhìn thẳng vào tâm bóng tới khi đầu gậy lướt qua mặt cỏ mới xoay đầu nhìn bóng.",
                        annotation_type="head_drift",
                        annotation_data={
                            "nose": nose_pt,
                            "addr_nose": addr_nose,
                            "addr_nose_y": addr_nose[1],
                        },
                    ))
                else:
                    max_pct = max(v_ratio, h_ratio) * 100.0
                    criteria.append(FaultCriterion(
                        code="head_movement",
                        title="Độ ổn định trục đầu",
                        status="passed",
                        measured_text=f"Lệch {max_pct:.1f}%",
                        standard_text=f"≤ {config.HEAD_VERTICAL_DRIFT_MAX_RATIO*100:.0f}%",
                        frame_idx=top_idx,
                        description=f"Trục đầu giữ cực kỳ ổn định (dao động chỉ {max_pct:.1f}%), bảo toàn tâm quay của cú swing.",
                        correction_tip="Giữ tâm nhìn ổn định rất tốt.",
                        annotation_type="head_drift",
                        annotation_data={
                            "nose": lm_2d[top_idx]["nose"][:2],
                            "addr_nose": addr_nose,
                            "addr_nose_y": addr_nose[1],
                        },
                    ))
        else:
            criteria.append(FaultCriterion(
                code="head_movement",
                title="Độ ổn định trục đầu",
                status="not_visible",
                measured_text="Chưa rõ đầu",
                standard_text="Thấy rõ khuôn mặt",
                frame_idx=top_idx,
                description="Không nhận diện rõ vùng đầu ở tư thế chuẩn bị.",
                correction_tip="Điều chỉnh góc quay thấy trọn đầu đến chân.",
                annotation_type="none",
            ))

        # ==========================================================
        # TIÊU CHÍ 3 (3D X-Z): XOAY VAI Ở ĐỈNH SWING (SHOULDER TURN)
        # ==========================================================
        req_sh = ["left_shoulder", "right_shoulder"]
        if self._is_visible(addr_3d, req_sh) and self._is_visible(lm_3d[top_idx], req_sh):
            sh_turn = calculate_shoulder_turn_xz_deg(
                addr_3d["left_shoulder"], addr_3d["right_shoulder"],
                lm_3d[top_idx]["left_shoulder"], lm_3d[top_idx]["right_shoulder"],
            )
            mid_sh_top_2d = midpoint_2d(lm_2d[top_idx]["left_shoulder"], lm_2d[top_idx]["right_shoulder"])

            if sh_turn < config.SHOULDER_TURN_MIN_DEG:
                criteria.append(FaultCriterion(
                    code="shoulder_turn",
                    title="Thiếu độ xoay vai (Arm Swing)",
                    status="error",
                    measured_text=f"Xoay {sh_turn:.1f}°",
                    standard_text=f"≥ {config.SHOULDER_TURN_MIN_DEG:.0f}°",
                    frame_idx=top_idx,
                    description=(
                        f"Ở Đỉnh Swing, vai mới chỉ xoay được {sh_turn:.1f}° (chuẩn PGA xoay từ 75° - 100°). "
                        "Xoay vai ít khiến người chơi chỉ dùng lực tay để đánh, mất khoảng cách và thiếu ổn định."
                    ),
                    correction_tip="Khi đưa gậy lên, hãy xoay hẳn vai trái xuống dưới cằm và hướng lưng về phía mục tiêu.",
                    annotation_type="shoulder_turn",
                    annotation_data={
                        "left_shoulder": lm_2d[top_idx]["left_shoulder"][:2],
                        "right_shoulder": lm_2d[top_idx]["right_shoulder"][:2],
                        "center": mid_sh_top_2d,
                        "turn_deg": sh_turn,
                    },
                ))
            else:
                criteria.append(FaultCriterion(
                    code="shoulder_turn",
                    title="Biên độ xoay vai (Shoulder Turn)",
                    status="passed",
                    measured_text=f"Xoay {sh_turn:.1f}°",
                    standard_text=f"≥ {config.SHOULDER_TURN_MIN_DEG:.0f}°",
                    frame_idx=top_idx,
                    description=f"Độ xoay vai ở đỉnh swing đạt {sh_turn:.1f}° rất tốt, tích trữ đầy đủ năng lượng xoắn cho cú Downswing.",
                    correction_tip="Tiếp tục giữ nhịp xoay vai mượt mà này.",
                    annotation_type="shoulder_turn",
                    annotation_data={
                        "left_shoulder": lm_2d[top_idx]["left_shoulder"][:2],
                        "right_shoulder": lm_2d[top_idx]["right_shoulder"][:2],
                        "center": mid_sh_top_2d,
                        "turn_deg": sh_turn,
                    },
                ))
        else:
            criteria.append(FaultCriterion(
                code="shoulder_turn",
                title="Biên độ xoay vai (Shoulder Turn)",
                status="not_visible",
                measured_text="N/A",
                standard_text=f"≥ {config.SHOULDER_TURN_MIN_DEG:.0f}°",
                frame_idx=top_idx,
                description="Không đủ dữ liệu khớp 2 bên vai để đo độ xoay.",
                correction_tip="Đảm bảo camera nhìn thấy rõ cả vai trái và vai phải.",
                annotation_type="none",
            ))

        # ==========================================================
        # TIÊU CHÍ 4 (THEO GÓC CAMERA): FACE-ON vs DOWN-THE-LINE
        # ==========================================================
        req_torso = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        if camera_view == "face_on":
            # --- 4A: NGƯỢC TRỤC CỘT SỐNG (REVERSE SPINE ANGLE) ---
            if self._is_visible(addr_2d, req_torso) and self._is_visible(lm_2d[top_idx], req_torso):
                target_dx = addr_2d[sk["lead_hip"]][0] - addr_2d[sk["trail_hip"]][0]
                target_dir_sign = 1.0 if target_dx >= 0 else -1.0
                lm_top = lm_2d[top_idx]
                mid_hip_top = midpoint_2d(lm_top["left_hip"], lm_top["right_hip"])
                mid_sh_top = midpoint_2d(lm_top["left_shoulder"], lm_top["right_shoulder"])
                away_tilt = calculate_spine_away_tilt_deg(mid_hip_top, mid_sh_top, target_dir_sign)

                if away_tilt < -config.REVERSE_SPINE_MAX_DEG:
                    criteria.append(FaultCriterion(
                        code="reverse_spine",
                        title="Ngược trục cột sống (Reverse Spine)",
                        status="error",
                        measured_text=f"Đổ ngược {abs(away_tilt):.1f}°",
                        standard_text="≥ -4° (Giữ trục)",
                        frame_idx=top_idx,
                        description=(
                            f"Tại đỉnh Swing, thân trên bị đổ ngược {abs(away_tilt):.1f}° về phía mục tiêu thay vì nghiêng nhẹ ra sau. "
                            "Đây là nguyên nhân chính gây đau lưng dưới và đánh chém bóng từ ngoài vào trong (Over-the-top)."
                        ),
                        correction_tip="Khi xoay lên đỉnh, giữ đầu hơi chếch sau bóng và dồn trọng tâm vào mặt trong đùi chân sau.",
                        annotation_type="spine_tilt",
                        annotation_data={"mid_hip": mid_hip_top, "mid_shoulder": mid_sh_top},
                    ))
                else:
                    criteria.append(FaultCriterion(
                        code="reverse_spine",
                        title="Trục cột sống ở Đỉnh Swing",
                        status="passed",
                        measured_text=f"{away_tilt:+.1f}°",
                        standard_text="≥ -4°",
                        frame_idx=top_idx,
                        description=f"Trục cột sống xoay đúng kỹ thuật ({away_tilt:+.1f}°), không bị đổ ngược về phía mục tiêu.",
                        correction_tip="Trục xoay cột sống rất an toàn và chuẩn kỹ thuật.",
                        annotation_type="spine_tilt",
                        annotation_data={"mid_hip": mid_hip_top, "mid_shoulder": mid_sh_top},
                    ))
            else:
                criteria.append(FaultCriterion(
                    code="reverse_spine",
                    title="Trục cột sống ở Đỉnh Swing",
                    status="not_visible",
                    measured_text="Chưa thấy hông",
                    standard_text="Thấy vai & hông",
                    frame_idx=top_idx,
                    description="Chưa quan sát được cả vai và hông trong khung hình.",
                    correction_tip="Lùi xa camera khoảng 2m - 2.5m để lấy toàn thân.",
                    annotation_type="none",
                ))

            # --- 4B: TRƯỢT / LẮC HÔNG NGANG (PELVIC SWAY) ---
            req_sway = [sk["trail_hip"], sk["trail_ankle"], sk["lead_ankle"]]
            backswing_frames = [
                i for i in valid_indices if address_idx <= i <= top_idx and self._is_visible(lm_2d[i], req_sway)
            ]
            if self._is_visible(addr_2d, req_sway) and backswing_frames:
                stance_w = max(abs(addr_2d[sk["lead_ankle"]][0] - addr_2d[sk["trail_ankle"]][0]), 40.0)
                away_sign = 1.0 if (addr_2d[sk["trail_ankle"]][0] - addr_2d[sk["lead_ankle"]][0]) >= 0 else -1.0
                trail_limit_x = addr_2d[sk["trail_ankle"]][0]

                def sway_px(idx):
                    return (lm_2d[idx][sk["trail_hip"]][0] - trail_limit_x) * away_sign

                worst_sway_idx = max(backswing_frames, key=sway_px)
                sway_ratio = max(0.0, sway_px(worst_sway_idx) / stance_w)

                if sway_ratio > config.HIP_SWAY_MAX_RATIO:
                    criteria.append(FaultCriterion(
                        code="pelvic_sway",
                        title="Trượt / Lắc hông ngang (Pelvic Sway)",
                        status="error",
                        measured_text=f"Trượt {sway_ratio*100:.1f}%",
                        standard_text=f"≤ {config.HIP_SWAY_MAX_RATIO*100:.0f}%",
                        frame_idx=worst_sway_idx,
                        description=(
                            f"Hông sau bị đẩy trượt ngang ra ngoài trục chân sau tới {sway_ratio*100:.1f}% bề rộng thế đứng "
                            "trong pha đưa gậy lên, làm mất lực xoắn và dễ đánh lệch tâm bóng."
                        ),
                        correction_tip="Xoay túi quần sau ra phía lưng thay vì đẩy hông trượt ngang qua chân trụ.",
                        annotation_type="hip_sway",
                        annotation_data={
                            "trail_hip": lm_2d[worst_sway_idx][sk["trail_hip"]][:2],
                            "trail_ankle": addr_2d[sk["trail_ankle"]][:2],
                            "limit_x": trail_limit_x,
                        },
                    ))
                else:
                    criteria.append(FaultCriterion(
                        code="pelvic_sway",
                        title="Kiểm soát trượt hông (Pelvic Sway)",
                        status="passed",
                        measured_text=f"{sway_ratio*100:.1f}%",
                        standard_text=f"≤ {config.HIP_SWAY_MAX_RATIO*100:.0f}%",
                        frame_idx=top_idx,
                        description="Hông xoay gọn gàng trong trục hai bàn chân, không bị lắc hông sang ngang.",
                        correction_tip="Giữ trục hông xoay rất vững.",
                        annotation_type="hip_sway",
                        annotation_data={
                            "trail_hip": lm_2d[top_idx][sk["trail_hip"]][:2],
                            "trail_ankle": addr_2d[sk["trail_ankle"]][:2],
                            "limit_x": trail_limit_x,
                        },
                    ))
            else:
                criteria.append(FaultCriterion(
                    code="pelvic_sway",
                    title="Kiểm soát trượt hông (Pelvic Sway)",
                    status="not_visible",
                    measured_text="Chưa thấy chân",
                    standard_text="Thấy hông & 2 chân",
                    frame_idx=top_idx,
                    description="Chưa thấy hai bàn chân trong khung hình.",
                    correction_tip="Chỉnh camera lùi ra để lấy toàn thân từ đầu tới gót chân.",
                    annotation_type="none",
                ))
        else:
            # --- GÓC QUAY NHÌN NGANG (DOWN-THE-LINE): MẤT GÓC LƯNG / EARLY EXTENSION ---
            if self._is_visible(addr_2d, req_torso) and self._is_visible(lm_2d[impact_idx], req_torso):
                addr_flex = calculate_spine_forward_flexion_deg(
                    midpoint_2d(addr_2d["left_hip"], addr_2d["right_hip"]),
                    midpoint_2d(addr_2d["left_shoulder"], addr_2d["right_shoulder"]),
                )
                imp_2d = lm_2d[impact_idx]
                imp_hip = midpoint_2d(imp_2d["left_hip"], imp_2d["right_hip"])
                imp_sh = midpoint_2d(imp_2d["left_shoulder"], imp_2d["right_shoulder"])
                imp_flex = calculate_spine_forward_flexion_deg(imp_hip, imp_sh)
                loss = addr_flex - imp_flex

                if addr_flex < config.DTL_SPINE_FLEXION_MIN_DEG:
                    criteria.append(FaultCriterion(
                        code="posture_loss",
                        title="Tư thế Setup đứng quá thẳng",
                        status="error",
                        measured_text=f"Cúi {addr_flex:.1f}°",
                        standard_text=f"≥ {config.DTL_SPINE_FLEXION_MIN_DEG:.0f}°",
                        frame_idx=address_idx,
                        description=f"Góc cúi thân người lúc chuẩn bị chỉ đạt {addr_flex:.1f}°, khiến mặt phẳng vung gậy bị quá ngang.",
                        correction_tip="Đẩy nhẹ hông ra sau và cúi thân người từ khớp hông khoảng 25° - 40° với lưng thẳng.",
                        annotation_type="spine_tilt",
                        annotation_data={
                            "mid_hip": midpoint_2d(addr_2d["left_hip"], addr_2d["right_hip"]),
                            "mid_shoulder": midpoint_2d(addr_2d["left_shoulder"], addr_2d["right_shoulder"]),
                        },
                    ))
                elif loss > config.DTL_POSTURE_LOSS_MAX_DEG:
                    criteria.append(FaultCriterion(
                        code="posture_loss",
                        title="Mất góc lưng / Nhổm người sớm (Early Extension)",
                        status="error",
                        measured_text=f"Mất góc {loss:.1f}°",
                        standard_text=f"Lệch ≤ {config.DTL_POSTURE_LOSS_MAX_DEG:.0f}°",
                        frame_idx=impact_idx,
                        description=f"Khi chạm bóng bạn bị ngẩng thẳng lưng lên {loss:.1f}° so với lúc chuẩn bị, đẩy hông lao về phía bóng.",
                        correction_tip="Giữ cảm giác mông chạm vào một bức tường tưởng tượng phía sau trong suốt pha Downswing.",
                        annotation_type="spine_tilt",
                        annotation_data={"mid_hip": imp_hip, "mid_shoulder": imp_sh},
                    ))
                else:
                    criteria.append(FaultCriterion(
                        code="posture_loss",
                        title="Duy trì góc cột sống (Posture Retention)",
                        status="passed",
                        measured_text=f"Setup {addr_flex:.0f}° -> Impact {imp_flex:.0f}°",
                        standard_text=f"Lệch ≤ {config.DTL_POSTURE_LOSS_MAX_DEG:.0f}°",
                        frame_idx=impact_idx,
                        description="Duy trì góc cúi cột sống xuyên suốt từ Setup đến lúc chạm bóng cực kỳ ổn định.",
                        correction_tip="Khả năng giữ tư thế lưng rất xuất sắc!",
                        annotation_type="spine_tilt",
                        annotation_data={"mid_hip": imp_hip, "mid_shoulder": imp_sh},
                    ))

        # ==========================================================
        # TIÊU CHÍ 5 (3D): ỔN ĐỊNH ĐẦU GỐI CHÂN TRỤ (KNEE STABILITY)
        # ==========================================================
        req_knee = [sk["lead_hip"], sk["lead_knee"], sk["lead_ankle"]]
        knee_frames = [i for i in eval_range if self._is_visible(lm_3d[i], req_knee)]
        if knee_frames:
            knee_angles_3d = {
                i: calculate_angle_3d(lm_3d[i][sk["lead_hip"]], lm_3d[i][sk["lead_knee"]], lm_3d[i][sk["lead_ankle"]])
                for i in knee_frames
            }
            min_k_idx = min(knee_frames, key=lambda i: knee_angles_3d[i])
            max_k_idx = max(knee_frames, key=lambda i: knee_angles_3d[i])
            min_k = knee_angles_3d[min_k_idx]
            max_k = knee_angles_3d[max_k_idx]

            if min_k < config.KNEE_MIN_ANGLE_3D:
                lm_k = lm_2d[min_k_idx]
                criteria.append(FaultCriterion(
                    code="knee_stability",
                    title="Chùng / Sụp đầu gối chân dẫn hướng",
                    status="error",
                    measured_text=f"{min_k:.1f}° (3D)",
                    standard_text=f"{config.KNEE_MIN_ANGLE_3D:.0f}° - {config.KNEE_MAX_ANGLE_3D:.0f}°",
                    frame_idx=min_k_idx,
                    description=f"Đầu gối chân trước bị sụp gập xuống quá sâu ({min_k:.1f}°), làm hạ thấp trọng tâm bất thường.",
                    correction_tip="Giữ độ khuỵu gối vừa phải làm trụ vững khi xoay vai lên đỉnh.",
                    annotation_type="knee_angle",
                    annotation_data={
                        "hip": lm_k[sk["lead_hip"]][:2],
                        "knee": lm_k[sk["lead_knee"]][:2],
                        "ankle": lm_k[sk["lead_ankle"]][:2],
                    },
                ))
            elif max_k > config.KNEE_MAX_ANGLE_3D and max_k_idx >= top_idx:
                lm_k = lm_2d[max_k_idx]
                criteria.append(FaultCriterion(
                    code="knee_stability",
                    title="Bật thẳng đầu gối sớm (Early Extension)",
                    status="error",
                    measured_text=f"{max_k:.1f}° (3D)",
                    standard_text=f"≤ {config.KNEE_MAX_ANGLE_3D:.0f}°",
                    frame_idx=max_k_idx,
                    description=f"Chân trước duỗi thẳng cứng quá sớm ({max_k:.1f}°) trước/tại lúc chạm bóng, làm người chơi bị nhổm dậy mất lực.",
                    correction_tip="Giữ độ nén đàn hồi của đầu gối cho tới khoảnh khắc gậy chạm bóng.",
                    annotation_type="knee_angle",
                    annotation_data={
                        "hip": lm_k[sk["lead_hip"]][:2],
                        "knee": lm_k[sk["lead_knee"]][:2],
                        "ankle": lm_k[sk["lead_ankle"]][:2],
                    },
                ))
            else:
                show_idx = impact_idx if impact_idx in knee_angles_3d else min_k_idx
                lm_k = lm_2d[show_idx]
                k_val = knee_angles_3d[show_idx]
                criteria.append(FaultCriterion(
                    code="knee_stability",
                    title="Độ ổn định chân trụ & đầu gối",
                    status="passed",
                    measured_text=f"{k_val:.1f}° (3D)",
                    standard_text=f"{config.KNEE_MIN_ANGLE_3D:.0f}° - {config.KNEE_MAX_ANGLE_3D:.0f}°",
                    frame_idx=show_idx,
                    description="Đầu gối giữ trụ đàn hồi rất chắc chắn, tạo bệ phóng lực vững vàng.",
                    correction_tip="Tiếp tục duy trì trụ chân vững chãi như hiện tại.",
                    annotation_type="knee_angle",
                    annotation_data={
                        "hip": lm_k[sk["lead_hip"]][:2],
                        "knee": lm_k[sk["lead_knee"]][:2],
                        "ankle": lm_k[sk["lead_ankle"]][:2],
                    },
                ))
        else:
            criteria.append(FaultCriterion(
                code="knee_stability",
                title="Độ ổn định chân trụ & đầu gối",
                status="not_visible",
                measured_text="Chưa thấy chân",
                standard_text="Thấy hông, gối & mắt cá",
                frame_idx=top_idx,
                description="Chưa thấy rõ khớp gối và mắt cá chân trong khung hình.",
                correction_tip="Đặt camera lấy toàn thân để kiểm tra chi tiết chân trụ.",
                annotation_type="none",
            ))

        # Sắp xếp: Lỗi lên đầu -> Đạt chuẩn -> Chưa rõ khớp
        order_map = {"error": 0, "passed": 1, "not_visible": 2}
        criteria.sort(key=lambda c: order_map.get(c.status, 3))

        evaluated = [c for c in criteria if c.status in ("error", "passed")]
        passed = [c for c in criteria if c.status == "passed"]

        return SwingAnalysisResult(
            person_detected=True,
            camera_view=camera_view,
            total_frames=total_frames,
            fps=actual_fps,
            address_idx=address_idx,
            top_idx=top_idx,
            impact_idx=impact_idx,
            finish_idx=finish_idx,
            score_passed=len(passed),
            score_evaluated=len(evaluated),
            criteria=criteria,
            landmarks_per_frame=lm_2d,
            arm_angles_per_frame=arm_angles_3d,
        )
