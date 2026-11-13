"""
GIAO DIỆN DESKTOP CHÍNH - VAR GOLF STUDIO (MODERN WHITE BENTO UI)
Thiết kế sáng hiện đại, nền trắng tinh tế, bo góc mềm mại (14-22px),
đổ bóng nhẹ, icon vector SVG và phân tích 2D + 3D chuẩn PGA/TPI.
"""

import copy
import os
import time
import cv2
import numpy as np

from PySide6.QtCore import QRectF, QSize, QThread, QTimer, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import config
from src.core.swing_analyzer import BatchSwingAnalyzer, FaultCriterion, SwingAnalysisResult
from src.ui.icons import svg_icon, svg_pixmap
from src.ui.theme import COLORS, FONT_MONO, apply_soft_shadow, build_app_stylesheet
from src.utils.drawing import draw_skeleton_overlay, render_annotated_snapshot


# ==============================
# WORKER ĐỌC CAMERA ĐA LUỒNG
# ==============================
class CameraWorker(QThread):
    frame_ready = Signal(object, float)  # (frame_bgr, timestamp_ms)
    camera_error = Signal(str)

    def __init__(self, camera_index: int = 0):
        super().__init__()
        self.camera_index = camera_index
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            self.camera_error.emit(
                f"Chưa tìm thấy Webcam (ID={self.camera_index}). "
                "Cậu/bạn hãy kiểm tra cáp USB hoặc quyền Camera trong Windows Settings."
            )
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

        start_time = time.perf_counter()
        while self._running:
            ret, frame = cap.read()
            if ret and frame is not None:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                self.frame_ready.emit(frame, elapsed_ms)
            else:
                time.sleep(0.01)

        cap.release()

    def stop(self):
        self._running = False
        self.wait(1500)


# ==============================
# WORKER PHÂN TÍCH VIDEO AI
# ==============================
class AnalysisWorker(QThread):
    progress_updated = Signal(int)
    analysis_finished = Signal(object)
    analysis_failed = Signal(str)

    def __init__(self, frames_bgr: list, timestamps_ms: list, handedness: str):
        super().__init__()
        self.frames_bgr = frames_bgr
        self.timestamps_ms = timestamps_ms
        self.handedness = handedness

    def run(self):
        try:
            analyzer = BatchSwingAnalyzer(handedness=self.handedness)
            result = analyzer.analyze_clip(
                self.frames_bgr,
                self.timestamps_ms,
                progress_callback=self.progress_updated.emit,
            )

            if self.frames_bgr:
                try:
                    os.makedirs(config.RECORDINGS_DIR, exist_ok=True)
                    h, w, _ = self.frames_bgr[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps_out = max(15.0, min(60.0, result.fps))
                    writer = cv2.VideoWriter(config.LAST_SWING_VIDEO_PATH, fourcc, fps_out, (w, h))
                    for fr in self.frames_bgr:
                        writer.write(fr)
                    writer.release()
                except Exception:
                    pass

            self.analysis_finished.emit(result)
        except Exception as e:
            self.analysis_failed.emit(str(e))


# ==============================
# HÀM TIỆN ÍCH ẢNH & BO GÓC MỀM
# ==============================
def bgr_to_pixmap(frame_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(q_img)


def round_pixmap(pix: QPixmap, radius: int = 16) -> QPixmap:
    """Bo tròn 4 góc cho khung hình Video/Snapshot trên giao diện trắng."""
    if pix.isNull():
        return pix
    rounded = QPixmap(pix.size())
    rounded.fill(Qt.transparent)
    painter = QPainter(rounded)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, pix.width(), pix.height()), radius, radius)
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, pix)
    painter.end()
    return rounded


def mirror_landmarks(lm_dict: dict | None, width: int) -> dict | None:
    if lm_dict is None:
        return None
    return {k: (float(width - x), y, z, vis) for k, (x, y, z, vis) in lm_dict.items()}


def mirror_criterion(criterion: FaultCriterion, width: int) -> FaultCriterion:
    c = copy.deepcopy(criterion)
    ad = c.annotation_data
    point_keys = (
        "shoulder",
        "elbow",
        "wrist",
        "nose",
        "addr_nose",
        "left_shoulder",
        "right_shoulder",
        "center",
        "trail_hip",
        "trail_ankle",
        "mid_hip",
        "mid_shoulder",
        "hip",
        "knee",
        "ankle",
    )
    for key in point_keys:
        if key in ad and ad[key] is not None:
            pt = ad[key]
            ad[key] = (float(width - pt[0]), pt[1])
    if "limit_x" in ad:
        ad["limit_x"] = float(width - ad["limit_x"])
    return c


# ==============================
# THẺ HIỂN THỊ TỪNG LỖI (BENTO WHITE CARD)
# ==============================
class FaultCardWidget(QFrame):
    clicked = Signal(object)

    def __init__(self, criterion: FaultCriterion, is_selected: bool = False):
        super().__init__()
        self.criterion = criterion
        self.setCursor(Qt.PointingHandCursor)
        self._build_ui(is_selected)

    def _build_ui(self, is_selected: bool):
        c = COLORS
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        if self.criterion.status == "error":
            accent = c.CRIMSON
            pill_bg = "#FEE2E2"
            pill_fg = c.CRIMSON_TEXT
            pill_txt = "Cần sửa lỗi"
            icon_name = "alert_triangle"
            card_bg = c.CRIMSON_SOFT if is_selected else c.BG_CARD
            border = f"2px solid {c.CRIMSON}" if is_selected else f"1px solid {c.BORDER_LIGHT}"
        elif self.criterion.status == "passed":
            accent = c.EMERALD
            pill_bg = "#D1FAE5"
            pill_fg = c.EMERALD_TEXT
            pill_txt = "Đạt chuẩn"
            icon_name = "check_circle"
            card_bg = c.EMERALD_SOFT if is_selected else c.BG_CARD
            border = f"2px solid {c.EMERALD}" if is_selected else f"1px solid {c.BORDER_LIGHT}"
        else:
            accent = c.AMBER
            pill_bg = "#FEF3C7"
            pill_fg = c.AMBER_TEXT
            pill_txt = "Chưa rõ khớp"
            icon_name = "eye_off"
            card_bg = c.AMBER_SOFT if is_selected else c.BG_CARD
            border = f"2px solid {c.AMBER}" if is_selected else f"1px solid {c.BORDER_LIGHT}"

        self.setStyleSheet(
            f"""
            FaultCardWidget {{
                background-color: {card_bg};
                border: {border};
                border-radius: 16px;
            }}
            FaultCardWidget:hover {{
                border: 1.5px solid {c.EMERALD_BRIGHT};
                background-color: {c.BG_SUBTLE};
            }}
            """
        )

        # Dòng 1: Pill trạng thái + Số đo dạng pill
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        badge_wrap = QFrame()
        badge_wrap.setStyleSheet(
            f"background-color: {pill_bg}; border-radius: 10px; border: none;"
        )
        bw_layout = QHBoxLayout(badge_wrap)
        bw_layout.setContentsMargins(8, 3, 9, 3)
        bw_layout.setSpacing(5)

        icon_lbl = QLabel()
        icon_lbl.setPixmap(svg_pixmap(icon_name, color=accent, size=14, stroke_width=2.2))
        icon_lbl.setFixedSize(15, 15)
        icon_lbl.setStyleSheet("background: transparent; border: none;")

        badge = QLabel(pill_txt)
        badge.setStyleSheet(
            f"color: {pill_fg}; font-weight: 700; font-size: 11px; "
            "background: transparent; border: none;"
        )
        bw_layout.addWidget(icon_lbl)
        bw_layout.addWidget(badge)
        badge_wrap.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        metric = QLabel(f"{self.criterion.measured_text}  ·  {self.criterion.standard_text}")
        metric.setWordWrap(True)
        metric.setStyleSheet(
            f"color: {c.TEXT_MUTED}; font-family: {FONT_MONO}; "
            "font-size: 11px; font-weight: 700; background: transparent; border: none;"
        )
        metric.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        top_row.addWidget(badge_wrap)
        top_row.addWidget(metric, stretch=1)
        layout.addLayout(top_row)

        # Dòng 2: Tên tiêu chí
        title = QLabel(self.criterion.title)
        title.setWordWrap(True)
        title.setStyleSheet(
            f"color: {c.TEXT_PRIMARY}; font-size: 14px; font-weight: 700; "
            "background: transparent; border: none;"
        )
        layout.addWidget(title)

        # Dòng 3: Mô tả chi tiết
        desc = QLabel(self.criterion.description)
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"color: {c.TEXT_SECONDARY}; font-size: 12px; "
            "background: transparent; border: none;"
        )
        layout.addWidget(desc)

        # Dòng 4: Hộp gợi ý sửa lỗi dịu mắt
        if self.criterion.status == "error" and self.criterion.correction_tip:
            tip_box = QFrame()
            tip_box.setStyleSheet(
                f"background-color: {c.AMBER_SOFT}; border: 1px solid {c.AMBER_BORDER}; "
                "border-radius: 10px;"
            )
            tip_row = QHBoxLayout(tip_box)
            tip_row.setContentsMargins(9, 6, 9, 6)
            tip_row.setSpacing(6)

            bulb_lbl = QLabel()
            bulb_lbl.setPixmap(svg_pixmap("bulb", color=c.AMBER, size=14, stroke_width=2.1))
            bulb_lbl.setFixedSize(16, 16)
            bulb_lbl.setStyleSheet("background: transparent; border: none;")

            tip = QLabel(self.criterion.correction_tip)
            tip.setWordWrap(True)
            tip.setStyleSheet(
                f"color: {c.AMBER_TEXT}; font-size: 12px; font-weight: 600; "
                "background: transparent; border: none;"
            )

            tip_row.addWidget(bulb_lbl)
            tip_row.addWidget(tip, stretch=1)
            layout.addWidget(tip_box)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.criterion)
        super().mousePressEvent(event)


# ==============================
# CỬA SỔ CHÍNH ỨNG DỤNG
# ==============================
class GolfAssistantWindow(QMainWindow):
    STATE_LIVE = "LIVE"
    STATE_COUNTDOWN = "COUNTDOWN"
    STATE_RECORDING = "RECORDING"
    STATE_ANALYZING = "ANALYZING"
    STATE_RESULT = "RESULT"

    def __init__(self, camera_index: int = config.DEFAULT_CAMERA_INDEX):
        super().__init__()
        self.camera_index = camera_index
        self.state = self.STATE_LIVE

        # Dữ liệu phiên quay & phân tích
        self.raw_recorded_frames: list[np.ndarray] = []
        self.raw_recorded_timestamps: list[float] = []
        self.analysis_result: SwingAnalysisResult | None = None
        self.selected_criterion: FaultCriterion | None = None

        # Tham số đếm ngược & ghi hình
        self.countdown_remaining = config.COUNTDOWN_SECONDS
        self.record_duration_sec = config.RECORD_SECONDS
        self.record_start_time = 0.0

        # Trạng thái phát video Slow-Motion
        self.replay_playing = False
        self.replay_frame_idx = 0
        self.replay_speed = 0.5

        self._init_ui()
        self._init_timers_and_shortcuts()
        self._start_camera()

    def _init_ui(self):
        self.setWindowTitle("VAR Golf Studio - Trợ Lý AI Phân Tích Tư Thế Golf")
        self.resize(1340, 860)
        self.setMinimumSize(1020, 690)
        self.setStyleSheet(build_app_stylesheet())

        central = QWidget()
        central.setObjectName("RootCanvas")
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(18, 16, 18, 18)
        root_layout.setSpacing(14)

        # 1. FLOATING WHITE TOP BAR
        root_layout.addWidget(self._build_top_bar())

        # 2. MAIN WORKSPACE STACK
        self.stacked = QStackedWidget()
        self.stacked.addWidget(self._build_live_page())
        self.stacked.addWidget(self._build_result_page())
        root_layout.addWidget(self.stacked, stretch=1)

    def _build_top_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(66)
        apply_soft_shadow(bar, blur=20, y_offset=4, alpha=16)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(12)

        # Rounded Logo Box
        icon_box = QFrame()
        icon_box.setObjectName("BrandIconBox")
        icon_box.setFixedSize(42, 42)
        ib_layout = QHBoxLayout(icon_box)
        ib_layout.setContentsMargins(0, 0, 0, 0)
        ib_layout.setAlignment(Qt.AlignCenter)
        logo_lbl = QLabel()
        logo_lbl.setPixmap(svg_pixmap("target", color=COLORS.EMERALD, size=22, stroke_width=2.2))
        logo_lbl.setStyleSheet("background: transparent; border: none;")
        ib_layout.addWidget(logo_lbl)

        brand_col = QVBoxLayout()
        brand_col.setSpacing(1)
        title = QLabel("VAR Golf Studio")
        title.setObjectName("BrandTitle")
        subtitle = QLabel("Phân tích cơ sinh học 2D + 3D chuẩn PGA/TPI")
        subtitle.setObjectName("BrandSubtitle")
        brand_col.addWidget(title)
        brand_col.addWidget(subtitle)

        self.lbl_status_pill = QLabel("Sẵn sàng")
        self.lbl_status_pill.setObjectName("StatusPill")

        layout.addWidget(icon_box)
        layout.addLayout(brand_col)
        layout.addSpacing(8)
        layout.addWidget(self.lbl_status_pill)
        layout.addStretch()

        # Handedness
        lbl_hand = QLabel("Tay thuận")
        lbl_hand.setObjectName("TopControlLabel")
        self.combo_handedness = QComboBox()
        self.combo_handedness.addItem("Thuận tay phải", "right")
        self.combo_handedness.addItem("Thuận tay trái", "left")

        # Duration
        lbl_dur = QLabel("Thời lượng")
        lbl_dur.setObjectName("TopControlLabel")
        self.combo_duration = QComboBox()
        self.combo_duration.addItem("10 giây (Khuyên dùng)", 10)
        self.combo_duration.addItem("8 giây", 8)
        self.combo_duration.addItem("12 giây", 12)
        self.combo_duration.addItem("5 giây (Thử nhanh)", 5)
        self.combo_duration.currentIndexChanged.connect(self._on_duration_changed)

        # Mirror
        self.chk_mirror = QCheckBox("Soi gương")
        self.chk_mirror.setChecked(True)

        layout.addWidget(lbl_hand)
        layout.addWidget(self.combo_handedness)
        layout.addSpacing(4)
        layout.addWidget(lbl_dur)
        layout.addWidget(self.combo_duration)
        layout.addSpacing(4)
        layout.addWidget(self.chk_mirror)
        return bar

    def _build_live_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        # Thẻ trắng nổi chứa khung Video
        stage = QFrame()
        stage.setObjectName("StageCard")
        apply_soft_shadow(stage, blur=26, y_offset=6, alpha=18)
        stage_layout = QVBoxLayout(stage)
        stage_layout.setContentsMargins(12, 12, 12, 12)

        self.live_viewer = QLabel("Đang kết nối tín hiệu Camera...")
        self.live_viewer.setAlignment(Qt.AlignCenter)
        self.live_viewer.setStyleSheet(
            f"background-color: {COLORS.BG_SUBTLE}; border-radius: 16px; "
            f"border: 1px dashed {COLORS.BORDER_MEDIUM}; "
            f"color: {COLORS.TEXT_MUTED}; font-size: 15px; font-weight: 600;"
        )
        self.live_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.live_viewer.setMinimumHeight(460)
        stage_layout.addWidget(self.live_viewer)
        layout.addWidget(stage, stretch=1)

        # Bottom Floating Control Dock
        dock = QFrame()
        dock.setObjectName("ControlDock")
        apply_soft_shadow(dock, blur=20, y_offset=4, alpha=14)
        dock_layout = QHBoxLayout(dock)
        dock_layout.setContentsMargins(20, 14, 20, 14)
        dock_layout.setSpacing(16)

        left_info = QVBoxLayout()
        left_info.setSpacing(6)
        self.lbl_live_status = QLabel(
            "Đứng toàn thân trong khung ngắm (góc chính diện hoặc nhìn ngang đều được AI tự nhận diện)."
        )
        self.lbl_live_status.setObjectName("LiveHintLabel")
        self.lbl_live_status.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedWidth(440)

        left_info.addWidget(self.lbl_live_status)
        left_info.addWidget(self.progress_bar)
        dock_layout.addLayout(left_info, stretch=1)

        # Buttons
        self.btn_start = QPushButton(" Bắt đầu quay & Phân tích [SPACE]")
        self.btn_start.setObjectName("PrimaryCTA")
        self.btn_start.setIcon(svg_icon("play", color="#FFFFFF", size=17))
        self.btn_start.setIconSize(QSize(17, 17))
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self._on_start_clicked)

        self.btn_stop_early = QPushButton(" Dừng sớm & Phân tích ngay")
        self.btn_stop_early.setObjectName("DangerCTA")
        self.btn_stop_early.setIcon(svg_icon("stop", color="#FFFFFF", size=15))
        self.btn_stop_early.setIconSize(QSize(15, 15))
        self.btn_stop_early.setCursor(Qt.PointingHandCursor)
        self.btn_stop_early.setVisible(False)
        self.btn_stop_early.clicked.connect(self._finish_recording_and_analyze)

        self.btn_cancel = QPushButton(" Hủy [ESC]")
        self.btn_cancel.setObjectName("SecondaryBtn")
        self.btn_cancel.setIcon(svg_icon("x_close", color=COLORS.TEXT_PRIMARY, size=15))
        self.btn_cancel.setIconSize(QSize(15, 15))
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)

        dock_layout.addWidget(self.btn_cancel)
        dock_layout.addWidget(self.btn_stop_early)
        dock_layout.addWidget(self.btn_start)
        layout.addWidget(dock)
        return page

    def _build_result_page(self) -> QWidget:
        page = QWidget()
        main_split = QHBoxLayout(page)
        main_split.setContentsMargins(0, 0, 0, 0)
        main_split.setSpacing(16)

        # === LEFT STAGE: SNAPSHOT & SLOW-MOTION ===
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        # Segmented Switcher Pill + Camera View Badge
        top_strip = QHBoxLayout()
        top_strip.setSpacing(10)

        seg_track = QFrame()
        seg_track.setObjectName("SegmentTrack")
        seg_layout = QHBoxLayout(seg_track)
        seg_layout.setContentsMargins(4, 4, 4, 4)
        seg_layout.setSpacing(4)

        self.btn_tab_snapshot = QPushButton(" Ảnh khoanh vùng lỗi")
        self.btn_tab_snapshot.setObjectName("SegmentBtn")
        self.btn_tab_snapshot.setIcon(svg_icon("image", color=COLORS.EMERALD_TEXT, size=16))
        self.btn_tab_snapshot.setCursor(Qt.PointingHandCursor)
        self.btn_tab_snapshot.clicked.connect(lambda: self._switch_result_tab(0))

        self.btn_tab_replay = QPushButton(" Xem lại Slow-Motion")
        self.btn_tab_replay.setObjectName("SegmentBtn")
        self.btn_tab_replay.setIcon(svg_icon("video", color=COLORS.TEXT_MUTED, size=16))
        self.btn_tab_replay.setCursor(Qt.PointingHandCursor)
        self.btn_tab_replay.clicked.connect(lambda: self._switch_result_tab(1))

        seg_layout.addWidget(self.btn_tab_snapshot)
        seg_layout.addWidget(self.btn_tab_replay)

        self.lbl_view_badge = QLabel("Góc quay: Chính diện (Face-On)")
        self.lbl_view_badge.setObjectName("ViewBadge")

        top_strip.addWidget(seg_track)
        top_strip.addStretch()
        top_strip.addWidget(self.lbl_view_badge)
        left_col.addLayout(top_strip)

        # White Rounded Stage
        stage = QFrame()
        stage.setObjectName("StageCard")
        apply_soft_shadow(stage, blur=26, y_offset=6, alpha=18)
        stage_layout = QVBoxLayout(stage)
        stage_layout.setContentsMargins(12, 12, 12, 12)

        self.result_viewer = QLabel()
        self.result_viewer.setAlignment(Qt.AlignCenter)
        self.result_viewer.setStyleSheet(
            f"background-color: {COLORS.BG_SUBTLE}; border-radius: 16px;"
        )
        self.result_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_viewer.setMinimumHeight(440)
        stage_layout.addWidget(self.result_viewer)
        left_col.addWidget(stage, stretch=1)

        # Stacked Controls under Result Stage
        self.result_controls_stack = QStackedWidget()
        self.result_controls_stack.setFixedHeight(74)

        # Tab 0: 4-Phase Timeline Bar
        snap_dock = QFrame()
        snap_dock.setObjectName("ControlDock")
        apply_soft_shadow(snap_dock, blur=18, y_offset=4, alpha=14)
        snap_ctrl = QHBoxLayout(snap_dock)
        snap_ctrl.setContentsMargins(16, 10, 16, 10)
        snap_ctrl.setSpacing(8)

        self.lbl_snap_info = QLabel("Đang soi tư thế...")
        self.lbl_snap_info.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 12px; font-weight: 600;"
        )
        snap_ctrl.addWidget(self.lbl_snap_info, stretch=1)

        self.btn_jump_addr = QPushButton("Chuẩn bị")
        self.btn_jump_top = QPushButton("Đỉnh Swing")
        self.btn_jump_imp = QPushButton("Chạm bóng")
        self.btn_jump_fin = QPushButton("Kết thúc")
        for btn, phase_key in (
            (self.btn_jump_addr, "address"),
            (self.btn_jump_top, "top"),
            (self.btn_jump_imp, "impact"),
            (self.btn_jump_fin, "finish"),
        ):
            btn.setObjectName("PhaseBtn")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, p=phase_key: self._show_phase_snapshot(p))
            snap_ctrl.addWidget(btn)

        # Tab 1: Slow-Motion Controls
        replay_dock = QFrame()
        replay_dock.setObjectName("ControlDock")
        apply_soft_shadow(replay_dock, blur=18, y_offset=4, alpha=14)
        replay_ctrl = QHBoxLayout(replay_dock)
        replay_ctrl.setContentsMargins(16, 10, 16, 10)
        replay_ctrl.setSpacing(10)

        self.btn_play_pause = QPushButton(" Phát")
        self.btn_play_pause.setObjectName("SecondaryBtn")
        self.btn_play_pause.setIcon(svg_icon("play", color=COLORS.TEXT_PRIMARY, size=15))
        self.btn_play_pause.setCursor(Qt.PointingHandCursor)
        self.btn_play_pause.clicked.connect(self._toggle_replay_play)

        self.replay_slider = QSlider(Qt.Horizontal)
        self.replay_slider.setMinimum(0)
        self.replay_slider.valueChanged.connect(self._on_replay_slider_moved)

        self.lbl_replay_time = QLabel("000 / 000")
        self.lbl_replay_time.setObjectName("MetricValue")

        self.combo_speed = QComboBox()
        self.combo_speed.addItem("0.25x Siêu chậm", 0.25)
        self.combo_speed.addItem("0.50x Chậm", 0.5)
        self.combo_speed.addItem("1.00x Thực", 1.0)
        self.combo_speed.setCurrentIndex(1)
        self.combo_speed.currentIndexChanged.connect(self._on_speed_changed)

        replay_ctrl.addWidget(self.btn_play_pause)
        replay_ctrl.addWidget(self.replay_slider, stretch=1)
        replay_ctrl.addWidget(self.lbl_replay_time)
        replay_ctrl.addWidget(self.combo_speed)

        self.result_controls_stack.addWidget(snap_dock)
        self.result_controls_stack.addWidget(replay_dock)
        left_col.addWidget(self.result_controls_stack)

        main_split.addLayout(left_col, stretch=62)

        # === RIGHT PANEL: WHITE ROUNDED BIOMECHANICS PANEL ===
        right_panel = QFrame()
        right_panel.setObjectName("RightPanel")
        right_panel.setMinimumWidth(400)
        right_panel.setMaximumWidth(480)
        apply_soft_shadow(right_panel, blur=26, y_offset=6, alpha=18)

        right_col = QVBoxLayout(right_panel)
        right_col.setContentsMargins(18, 16, 18, 18)
        right_col.setSpacing(12)

        # Header row
        diag_header = QHBoxLayout()
        lbl_diag = QLabel("CHẨN ĐOÁN TƯ THẾ GOLF")
        lbl_diag.setObjectName("SectionHeading")
        self.lbl_score_pill = QLabel("Đạt 0/0")
        self.lbl_score_pill.setObjectName("ScorePill")
        diag_header.addWidget(lbl_diag)
        diag_header.addStretch()
        diag_header.addWidget(self.lbl_score_pill)
        right_col.addLayout(diag_header)

        # Summary Banner
        self.summary_box = QFrame()
        sum_layout = QHBoxLayout(self.summary_box)
        sum_layout.setContentsMargins(14, 12, 14, 12)
        sum_layout.setSpacing(10)
        self.lbl_summary_icon = QLabel()
        self.lbl_summary_icon.setFixedSize(22, 22)
        self.lbl_summary_icon.setStyleSheet("background: transparent; border: none;")
        self.lbl_summary_banner = QLabel("Đang tổng hợp...")
        self.lbl_summary_banner.setWordWrap(True)
        self.lbl_summary_banner.setStyleSheet(
            "font-size: 13px; font-weight: 800; background: transparent; border: none;"
        )
        sum_layout.addWidget(self.lbl_summary_icon)
        sum_layout.addWidget(self.lbl_summary_banner, stretch=1)
        right_col.addWidget(self.summary_box)

        lbl_hint_card = QLabel("Nhấn vào từng thẻ để khoanh vùng vị trí lỗi trên ảnh:")
        lbl_hint_card.setWordWrap(True)
        lbl_hint_card.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 12px;")
        right_col.addWidget(lbl_hint_card)

        # Scrollable Cards Container (ràng buộc chiều rộng theo viewport để card không bị tràn mép)
        class _FittedScrollArea(QScrollArea):
            def resizeEvent(self, ev):
                super().resizeEvent(ev)
                if self.widget() is not None:
                    self.widget().setMaximumWidth(max(100, self.viewport().width()))

        scroll = _FittedScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.cards_container = QWidget()
        self.cards_container.setStyleSheet("background: transparent;")
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 2, 8, 2)
        self.cards_layout.setSpacing(10)
        self.cards_layout.addStretch()
        scroll.setWidget(self.cards_container)
        right_col.addWidget(scroll, stretch=1)

        # Retry CTA
        self.btn_retry = QPushButton(" Đánh cú mới [R]")
        self.btn_retry.setObjectName("PrimaryCTA")
        self.btn_retry.setIcon(svg_icon("rotate_ccw", color="#FFFFFF", size=17))
        self.btn_retry.setIconSize(QSize(17, 17))
        self.btn_retry.setCursor(Qt.PointingHandCursor)
        self.btn_retry.clicked.connect(self._restart_to_live)
        right_col.addWidget(self.btn_retry)

        main_split.addWidget(right_panel, stretch=38)
        return page

    # --- TIMERS & SHORTCUTS ---
    def _init_timers_and_shortcuts(self):
        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self._on_countdown_tick)

        self.replay_timer = QTimer(self)
        self.replay_timer.timeout.connect(self._on_replay_tick)

        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self._on_space_shortcut)
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self._on_cancel_clicked)
        QShortcut(QKeySequence("R"), self).activated.connect(self._on_retry_shortcut)

    def _on_space_shortcut(self):
        if self.state == self.STATE_LIVE:
            self._on_start_clicked()
        elif self.state == self.STATE_RECORDING:
            self._finish_recording_and_analyze()
        elif self.state == self.STATE_RESULT and self.result_controls_stack.currentIndex() == 1:
            self._toggle_replay_play()

    def _on_retry_shortcut(self):
        if self.state == self.STATE_RESULT:
            self._restart_to_live()

    def _set_status_pill(self, text: str, fg: str, bg: str, border: str):
        self.lbl_status_pill.setText(text)
        self.lbl_status_pill.setStyleSheet(
            f"background-color: {bg}; color: {fg}; border: 1px solid {border}; "
            "border-radius: 14px; padding: 4px 12px; font-size: 11px; font-weight: 700;"
        )

    # --- CAMERA & LIVE LOOP ---
    def _start_camera(self):
        self.cam_worker = CameraWorker(self.camera_index)
        self.cam_worker.frame_ready.connect(self._on_camera_frame)
        self.cam_worker.camera_error.connect(self._on_camera_error)
        self.cam_worker.start()

    def _on_camera_error(self, msg: str):
        self.live_viewer.setText(f"Chưa kết nối được Camera\n\n{msg}")
        self.live_viewer.setStyleSheet(
            f"background-color: {COLORS.CRIMSON_SOFT}; border-radius: 16px; "
            f"border: 1px dashed {COLORS.CRIMSON_BORDER}; "
            f"color: {COLORS.CRIMSON_TEXT}; font-size: 14px; font-weight: 600;"
        )
        self.btn_start.setEnabled(False)
        self._set_status_pill("Mất kết nối Camera", COLORS.CRIMSON_TEXT, COLORS.CRIMSON_SOFT, COLORS.CRIMSON_BORDER)

    def _on_duration_changed(self):
        self.record_duration_sec = int(self.combo_duration.currentData())

    def _on_camera_frame(self, frame_bgr: np.ndarray, timestamp_ms: float):
        if self.state in (self.STATE_ANALYZING, self.STATE_RESULT):
            return

        if self.state == self.STATE_RECORDING:
            self.raw_recorded_frames.append(frame_bgr.copy())
            self.raw_recorded_timestamps.append(timestamp_ms)

            elapsed = time.perf_counter() - self.record_start_time
            remaining = max(0.0, self.record_duration_sec - elapsed)
            self.lbl_live_status.setText(
                f"Đang ghi hình cú đánh Golf... Còn {remaining:.1f} giây (Nhấn SPACE để phân tích sớm)"
            )
            if elapsed >= self.record_duration_sec:
                self._finish_recording_and_analyze()
                return

        if not getattr(self, "_live_has_video", False):
            self._live_has_video = True
            self.live_viewer.setStyleSheet("background-color: transparent; border: none; border-radius: 16px;")

        disp = cv2.flip(frame_bgr, 1) if self.chk_mirror.isChecked() else frame_bgr
        pix = bgr_to_pixmap(disp)
        if not self.live_viewer.size().isEmpty():
            pix = pix.scaled(self.live_viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        pix = self._paint_live_hud(pix)
        self.live_viewer.setPixmap(round_pixmap(pix, radius=16))

    def _paint_live_hud(self, pix: QPixmap) -> QPixmap:
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w, h = pix.width(), pix.height()

        # Subtle Corner Alignment Reticle Brackets
        if self.state in (self.STATE_LIVE, self.STATE_COUNTDOWN, self.STATE_RECORDING):
            bracket_color = QColor("#10B981" if self.state == self.STATE_RECORDING else "#FFFFFF")
            bracket_color.setAlpha(200 if self.state == self.STATE_RECORDING else 135)
            painter.setPen(QPen(bracket_color, 2.5, Qt.SolidLine, Qt.RoundCap))
            pad, arm = 20, 24
            painter.drawLine(pad, pad, pad + arm, pad)
            painter.drawLine(pad, pad, pad, pad + arm)
            painter.drawLine(w - pad, pad, w - pad - arm, pad)
            painter.drawLine(w - pad, pad, w - pad, pad + arm)
            painter.drawLine(pad, h - pad, pad + arm, h - pad)
            painter.drawLine(pad, h - pad, pad, h - pad - arm)
            painter.drawLine(w - pad, h - pad, w - pad - arm, h - pad)
            painter.drawLine(w - pad, h - pad, w - pad, h - pad - arm)

        if self.state == self.STATE_COUNTDOWN:
            painter.fillRect(0, 0, w, h, QColor(15, 23, 42, 115))

            cx, cy = w // 2, h // 2 - 14
            r = 68
            # Vòng tròn trắng mờ làm nền số đếm ngược
            painter.setBrush(QColor(255, 255, 255, 235))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)

            frac = self.countdown_remaining / max(1, config.COUNTDOWN_SECONDS)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#E2E8F0"), 6))
            painter.drawEllipse(cx - r + 5, cy - r + 5, (r - 5) * 2, (r - 5) * 2)
            painter.setPen(QPen(QColor(COLORS.EMERALD_BRIGHT), 6, Qt.SolidLine, Qt.RoundCap))
            painter.drawArc(
                cx - r + 5,
                cy - r + 5,
                (r - 5) * 2,
                (r - 5) * 2,
                90 * 16,
                int(-360 * frac * 16),
            )

            painter.setPen(QColor(COLORS.TEXT_PRIMARY))
            painter.setFont(QFont("Consolas", 56, QFont.Bold))
            painter.drawText(cx - 55, cy - 48, 110, 96, Qt.AlignCenter, str(self.countdown_remaining))

            # Pill hướng dẫn dưới vòng tròn
            painter.setBrush(QColor(255, 255, 255, 235))
            painter.setPen(Qt.NoPen)
            pill_w = min(380, w - 40)
            painter.drawRoundedRect((w - pill_w) // 2, cy + 84, pill_w, 38, 19, 19)
            painter.setPen(QColor(COLORS.TEXT_PRIMARY))
            painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
            painter.drawText(
                (w - pill_w) // 2, cy + 84, pill_w, 38, Qt.AlignCenter, "Vào tư thế chuẩn bị đánh (Address)"
            )

        elif self.state == self.STATE_RECORDING:
            elapsed = time.perf_counter() - self.record_start_time
            remaining = max(0.0, self.record_duration_sec - elapsed)
            ratio = min(1.0, elapsed / max(0.1, float(self.record_duration_sec)))

            # Glassmorphic white REC pill
            painter.setBrush(QColor(255, 255, 255, 235))
            painter.setPen(QPen(QColor(COLORS.CRIMSON_BORDER), 1.5))
            painter.drawRoundedRect(26, 26, 172, 36, 18, 18)

            painter.setBrush(QColor(COLORS.CRIMSON))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(39, 38, 12, 12)

            painter.setPen(QColor(COLORS.TEXT_PRIMARY))
            painter.setFont(QFont("Consolas", 13, QFont.Bold))
            painter.drawText(58, 26, 134, 36, Qt.AlignVCenter | Qt.AlignLeft, f"REC  {remaining:04.1f}s")

            # Bottom progress bar
            bar_w = int((w - 52) * ratio)
            painter.setBrush(QColor(255, 255, 255, 160))
            painter.drawRoundedRect(26, h - 22, w - 52, 6, 3, 3)
            painter.setBrush(QColor(COLORS.EMERALD_BRIGHT))
            painter.drawRoundedRect(26, h - 22, max(10, bar_w), 6, 3, 3)

        painter.end()
        return pix

    # --- CAPTURE FLOW ---
    def _on_start_clicked(self):
        if self.state != self.STATE_LIVE:
            return
        self.record_duration_sec = int(self.combo_duration.currentData())
        self.countdown_remaining = config.COUNTDOWN_SECONDS
        self.state = self.STATE_COUNTDOWN

        self.btn_start.setVisible(False)
        self.btn_cancel.setVisible(True)
        self.btn_stop_early.setVisible(False)
        self.combo_handedness.setEnabled(False)
        self.combo_duration.setEnabled(False)

        self._set_status_pill(
            f"Chuẩn bị {self.countdown_remaining}s",
            COLORS.AMBER_TEXT,
            COLORS.AMBER_SOFT,
            COLORS.AMBER_BORDER,
        )
        self.lbl_live_status.setText(f"Chuẩn bị đánh... Bắt đầu quay sau {self.countdown_remaining} giây!")
        self.countdown_timer.start()

    def _on_countdown_tick(self):
        self.countdown_remaining -= 1
        if self.countdown_remaining > 0:
            self._set_status_pill(
                f"Chuẩn bị {self.countdown_remaining}s",
                COLORS.AMBER_TEXT,
                COLORS.AMBER_SOFT,
                COLORS.AMBER_BORDER,
            )
            self.lbl_live_status.setText(f"Chuẩn bị đánh... Bắt đầu quay sau {self.countdown_remaining} giây!")
        else:
            self.countdown_timer.stop()
            self._begin_recording()

    def _begin_recording(self):
        self.raw_recorded_frames.clear()
        self.raw_recorded_timestamps.clear()
        self.record_start_time = time.perf_counter()
        self.state = self.STATE_RECORDING
        self.btn_stop_early.setVisible(True)
        self._set_status_pill("Đang ghi hình", COLORS.CRIMSON_TEXT, COLORS.CRIMSON_SOFT, COLORS.CRIMSON_BORDER)
        self.lbl_live_status.setText("ĐANG QUAY! Hãy thực hiện cú đánh Golf tự nhiên...")

    def _on_cancel_clicked(self):
        if self.state in (self.STATE_COUNTDOWN, self.STATE_RECORDING):
            self.countdown_timer.stop()
            self.raw_recorded_frames.clear()
            self.raw_recorded_timestamps.clear()
            self.state = self.STATE_LIVE
            self._reset_live_controls()
        elif self.state == self.STATE_RESULT:
            self._restart_to_live()

    def _reset_live_controls(self):
        self.btn_start.setVisible(True)
        self.btn_start.setEnabled(True)
        self.btn_cancel.setVisible(False)
        self.btn_stop_early.setVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.combo_handedness.setEnabled(True)
        self.combo_duration.setEnabled(True)
        self._set_status_pill("Sẵn sàng", COLORS.EMERALD_TEXT, COLORS.EMERALD_SOFT, COLORS.EMERALD_BORDER)
        self.lbl_live_status.setText(
            "Đứng toàn thân trong khung ngắm (góc chính diện hoặc nhìn ngang đều được AI tự nhận diện)."
        )

    def _finish_recording_and_analyze(self):
        if self.state != self.STATE_RECORDING:
            return
        self.state = self.STATE_ANALYZING
        self.btn_stop_early.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._set_status_pill("Đang phân tích", COLORS.BLUE_TEXT, COLORS.BLUE_SOFT, COLORS.BLUE_BORDER)
        self.lbl_live_status.setText(
            f"AI đang dựng khung xương 2D+3D và chấm điểm tư thế ({len(self.raw_recorded_frames)} khung hình)..."
        )

        handedness = self.combo_handedness.currentData()
        self.analysis_worker = AnalysisWorker(
            self.raw_recorded_frames, self.raw_recorded_timestamps, handedness
        )
        self.analysis_worker.progress_updated.connect(self.progress_bar.setValue)
        self.analysis_worker.analysis_finished.connect(self._on_analysis_finished)
        self.analysis_worker.analysis_failed.connect(self._on_analysis_failed)
        self.analysis_worker.start()

    def _on_analysis_failed(self, err: str):
        QMessageBox.critical(self, "Lỗi Phân Tích AI", f"Đã xảy ra lỗi khi phân tích:\n{err}")
        self.state = self.STATE_LIVE
        self._reset_live_controls()

    def _on_analysis_finished(self, result: SwingAnalysisResult):
        self.analysis_result = result
        if not result.person_detected or not result.criteria:
            QMessageBox.information(
                self,
                "Không nhận diện rõ người chơi",
                "AI không thấy rõ toàn thân người chơi trong đoạn video vừa quay.\n"
                "Hãy đảm bảo đủ ánh sáng và đứng lùi để thấy từ đầu tới chân nhé ạ!",
            )
            self.state = self.STATE_LIVE
            self._reset_live_controls()
            return

        self.state = self.STATE_RESULT
        self._set_status_pill("Hoàn tất phân tích", COLORS.EMERALD_TEXT, COLORS.EMERALD_SOFT, COLORS.EMERALD_BORDER)
        self.stacked.setCurrentIndex(1)
        self._populate_result_ui()
        QTimer.singleShot(0, self._refresh_active_result_view)

    def _refresh_active_result_view(self):
        if self.state != self.STATE_RESULT:
            return
        if self.result_controls_stack.currentIndex() == 0:
            self._render_selected_snapshot()
        else:
            self._render_replay_frame(self.replay_frame_idx)

    # --- RESULT PRESENTATION ---
    def _populate_result_ui(self):
        res = self.analysis_result
        errors = [c for c in res.criteria if c.status == "error"]

        view_txt = (
            "Góc quay: Chính diện (Face-On)"
            if res.camera_view == "face_on"
            else "Góc quay: Nhìn ngang (Down-the-Line)"
        )
        self.lbl_view_badge.setText(view_txt)
        self.lbl_score_pill.setText(f"Đạt {res.score_passed}/{res.score_evaluated}")

        if errors:
            self.lbl_summary_icon.setPixmap(svg_pixmap("alert_triangle", color=COLORS.CRIMSON, size=20))
            self.lbl_summary_banner.setText(
                f"Phát hiện {len(errors)} lỗi kỹ thuật  (Đạt {res.score_passed}/{res.score_evaluated})"
            )
            self.lbl_summary_banner.setStyleSheet(
                f"color: {COLORS.CRIMSON_TEXT}; font-size: 13px; font-weight: 800; background: transparent; border: none;"
            )
            self.summary_box.setStyleSheet(
                f"background-color: {COLORS.CRIMSON_SOFT}; border: 1px solid {COLORS.CRIMSON_BORDER}; border-radius: 16px;"
            )
        else:
            self.lbl_summary_icon.setPixmap(svg_pixmap("check_circle", color=COLORS.EMERALD, size=20))
            self.lbl_summary_banner.setText(
                f"Xuất sắc! Đạt chuẩn {res.score_passed}/{res.score_evaluated} tiêu chí"
            )
            self.lbl_summary_banner.setStyleSheet(
                f"color: {COLORS.EMERALD_TEXT}; font-size: 13px; font-weight: 800; background: transparent; border: none;"
            )
            self.summary_box.setStyleSheet(
                f"background-color: {COLORS.EMERALD_SOFT}; border: 1px solid {COLORS.EMERALD_BORDER}; border-radius: 16px;"
            )

        self.selected_criterion = errors[0] if errors else res.criteria[0]
        self._rebuild_fault_cards()

        self.replay_slider.blockSignals(True)
        self.replay_slider.setMaximum(max(0, res.total_frames - 1))
        self.replay_slider.setValue(res.top_idx)
        self.replay_slider.blockSignals(False)
        self.replay_frame_idx = res.top_idx
        self.replay_playing = False
        self._update_play_pause_button()

        self._switch_result_tab(0)

    def _rebuild_fault_cards(self):
        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for crit in self.analysis_result.criteria:
            is_sel = self.selected_criterion is not None and crit.code == self.selected_criterion.code
            card = FaultCardWidget(crit, is_selected=is_sel)
            card.clicked.connect(self._on_fault_card_clicked)
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)

    def _on_fault_card_clicked(self, criterion: FaultCriterion):
        self.selected_criterion = criterion
        self._rebuild_fault_cards()
        self._switch_result_tab(0)

    def _switch_result_tab(self, tab_idx: int):
        self.result_controls_stack.setCurrentIndex(tab_idx)
        if tab_idx == 0:
            self.replay_timer.stop()
            self.replay_playing = False
            self._update_play_pause_button()

            self.btn_tab_snapshot.setProperty("active", "emerald")
            self.btn_tab_snapshot.setIcon(svg_icon("image", color=COLORS.EMERALD_TEXT, size=16))
            self.btn_tab_replay.setProperty("active", "false")
            self.btn_tab_replay.setIcon(svg_icon("video", color=COLORS.TEXT_MUTED, size=16))
            self._refresh_button_style(self.btn_tab_snapshot)
            self._refresh_button_style(self.btn_tab_replay)
            self._render_selected_snapshot()
        else:
            self.btn_tab_snapshot.setProperty("active", "false")
            self.btn_tab_snapshot.setIcon(svg_icon("image", color=COLORS.TEXT_MUTED, size=16))
            self.btn_tab_replay.setProperty("active", "true")
            self.btn_tab_replay.setIcon(svg_icon("video", color=COLORS.BLUE_TEXT, size=16))
            self._refresh_button_style(self.btn_tab_snapshot)
            self._refresh_button_style(self.btn_tab_replay)
            self._render_replay_frame(self.replay_frame_idx)

    @staticmethod
    def _refresh_button_style(btn: QPushButton):
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def _get_phase_label(self, frame_idx: int) -> str:
        res = self.analysis_result
        if abs(frame_idx - res.address_idx) <= 3:
            return f"CHUẨN BỊ | Frame {frame_idx}/{res.total_frames}"
        if abs(frame_idx - res.top_idx) <= 2:
            return f"ĐỈNH SWING | Frame {frame_idx}/{res.total_frames}"
        if abs(frame_idx - res.impact_idx) <= 2:
            return f"CHẠM BÓNG | Frame {frame_idx}/{res.total_frames}"
        if abs(frame_idx - res.finish_idx) <= 3:
            return f"KẾT THÚC | Frame {frame_idx}/{res.total_frames}"
        return f"KHOẢNH KHẮC LỖI | Frame {frame_idx}/{res.total_frames}"

    def _render_selected_snapshot(self, override_frame_idx: int | None = None):
        if not self.analysis_result or not self.selected_criterion:
            return

        res = self.analysis_result
        crit = self.selected_criterion
        idx = override_frame_idx if override_frame_idx is not None else crit.frame_idx
        idx = max(0, min(idx, res.total_frames - 1))

        raw_frame = self.raw_recorded_frames[idx]
        lm = res.landmarks_per_frame[idx]
        h, w, _ = raw_frame.shape

        if self.chk_mirror.isChecked():
            disp_frame = cv2.flip(raw_frame, 1)
            disp_lm = mirror_landmarks(lm, w)
            disp_crit = mirror_criterion(crit, w)
        else:
            disp_frame = raw_frame
            disp_lm = lm
            disp_crit = crit

        phase_txt = self._get_phase_label(idx)
        annotated = render_annotated_snapshot(disp_frame, disp_lm, disp_crit, phase_label=phase_txt)
        pix = bgr_to_pixmap(annotated)
        if not self.result_viewer.size().isEmpty():
            pix = pix.scaled(self.result_viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.result_viewer.setPixmap(round_pixmap(pix, radius=16))
        self.lbl_snap_info.setText(f"Đang soi: {crit.title} ({phase_txt})")

    def _show_phase_snapshot(self, phase: str):
        if not self.analysis_result:
            return
        phase_map = {
            "address": self.analysis_result.address_idx,
            "top": self.analysis_result.top_idx,
            "impact": self.analysis_result.impact_idx,
            "finish": self.analysis_result.finish_idx,
        }
        idx = phase_map.get(phase, self.analysis_result.top_idx)
        self._render_selected_snapshot(override_frame_idx=idx)

    # --- SLOW MOTION REPLAY ---
    def _update_play_pause_button(self):
        if self.replay_playing:
            self.btn_play_pause.setText(" Tạm dừng")
            self.btn_play_pause.setIcon(svg_icon("pause", color=COLORS.TEXT_PRIMARY, size=15))
        else:
            self.btn_play_pause.setText(" Phát")
            self.btn_play_pause.setIcon(svg_icon("play", color=COLORS.TEXT_PRIMARY, size=15))

    def _on_speed_changed(self):
        self.replay_speed = float(self.combo_speed.currentData())
        if self.replay_playing:
            self._start_replay_timer()

    def _start_replay_timer(self):
        base_fps = self.analysis_result.fps if self.analysis_result else 30.0
        interval_ms = max(15, int(1000.0 / (base_fps * self.replay_speed)))
        self.replay_timer.start(interval_ms)

    def _toggle_replay_play(self):
        if not self.analysis_result:
            return
        self.replay_playing = not self.replay_playing
        self._update_play_pause_button()
        if self.replay_playing:
            if self.replay_frame_idx >= self.analysis_result.total_frames - 1:
                self.replay_frame_idx = 0
            self._start_replay_timer()
        else:
            self.replay_timer.stop()

    def _on_replay_tick(self):
        if not self.analysis_result:
            return
        self.replay_frame_idx += 1
        if self.replay_frame_idx >= self.analysis_result.total_frames:
            self.replay_frame_idx = 0
        self.replay_slider.blockSignals(True)
        self.replay_slider.setValue(self.replay_frame_idx)
        self.replay_slider.blockSignals(False)
        self._render_replay_frame(self.replay_frame_idx)

    def _on_replay_slider_moved(self, val: int):
        self.replay_frame_idx = val
        if self.result_controls_stack.currentIndex() == 1:
            self._render_replay_frame(val)

    def _render_replay_frame(self, idx: int):
        if not self.analysis_result:
            return
        res = self.analysis_result
        idx = max(0, min(idx, res.total_frames - 1))

        raw_frame = self.raw_recorded_frames[idx]
        lm = res.landmarks_per_frame[idx]
        angle = res.arm_angles_per_frame[idx]
        handedness = self.combo_handedness.currentData()
        h, w, _ = raw_frame.shape

        if self.chk_mirror.isChecked():
            disp_frame = cv2.flip(raw_frame, 1)
            disp_lm = mirror_landmarks(lm, w)
            disp_hand = "left" if handedness == "right" else "right"
        else:
            disp_frame = raw_frame
            disp_lm = lm
            disp_hand = handedness

        drawn = draw_skeleton_overlay(disp_frame, disp_lm, arm_angle=angle, handedness=disp_hand)
        pix = bgr_to_pixmap(drawn)
        if not self.result_viewer.size().isEmpty():
            pix = pix.scaled(self.result_viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.result_viewer.setPixmap(round_pixmap(pix, radius=16))
        self.lbl_replay_time.setText(f"{idx + 1:03d} / {res.total_frames:03d}")

    def _restart_to_live(self):
        self.replay_timer.stop()
        self.replay_playing = False
        self.raw_recorded_frames.clear()
        self.raw_recorded_timestamps.clear()
        self.analysis_result = None
        self.selected_criterion = None
        self.state = self.STATE_LIVE
        self._reset_live_controls()
        self.stacked.setCurrentIndex(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.state == self.STATE_RESULT:
            if self.result_controls_stack.currentIndex() == 0:
                self._render_selected_snapshot()
            else:
                self._render_replay_frame(self.replay_frame_idx)

    def closeEvent(self, event):
        self.countdown_timer.stop()
        self.replay_timer.stop()
        if hasattr(self, "cam_worker"):
            self.cam_worker.stop()
        super().closeEvent(event)
