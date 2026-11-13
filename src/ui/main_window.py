"""
MAIN DESKTOP UI - VAR GOLF STUDIO (FULL-BLEED BORDERLESS KIOSK UI)
- Zero top bar, zero window frame/borders: 100% full-bleed camera viewport
- Controlled by 1 physical button / foot pedal ([SPACE])
- Horizontal auto-sliding diagnostic carousel (4.0s hold per slide)
"""

import copy
import os
import time
import cv2
import numpy as np

from PySide6.QtCore import (
    QEasingCurve,
    QParallelAnimationGroup,
    QPoint,
    QPropertyAnimation,
    QRectF,
    QSize,
    QThread,
    QTimer,
    Qt,
    Signal,
)
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
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import config
from src.core.swing_analyzer import BatchSwingAnalyzer, FaultCriterion, SwingAnalysisResult
from src.ui.icons import svg_icon, svg_pixmap
from src.ui.theme import COLORS, FONT_MONO, apply_soft_shadow, build_app_stylesheet
from src.utils.drawing import render_annotated_snapshot


# ==============================
# MULTI-THREADED CAMERA WORKER
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
                f"Unable to connect to Webcam (ID={self.camera_index}). "
                "Check the USB cable or Camera permissions in Windows Settings."
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
# AI SWING ANALYSIS WORKER
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
# IMAGE & MIRRORING HELPERS
# ==============================
def bgr_to_pixmap(frame_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(q_img)


def round_pixmap(pix: QPixmap, radius: int = 16) -> QPixmap:
    """Clip all 4 corners of a Snapshot QPixmap cleanly."""
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
# SINGLE DIAGNOSTIC SLIDE (IMAGE + TEXT)
# ==============================
class SwingSlideWidget(QFrame):
    """Single full-stage slide pairing an annotated joint snapshot with large readable coaching text."""

    def __init__(
        self,
        annotated_bgr: np.ndarray,
        criterion: FaultCriterion,
        slide_index: int,
        total_slides: int,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("SlideCard")
        self._source_pixmap = bgr_to_pixmap(annotated_bgr)
        self.criterion = criterion
        self._build_ui(slide_index, total_slides)

    def _build_ui(self, slide_index: int, total_slides: int):
        c = COLORS
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 22, 16)
        root.setSpacing(24)

        # Left: Circled Fault Snapshot
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet(f"background-color: {c.BG_SUBTLE}; border-radius: 16px;")
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_label.setMinimumSize(320, 260)
        root.addWidget(self.img_label, stretch=64)

        # Right: Large Readable Coaching Card
        text_col = QVBoxLayout()
        text_col.setContentsMargins(4, 10, 4, 10)
        text_col.setSpacing(15)

        if self.criterion.status == "error":
            accent, pill_bg, pill_fg = c.CRIMSON, "#FEE2E2", c.CRIMSON_TEXT
            pill_txt, icon_name = "FAULT DETECTED", "alert_triangle"
        elif self.criterion.status == "passed":
            accent, pill_bg, pill_fg = c.EMERALD, "#D1FAE5", c.EMERALD_TEXT
            pill_txt, icon_name = "PRO STANDARD", "check_circle"
        else:
            accent, pill_bg, pill_fg = c.AMBER, "#FEF3C7", c.AMBER_TEXT
            pill_txt, icon_name = "NOT VISIBLE", "eye_off"

        # Status Pill + Slide Number
        top_row = QHBoxLayout()
        badge_wrap = QFrame()
        badge_wrap.setStyleSheet(f"background-color: {pill_bg}; border-radius: 12px; border: none;")
        bw = QHBoxLayout(badge_wrap)
        bw.setContentsMargins(12, 6, 14, 6)
        bw.setSpacing(7)

        icon_lbl = QLabel()
        icon_lbl.setPixmap(svg_pixmap(icon_name, color=accent, size=17, stroke_width=2.3))
        icon_lbl.setStyleSheet("background: transparent; border: none;")

        badge = QLabel(pill_txt)
        badge.setStyleSheet(
            f"color: {pill_fg}; font-weight: 800; font-size: 12px; letter-spacing: 0.6px; "
            "background: transparent; border: none;"
        )
        bw.addWidget(icon_lbl)
        bw.addWidget(badge)
        badge_wrap.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        counter = QLabel(f"{slide_index + 1} / {total_slides}")
        counter.setStyleSheet(
            f"background-color: {c.BG_MUTED}; color: {c.TEXT_SECONDARY}; "
            f"font-family: {FONT_MONO}; font-size: 13px; font-weight: 800; "
            "padding: 6px 12px; border-radius: 11px;"
        )
        counter.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        top_row.addWidget(badge_wrap)
        top_row.addStretch()
        top_row.addWidget(counter)
        text_col.addLayout(top_row)

        # Big Title
        title = QLabel(self.criterion.title)
        title.setWordWrap(True)
        title.setStyleSheet(
            f"color: {c.TEXT_PRIMARY}; font-size: 24px; font-weight: 800; "
            "line-height: 1.2; background: transparent; border: none;"
        )
        text_col.addWidget(title)

        # Metric Comparison Box
        metric_box = QFrame()
        metric_box.setStyleSheet(
            f"background-color: {c.BG_SUBTLE}; border: 1px solid {c.BORDER_LIGHT}; border-radius: 14px;"
        )
        mb_layout = QHBoxLayout(metric_box)
        mb_layout.setContentsMargins(16, 12, 16, 12)
        mb_layout.setSpacing(16)

        measured_col = QVBoxLayout()
        measured_col.setSpacing(2)
        m_hdr = QLabel("MEASURED")
        m_hdr.setStyleSheet(f"color: {c.TEXT_MUTED}; font-size: 10px; font-weight: 800; letter-spacing: 0.7px; border: none;")
        m_val = QLabel(self.criterion.measured_text)
        m_val.setStyleSheet(f"color: {accent}; font-family: {FONT_MONO}; font-size: 17px; font-weight: 800; border: none;")
        measured_col.addWidget(m_hdr)
        measured_col.addWidget(m_val)

        target_col = QVBoxLayout()
        target_col.setSpacing(2)
        t_hdr = QLabel("PGA TARGET")
        t_hdr.setStyleSheet(f"color: {c.TEXT_MUTED}; font-size: 10px; font-weight: 800; letter-spacing: 0.7px; border: none;")
        t_val = QLabel(self.criterion.standard_text)
        t_val.setStyleSheet(f"color: {c.TEXT_PRIMARY}; font-family: {FONT_MONO}; font-size: 17px; font-weight: 800; border: none;")
        target_col.addWidget(t_hdr)
        target_col.addWidget(t_val)

        mb_layout.addLayout(measured_col, stretch=1)
        mb_layout.addLayout(target_col, stretch=1)
        text_col.addWidget(metric_box)

        # Explanation
        desc = QLabel(self.criterion.description)
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"color: {c.TEXT_SECONDARY}; font-size: 16px; font-weight: 500; "
            "line-height: 1.45; background: transparent; border: none;"
        )
        text_col.addWidget(desc)

        # Coaching Tip Card
        if self.criterion.correction_tip:
            is_err = self.criterion.status == "error"
            tip_bg = c.AMBER_SOFT if is_err else c.EMERALD_SOFT
            tip_border = c.AMBER_BORDER if is_err else c.EMERALD_BORDER
            tip_fg = c.AMBER_TEXT if is_err else c.EMERALD_TEXT
            tip_icon = "bulb" if is_err else "check_circle"
            tip_icon_color = c.AMBER if is_err else c.EMERALD

            tip_box = QFrame()
            tip_box.setStyleSheet(
                f"background-color: {tip_bg}; border: 1px solid {tip_border}; border-radius: 14px;"
            )
            tip_row = QHBoxLayout(tip_box)
            tip_row.setContentsMargins(14, 12, 14, 12)
            tip_row.setSpacing(10)

            bulb_lbl = QLabel()
            bulb_lbl.setPixmap(svg_pixmap(tip_icon, color=tip_icon_color, size=18, stroke_width=2.1))
            bulb_lbl.setFixedSize(20, 20)
            bulb_lbl.setStyleSheet("background: transparent; border: none;")

            tip = QLabel(self.criterion.correction_tip)
            tip.setWordWrap(True)
            tip.setStyleSheet(
                f"color: {tip_fg}; font-size: 15px; font-weight: 700; "
                "line-height: 1.35; background: transparent; border: none;"
            )
            tip_row.addWidget(bulb_lbl)
            tip_row.addWidget(tip, stretch=1)
            text_col.addWidget(tip_box)

        text_col.addStretch()
        root.addLayout(text_col, stretch=36)

    def refresh_image(self):
        if self._source_pixmap.isNull() or self.img_label.size().isEmpty():
            return
        scaled = self._source_pixmap.scaled(
            self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.img_label.setPixmap(round_pixmap(scaled, radius=16))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_image()


# ==============================
# HORIZONTAL AUTO-SLIDING CAROUSEL
# ==============================
class HorizontalCarouselWidget(QWidget):
    """Auto-sliding horizontal carousel holding each slide for config.SLIDE_HOLD_SECONDS."""

    slide_changed = Signal(int, int)
    hold_progress = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.slides_data: list[tuple[np.ndarray, FaultCriterion]] = []
        self.current_idx = 0
        self.current_slide: SwingSlideWidget | None = None
        self.next_slide: SwingSlideWidget | None = None
        self._anim_group: QParallelAnimationGroup | None = None
        self._elapsed_ms = 0
        self._hold_ms = int(config.SLIDE_HOLD_SECONDS * 1000)

        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(50)
        self._tick_timer.timeout.connect(self._on_tick)

    def set_slides(self, slides: list[tuple[np.ndarray, FaultCriterion]]):
        self.stop()
        if self.current_slide is not None:
            self.current_slide.deleteLater()
            self.current_slide = None

        self.slides_data = slides
        self.current_idx = 0
        self._elapsed_ms = 0

        if not self.slides_data:
            return

        self.current_slide = self._create_slide_widget(0)
        self.current_slide.setGeometry(self.rect())
        self.current_slide.show()
        self.current_slide.refresh_image()

        self.slide_changed.emit(0, len(self.slides_data))
        self.hold_progress.emit(0)

        if len(self.slides_data) > 1:
            self._tick_timer.start()

    def stop(self):
        self._tick_timer.stop()
        if self._anim_group is not None:
            self._anim_group.stop()
            self._anim_group = None
        if self.next_slide is not None:
            self.next_slide.deleteLater()
            self.next_slide = None

    def step_slide(self, delta: int):
        if len(self.slides_data) <= 1 or self._anim_group is not None:
            return
        self._elapsed_ms = 0
        self.hold_progress.emit(0)
        target = (self.current_idx + delta) % len(self.slides_data)
        self._animate_to(target, direction=1 if delta >= 0 else -1)

    def _create_slide_widget(self, idx: int) -> SwingSlideWidget:
        img, crit = self.slides_data[idx]
        return SwingSlideWidget(img, crit, idx, len(self.slides_data), parent=self)

    def _on_tick(self):
        if len(self.slides_data) <= 1 or self._anim_group is not None:
            return
        self._elapsed_ms += 50
        pct = min(100, int((self._elapsed_ms / max(1, self._hold_ms)) * 100))
        self.hold_progress.emit(pct)
        if self._elapsed_ms >= self._hold_ms:
            self._elapsed_ms = 0
            self.hold_progress.emit(0)
            target = (self.current_idx + 1) % len(self.slides_data)
            self._animate_to(target, direction=1)

    def _animate_to(self, target_idx: int, direction: int = 1):
        w = max(1, self.width())
        h = max(1, self.height())

        self.next_slide = self._create_slide_widget(target_idx)
        self.next_slide.setGeometry(direction * w, 0, w, h)
        self.next_slide.show()
        self.next_slide.refresh_image()

        anim_old = QPropertyAnimation(self.current_slide, b"pos", self)
        anim_old.setDuration(420)
        anim_old.setStartValue(QPoint(0, 0))
        anim_old.setEndValue(QPoint(-direction * w, 0))
        anim_old.setEasingCurve(QEasingCurve.OutCubic)

        anim_new = QPropertyAnimation(self.next_slide, b"pos", self)
        anim_new.setDuration(420)
        anim_new.setStartValue(QPoint(direction * w, 0))
        anim_new.setEndValue(QPoint(0, 0))
        anim_new.setEasingCurve(QEasingCurve.OutCubic)

        group = QParallelAnimationGroup(self)
        group.addAnimation(anim_old)
        group.addAnimation(anim_new)
        group.finished.connect(lambda: self._finish_slide(target_idx))
        self._anim_group = group
        group.start()

    def _finish_slide(self, target_idx: int):
        if self.current_slide is not None:
            self.current_slide.deleteLater()
        self.current_slide = self.next_slide
        self.next_slide = None
        self._anim_group = None
        self.current_idx = target_idx
        if self.current_slide is not None:
            self.current_slide.setGeometry(self.rect())
            self.current_slide.refresh_image()
        self.slide_changed.emit(self.current_idx, len(self.slides_data))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_slide is not None and self._anim_group is None:
            self.current_slide.setGeometry(self.rect())
            self.current_slide.refresh_image()


# ==============================
# MAIN APPLICATION WINDOW
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

        # Kiosk config (toggled via keyboard if needed)
        self.handedness = config.GOLFER_HANDEDNESS
        self.mirror_enabled = config.MIRROR_CAMERA_DEFAULT
        self.record_duration_sec = config.RECORD_SECONDS

        # Session data
        self.raw_recorded_frames: list[np.ndarray] = []
        self.raw_recorded_timestamps: list[float] = []
        self.analysis_result: SwingAnalysisResult | None = None
        self.countdown_remaining = config.COUNTDOWN_SECONDS
        self.record_start_time = 0.0
        self._dot_labels: list[QLabel] = []

        self._init_ui()
        self._init_timers_and_shortcuts()
        self._start_camera()

    def _init_ui(self):
        self.setWindowTitle("VAR Golf Studio")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.resize(1366, 768)
        self.setStyleSheet(build_app_stylesheet())

        # Zero outer window margins — edge-to-edge full screen
        self.stacked = QStackedWidget()
        self.stacked.setContentsMargins(0, 0, 0, 0)
        self.stacked.addWidget(self._build_live_page())
        self.stacked.addWidget(self._build_result_page())
        self.setCentralWidget(self.stacked)

    def _build_live_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        # Layer 0: Full-bleed edge-to-edge Camera Feed
        self.live_viewer = QLabel("Initializing Camera Feed...")
        self.live_viewer.setAlignment(Qt.AlignCenter)
        self.live_viewer.setStyleSheet(
            f"background-color: {COLORS.BG_CANVAS}; color: {COLORS.TEXT_MUTED}; "
            "font-size: 18px; font-weight: 600;"
        )
        self.live_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.live_viewer, 0, 0)

        # Layer 1: Minimal Floating Glass HUD Overlay
        hud = QWidget()
        hud.setAttribute(Qt.WA_TranslucentBackground)
        hud_layout = QVBoxLayout(hud)
        hud_layout.setContentsMargins(20, 18, 20, 22)
        hud_layout.setSpacing(12)

        # Top-left floating status pill + Top-right exit button (no full bar)
        top_row = QHBoxLayout()
        status_pill = QFrame()
        status_pill.setObjectName("FloatingPill")
        apply_soft_shadow(status_pill, blur=16, y_offset=3, alpha=22)
        sp_layout = QHBoxLayout(status_pill)
        sp_layout.setContentsMargins(10, 6, 14, 6)
        sp_layout.setSpacing(8)

        logo_lbl = QLabel()
        logo_lbl.setPixmap(svg_pixmap("target", color=COLORS.EMERALD, size=18, stroke_width=2.2))
        logo_lbl.setStyleSheet("background: transparent;")

        self.lbl_status_pill = QLabel("Ready")
        self._set_status_pill("Ready", COLORS.EMERALD_TEXT, COLORS.EMERALD_SOFT, COLORS.EMERALD_BORDER)

        self.lbl_mode_pill = QLabel("Right-Handed  ·  10s")
        self.lbl_mode_pill.setObjectName("FloatingMeta")

        sp_layout.addWidget(logo_lbl)
        sp_layout.addWidget(self.lbl_status_pill)
        sp_layout.addWidget(self.lbl_mode_pill)

        self.btn_exit = QPushButton()
        self.btn_exit.setObjectName("ExitBtn")
        self.btn_exit.setIcon(svg_icon("x_close", color=COLORS.TEXT_MUTED, size=16))
        self.btn_exit.setIconSize(QSize(16, 16))
        self.btn_exit.setToolTip("Exit [Ctrl+Q]")
        self.btn_exit.setCursor(Qt.PointingHandCursor)
        self.btn_exit.clicked.connect(self.close)

        top_row.addWidget(status_pill)
        top_row.addStretch()
        top_row.addWidget(self.btn_exit)
        hud_layout.addLayout(top_row)

        hud_layout.addStretch(1)

        # Center Analyzing Card (shown only while processing)
        self.analyze_modal = QFrame()
        self.analyze_modal.setObjectName("AnalyzeModal")
        self.analyze_modal.setFixedWidth(480)
        self.analyze_modal.setVisible(False)
        apply_soft_shadow(self.analyze_modal, blur=28, y_offset=8, alpha=30)
        am_layout = QVBoxLayout(self.analyze_modal)
        am_layout.setContentsMargins(24, 22, 24, 22)
        am_layout.setSpacing(12)

        self.lbl_analyze_title = QLabel("Analyzing 2D + 3D Swing Biomechanics...")
        self.lbl_analyze_title.setAlignment(Qt.AlignCenter)
        self.lbl_analyze_title.setStyleSheet(
            f"color: {COLORS.TEXT_PRIMARY}; font-size: 16px; font-weight: 800; background: transparent;"
        )
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        am_layout.addWidget(self.lbl_analyze_title)
        am_layout.addWidget(self.progress_bar)

        center_row = QHBoxLayout()
        center_row.addStretch()
        center_row.addWidget(self.analyze_modal)
        center_row.addStretch()
        hud_layout.addLayout(center_row)

        hud_layout.addStretch(1)

        # Bottom-center Floating Action Pill
        self.bottom_pill = QFrame()
        self.bottom_pill.setObjectName("FloatingPill")
        apply_soft_shadow(self.bottom_pill, blur=20, y_offset=4, alpha=24)
        bp_layout = QHBoxLayout(self.bottom_pill)
        bp_layout.setContentsMargins(18, 10, 12, 10)
        bp_layout.setSpacing(14)

        self.lbl_live_status = QLabel("Press [SPACE] or Foot Pedal to start 5s countdown")
        self.lbl_live_status.setObjectName("LiveHintLabel")

        self.btn_start = QPushButton(" Start Recording [SPACE]")
        self.btn_start.setObjectName("PrimaryCTA")
        self.btn_start.setIcon(svg_icon("play", color="#FFFFFF", size=17))
        self.btn_start.setIconSize(QSize(17, 17))
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self._on_start_clicked)

        self.btn_stop_early = QPushButton(" Analyze Now [SPACE]")
        self.btn_stop_early.setObjectName("DangerCTA")
        self.btn_stop_early.setIcon(svg_icon("stop", color="#FFFFFF", size=15))
        self.btn_stop_early.setIconSize(QSize(15, 15))
        self.btn_stop_early.setCursor(Qt.PointingHandCursor)
        self.btn_stop_early.setVisible(False)
        self.btn_stop_early.clicked.connect(self._finish_recording_and_analyze)

        self.btn_cancel = QPushButton(" Cancel [ESC]")
        self.btn_cancel.setObjectName("SecondaryBtn")
        self.btn_cancel.setIcon(svg_icon("x_close", color=COLORS.TEXT_PRIMARY, size=15))
        self.btn_cancel.setIconSize(QSize(15, 15))
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)

        bp_layout.addWidget(self.lbl_live_status)
        bp_layout.addWidget(self.btn_cancel)
        bp_layout.addWidget(self.btn_stop_early)
        bp_layout.addWidget(self.btn_start)

        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        bottom_row.addWidget(self.bottom_pill)
        bottom_row.addStretch()
        hud_layout.addLayout(bottom_row)

        grid.addWidget(hud, 0, 0)
        return page

    def _build_result_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("RootCanvas")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        # Horizontal Sliding Carousel Stage
        self.carousel = HorizontalCarouselWidget()
        self.carousel.slide_changed.connect(self._on_carousel_slide_changed)
        self.carousel.hold_progress.connect(self._on_carousel_hold_progress)
        layout.addWidget(self.carousel, stretch=1)

        # Minimal Bottom Footer
        footer = QFrame()
        footer.setObjectName("ControlDock")
        apply_soft_shadow(footer, blur=20, y_offset=4, alpha=14)
        footer_col = QVBoxLayout(footer)
        footer_col.setContentsMargins(20, 10, 20, 12)
        footer_col.setSpacing(8)

        self.slide_timer_bar = QProgressBar()
        self.slide_timer_bar.setObjectName("SlideTimerBar")
        self.slide_timer_bar.setTextVisible(False)
        self.slide_timer_bar.setRange(0, 100)
        self.slide_timer_bar.setValue(0)
        footer_col.addWidget(self.slide_timer_bar)

        row = QHBoxLayout()
        row.setSpacing(12)

        self.lbl_summary_pill = QLabel("Swing Report")
        self.lbl_summary_pill.setStyleSheet(
            f"color: {COLORS.TEXT_PRIMARY}; font-size: 14px; font-weight: 800;"
        )

        self.lbl_view_pill = QLabel("")
        self.lbl_view_pill.setObjectName("MetaPill")

        self.lbl_score_pill = QLabel("")
        self.lbl_score_pill.setObjectName("ScorePill")

        self.dots_container = QHBoxLayout()
        self.dots_container.setSpacing(6)

        self.btn_retry = QPushButton(" New Swing [SPACE]")
        self.btn_retry.setObjectName("PrimaryCTA")
        self.btn_retry.setIcon(svg_icon("rotate_ccw", color="#FFFFFF", size=17))
        self.btn_retry.setIconSize(QSize(17, 17))
        self.btn_retry.setCursor(Qt.PointingHandCursor)
        self.btn_retry.clicked.connect(self._restart_to_live)

        self.btn_result_exit = QPushButton()
        self.btn_result_exit.setObjectName("ExitBtn")
        self.btn_result_exit.setIcon(svg_icon("x_close", color=COLORS.TEXT_MUTED, size=16))
        self.btn_result_exit.setIconSize(QSize(16, 16))
        self.btn_result_exit.setToolTip("Exit [Ctrl+Q]")
        self.btn_result_exit.setCursor(Qt.PointingHandCursor)
        self.btn_result_exit.clicked.connect(self.close)

        row.addWidget(self.lbl_summary_pill)
        row.addWidget(self.lbl_view_pill)
        row.addWidget(self.lbl_score_pill)
        row.addSpacing(8)
        row.addLayout(self.dots_container)
        row.addStretch()
        row.addWidget(self.btn_retry)
        row.addWidget(self.btn_result_exit)
        footer_col.addLayout(row)

        layout.addWidget(footer)
        return page

    # --- TIMERS & KEYBOARD CONTROLS ---
    def _init_timers_and_shortcuts(self):
        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self._on_countdown_tick)

        self._notice_reset_timer = QTimer(self)
        self._notice_reset_timer.setSingleShot(True)
        self._notice_reset_timer.timeout.connect(self._reset_live_controls)

        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self._on_space_shortcut)
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self._on_cancel_clicked)
        QShortcut(QKeySequence("R"), self).activated.connect(self._on_retry_shortcut)
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(lambda: self._step_carousel(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(lambda: self._step_carousel(1))
        QShortcut(QKeySequence("H"), self).activated.connect(self._toggle_handedness)
        QShortcut(QKeySequence("M"), self).activated.connect(self._toggle_mirror)
        QShortcut(QKeySequence(Qt.Key_F11), self).activated.connect(self._toggle_fullscreen)
        QShortcut(QKeySequence("Ctrl+Q"), self).activated.connect(self.close)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _toggle_handedness(self):
        if self.state != self.STATE_LIVE:
            return
        self.handedness = "left" if self.handedness == "right" else "right"
        hand_lbl = "Right-Handed" if self.handedness == "right" else "Left-Handed"
        self.lbl_mode_pill.setText(f"{hand_lbl}  ·  {self.record_duration_sec}s")

    def _toggle_mirror(self):
        self.mirror_enabled = not self.mirror_enabled

    def _step_carousel(self, delta: int):
        if self.state == self.STATE_RESULT:
            self.carousel.step_slide(delta)

    def _on_space_shortcut(self):
        if self.state == self.STATE_LIVE:
            self._on_start_clicked()
        elif self.state == self.STATE_RECORDING:
            self._finish_recording_and_analyze()
        elif self.state == self.STATE_RESULT:
            self._restart_to_live()

    def _on_retry_shortcut(self):
        if self.state == self.STATE_RESULT:
            self._restart_to_live()

    def _set_status_pill(self, text: str, fg: str, bg: str, border: str):
        self.lbl_status_pill.setText(text)
        self.lbl_status_pill.setStyleSheet(
            f"background-color: {bg}; color: {fg}; border: 1px solid {border}; "
            "border-radius: 11px; padding: 3px 10px; font-size: 11px; font-weight: 800;"
        )

    # --- CAMERA & LIVE LOOP ---
    def _start_camera(self):
        self.cam_worker = CameraWorker(self.camera_index)
        self.cam_worker.frame_ready.connect(self._on_camera_frame)
        self.cam_worker.camera_error.connect(self._on_camera_error)
        self.cam_worker.start()

    def _on_camera_error(self, msg: str):
        self.live_viewer.setText(f"Camera Offline\n\n{msg}")
        self.live_viewer.setStyleSheet(
            f"background-color: {COLORS.CRIMSON_SOFT}; color: {COLORS.CRIMSON_TEXT}; "
            "font-size: 16px; font-weight: 600;"
        )
        self.btn_start.setEnabled(False)
        self._set_status_pill("Camera Offline", COLORS.CRIMSON_TEXT, COLORS.CRIMSON_SOFT, COLORS.CRIMSON_BORDER)

    def _on_camera_frame(self, frame_bgr: np.ndarray, timestamp_ms: float):
        if self.state in (self.STATE_ANALYZING, self.STATE_RESULT):
            return

        if self.state == self.STATE_RECORDING:
            self.raw_recorded_frames.append(frame_bgr.copy())
            self.raw_recorded_timestamps.append(timestamp_ms)

            elapsed = time.perf_counter() - self.record_start_time
            remaining = max(0.0, self.record_duration_sec - elapsed)
            self.lbl_live_status.setText(f"Recording Golf Swing... {remaining:.1f}s left")
            if elapsed >= self.record_duration_sec:
                self._finish_recording_and_analyze()
                return

        disp = cv2.flip(frame_bgr, 1) if self.mirror_enabled else frame_bgr
        pix = bgr_to_pixmap(disp)
        target_sz = self.live_viewer.size()
        if not target_sz.isEmpty():
            # Full-bleed edge-to-edge fill with center crop if aspect ratio differs
            pix = pix.scaled(target_sz, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            if pix.width() > target_sz.width() or pix.height() > target_sz.height():
                x_off = max(0, (pix.width() - target_sz.width()) // 2)
                y_off = max(0, (pix.height() - target_sz.height()) // 2)
                pix = pix.copy(x_off, y_off, target_sz.width(), target_sz.height())

        pix = self._paint_live_hud(pix)
        self.live_viewer.setPixmap(pix)

    def _paint_live_hud(self, pix: QPixmap) -> QPixmap:
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w, h = pix.width(), pix.height()

        if self.state == self.STATE_COUNTDOWN:
            painter.fillRect(0, 0, w, h, QColor(15, 23, 42, 110))

            cx, cy = w // 2, h // 2 - 14
            r = 74
            painter.setBrush(QColor(255, 255, 255, 238))
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
            painter.setFont(QFont("Consolas", 60, QFont.Bold))
            painter.drawText(cx - 55, cy - 50, 110, 100, Qt.AlignCenter, str(self.countdown_remaining))

            painter.setBrush(QColor(255, 255, 255, 238))
            painter.setPen(Qt.NoPen)
            pill_w = min(360, w - 40)
            painter.drawRoundedRect((w - pill_w) // 2, cy + 92, pill_w, 40, 20, 20)
            painter.setPen(QColor(COLORS.TEXT_PRIMARY))
            painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
            painter.drawText(
                (w - pill_w) // 2, cy + 92, pill_w, 40, Qt.AlignCenter, "GET INTO ADDRESS POSTURE"
            )

        elif self.state == self.STATE_RECORDING:
            elapsed = time.perf_counter() - self.record_start_time
            remaining = max(0.0, self.record_duration_sec - elapsed)
            ratio = min(1.0, elapsed / max(0.1, float(self.record_duration_sec)))

            # REC Pill top-center
            pill_w = 176
            rx = (w - pill_w) // 2
            painter.setBrush(QColor(255, 255, 255, 235))
            painter.setPen(QPen(QColor(COLORS.CRIMSON_BORDER), 1.5))
            painter.drawRoundedRect(rx, 20, pill_w, 36, 18, 18)

            painter.setBrush(QColor(COLORS.CRIMSON))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(rx + 14, 32, 12, 12)

            painter.setPen(QColor(COLORS.TEXT_PRIMARY))
            painter.setFont(QFont("Consolas", 13, QFont.Bold))
            painter.drawText(rx + 34, 20, 134, 36, Qt.AlignVCenter | Qt.AlignLeft, f"REC  {remaining:04.1f}s")

            # Edge-to-edge thin bottom recording bar
            bar_w = int(w * ratio)
            painter.setBrush(QColor(255, 255, 255, 140))
            painter.drawRect(0, h - 5, w, 5)
            painter.setBrush(QColor(COLORS.EMERALD_BRIGHT))
            painter.drawRect(0, h - 5, max(8, bar_w), 5)

        painter.end()
        return pix

    # --- CAPTURE FLOW ---
    def _on_start_clicked(self):
        if self.state != self.STATE_LIVE:
            return
        self._notice_reset_timer.stop()
        self.countdown_remaining = config.COUNTDOWN_SECONDS
        self.state = self.STATE_COUNTDOWN

        self.btn_start.setVisible(False)
        self.btn_cancel.setVisible(True)
        self.btn_stop_early.setVisible(False)

        self._set_status_pill(
            f"Countdown {self.countdown_remaining}s",
            COLORS.AMBER_TEXT,
            COLORS.AMBER_SOFT,
            COLORS.AMBER_BORDER,
        )
        self.lbl_live_status.setText(f"Get ready... Recording starts in {self.countdown_remaining}s!")
        self.countdown_timer.start()

    def _on_countdown_tick(self):
        self.countdown_remaining -= 1
        if self.countdown_remaining > 0:
            self._set_status_pill(
                f"Countdown {self.countdown_remaining}s",
                COLORS.AMBER_TEXT,
                COLORS.AMBER_SOFT,
                COLORS.AMBER_BORDER,
            )
            self.lbl_live_status.setText(f"Get ready... Recording starts in {self.countdown_remaining}s!")
        else:
            self.countdown_timer.stop()
            self._begin_recording()

    def _begin_recording(self):
        self.raw_recorded_frames.clear()
        self.raw_recorded_timestamps.clear()
        self.record_start_time = time.perf_counter()
        self.state = self.STATE_RECORDING
        self.btn_stop_early.setVisible(True)
        self._set_status_pill("Recording", COLORS.CRIMSON_TEXT, COLORS.CRIMSON_SOFT, COLORS.CRIMSON_BORDER)
        self.lbl_live_status.setText("RECORDING! Perform your natural golf swing...")

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
        self.analyze_modal.setVisible(False)
        self.bottom_pill.setVisible(True)
        self.btn_start.setVisible(True)
        self.btn_start.setEnabled(True)
        self.btn_cancel.setVisible(False)
        self.btn_stop_early.setVisible(False)
        self.progress_bar.setValue(0)
        self._set_status_pill("Ready", COLORS.EMERALD_TEXT, COLORS.EMERALD_SOFT, COLORS.EMERALD_BORDER)
        self.lbl_live_status.setText("Press [SPACE] or Foot Pedal to start 5s countdown")

    def _finish_recording_and_analyze(self):
        if self.state != self.STATE_RECORDING:
            return
        self.state = self.STATE_ANALYZING
        self.bottom_pill.setVisible(False)
        self.analyze_modal.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_analyze_title.setText(
            f"Analyzing 2D + 3D Biomechanics ({len(self.raw_recorded_frames)} frames)..."
        )
        self._set_status_pill("Analyzing", COLORS.BLUE_TEXT, COLORS.BLUE_SOFT, COLORS.BLUE_BORDER)

        self.analysis_worker = AnalysisWorker(
            self.raw_recorded_frames, self.raw_recorded_timestamps, self.handedness
        )
        self.analysis_worker.progress_updated.connect(self.progress_bar.setValue)
        self.analysis_worker.analysis_finished.connect(self._on_analysis_finished)
        self.analysis_worker.analysis_failed.connect(self._on_analysis_failed)
        self.analysis_worker.start()

    def _show_nonblocking_live_notice(self, pill_txt: str, status_msg: str):
        """Show auto-recovering status on live screen so kiosk never blocks on mouse dialogs."""
        self.state = self.STATE_LIVE
        self._reset_live_controls()
        self._set_status_pill(pill_txt, COLORS.AMBER_TEXT, COLORS.AMBER_SOFT, COLORS.AMBER_BORDER)
        self.lbl_live_status.setText(status_msg)
        self._notice_reset_timer.start(4000)

    def _on_analysis_failed(self, err: str):
        self._show_nonblocking_live_notice(
            "Analysis Retry",
            f"Analysis interrupted ({err}). Press [SPACE] to try again.",
        )

    def _on_analysis_finished(self, result: SwingAnalysisResult):
        self.analysis_result = result
        if not result.person_detected or not result.criteria:
            self._show_nonblocking_live_notice(
                "No Player Detected",
                "Player was not clearly visible head-to-toe. Step back slightly and press [SPACE].",
            )
            return

        self.state = self.STATE_RESULT
        self.stacked.setCurrentIndex(1)
        QTimer.singleShot(0, self._populate_result_carousel)

    # --- RESULT CAROUSEL ---
    def _get_phase_label(self, frame_idx: int) -> str:
        res = self.analysis_result
        if abs(frame_idx - res.address_idx) <= 3:
            return f"ADDRESS | Frame {frame_idx}/{res.total_frames}"
        if abs(frame_idx - res.top_idx) <= 2:
            return f"TOP OF SWING | Frame {frame_idx}/{res.total_frames}"
        if abs(frame_idx - res.impact_idx) <= 2:
            return f"IMPACT | Frame {frame_idx}/{res.total_frames}"
        if abs(frame_idx - res.finish_idx) <= 3:
            return f"FINISH | Frame {frame_idx}/{res.total_frames}"
        return f"FAULT MOMENT | Frame {frame_idx}/{res.total_frames}"

    def _render_criterion_image(self, crit: FaultCriterion) -> np.ndarray:
        res = self.analysis_result
        idx = max(0, min(crit.frame_idx, res.total_frames - 1))
        raw_frame = self.raw_recorded_frames[idx]
        lm = res.landmarks_per_frame[idx]
        _, w, _ = raw_frame.shape

        if self.mirror_enabled:
            disp_frame = cv2.flip(raw_frame, 1)
            disp_lm = mirror_landmarks(lm, w)
            disp_crit = mirror_criterion(crit, w)
        else:
            disp_frame = raw_frame
            disp_lm = lm
            disp_crit = crit

        return render_annotated_snapshot(
            disp_frame, disp_lm, disp_crit, phase_label=self._get_phase_label(idx)
        )

    def _populate_result_carousel(self):
        if self.state != self.STATE_RESULT or not self.analysis_result:
            return
        res = self.analysis_result

        self.lbl_view_pill.setText(
            "Face-On View" if res.camera_view == "face_on" else "Down-the-Line (DTL)"
        )
        self.lbl_score_pill.setText(f"Passed {res.score_passed}/{res.score_evaluated}")

        errors = [c for c in res.criteria if c.status == "error"]
        if errors:
            noun = "Fault" if len(errors) == 1 else "Faults"
            self.lbl_summary_pill.setText(f"{len(errors)} {noun} Detected")
            self.lbl_summary_pill.setStyleSheet(
                f"color: {COLORS.CRIMSON_TEXT}; font-size: 15px; font-weight: 800;"
            )
        else:
            self.lbl_summary_pill.setText("Clean Swing!")
            self.lbl_summary_pill.setStyleSheet(
                f"color: {COLORS.EMERALD_TEXT}; font-size: 15px; font-weight: 800;"
            )

        slides = [(self._render_criterion_image(crit), crit) for crit in res.criteria]
        self._rebuild_pagination_dots(len(slides))
        self.carousel.set_slides(slides)

    def _rebuild_pagination_dots(self, count: int):
        while self.dots_container.count():
            item = self.dots_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._dot_labels.clear()

        for _ in range(count):
            dot = QLabel()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(f"background-color: {COLORS.BORDER_MEDIUM}; border-radius: 5px;")
            self._dot_labels.append(dot)
            self.dots_container.addWidget(dot)

    def _on_carousel_slide_changed(self, current: int, _total: int):
        for i, dot in enumerate(self._dot_labels):
            if i == current:
                dot.setFixedSize(26, 10)
                dot.setStyleSheet(f"background-color: {COLORS.EMERALD}; border-radius: 5px;")
            else:
                dot.setFixedSize(10, 10)
                dot.setStyleSheet(f"background-color: {COLORS.BORDER_MEDIUM}; border-radius: 5px;")

    def _on_carousel_hold_progress(self, pct: int):
        self.slide_timer_bar.setValue(pct)

    def _restart_to_live(self):
        self.carousel.stop()
        self.raw_recorded_frames.clear()
        self.raw_recorded_timestamps.clear()
        self.analysis_result = None
        self.state = self.STATE_LIVE
        self._reset_live_controls()
        self.stacked.setCurrentIndex(0)

    def closeEvent(self, event):
        self.countdown_timer.stop()
        self._notice_reset_timer.stop()
        self.carousel.stop()
        if hasattr(self, "cam_worker"):
            self.cam_worker.stop()
        super().closeEvent(event)
