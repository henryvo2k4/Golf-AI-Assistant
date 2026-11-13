"""
GIAO DIỆN DESKTOP CHÍNH - HỆ THỐNG TRỢ LÝ AI GOLF (VAR GOLF)
Xây dựng bằng PySide6 (Qt6) với thiết kế Dark Mode hiện đại, tối ưu cho màn hình 10.5" và PC.
"""

import copy
import os
import time
import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
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
        # Trên Windows, thử DirectShow trước để mở nhanh hơn
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            self.camera_error.emit(
                f"Không thể kết nối với Webcam (ID={self.camera_index}). "
                "Vui lòng kiểm tra dây cắm USB hoặc quyền truy cập Camera trong Windows!"
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

            # Lưu lại file video MP4 cú đánh gần nhất để lưu trữ/kiểm tra
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
# HÀM TIỆN ÍCH CHUYỂN ĐỔI ẢNH
# ==============================
def bgr_to_pixmap(frame_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(q_img)


def mirror_landmarks(lm_dict: dict | None, width: int) -> dict | None:
    if lm_dict is None:
        return None
    mirrored = {}
    for k, (x, y, z, vis) in lm_dict.items():
        mirrored[k] = (float(width - x), y, z, vis)
    return mirrored


def mirror_criterion(criterion: FaultCriterion, width: int) -> FaultCriterion:
    c = copy.deepcopy(criterion)
    ad = c.annotation_data
    for key in ("shoulder", "elbow", "wrist", "nose", "trail_hip", "trail_ankle", "mid_hip", "mid_shoulder", "hip", "knee", "ankle"):
        if key in ad and ad[key] is not None:
            pt = ad[key]
            ad[key] = (float(width - pt[0]), pt[1])
    if "limit_x" in ad:
        ad["limit_x"] = float(width - ad["limit_x"])
    return c


# ==============================
# THẺ HIỂN THỊ TỪNG LỖI (UI CARD)
# ==============================
class FaultCardWidget(QFrame):
    clicked = Signal(object)

    def __init__(self, criterion: FaultCriterion, is_selected: bool = False):
        super().__init__()
        self.criterion = criterion
        self.setCursor(Qt.PointingHandCursor)
        self._build_ui(is_selected)

    def _build_ui(self, is_selected: bool):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        if self.criterion.status == "error":
            badge_bg = "#DC2626"
            badge_txt = "LỖI KỸ THUẬT"
            border_color = "#EF4444" if is_selected else "#7F1D1D"
            bg_color = "#2A1519" if is_selected else "#1E161A"
        elif self.criterion.status == "passed":
            badge_bg = "#059669"
            badge_txt = "ĐẠT CHUẨN PGA"
            border_color = "#10B981" if is_selected else "#064E3B"
            bg_color = "#122922" if is_selected else "#13201E"
        else:
            badge_bg = "#D97706"
            badge_txt = "CHƯA RÕ KHỚP"
            border_color = "#F59E0B" if is_selected else "#475569"
            bg_color = "#242016" if is_selected else "#1E293B"

        self.setStyleSheet(
            f"""
            FaultCardWidget {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 10px;
            }}
            FaultCardWidget:hover {{
                border: 2px solid #38BDF8;
            }}
            """
        )

        # Dòng 1: Badge + Số đo
        top_row = QHBoxLayout()
        badge = QLabel(badge_txt)
        badge.setStyleSheet(
            f"background-color: {badge_bg}; color: white; font-weight: bold; "
            "font-size: 11px; padding: 3px 8px; border-radius: 5px;"
        )
        badge.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        metric = QLabel(f"Đo: {self.criterion.measured_text}  |  Chuẩn: {self.criterion.standard_text}")
        metric.setStyleSheet("color: #CBD5E1; font-size: 12px; font-weight: 600;")
        metric.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        top_row.addWidget(badge)
        top_row.addStretch()
        top_row.addWidget(metric)
        layout.addLayout(top_row)

        # Dòng 2: Tên lỗi
        title = QLabel(self.criterion.title)
        title.setWordWrap(True)
        title.setStyleSheet("color: #F8FAFC; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        # Dòng 3: Giải thích chi tiết
        desc = QLabel(self.criterion.description)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #94A3B8; font-size: 12px;")
        layout.addWidget(desc)

        # Dòng 4: Mẹo sửa (nếu là lỗi)
        if self.criterion.status == "error":
            tip = QLabel(f"💡 Cách sửa: {self.criterion.correction_tip}")
            tip.setWordWrap(True)
            tip.setStyleSheet("color: #FCD34D; font-size: 12px; font-weight: 500; margin-top: 2px;")
            layout.addWidget(tip)

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
        self.replay_speed = 0.5  # Mặc định 0.5x chậm

        self._init_ui()
        self._init_timers_and_shortcuts()
        self._start_camera()

    def _init_ui(self):
        self.setWindowTitle("Trợ Lý AI Phân Tích Tư Thế Golf (VAR Golf)")
        self.resize(1280, 820)
        self.setMinimumSize(960, 650)

        # Giao diện Dark Mode hiện đại
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #0F172A;
                color: #F8FAFC;
                font-family: "Segoe UI", Arial, sans-serif;
            }
            QPushButton {
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 14px;
            }
            QComboBox, QCheckBox {
                font-size: 13px;
                color: #E2E8F0;
            }
            QComboBox {
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 5px 10px;
            }
            QProgressBar {
                border: 1px solid #334155;
                border-radius: 6px;
                background-color: #1E293B;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #10B981;
                border-radius: 5px;
            }
            """
        )

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 12, 16, 14)
        main_layout.setSpacing(12)

        # --- HEADER BAR ---
        header = QFrame()
        header.setStyleSheet("background-color: #1E293B; border-radius: 10px;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 10)

        app_title = QLabel("⛳  TRỢ LÝ AI PHÂN TÍCH TƯ THẾ GOLF (VAR GOLF)")
        app_title.setStyleSheet("font-size: 17px; font-weight: bold; color: #34D399;")
        header_layout.addWidget(app_title)

        header_layout.addStretch()

        # Tay thuận
        lbl_hand = QLabel("Tay thuận:")
        self.combo_handedness = QComboBox()
        self.combo_handedness.addItem("Thuận tay phải", "right")
        self.combo_handedness.addItem("Thuận tay trái", "left")
        header_layout.addWidget(lbl_hand)
        header_layout.addWidget(self.combo_handedness)

        # Thời gian quay
        lbl_dur = QLabel("  Thời gian quay:")
        self.combo_duration = QComboBox()
        self.combo_duration.addItem("10 giây (Khuyên dùng)", 10)
        self.combo_duration.addItem("8 giây", 8)
        self.combo_duration.addItem("12 giây", 12)
        self.combo_duration.addItem("5 giây (Test nhanh)", 5)
        self.combo_duration.currentIndexChanged.connect(self._on_duration_changed)
        header_layout.addWidget(lbl_dur)
        header_layout.addWidget(self.combo_duration)

        # Lật gương
        self.chk_mirror = QCheckBox("Lật gương Camera")
        self.chk_mirror.setChecked(config.MIRROR_CAMERA_DEFAULT)
        header_layout.addSpacing(12)
        header_layout.addWidget(self.chk_mirror)

        main_layout.addWidget(header)

        # --- STACKED PAGES: PAGE 0 (CAMERA LIVE/RECORD/ANALYZE) | PAGE 1 (RESULT) ---
        self.stacked = QStackedWidget()
        main_layout.addWidget(self.stacked, stretch=1)

        self._build_camera_page()
        self._build_result_page()

    def _build_camera_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Khung hiển thị Camera Live
        self.camera_label = QLabel("Đang khởi động Camera...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet(
            "background-color: #020617; border: 2px solid #334155; border-radius: 12px; font-size: 16px; color: #94A3B8;"
        )
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setMinimumHeight(420)
        layout.addWidget(self.camera_label, stretch=1)

        # Thanh tiến trình ghi hình / phân tích
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(26)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Thanh nút điều khiển dưới cùng
        ctrl_bar = QHBoxLayout()
        ctrl_bar.setSpacing(12)

        self.status_hint = QLabel("💡 Đứng cách camera ~2m - 2.5m (thấy toàn thân để bắt đủ 5 lỗi). Bấm nút hoặc dậm bàn đạp để bắt đầu!")
        self.status_hint.setStyleSheet("color: #CBD5E1; font-size: 13px;")
        self.status_hint.setWordWrap(True)
        ctrl_bar.addWidget(self.status_hint, stretch=1)

        self.btn_action = QPushButton("▶  BẮT ĐẦU SWING (SPACE)")
        self.btn_action.setFixedHeight(52)
        self.btn_action.setMinimumWidth(280)
        self.btn_action.setStyleSheet(
            "background-color: #10B981; color: #022C22; font-size: 16px;"
        )
        self.btn_action.clicked.connect(self._on_action_button_clicked)
        ctrl_bar.addWidget(self.btn_action)

        self.btn_cancel = QPushButton("✕  HỦY (ESC)")
        self.btn_cancel.setFixedHeight(52)
        self.btn_cancel.setStyleSheet("background-color: #475569; color: white;")
        self.btn_cancel.clicked.connect(self._cancel_to_live)
        self.btn_cancel.setVisible(False)
        ctrl_bar.addWidget(self.btn_cancel)

        layout.addLayout(ctrl_bar)
        self.stacked.addWidget(page)

    def _build_result_page(self):
        page = QWidget()
        root_layout = QHBoxLayout(page)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(14)

        # --- CỘT TRÁI (68%): ẢNH KHOANH VÙNG LỖI & VIDEO SLOW-MOTION ---
        left_panel = QFrame()
        left_panel.setStyleSheet("background-color: #1E293B; border-radius: 12px;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        # Thanh chuyển chế độ Ảnh Tĩnh <-> Video Slow-Motion
        mode_bar = QHBoxLayout()
        self.btn_tab_snapshot = QPushButton("📸  ẢNH KHOANH VÙNG LỖI")
        self.btn_tab_replay = QPushButton("🎬  XEM LẠI SLOW-MOTION")
        self.btn_tab_snapshot.setFixedHeight(38)
        self.btn_tab_replay.setFixedHeight(38)
        self.btn_tab_snapshot.clicked.connect(lambda: self._switch_result_tab(0))
        self.btn_tab_replay.clicked.connect(lambda: self._switch_result_tab(1))
        mode_bar.addWidget(self.btn_tab_snapshot)
        mode_bar.addWidget(self.btn_tab_replay)
        mode_bar.addStretch()
        left_layout.addLayout(mode_bar)

        # Khung hiển thị nội dung kết quả (Ảnh hoặc Video)
        self.result_viewer = QLabel()
        self.result_viewer.setAlignment(Qt.AlignCenter)
        self.result_viewer.setStyleSheet("background-color: #020617; border-radius: 10px;")
        self.result_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_viewer.setMinimumHeight(420)
        left_layout.addWidget(self.result_viewer, stretch=1)

        # Bộ điều khiển riêng cho Tab Ảnh Tĩnh vs Video Slow-Motion
        self.result_controls_stack = QStackedWidget()
        left_layout.addWidget(self.result_controls_stack)

        # Tab 0 Controls: Nút chuyển nhanh khung hình các pha
        snap_ctrl = QWidget()
        snap_ctrl_layout = QHBoxLayout(snap_ctrl)
        snap_ctrl_layout.setContentsMargins(0, 4, 0, 0)
        self.lbl_snap_info = QLabel("Đang hiển thị khoảnh khắc xảy ra lỗi được khoanh tròn đỏ.")
        self.lbl_snap_info.setStyleSheet("color: #CBD5E1; font-size: 13px;")
        snap_ctrl_layout.addWidget(self.lbl_snap_info)
        snap_ctrl_layout.addStretch()

        self.btn_jump_top = QPushButton("🏔️ Đỉnh Swing")
        self.btn_jump_top.setStyleSheet("background-color: #334155; color: white; font-size: 12px; padding: 6px 12px;")
        self.btn_jump_top.clicked.connect(lambda: self._show_phase_snapshot("top"))
        snap_ctrl_layout.addWidget(self.btn_jump_top)

        self.btn_jump_impact = QPushButton("💥 Chạm bóng")
        self.btn_jump_impact.setStyleSheet("background-color: #334155; color: white; font-size: 12px; padding: 6px 12px;")
        self.btn_jump_impact.clicked.connect(lambda: self._show_phase_snapshot("impact"))
        snap_ctrl_layout.addWidget(self.btn_jump_impact)
        self.result_controls_stack.addWidget(snap_ctrl)

        # Tab 1 Controls: Video Slow-Motion Controls
        replay_ctrl = QWidget()
        replay_ctrl_layout = QHBoxLayout(replay_ctrl)
        replay_ctrl_layout.setContentsMargins(0, 4, 0, 0)

        self.btn_play_pause = QPushButton("▶ Phát")
        self.btn_play_pause.setStyleSheet("background-color: #3B82F6; color: white; min-width: 90px;")
        self.btn_play_pause.clicked.connect(self._toggle_replay_play)
        replay_ctrl_layout.addWidget(self.btn_play_pause)

        self.replay_slider = QSlider(Qt.Horizontal)
        self.replay_slider.setMinimum(0)
        self.replay_slider.valueChanged.connect(self._on_replay_slider_moved)
        replay_ctrl_layout.addWidget(self.replay_slider, stretch=1)

        self.lbl_replay_time = QLabel("0 / 0")
        self.lbl_replay_time.setStyleSheet("color: #CBD5E1; font-size: 12px; min-width: 75px;")
        replay_ctrl_layout.addWidget(self.lbl_replay_time)

        self.combo_speed = QComboBox()
        self.combo_speed.addItem("0.25x (Siêu chậm)", 0.25)
        self.combo_speed.addItem("0.5x (Chậm)", 0.5)
        self.combo_speed.addItem("1.0x (Thực tế)", 1.0)
        self.combo_speed.setCurrentIndex(1)
        self.combo_speed.currentIndexChanged.connect(self._on_speed_changed)
        replay_ctrl_layout.addWidget(self.combo_speed)

        self.result_controls_stack.addWidget(replay_ctrl)

        root_layout.addWidget(left_panel, stretch=68)

        # --- CỘT PHẢI (32%): BẢNG DANH SÁCH LỖI & NÚT ĐÁNH LẠI ---
        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: #1E293B; border-radius: 12px;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)

        self.lbl_summary_banner = QLabel("KẾT QUẢ PHÂN TÍCH")
        self.lbl_summary_banner.setAlignment(Qt.AlignCenter)
        self.lbl_summary_banner.setWordWrap(True)
        self.lbl_summary_banner.setStyleSheet(
            "background-color: #0F172A; padding: 12px; border-radius: 8px; font-size: 16px; font-weight: bold;"
        )
        right_layout.addWidget(self.lbl_summary_banner)

        lbl_tip_click = QLabel("👆 Bấm vào từng thẻ bên dưới để soi ảnh khoanh vùng:")
        lbl_tip_click.setStyleSheet("color: #94A3B8; font-size: 12px;")
        right_layout.addWidget(lbl_tip_click)

        # Vùng cuộn chứa danh sách 5 thẻ lỗi
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background-color: transparent;")
        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 4, 0)
        self.cards_layout.setSpacing(10)
        self.cards_layout.addStretch()
        scroll.setWidget(self.cards_container)
        right_layout.addWidget(scroll, stretch=1)

        # Nút đánh lượt mới
        self.btn_new_swing = QPushButton("🔄  ĐÁNH LƯỢT MỚI (SPACE)")
        self.btn_new_swing.setFixedHeight(52)
        self.btn_new_swing.setStyleSheet("background-color: #10B981; color: #022C22; font-size: 16px;")
        self.btn_new_swing.clicked.connect(self._restart_to_live)
        right_layout.addWidget(self.btn_new_swing)

        root_layout.addWidget(right_panel, stretch=32)
        self.stacked.addWidget(page)

    def _init_timers_and_shortcuts(self):
        # Timer đếm ngược từng giây
        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self._on_countdown_tick)

        # Timer phát lại video Slow-Motion
        self.replay_timer = QTimer(self)
        self.replay_timer.timeout.connect(self._on_replay_tick)

        # Phím tắt Space (Bàn đạp chân USB) và ESC
        self.shortcut_space = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut_space.activated.connect(self._on_space_pressed)

        self.shortcut_esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.shortcut_esc.activated.connect(self._cancel_to_live)

    def _start_camera(self):
        self.cam_worker = CameraWorker(self.camera_index)
        self.cam_worker.frame_ready.connect(self._on_camera_frame)
        self.cam_worker.camera_error.connect(self._on_camera_error)
        self.cam_worker.start()

    def _on_camera_error(self, msg: str):
        self.camera_label.setText(f"⚠️ {msg}")
        QMessageBox.warning(self, "Lỗi Camera", msg)

    def _on_duration_changed(self):
        self.record_duration_sec = int(self.combo_duration.currentData())

    def _on_space_pressed(self):
        if self.state == self.STATE_RESULT:
            self._restart_to_live()
        else:
            self._on_action_button_clicked()

    def _on_action_button_clicked(self):
        if self.state == self.STATE_LIVE:
            self._begin_countdown()
        elif self.state == self.STATE_RECORDING:
            self._finish_recording_and_analyze()

    def _begin_countdown(self):
        self.state = self.STATE_COUNTDOWN
        self.countdown_remaining = config.COUNTDOWN_SECONDS
        self.btn_action.setEnabled(False)
        self.btn_action.setText(f"⏳ CHUẨN BỊ... ({self.countdown_remaining}s)")
        self.btn_action.setStyleSheet("background-color: #F59E0B; color: #1E293B; font-size: 16px;")
        self.btn_cancel.setVisible(True)
        self.combo_handedness.setEnabled(False)
        self.combo_duration.setEnabled(False)
        self.countdown_timer.start()

    def _on_countdown_tick(self):
        self.countdown_remaining -= 1
        if self.countdown_remaining > 0:
            self.btn_action.setText(f"⏳ CHUẨN BỊ... ({self.countdown_remaining}s)")
        else:
            self.countdown_timer.stop()
            self._begin_recording()

    def _begin_recording(self):
        self.state = self.STATE_RECORDING
        self.raw_recorded_frames = []
        self.raw_recorded_timestamps = []
        self.record_start_time = time.perf_counter()

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #EF4444; border-radius: 5px; }")

        self.btn_action.setEnabled(True)
        self.btn_action.setText("■  ĐÃ ĐÁNH XONG - PHÂN TÍCH NGAY (SPACE)")
        self.btn_action.setStyleSheet("background-color: #EF4444; color: white; font-size: 15px;")

    def _cancel_to_live(self):
        if self.state in (self.STATE_COUNTDOWN, self.STATE_RECORDING):
            self.countdown_timer.stop()
            self.state = self.STATE_LIVE
            self.raw_recorded_frames.clear()
            self.raw_recorded_timestamps.clear()
            self._reset_live_controls()

    def _reset_live_controls(self):
        self.progress_bar.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.btn_action.setEnabled(True)
        self.btn_action.setText("▶  BẮT ĐẦU SWING (SPACE)")
        self.btn_action.setStyleSheet("background-color: #10B981; color: #022C22; font-size: 16px;")
        self.combo_handedness.setEnabled(True)
        self.combo_duration.setEnabled(True)

    def _on_camera_frame(self, frame_bgr: np.ndarray, timestamp_ms: float):
        if self.state not in (self.STATE_LIVE, self.STATE_COUNTDOWN, self.STATE_RECORDING):
            return

        # Nếu đang ghi hình: lưu ảnh GỐC (không lật gương) để MediaPipe nhận diện chuẩn Trái/Phải
        elapsed_rec = 0.0
        if self.state == self.STATE_RECORDING:
            elapsed_rec = time.perf_counter() - self.record_start_time
            self.raw_recorded_frames.append(frame_bgr.copy())
            self.raw_recorded_timestamps.append(elapsed_rec * 1000.0)

            pct = min(100, int((elapsed_rec / self.record_duration_sec) * 100))
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f"● ĐANG GHI HÌNH: {elapsed_rec:.1f}s / {self.record_duration_sec}s")

            if elapsed_rec >= self.record_duration_sec:
                self._finish_recording_and_analyze()
                return

        # Tạo khung hình hiển thị trên màn hình
        display_frame = cv2.flip(frame_bgr, 1) if self.chk_mirror.isChecked() else frame_bgr.copy()
        pixmap = bgr_to_pixmap(display_frame)

        # Vẽ Overlay Đếm ngược hoặc Đang ghi hình bằng QPainter cực nét
        if self.state == self.STATE_COUNTDOWN:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            w, h = pixmap.width(), pixmap.height()

            # Làm tối nhẹ nền để số đếm ngược nổi bật
            painter.fillRect(0, 0, w, h, QColor(0, 0, 0, 95))

            # Vòng tròn hổ phách ở giữa
            cx, cy, r = w // 2, h // 2 - 20, 85
            painter.setPen(QPen(QColor("#F59E0B"), 6))
            painter.setBrush(QColor(15, 23, 42, 210))
            painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)

            # Số đếm ngược
            painter.setPen(QColor("#FBBF24"))
            painter.setFont(QFont("Segoe UI", 64, QFont.Bold))
            painter.drawText(cx - r, cy - r, r * 2, r * 2, Qt.AlignCenter, str(self.countdown_remaining))

            # Dòng nhắc nhở
            painter.setPen(QColor("#FFFFFF"))
            painter.setFont(QFont("Segoe UI", 20, QFont.Bold))
            painter.drawText(0, cy + r + 15, w, 45, Qt.AlignCenter, "HÃY VÀO VỊ TRÍ THẢM & CHUẨN BỊ TƯ THẾ!")
            painter.end()

        elif self.state == self.STATE_RECORDING:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            w = pixmap.width()
            painter.setBrush(QColor(185, 28, 28, 220))
            painter.setPen(QPen(QColor("#FCA5A5"), 2))
            painter.drawRoundedRect(20, 20, 320, 46, 10, 10)
            painter.setPen(QColor("#FFFFFF"))
            painter.setFont(QFont("Segoe UI", 15, QFont.Bold))
            remain = max(0.0, self.record_duration_sec - elapsed_rec)
            painter.drawText(34, 20, 300, 46, Qt.AlignVCenter, f"● ĐANG QUAY SWING ({remain:.1f}s)")
            painter.end()

        scaled = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled)

    def _finish_recording_and_analyze(self):
        if self.state != self.STATE_RECORDING:
            return

        self.state = self.STATE_ANALYZING
        self.btn_action.setEnabled(False)
        self.btn_action.setText("🧠 ĐANG PHÂN TÍCH CÚ SWING BẰNG AI...")
        self.btn_action.setStyleSheet("background-color: #3B82F6; color: white; font-size: 15px;")
        self.btn_cancel.setVisible(False)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #3B82F6; border-radius: 5px; }")
        self.progress_bar.setFormat("Đang quét khung xương MediaPipe AI: %p%")

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
                "Không tìm thấy người chơi",
                "AI không nhìn thấy rõ cơ thể người chơi trong đoạn video vừa quay.\n"
                "Hãy đảm bảo ánh sáng đủ và đứng lùi ra trước camera nhé ạ!",
            )
            self.state = self.STATE_LIVE
            self._reset_live_controls()
            return

        self.state = self.STATE_RESULT
        self._populate_result_ui()
        self.stacked.setCurrentIndex(1)

    def _populate_result_ui(self):
        res = self.analysis_result
        errors = [c for c in res.criteria if c.status == "error"]

        if errors:
            self.lbl_summary_banner.setText(
                f"⚠️  PHÁT HIỆN {len(errors)} LỖI KỸ THUẬT  (Đạt {res.score_passed}/{res.score_evaluated})"
            )
            self.lbl_summary_banner.setStyleSheet(
                "background-color: #7F1D1D; color: #FEF2F2; border: 2px solid #EF4444; "
                "padding: 12px; border-radius: 8px; font-size: 15px; font-weight: bold;"
            )
        else:
            self.lbl_summary_banner.setText(
                f"✅  XUẤT SẮC! ĐẠT CHUẨN {res.score_passed}/{res.score_evaluated} TIÊU CHÍ"
            )
            self.lbl_summary_banner.setStyleSheet(
                "background-color: #064E3B; color: #ECFDF5; border: 2px solid #10B981; "
                "padding: 12px; border-radius: 8px; font-size: 15px; font-weight: bold;"
            )

        # Chọn thẻ mặc định (lỗi đầu tiên nếu có, hoặc tiêu chí đầu tiên)
        self.selected_criterion = errors[0] if errors else res.criteria[0]
        self._rebuild_fault_cards()

        # Cấu hình thanh tua Slow-Motion
        self.replay_slider.blockSignals(True)
        self.replay_slider.setMaximum(max(0, res.total_frames - 1))
        self.replay_slider.setValue(res.top_idx)
        self.replay_slider.blockSignals(False)
        self.replay_frame_idx = res.top_idx
        self.replay_playing = False
        self.btn_play_pause.setText("▶ Phát")

        # Mặc định mở Tab 0: Ảnh khoanh vùng lỗi
        self._switch_result_tab(0)

    def _rebuild_fault_cards(self):
        # Xóa card cũ
        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for crit in self.analysis_result.criteria:
            is_sel = (self.selected_criterion is not None and crit.code == self.selected_criterion.code)
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
            # Dừng video replay nếu đang phát
            self.replay_timer.stop()
            self.replay_playing = False
            self.btn_play_pause.setText("▶ Phát")
            self.btn_tab_snapshot.setStyleSheet("background-color: #10B981; color: #022C22;")
            self.btn_tab_replay.setStyleSheet("background-color: #334155; color: #CBD5E1;")
            self._render_selected_snapshot()
        else:
            self.btn_tab_snapshot.setStyleSheet("background-color: #334155; color: #CBD5E1;")
            self.btn_tab_replay.setStyleSheet("background-color: #3B82F6; color: white;")
            self._render_replay_frame(self.replay_frame_idx)

    def _get_phase_label(self, frame_idx: int) -> str:
        res = self.analysis_result
        if abs(frame_idx - res.top_idx) <= 2:
            return f"🏔️ ĐỈNH SWING (Frame {frame_idx}/{res.total_frames})"
        if abs(frame_idx - res.impact_idx) <= 2:
            return f"💥 CHẠM BÓNG (Frame {frame_idx}/{res.total_frames})"
        if abs(frame_idx - res.address_idx) <= 3:
            return f"📍 CHUẨN BỊ (Frame {frame_idx}/{res.total_frames})"
        return f"⏱️ KHOẢNH KHẮC LỖI (Frame {frame_idx}/{res.total_frames})"

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
        self.result_viewer.setPixmap(
            pix.scaled(self.result_viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.lbl_snap_info.setText(f"Đang soi: {crit.title} ({phase_txt})")

    def _show_phase_snapshot(self, phase: str):
        if not self.analysis_result:
            return
        idx = self.analysis_result.top_idx if phase == "top" else self.analysis_result.impact_idx
        self._render_selected_snapshot(override_frame_idx=idx)

    # --- SLOW MOTION REPLAY ---
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
        if self.replay_playing:
            if self.replay_frame_idx >= self.analysis_result.total_frames - 1:
                self.replay_frame_idx = 0
            self.btn_play_pause.setText("⏸ Tạm dừng")
            self._start_replay_timer()
        else:
            self.btn_play_pause.setText("▶ Phát")
            self.replay_timer.stop()

    def _on_replay_tick(self):
        if not self.analysis_result:
            return
        self.replay_frame_idx += 1
        if self.replay_frame_idx >= self.analysis_result.total_frames:
            self.replay_frame_idx = 0  # Lặp lại tự động
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
        self.result_viewer.setPixmap(
            pix.scaled(self.result_viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.lbl_replay_time.setText(f"{idx + 1} / {res.total_frames}")

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
        if self.state == self.STATE_RESULT and self.result_controls_stack.currentIndex() == 0:
            self._render_selected_snapshot()

    def closeEvent(self, event):
        self.countdown_timer.stop()
        self.replay_timer.stop()
        if hasattr(self, "cam_worker"):
            self.cam_worker.stop()
        super().closeEvent(event)
