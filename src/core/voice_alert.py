"""
MODULE PHÁT ÂM THANH & GIỌNG NÓI TTS ĐA LUỒNG (NON-BLOCKING)
Đảm bảo khi phát âm thanh cảnh báo không làm giật khung hình video camera.
"""

import threading
import queue
import time
import os

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class VoiceAlertService:
    def __init__(self, speech_rate=160, enabled=True):
        self.enabled = enabled
        self.speech_rate = speech_rate
        self.msg_queue = queue.Queue()
        self.is_running = True

        # Khởi động luồng chạy nền (Worker Thread)
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        """
        Vòng lặp chạy nền nhận thông điệp từ hàng đợi và phát âm thanh.
        """
        engine = None
        if HAS_PYTTSX3:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', self.speech_rate)
            except Exception as e:
                print(f"[VoiceAlert] Khong the khoi tao pyttsx3: {e}")
                engine = None

        while self.is_running:
            try:
                item = self.msg_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            msg_type, content = item
            if not self.enabled:
                self.msg_queue.task_done()
                continue

            try:
                if msg_type == "BEEP_ERROR":
                    if HAS_WINSOUND:
                        # Tiếng beep cảnh báo trầm (400Hz, 400ms)
                        winsound.Beep(400, 400)
                elif msg_type == "BEEP_SUCCESS":
                    if HAS_WINSOUND:
                        # Tiếng ting thành công cao vút (1000Hz, 250ms)
                        winsound.Beep(1000, 250)
                elif msg_type == "SPEECH":
                    if engine:
                        engine.say(content)
                        engine.runAndWait()
                    else:
                        print(f"\n📢 [LOA PHÁT THANH]: {content}\n")
            except Exception as ex:
                print(f"[VoiceAlert Error]: {ex}")

            self.msg_queue.task_done()

    def alert_bent_arm(self, angle: float):
        """
        Gửi cảnh báo lỗi gập tay trái vào hàng đợi.
        """
        self.msg_queue.put(("BEEP_ERROR", None))
        msg = f"Cảnh báo: Bạn bị gập tay trái ở đỉnh swing, góc chỉ đạt {int(angle)} độ. Hãy giữ thẳng tay!"
        self.msg_queue.put(("SPEECH", msg))

    def alert_good_swing(self, angle: float):
        """
        Gửi thông báo tư thế chuẩn vào hàng đợi.
        """
        self.msg_queue.put(("BEEP_SUCCESS", None))
        msg = f"Tư thế rất tốt, tay trái thẳng {int(angle)} độ đạt chuẩn PGA!"
        self.msg_queue.put(("SPEECH", msg))

    def shutdown(self):
        self.is_running = False
