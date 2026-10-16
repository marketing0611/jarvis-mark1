"""
Webcam capture and face tracking for the VISION view. Runs locally with OpenCV:
YuNet (core/models) gives the face box plus eye/nose/mouth points; if the model is
missing, a Haar cascade bundled with OpenCV is used and the points are estimated.
"""
from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH   = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
FRAME_SIZE   = (1280, 720)
DETECT_WIDTH = 480
_SMOOTH      = 0.45           # weight of a new detection in the moving average
_LOST_AFTER  = 0.6            # seconds without a detection before the face is dropped

# Cold, high-contrast "lit by the HUD" grade (BGR matrix), like the helmet shots in the films.
_GRADE = np.array([[1.10, 0.10, 0.02], [0.06, 0.95, 0.0], [0.0, 0.10, 0.62]], dtype=np.float32) * 0.92

Point = tuple[float, float]


@dataclass
class Face:
    box: tuple[float, float, float, float]       # x, y, w, h in frame pixels
    left_eye: Point                              # screen-left eye (the person's left in the mirrored view)
    right_eye: Point
    nose: Point
    mouth_left: Point
    mouth_right: Point
    score: float

    def as_array(self) -> np.ndarray:
        return np.array([*self.box, *self.left_eye, *self.right_eye, *self.nose,
                         *self.mouth_left, *self.mouth_right, self.score], dtype=np.float32)

    @classmethod
    def from_array(cls, a: np.ndarray) -> "Face":
        return cls(tuple(a[0:4]), tuple(a[4:6]), tuple(a[6:8]), tuple(a[8:10]),
                   tuple(a[10:12]), tuple(a[12:14]), float(a[14]))


class _YuNet:
    def __init__(self):
        self._net = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (320, 320),
                                              score_threshold=0.75, nms_threshold=0.3, top_k=5)

    def detect(self, small: np.ndarray) -> list[np.ndarray]:
        h, w = small.shape[:2]
        self._net.setInputSize((w, h))
        _, faces = self._net.detect(small)
        if faces is None:
            return []
        out = []
        for f in faces:
            x, y, fw, fh = f[0:4]
            eyes = sorted([f[4:6], f[6:8]], key=lambda p: p[0])          # screen-left first
            mouth = sorted([f[10:12], f[12:14]], key=lambda p: p[0])
            out.append(np.array([x, y, fw, fh, *eyes[0], *eyes[1], *f[8:10],
                                 *mouth[0], *mouth[1], f[14]], dtype=np.float32))
        return out


class _Haar:
    def __init__(self):
        self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, small: np.ndarray) -> list[np.ndarray]:
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        out = []
        for x, y, w, h in self._cascade.detectMultiScale(gray, 1.15, 5, minSize=(48, 48)):
            pts = [(0.32, 0.40), (0.68, 0.40), (0.5, 0.62), (0.36, 0.80), (0.64, 0.80)]
            coords = [v for u, t in pts for v in (x + u * w, y + t * h)]
            out.append(np.array([x, y, w, h, *coords, 0.8], dtype=np.float32))
        return out


class Camera:
    """Background capture + tracking. `latest()` returns the newest (frame_id, RGB frame, face)."""

    def __init__(self, index: int = 0):
        self.index = index
        self.error: str | None = None
        self.status = "OFF"
        self.detector = "-"
        self.fps = 0.0
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._face: np.ndarray | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)              # still releasing the previous session
        self._running = True
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="jarvis-camera")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def latest(self) -> tuple[int, np.ndarray | None, Face | None]:
        with self._lock:
            face = Face.from_array(self._face) if self._face is not None else None
            return self._frame_id, self._frame, face

    def _open(self):
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        # macOS asks for camera permission on the first attempt and fails it; keep trying briefly.
        for _ in range(8):
            if not self._running:
                return None
            cap = cv2.VideoCapture(self.index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
                return cap
            cap.release()
            time.sleep(1.5)
        return None

    def _run(self) -> None:
        self.status = "STARTING"
        cap = self._open()
        if cap is None:
            if self._running:
                self.error = "camera"
                self.status = "ERROR"
            self._running = False
            return
        try:
            detector = _YuNet() if MODEL_PATH.exists() else _Haar()
        except Exception:
            detector = _Haar()
        self.detector = "YUNET" if isinstance(detector, _YuNet) else "HAAR"

        smooth: np.ndarray | None = None
        last_seen = 0.0
        frames, t0 = 0, time.time()
        self.status = "LIVE"
        while self._running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            frame = cv2.flip(frame, 1)
            frame = cv2.convertScaleAbs(cv2.transform(frame, _GRADE), alpha=1.18, beta=-18)
            h, w = frame.shape[:2]
            scale = DETECT_WIDTH / w
            small = cv2.resize(frame, (DETECT_WIDTH, int(h * scale)))
            found = detector.detect(small)
            now = time.time()
            if found:
                best = max(found, key=lambda f: f[2] * f[3])        # nearest face
                best[:14] /= scale
                smooth = best if smooth is None else smooth + _SMOOTH * (best - smooth)
                last_seen = now
            elif smooth is not None and now - last_seen > _LOST_AFTER:
                smooth = None

            frames += 1
            if now - t0 >= 1.0:
                self.fps, frames, t0 = frames / (now - t0), 0, now
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame = rgb
                self._frame_id += 1
                self._face = None if smooth is None else smooth.copy()
        cap.release()
        self.status = "OFF"
        with self._lock:
            self._frame, self._face = None, None
