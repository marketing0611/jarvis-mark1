"""
J.A.R.V.I.S — desktop interface (PyQt6).

Centre: the animated arc-reactor HUD with a live audio visualiser and system readouts.
Right: a compact conversation panel with quick-command chips and the input bar.

Every public method on `JarvisUI` is thread-safe: calls are marshalled onto the
Qt main thread through a signal.
"""
from __future__ import annotations

import math
import platform
import random
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import psutil
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QConicalGradient, QFont, QFontDatabase, QImage, QKeySequence,
    QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QRadialGradient,
    QShortcut, QTextDocument, QTransform,
)
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

_OS = platform.system()

_DEFAULT_W, _DEFAULT_H = 1320, 840
_MIN_W, _MIN_H = 1040, 680
_CHAT_W = 430


class C:
    BG        = "#04080c"
    SURFACE   = "#08121a"
    SURFACE_2 = "#0c1923"
    SURFACE_3 = "#112230"
    BORDER    = "#15303e"
    BORDER_HI = "#21506a"
    PRI       = "#3fd6ff"
    PRI_DIM   = "#1d7f9b"
    PRI_GHOST = "#0a2734"
    ACC       = "#ff8a3d"
    AMBER     = "#ffc857"
    GREEN     = "#3ef0a1"
    RED       = "#ff4d6d"
    TEXT      = "#e4f5fc"
    TEXT_2    = "#9fbccb"
    TEXT_3    = "#58788a"
    USER_BG   = "#0f2d3c"
    USER_BD   = "#1f5a73"


def qcol(h: str, a: int = 255) -> QColor:
    c = QColor(h)
    c.setAlpha(a)
    return c


def _font(size: float, weight: QFont.Weight = QFont.Weight.Normal, mono: bool = False,
          spacing: float = 0.0) -> QFont:
    kind = QFontDatabase.SystemFont.FixedFont if mono else QFontDatabase.SystemFont.GeneralFont
    f = QFontDatabase.systemFont(kind)
    f.setPointSizeF(size)
    f.setWeight(weight)
    if spacing:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return f


# ── State shown in the header pill and the HUD ───────────────────────────────

def display_state(*, speaking: bool, busy: bool, muted: bool, listener: str) -> tuple[str, str]:
    if speaking:
        return "SPEAKING", C.ACC
    if busy:
        return "THINKING", C.AMBER
    if listener == "TRANSCRIBING":
        return "TRANSCRIBING", C.PRI
    if listener == "HEARING" and not muted:
        return "HEARING YOU", C.GREEN
    if muted:
        return "MIC OFF", C.RED
    if listener == "LOADING":
        return "LOADING VOICE", C.TEXT_2
    if listener.startswith("ERROR"):
        return "VOICE OFFLINE", C.RED
    if listener == "READY":
        return "LISTENING", C.GREEN
    return "STANDBY", C.PRI


# ── System metrics ───────────────────────────────────────────────────────────

class _SysMetrics:
    def __init__(self):
        self.cpu = self.mem = self.net = 0.0
        self._last = psutil.net_io_counters()
        self._last_t = time.time()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                self.cpu = psutil.cpu_percent(interval=None)
                self.mem = psutil.virtual_memory().percent
                now, nc = time.time(), psutil.net_io_counters()
                dt = max(now - self._last_t, 1e-3)
                self.net = (nc.bytes_sent + nc.bytes_recv
                            - self._last.bytes_sent - self._last.bytes_recv) / dt / 1048576
                self._last, self._last_t = nc, now
            except Exception:
                pass
            time.sleep(1.5)


_metrics = _SysMetrics()


# ── Arc reactor HUD ──────────────────────────────────────────────────────────

_STATE_MN = {
    "SPEAKING": "Ярьж байна", "THINKING": "Бодож байна", "TRANSCRIBING": "Таньж байна",
    "HEARING YOU": "Таныг сонсож байна", "MIC OFF": "Микрофон унтраастай",
    "LOADING VOICE": "Дуу таних модель ачаалж байна", "VOICE OFFLINE": "Дуу таних боломжгүй",
    "LISTENING": "Сонсож байна", "STANDBY": "Бэлэн байна",
}


def _uptime() -> str:
    up = time.time() - psutil.boot_time()
    return (f"{int(up // 86400)}D {int(up % 86400 // 3600)}H" if up >= 86400
            else f"{int(up // 3600)}H {int(up % 3600 // 60):02d}M")


class HudCanvas(QWidget):
    """The arc reactor: rings, gauges, a radar sweep and a circular audio visualiser."""

    clicked = pyqtSignal()
    BARS = 96

    def __init__(self, face_path: str, parent=None):
        super().__init__(parent)
        self.setMinimumSize(460, 460)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to stop JARVIS speaking  (Esc)")
        self.label, self.color = "STANDBY", C.PRI
        self.speaking = self.listening = self.thinking = False
        self.voice_on = True
        self.mic_level = 0.0
        self._bars = [0.0] * (self.BARS // 2)
        self._t = 0.0
        self._rings = [0.0, 120.0, 240.0, 0.0]
        self._sweep = 0.0
        self._glow = 0.5
        self._pulses: list[float] = []
        self._static: QPixmap | None = None
        self._face = self._load_face(face_path)
        self._readouts: dict[str, list[tuple[str, str]]] = {}
        self._refresh_readouts()
        frame = QTimer(self)
        frame.timeout.connect(self._step)
        frame.start(33)                                 # 30 fps keeps CPU use low
        slow = QTimer(self)
        slow.timeout.connect(self._refresh_readouts)
        slow.start(1000)

    @staticmethod
    def _load_face(path: str) -> QPixmap | None:
        if not Path(path).exists():
            return None
        px = QPixmap(path)
        return px if not px.isNull() else None

    def mousePressEvent(self, _):
        self.clicked.emit()

    def resizeEvent(self, e):
        self._static = None
        super().resizeEvent(e)

    def _geometry(self) -> tuple[float, float, float]:
        W, H = self.width(), self.height()
        R = max(120.0, min((H - 130) / 2, (W - 80) / 2))
        return W / 2, (H - 110) / 2 + 12, R

    def _refresh_readouts(self):
        net = _metrics.net
        b = psutil.sensors_battery()
        self._readouts = {
            "tl": [("CORE", "ONLINE"), ("MODE", "LOCAL DEMO")],
            "tr": [("LANGUAGE", "МОН · ENG"), ("VOICE", "ON" if self.voice_on else "OFF")],
            "bl": [("NETWORK", f"{net * 1024:.0f} KB/s" if net < 1 else f"{net:.1f} MB/s"),
                   ("UPTIME", _uptime())],
            "br": [("BATTERY", f"{b.percent:.0f}%{' +' if b.power_plugged else ''}" if b else "AC"),
                   ("SYSTEM", {"Darwin": "MACOS", "Windows": "WINDOWS"}.get(_OS, _OS.upper()))],
        }

    def _step(self):
        self._t += 0.033
        if self.speaking:
            energy = 0.75 + 0.25 * random.random()
        elif self.listening:
            energy = 0.35 + 0.65 * self.mic_level
        elif self.thinking:
            energy = 0.55 + 0.15 * math.sin(self._t * 6)
        else:
            energy = 0.4 + 0.08 * math.sin(self._t * 1.6)
        self._glow += (energy - self._glow) * 0.3

        speed = 4.4 if self.thinking else 2.8 if self.speaking else 1.0
        for i, d in enumerate((speed, -speed * 0.7, speed * 1.4, -0.4)):
            self._rings[i] = (self._rings[i] + d) % 360
        self._sweep = (self._sweep + (7 if self.thinking else 3.5)) % 360

        for i in range(len(self._bars)):
            if self.speaking:
                target = (0.35 + 0.65 * random.random()) * (0.55 + 0.45 * math.sin(self._t * 9 + i * 0.5) ** 2)
            elif self.listening:
                target = self.mic_level * (0.55 + 0.45 * random.random())
            elif self.thinking:
                target = 0.2 + 0.15 * math.sin(self._t * 6 + i * 0.35)
            else:
                target = 0.05 + 0.03 * math.sin(self._t * 2 + i * 0.25)
            self._bars[i] += (target - self._bars[i]) * 0.35

        _, _, R = self._geometry()
        grow = R * (0.014 if self.speaking else 0.007)
        self._pulses = [r + grow for r in self._pulses if r + grow < R * 0.95]
        if (self.speaking or (self.listening and self.mic_level > 0.35)) and random.random() < 0.1:
            self._pulses.append(R * 0.32)
        self.update()

    # painting
    def _static_layer(self) -> QPixmap:
        """Grid, crosshair, brackets and the degree ring — redrawn only on resize."""
        if self._static is not None:
            return self._static
        px = QPixmap(self.size() * self.devicePixelRatioF())
        px.setDevicePixelRatio(self.devicePixelRatioF())
        px.fill(qcol(C.BG))
        p = QPainter(px)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        cx, cy, R = self._geometry()

        p.setPen(QPen(qcol(C.PRI, 22), 1.6))
        step = 28
        for x in range(int(cx % step), W, step):
            for y in range(int(cy % step), H, step):
                p.drawPoint(QPointF(x, y))

        p.setPen(QPen(qcol(C.PRI, 40), 1))
        for sign in (-1, 1):
            p.drawLine(QPointF(cx + sign * R * 1.06, cy), QPointF(cx + sign * (W / 2 - 16), cy))
            p.drawLine(QPointF(cx, cy + sign * R * 1.06), QPointF(cx, cy + sign * (cy - 8)))

        p.setPen(QPen(qcol(C.PRI, 130), 1.5))
        b, L = R * 1.1, 22
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            x, y = cx + sx * b, cy + sy * b
            p.drawLine(QPointF(x, y), QPointF(x - sx * L, y))
            p.drawLine(QPointF(x, y), QPointF(x, y - sy * L))

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(qcol(C.PRI, 80), 1))
        p.drawEllipse(QPointF(cx, cy), R, R)
        p.setPen(QPen(qcol(C.PRI, 40), 1))
        p.drawEllipse(QPointF(cx, cy), R * 0.955, R * 0.955)
        for deg in range(0, 360, 2):
            rad = math.radians(deg)
            length, alpha = (13, 170) if deg % 30 == 0 else (8, 110) if deg % 10 == 0 else (4, 60)
            p.setPen(QPen(qcol(C.PRI, alpha), 1))
            sx, sy = math.sin(rad), -math.cos(rad)
            p.drawLine(QPointF(cx + R * sx, cy + R * sy), QPointF(cx + (R - length) * sx, cy + (R - length) * sy))
        p.setPen(qcol(C.PRI, 120))
        p.setFont(_font(7.5, QFont.Weight.Bold, mono=True))
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            x, y = cx + (R - 26) * math.sin(rad), cy - (R - 26) * math.cos(rad)
            p.drawText(QRectF(x - 16, y - 7, 32, 14), Qt.AlignmentFlag.AlignCenter, f"{deg:03d}")
        p.end()
        self._static = px
        return px

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.drawPixmap(0, 0, self._static_layer())
        W, H = self.width(), self.height()
        cx, cy, R = self._geometry()
        col, g = self.color, self._glow
        c = QPointF(cx, cy)

        glow = QRadialGradient(c, R * 1.1)
        glow.setColorAt(0.0, qcol(col, int(80 * g)))
        glow.setColorAt(0.45, qcol(col, int(22 * g)))
        glow.setColorAt(1.0, qcol(col, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(c, R * 1.1, R * 1.1)

        # radar sweep with a fading trail
        cone = QConicalGradient(c, -self._sweep)
        cone.setColorAt(0.0, qcol(col, int(70 * g)))
        cone.setColorAt(0.14, qcol(col, 0))
        cone.setColorAt(1.0, qcol(col, 0))
        p.setBrush(QBrush(cone))
        p.drawEllipse(c, R * 0.95, R * 0.95)
        rad = math.radians(-self._sweep)
        p.setPen(QPen(qcol(col, 140), 1.2))
        p.drawLine(QPointF(cx + R * 0.32 * math.cos(rad), cy - R * 0.32 * math.sin(rad)),
                   QPointF(cx + R * 0.95 * math.cos(rad), cy - R * 0.95 * math.sin(rad)))

        p.setBrush(Qt.BrushStyle.NoBrush)
        for r in self._pulses:
            p.setPen(QPen(qcol(col, max(0, int(150 * (1 - r / (R * 0.95))))), 1.2))
            p.drawEllipse(c, r, r)

        # dashed ring and rotating arc segments
        rr = R * 0.9
        p.setPen(QPen(qcol(col, 110), 2.5, cap=Qt.PenCapStyle.FlatCap))
        rect = QRectF(cx - rr, cy - rr, rr * 2, rr * 2)
        for k in range(90):
            p.drawArc(rect, int((self._rings[3] + k * 4) * 16), 2 * 16)
        for idx, (frac, width, arc, gap) in enumerate([(0.835, 3.5, 70, 50), (0.78, 2, 40, 20), (0.355, 1.5, 110, 70)]):
            rr = R * frac
            rect = QRectF(cx - rr, cy - rr, rr * 2, rr * 2)
            p.setPen(QPen(qcol(col, max(0, int(120 + 120 * g) - idx * 25)), width, cap=Qt.PenCapStyle.RoundCap))
            a = self._rings[idx]
            while a < self._rings[idx] + 360:
                p.drawArc(rect, int(a * 16), int(arc * 16))
                a += arc + gap

        # CPU / memory gauges (270° sweep, gap at the bottom)
        for frac, value, alpha in ((0.705, _metrics.cpu, 235), (0.66, _metrics.mem, 165)):
            rr = R * frac
            rect = QRectF(cx - rr, cy - rr, rr * 2, rr * 2)
            p.setPen(QPen(qcol(col, 34), 5, cap=Qt.PenCapStyle.FlatCap))
            p.drawArc(rect, 225 * 16, -270 * 16)
            p.setPen(QPen(qcol(col, alpha), 5, cap=Qt.PenCapStyle.FlatCap))
            p.drawArc(rect, 225 * 16, int(-270 * 16 * min(value, 100) / 100))
        p.setFont(_font(8.5, QFont.Weight.Bold, mono=True, spacing=1.5))
        p.setPen(qcol(C.TEXT_2))
        y = cy + R * 0.6
        p.drawText(QRectF(cx - R * 0.5, y, R * 0.42, 16), Qt.AlignmentFlag.AlignRight,
                   f"CPU {_metrics.cpu:.0f}%")
        p.drawText(QRectF(cx + R * 0.08, y, R * 0.42, 16), Qt.AlignmentFlag.AlignLeft,
                   f"MEM {_metrics.mem:.0f}%")

        # circular audio visualiser
        n, half = self.BARS, self.BARS // 2
        r0, width = R * 0.405, max(2.0, R * 0.012)
        for k in range(n):
            lvl = self._bars[k if k < half else n - 1 - k]
            rad = math.radians(k * 360 / n)
            sx, sy = math.sin(rad), -math.cos(rad)
            r1 = r0 + R * (0.02 + 0.19 * lvl)
            p.setPen(QPen(qcol(col, int(110 + 145 * min(lvl, 1))), width, cap=Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(cx + r0 * sx, cy + r0 * sy), QPointF(cx + r1 * sx, cy + r1 * sy))

        # core
        if self._face:
            size = int(R * 0.6)
            clip = QPainterPath()
            clip.addEllipse(c, size / 2, size / 2)
            p.setClipPath(clip)
            p.drawPixmap(int(cx - size / 2), int(cy - size / 2),
                         self._face.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                           Qt.TransformationMode.SmoothTransformation))
            p.setClipping(False)
        else:
            self._paint_reactor(p, cx, cy, R * 0.3, col, g, self._rings[2] * 0.25)

        paint_state(p, W, cy + R + 18, self.label, col)
        paint_readouts(p, W, H, self._readouts)

    @staticmethod
    def _paint_reactor(p: QPainter, cx: float, cy: float, r: float, col: str, g: float, spin: float):
        c = QPointF(cx, cy)
        grad = QRadialGradient(c, r)
        grad.setColorAt(0.0, qcol("#ffffff", 240))
        grad.setColorAt(0.22, qcol(col, 235))
        grad.setColorAt(0.7, qcol(col, int(60 + 70 * g)))
        grad.setColorAt(1.0, qcol(col, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(c, r, r)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(qcol(col, 170), 1.5))
        p.drawEllipse(c, r * 1.08, r * 1.08)
        p.setPen(QPen(qcol(col, 90), 1))
        p.drawEllipse(c, r * 1.16, r * 1.16)
        # ten coil segments, slowly turning
        p.setPen(QPen(qcol(col, 210), 1.3))
        p.setBrush(QBrush(qcol(C.SURFACE, 205)))
        for k in range(10):
            a0 = math.radians(spin + k * 36 + 4)
            a1 = math.radians(spin + k * 36 + 32)
            r1, r2 = r * 0.6, r * 0.92
            p.drawPolygon(QPolygonF([QPointF(cx + r1 * math.cos(a0), cy + r1 * math.sin(a0)),
                                     QPointF(cx + r2 * math.cos(a0), cy + r2 * math.sin(a0)),
                                     QPointF(cx + r2 * math.cos(a1), cy + r2 * math.sin(a1)),
                                     QPointF(cx + r1 * math.cos(a1), cy + r1 * math.sin(a1))]))
        p.setPen(QPen(qcol("#ffffff", 225), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, r * 0.42, r * 0.42)
        p.setPen(QPen(qcol(col, 200), 1))
        p.drawEllipse(c, r * 0.5, r * 0.5)


# ── Shared HUD painting ──────────────────────────────────────────────────────

def paint_readouts(p: QPainter, W: int, H: int, readouts: dict[str, list[tuple[str, str]]],
                   mx: float = 26, my: float = 26):
    """Two label/value pairs in each corner, with an L-shaped bracket."""
    for corner, items in readouts.items():
        right, top = corner[1] == "r", corner[0] == "t"
        y0 = my if top else H - my - 2 * 42 + 6
        x = W - mx if right else mx
        bx = W - mx + 10 if right else mx - 10
        by = y0 - 8 if top else y0 + 2 * 42 - 4
        sx, sy = (-1 if right else 1), (1 if top else -1)
        p.setPen(QPen(qcol(C.PRI, 120), 1.5))
        p.drawLine(QPointF(bx, by), QPointF(bx + sx * 16, by))
        p.drawLine(QPointF(bx, by), QPointF(bx, by + sy * 16))
        align = Qt.AlignmentFlag.AlignRight if right else Qt.AlignmentFlag.AlignLeft
        for i, (label, value) in enumerate(items):
            y = y0 + i * 42
            left = x - 190 if right else x
            p.setPen(qcol(C.TEXT_3))
            p.setFont(_font(7.5, QFont.Weight.Bold, mono=True, spacing=2))
            p.drawText(QRectF(left, y, 190, 14), align, label)
            p.setPen(qcol(C.TEXT))
            p.setFont(_font(12, QFont.Weight.DemiBold, mono=True, spacing=1))
            p.drawText(QRectF(left, y + 14, 190, 20), align, value)


def paint_state(p: QPainter, W: int, y: float, label: str, color: str):
    p.setPen(qcol(color))
    p.setFont(_font(17, QFont.Weight.Bold, mono=True, spacing=8))
    p.drawText(QRectF(0, y, W, 28), Qt.AlignmentFlag.AlignCenter, label)
    p.setPen(qcol(C.TEXT_2))
    p.setFont(_font(12))
    p.drawText(QRectF(0, y + 30, W, 22), Qt.AlignmentFlag.AlignCenter, _STATE_MN.get(label, ""))


# ── Vision: inside-the-helmet camera HUD ─────────────────────────────────────

ICE, ICE_DIM, WARN = "#8fe3ff", "#3d8fb3", "#ff4a4a"

_TRACE = [
    "track.lock(eye='L', conf={conf:.2f})", "mesh.align(pts=9) ........ OK", "bio::scan → MATCH {match:.1f}%",
    "thermal.read() → 36.{r}°C", "pupil.dilation = 4.{r} mm", "gaze.vec = ({gx:+.2f}, {gy:+.2f})",
    "frame #{frame:06d} · {fps:.0f} fps", "dist ≈ {dist:.0f} cm", "jarvis.vision.sync() ✓",
    "face.box [{bx:.0f},{by:.0f},{bw:.0f},{bh:.0f}]", "neural.infer {ms:.1f} ms", "stark.protocol XXXIX ✓",
    "retina.map → 0x{hx:04X}", "head.yaw {yaw:+.1f}°  roll {roll:+.1f}°",
]
# (u, v) spots on the face box where little code glyphs float
_GLYPH_SPOTS = [(0.30, 0.16), (0.55, 0.10), (0.74, 0.20), (0.14, 0.52), (0.88, 0.50), (0.50, 0.48),
                (0.22, 0.74), (0.80, 0.74), (0.50, 0.92), (0.40, 0.28), (0.66, 0.62)]
_DIAGNOSTICS = [("THRUSTERS", "STANDBY"), ("REPULSORS", "ONLINE"), ("FLIGHT CTRL", "READY"),
                ("ARMOR", "98%"), ("LIFE SUPPORT", "NOMINAL")]


def _visor_path(W: float, H: float) -> QPainterPath:
    """Outline of the faceplate window as seen from inside the helmet."""
    def P(x, y):
        return QPointF(W * x, H * y)
    path = QPainterPath(P(0.07, 0.12))
    path.cubicTo(P(0.25, 0.025), P(0.75, 0.025), P(0.93, 0.12))
    path.cubicTo(P(0.99, 0.30), P(0.995, 0.62), P(0.93, 0.84))
    path.cubicTo(P(0.86, 0.95), P(0.70, 0.985), P(0.5, 0.985))
    path.cubicTo(P(0.30, 0.985), P(0.14, 0.95), P(0.07, 0.84))
    path.cubicTo(P(0.005, 0.62), P(0.01, 0.30), P(0.07, 0.12))
    path.closeSubpath()
    return path


def _tilt(p: QPainter, rect: QRectF, side: int, amount: float = 0.06):
    """Project `rect` onto a trapezoid so a panel looks wrapped around the visor."""
    src = QPolygonF([rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()])
    d = rect.height() * amount
    if side < 0:
        dst = [QPointF(rect.left(), rect.top() + d), rect.topRight(), rect.bottomRight(),
               QPointF(rect.left(), rect.bottom() - d)]
    else:
        dst = [rect.topLeft(), QPointF(rect.right(), rect.top() + d),
               QPointF(rect.right(), rect.bottom() - d), rect.bottomLeft()]
    t = QTransform()
    if QTransform.quadToQuad(src, QPolygonF(dst), t):
        p.setTransform(t)


class VisionView(QWidget):
    """Mirrored webcam seen from inside an Iron-Man helmet; the overlay follows the face."""

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        from core.vision import Camera
        self.setMinimumSize(460, 460)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.label, self.color = "STANDBY", C.PRI
        self.camera = Camera()
        self._image: QImage | None = None
        self._frame_id = -1
        self._frame_size = (1280, 720)
        self._face = None                      # smoothed display copy (numpy array)
        self._lost_t = 0.0
        self._t = 0.0
        self._yaw = self._roll = self._pitch = 0.0
        self._left: list[str] = []
        self._right: list[str] = []
        self._next_line = 0.0
        self._glyphs = ["" for _ in _GLYPH_SPOTS]
        self._helmet: QPixmap | None = None
        tmr = QTimer(self)
        tmr.timeout.connect(self._step)
        tmr.start(33)

    # camera lifecycle follows visibility
    def showEvent(self, e):
        self.camera.start()
        super().showEvent(e)

    def hideEvent(self, e):
        self.camera.stop()
        self._image, self._face = None, None
        super().hideEvent(e)

    def mousePressEvent(self, _):
        if self.camera.error:
            self.camera.start()
        else:
            self.clicked.emit()

    def resizeEvent(self, e):
        self._helmet = None
        super().resizeEvent(e)

    def _mapping(self) -> tuple[float, float, float]:
        fw, fh = self._frame_size
        s = max(self.width() / fw, self.height() / fh)
        return s, (self.width() - fw * s) / 2, (self.height() - fh * s) / 2

    def _step(self):
        self._t += 0.033
        fid, frame, face = self.camera.latest()
        if frame is not None and fid != self._frame_id:
            h, w = frame.shape[:2]
            self._image = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            self._frame_id, self._frame_size = fid, (w, h)
        if face is not None:
            a = face.as_array()
            self._face = a if self._face is None else self._face + 0.5 * (a - self._face)
            self._lost_t = 0.0
        elif self._face is not None:
            self._lost_t += 0.033
            if self._lost_t > 0.4:
                self._face = None
        m = self._metrics()
        for attr, target in (("_yaw", m["yaw"]), ("_roll", m["roll"]), ("_pitch", m["gy"] * 20)):
            setattr(self, attr, getattr(self, attr) + 0.2 * (target - getattr(self, attr)))
        if self._t >= self._next_line:
            self._next_line = self._t + 0.22
            self._left = (self._left + [random.choice(_TRACE).format(**m)])[-8:]
            self._right = (self._right + [self._hex_line()])[-8:]
            self._glyphs = [random.choice([f"{random.randint(0, 0xFFFF):04X}", f"{random.random():.3f}",
                                           f"Δ{random.randint(1, 99):02d}", f"#{random.randint(0, 255):02X}"])
                            if random.random() < 0.5 else g for g in self._glyphs]
        self.update()

    def _metrics(self) -> dict:
        from core.vision import Face
        f = Face.from_array(self._face) if self._face is not None else None
        bx, by, bw, bh = f.box if f else (0, 0, 1, 1)
        fw = self._frame_size[0]
        dx = ((f.left_eye[0] + f.right_eye[0]) / 2 - (bx + bw / 2)) / bw if f else 0.0
        dy = (f.nose[1] - (by + bh * 0.55)) / bh if f else 0.0
        roll = math.degrees(math.atan2(f.right_eye[1] - f.left_eye[1], f.right_eye[0] - f.left_eye[0])) if f else 0.0
        return dict(conf=f.score if f else 0.0, match=96 + 3.5 * random.random(), r=random.randint(1, 9),
                    gx=dx * 2, gy=dy * 2, frame=self._frame_id, fps=self.camera.fps,
                    dist=15 * fw * 0.9 / max(bw, 1), bx=bx, by=by, bw=bw, bh=bh,
                    ms=2 + 2 * random.random(), hx=random.randint(0, 0xFFFF), yaw=dx * -60, roll=roll)

    @staticmethod
    def _hex_line() -> str:
        return f"0x{random.randint(0, 0xFFFF):04X}  " + " ".join(f"{random.randint(0, 255):02X}" for _ in range(6))

    # painting
    def _helmet_layer(self) -> QPixmap:
        """Dark helmet interior around the visor, its glowing rim, glass glare and scanlines."""
        if self._helmet is not None:
            return self._helmet
        W, H = self.width(), self.height()
        px = QPixmap(self.size() * self.devicePixelRatioF())
        px.setDevicePixelRatio(self.devicePixelRatioF())
        px.fill(Qt.GlobalColor.transparent)
        p = QPainter(px)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        visor = _visor_path(W, H)

        p.setPen(QPen(QColor(0, 0, 0, 34), 1))
        for y in range(0, H, 3):
            p.drawLine(0, y, W, y)

        glare = QLinearGradient(QPointF(0, 0), QPointF(W * 0.45, H * 0.6))
        glare.setColorAt(0.0, QColor(255, 255, 255, 16))
        glare.setColorAt(0.5, QColor(255, 255, 255, 4))
        glare.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(glare))
        p.drawPolygon(QPolygonF([QPointF(0, 0), QPointF(W * 0.55, 0), QPointF(0, H * 0.75)]))

        outside = QPainterPath()
        outside.addRect(QRectF(self.rect()))
        outside = outside.subtracted(visor)
        shade = QRadialGradient(QPointF(W / 2, H / 2), max(W, H) * 0.7)
        shade.setColorAt(0.0, QColor(4, 10, 16, 235))
        shade.setColorAt(1.0, QColor(0, 0, 0, 255))
        p.setBrush(QBrush(shade))
        p.drawPath(outside)

        p.setBrush(Qt.BrushStyle.NoBrush)
        for width, alpha in ((9, 18), (4, 40), (1.6, 150)):
            p.setPen(QPen(qcol(ICE, alpha), width))
            p.drawPath(visor)
        p.end()
        self._helmet = px
        return px

    def paintEvent(self, _):
        from core.vision import Face
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), QColor(0, 0, 0))
        face = Face.from_array(self._face) if self._face is not None else None
        s, ox, oy = self._mapping()

        if self._image is not None:
            fw, fh = self._frame_size
            p.drawImage(QRectF(ox, oy, fw * s, fh * s), self._image)
            # black out everything but a soft spotlight on the head
            if face:
                bx, by, bw, bh = face.box
                cx, cy = ox + (bx + bw / 2) * s, oy + (by + bh * 0.42) * s
                rx, ry = bw * s * 0.75, bh * s * 0.92
                stops = ((0.0, 0), (0.6, 0), (0.84, 205), (1.0, 255))
            else:
                cx, cy, rx, ry = W / 2, H * 0.45, W * 0.3, H * 0.42
                stops = ((0.0, 150), (1.0, 255))
            spot = QRadialGradient(QPointF(0, 0), 1.0)
            for at, alpha in stops:
                spot.setColorAt(at, QColor(0, 0, 0, alpha))
            brush = QBrush(spot)
            brush.setTransform(QTransform().translate(cx, cy).scale(rx, ry))
            p.fillRect(self.rect(), brush)
            p.fillRect(self.rect(), qcol(ICE, 10))          # the HUD's glow on the skin

        p.drawPixmap(0, 0, self._helmet_layer())
        self._paint_heading(p, W, H)
        self._paint_horizon(p, W, H)
        if self._image is None:
            self._paint_placeholder(p, W, H)
        elif face:
            self._paint_face(p, face, s, ox, oy)
        else:
            self._paint_searching(p, W, H)
        self._paint_clusters(p, W, H)

        tracking = "LOCKED" if face else "SEARCHING" if self._image is not None else "—"
        paint_readouts(p, W, H, {
            "tl": [("VISION", "ONLINE" if self._image is not None else "OFFLINE"), ("TRACKING", tracking)],
            "tr": [("FPS", f"{self.camera.fps:.0f}"), ("DETECTOR", self.camera.detector)],
        }, mx=W * 0.1, my=H * 0.12)
        paint_state(p, W, H * 0.885 - 30, self.label, self.color)

    def _paint_heading(self, p: QPainter, W: int, H: int):
        """Compass tape along the top that turns with the head."""
        cx, y, half = W / 2, H * 0.11, W * 0.17
        heading = (47 + self._yaw * 1.5) % 360
        ppd = half / 30
        p.setFont(_font(7.5, QFont.Weight.Bold, mono=True, spacing=1))
        for d in range(int(heading) - 30, int(heading) + 31):
            if d % 5:
                continue
            x = cx + (d - heading) * ppd
            fade = max(0, 1 - abs(x - cx) / half)
            p.setPen(QPen(qcol(ICE, int(200 * fade)), 1.2))
            p.drawLine(QPointF(x, y), QPointF(x, y + (9 if d % 15 == 0 else 5)))
            if d % 15 == 0:
                label = {0: "N", 90: "E", 180: "S", 270: "W"}.get(d % 360, f"{d % 360:03d}")
                p.drawText(QRectF(x - 20, y + 11, 40, 12), Qt.AlignmentFlag.AlignCenter, label)
        p.setPen(QPen(qcol(ICE, 220), 1.2))
        p.setBrush(QBrush(qcol(ICE, 220)))
        p.drawPolygon(QPolygonF([QPointF(cx - 5, y - 9), QPointF(cx + 5, y - 9), QPointF(cx, y - 2)]))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setFont(_font(9, QFont.Weight.Bold, mono=True, spacing=2))
        p.drawText(QRectF(cx - 60, y - 28, 120, 16), Qt.AlignmentFlag.AlignCenter, f"HDG {heading:03.0f}")

    def _paint_horizon(self, p: QPainter, W: int, H: int):
        """Artificial horizon that rolls and pitches with the head."""
        cx, cy = W / 2, H * 0.5
        p.save()
        p.translate(cx, cy + self._pitch * 3)
        p.rotate(self._roll)
        inner, outer = W * 0.19, W * 0.3
        p.setPen(QPen(qcol(ICE, 150), 1.6))
        for sign in (-1, 1):
            p.drawLine(QPointF(sign * inner, 0), QPointF(sign * outer, 0))
            p.drawLine(QPointF(sign * inner, 0), QPointF(sign * inner, 8))
        p.setFont(_font(7, QFont.Weight.Bold, mono=True))
        for step in (-20, -10, 10, 20):
            y = -step * 3.2
            p.setPen(QPen(qcol(ICE, 70), 1, Qt.PenStyle.DashLine if step < 0 else Qt.PenStyle.SolidLine))
            for sign in (-1, 1):
                p.drawLine(QPointF(sign * inner, y), QPointF(sign * (inner + W * 0.05), y))
            p.drawText(QRectF(outer - W * 0.06, y - 7, 30, 14), Qt.AlignmentFlag.AlignLeft, f"{step:+d}")
        p.restore()

    def _paint_clusters(self, p: QPainter, W: int, H: int):
        """Instrument panels on both sides, angled as if wrapped around the visor."""
        pw = min(236.0, W * 0.21)
        top = H * 0.3
        drift = 0.0
        if self._face is not None:
            s, ox, _ = self._mapping()
            drift = ((ox + (self._face[0] + self._face[2] / 2) * s) - W / 2) * 0.03   # slight parallax
        lx, rx = W * 0.06 + drift, W * 0.94 - pw + drift

        p.save()
        _tilt(p, QRectF(lx, top, pw, H * 0.5), -1)
        self._paint_gauges(p, lx, top, pw)
        self._paint_panel(p, lx, top + 108, pw, 8 * 15 + 34, "BIOMETRIC TRACE", self._left)
        p.restore()

        p.save()
        _tilt(p, QRectF(rx, top, pw, H * 0.5), 1)
        self._paint_diagnostics(p, rx, top, pw)
        self._paint_panel(p, rx, top + 108, pw, 8 * 15 + 34, "NEURAL STREAM", self._right)
        p.restore()

    def _paint_gauges(self, p: QPainter, x: float, y: float, w: float):
        r = 34
        c = QPointF(x + r + 6, y + r + 10)
        for frac, value, alpha in ((1.0, _metrics.cpu, 230), (0.72, _metrics.mem, 160)):
            rr = r * frac
            rect = QRectF(c.x() - rr, c.y() - rr, rr * 2, rr * 2)
            p.setPen(QPen(qcol(ICE, 40), 5, cap=Qt.PenCapStyle.FlatCap))
            p.drawArc(rect, 90 * 16, -360 * 16)
            p.setPen(QPen(qcol(ICE, alpha), 5, cap=Qt.PenCapStyle.FlatCap))
            p.drawArc(rect, 90 * 16, int(-360 * 16 * min(value, 100) / 100))
        p.setPen(qcol("#ffffff"))
        p.setFont(_font(16, QFont.Weight.Bold, mono=True))
        p.drawText(QRectF(x + 2 * r + 22, y + 6, w - 2 * r - 22, 24), Qt.AlignmentFlag.AlignLeft, "POWER")
        p.setFont(_font(8, QFont.Weight.Bold, mono=True, spacing=1.5))
        p.setPen(qcol(ICE))
        net = _metrics.net
        for i, line in enumerate((f"CPU  {_metrics.cpu:5.0f}%", f"MEM  {_metrics.mem:5.0f}%",
                                  f"NET  {net * 1024:.0f} KB/s" if net < 1 else f"NET  {net:.1f} MB/s")):
            p.drawText(QRectF(x + 2 * r + 22, y + 36 + i * 15, w, 14), Qt.AlignmentFlag.AlignLeft, line)

    def _paint_diagnostics(self, p: QPainter, x: float, y: float, w: float):
        p.setPen(qcol("#ffffff"))
        p.setFont(_font(16, QFont.Weight.Bold, mono=True))
        p.drawText(QRectF(x, y + 6, w, 24), Qt.AlignmentFlag.AlignLeft, "COOLANT")
        blink = int(self._t * 2) % 2 == 0
        p.setPen(qcol(WARN, 255 if blink else 150))
        p.setFont(_font(8, QFont.Weight.Bold, mono=True, spacing=2))
        p.drawText(QRectF(x, y + 30, w, 14), Qt.AlignmentFlag.AlignLeft, "▲ SYSTEM DIAGNOSTIC")
        p.setFont(_font(8, mono=True))
        for i, (name, value) in enumerate(_DIAGNOSTICS[:4]):
            row = QRectF(x, y + 48 + i * 14, w, 14)
            p.setPen(qcol(C.TEXT_2))
            p.drawText(row, Qt.AlignmentFlag.AlignLeft, name)
            p.setPen(qcol(ICE))
            p.drawText(row, Qt.AlignmentFlag.AlignRight, value)

    @staticmethod
    def _paint_panel(p: QPainter, x: float, y: float, w: float, h: float, title: str, lines: list[str]):
        p.setPen(QPen(qcol(ICE, 110), 1))
        p.setBrush(QBrush(QColor(0, 8, 14, 150)))
        p.drawRect(QRectF(x, y, w, h))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(qcol(ICE, 210)))
        p.drawRect(QRectF(x, y, 4, 24))
        p.setPen(qcol(ICE))
        p.setFont(_font(8, QFont.Weight.Bold, mono=True, spacing=2))
        p.drawText(QRectF(x + 12, y + 5, w - 20, 14), Qt.AlignmentFlag.AlignLeft, title)
        p.setFont(_font(8, mono=True))
        for i, line in enumerate(lines):
            alpha = 90 + int(165 * (i + 1) / max(len(lines), 1))
            p.setPen(qcol("#ffffff" if i == len(lines) - 1 else C.TEXT_2, alpha))
            p.drawText(QRectF(x + 12, y + 28 + i * 15, w - 20, 14), Qt.AlignmentFlag.AlignLeft, line)
        p.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_placeholder(self, p: QPainter, W: int, H: int):
        cx, cy = W / 2, H / 2 - 30
        self._paint_reticle(p, cx, cy, min(W, H) * 0.12, spin=self._t * 40)
        p.setFont(_font(12))
        p.setPen(qcol(C.TEXT_2))
        if self.camera.error:
            text = ("Камер нээгдсэнгүй. System Settings → Privacy & Security → Camera хэсэгт\n"
                    "Terminal (эсвэл VS Code)-д зөвшөөрөл өгөөд энд дарж дахин оролдоно уу.\n\n"
                    "Camera unavailable — allow camera access, then click to retry.")
        else:
            text = "Камер асааж байна…  ·  Starting camera…"
        p.drawText(QRectF(W * 0.15, cy + min(W, H) * 0.18, W * 0.7, 90),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, text)

    def _paint_searching(self, p: QPainter, W: int, H: int):
        cx, cy = W / 2, H * 0.45
        r = min(W, H) * 0.14
        self._paint_reticle(p, cx, cy, r, spin=self._t * 60)
        p.setPen(qcol(ICE, 220))
        p.setFont(_font(10, QFont.Weight.Bold, mono=True, spacing=4))
        p.drawText(QRectF(0, cy + r * 2.1, W, 18), Qt.AlignmentFlag.AlignCenter, "SCANNING FOR SUBJECT")
        p.setPen(qcol(C.TEXT_2))
        p.setFont(_font(11))
        p.drawText(QRectF(0, cy + r * 2.1 + 20, W, 18), Qt.AlignmentFlag.AlignCenter, "Царай хайж байна…")

    def _paint_reticle(self, p: QPainter, cx: float, cy: float, r: float, spin: float):
        c = QPointF(cx, cy)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(qcol(ICE, 120), 1.5))
        rect = QRectF(cx - r * 1.7, cy - r * 1.7, r * 3.4, r * 3.4)
        for k in range(24):
            p.drawArc(rect, int((spin + k * 15) * 16), 7 * 16)
        p.setPen(QPen(qcol(ICE, 235), 2.2, cap=Qt.PenCapStyle.RoundCap))
        rect = QRectF(cx - r * 1.32, cy - r * 1.32, r * 2.64, r * 2.64)
        for base in (-spin * 1.6, -spin * 1.6 + 180):
            p.drawArc(rect, int(base * 16), 80 * 16)
        p.setPen(QPen(qcol(ICE, 215), 1.4))
        p.drawEllipse(c, r, r)
        for deg in range(0, 360, 15):
            rad = math.radians(deg)
            inner = r * (0.8 if deg % 45 == 0 else 0.88)
            p.drawLine(QPointF(cx + r * math.cos(rad), cy + r * math.sin(rad)),
                       QPointF(cx + inner * math.cos(rad), cy + inner * math.sin(rad)))
        p.drawEllipse(c, r * 0.42, r * 0.42)
        p.setBrush(QBrush(qcol(ICE, 235)))
        p.drawEllipse(c, 2.2, 2.2)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p.drawLine(QPointF(cx + dx * r * 1.05, cy + dy * r * 1.05), QPointF(cx + dx * r * 1.5, cy + dy * r * 1.5))
        p.setBrush(QBrush(qcol(ICE, 200)))
        p.setPen(Qt.PenStyle.NoPen)
        for k in range(4):
            a = math.radians(spin * 0.5 + 45 + k * 90)
            tip = QPointF(cx + r * 1.85 * math.cos(a), cy + r * 1.85 * math.sin(a))
            side = QPointF(-math.sin(a), math.cos(a))
            back = QPointF(cx + r * 2.1 * math.cos(a), cy + r * 2.1 * math.sin(a))
            p.drawPolygon(QPolygonF([tip, back + side * (r * 0.12), back - side * (r * 0.12)]))
        p.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_face(self, p: QPainter, f, s: float, ox: float, oy: float):
        def pt(xy):
            return QPointF(ox + xy[0] * s, oy + xy[1] * s)
        bx, by, bw, bh = f.box
        x0, y0, w, h = ox + bx * s, oy + by * s, bw * s, bh * s
        pad = w * 0.12
        L, T, R_, B = x0 - pad, y0 - pad * 1.3, x0 + w + pad, y0 + h + pad
        pulse = 0.75 + 0.25 * math.sin(self._t * 5)

        arm = w * 0.18
        p.setPen(QPen(qcol(ICE, int(235 * pulse)), 2.2, cap=Qt.PenCapStyle.SquareCap))
        for cx_, cy_, sx, sy in ((L, T, 1, 1), (R_, T, -1, 1), (L, B, 1, -1), (R_, B, -1, -1)):
            p.drawLine(QPointF(cx_, cy_), QPointF(cx_ + sx * arm, cy_))
            p.drawLine(QPointF(cx_, cy_), QPointF(cx_, cy_ + sy * arm))
        m = self._metrics()
        p.setFont(_font(8.5, QFont.Weight.Bold, mono=True, spacing=2))
        p.setPen(qcol(ICE))
        p.drawText(QRectF(L, T - 22, R_ - L, 16), Qt.AlignmentFlag.AlignLeft, "SUBJECT LOCKED")
        p.drawText(QRectF(L, T - 22, R_ - L, 16), Qt.AlignmentFlag.AlignRight, f"CONF {m['conf']:.0%}")
        if B + 24 < self.height() * 0.885 - 36:            # keep clear of the state label
            p.setPen(qcol(C.TEXT_2))
            p.drawText(QRectF(L, B + 6, R_ - L, 16), Qt.AlignmentFlag.AlignCenter,
                       f"DIST ≈ {m['dist']:.0f} CM   ·   ROLL {m['roll']:+.0f}°")

        sy_ = T + ((self._t * 0.55) % 1.0) * (B - T)
        grad = QLinearGradient(QPointF(L, 0), QPointF(R_, 0))
        grad.setColorAt(0.0, qcol(ICE, 0))
        grad.setColorAt(0.5, qcol(ICE, 160))
        grad.setColorAt(1.0, qcol(ICE, 0))
        p.setPen(QPen(QBrush(grad), 1.4))
        p.drawLine(QPointF(L, sy_), QPointF(R_, sy_))

        le, re, no, ml, mr = pt(f.left_eye), pt(f.right_eye), pt(f.nose), pt(f.mouth_left), pt(f.mouth_right)
        brow = QPointF((le.x() + re.x()) / 2, y0 + h * 0.18)
        chin = QPointF(no.x(), y0 + h * 0.98)
        ck_l, ck_r = QPointF(x0 + w * 0.12, y0 + h * 0.58), QPointF(x0 + w * 0.88, y0 + h * 0.58)
        p.setPen(QPen(qcol(ICE, 60), 1))
        for a, b in ((le, re), (le, no), (re, no), (no, ml), (no, mr), (ml, mr), (le, brow), (re, brow),
                     (le, ck_l), (ck_l, ml), (re, ck_r), (ck_r, mr), (ml, chin), (mr, chin)):
            p.drawLine(a, b)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(qcol(ICE, 210)))
        for q in (le, re, no, ml, mr, brow, chin, ck_l, ck_r):
            p.drawEllipse(q, 2.3, 2.3)

        p.setFont(_font(7.5, QFont.Weight.Bold, mono=True))
        for i, ((u, v), g) in enumerate(zip(_GLYPH_SPOTS, self._glyphs)):
            if g:
                p.setPen(qcol(ICE, int(110 + 100 * math.sin(self._t * 3 + i))))
                p.drawText(QPointF(x0 + u * w, y0 + v * h), g)

        er = w * 0.15
        self._paint_reticle(p, le.x(), le.y(), er, spin=self._t * 90)
        p.setPen(QPen(qcol(ICE, 150), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(re, er * 0.55, er * 0.55)
        p.drawLine(QPointF(re.x() - er * 0.8, re.y()), QPointF(re.x() - er * 0.6, re.y()))
        p.drawLine(QPointF(re.x() + er * 0.6, re.y()), QPointF(re.x() + er * 0.8, re.y()))
        k1 = QPointF(le.x() - er * 1.5, le.y() - er * 1.5)
        k2 = QPointF(k1.x() - er * 0.5, k1.y() - er * 0.5)
        k3 = QPointF(k2.x() - 70, k2.y())
        p.setPen(QPen(qcol(ICE, 200), 1.2))
        p.drawPolyline(QPolygonF([k1, k2, k3]))
        p.setFont(_font(8.5, QFont.Weight.Bold, mono=True, spacing=1.5))
        p.setPen(qcol(ICE))
        p.drawText(QRectF(k2.x() - 160, k3.y() - 34, 160, 16), Qt.AlignmentFlag.AlignRight, "OCULAR LOCK · L")
        p.setPen(qcol(C.TEXT_2))
        p.drawText(QRectF(k2.x() - 160, k3.y() - 18, 160, 16), Qt.AlignmentFlag.AlignRight, f"PUPIL 4.{m['r']} MM")


# ── Small widgets ────────────────────────────────────────────────────────────

class IconButton(QPushButton):
    """Round button with a vector icon: 'mic', 'speaker' or 'send'."""

    def __init__(self, kind: str, size: int = 38, parent=None):
        super().__init__(parent)
        self.kind, self.on, self.accent = kind, True, C.PRI
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._hover = False

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        cx, cy = W / 2, H / 2
        filled = self.kind == "send"
        fg = self.accent if self.on else C.RED
        if filled:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(qcol(C.PRI if not self._hover else "#7fe5ff")))
        else:
            p.setPen(QPen(qcol(fg, 160 if self._hover else 90), 1))
            p.setBrush(QBrush(qcol(fg, 40 if self._hover else 22)))
        p.drawEllipse(QRectF(1, 1, W - 2, H - 2))

        ink = qcol(C.BG) if filled else qcol(fg)
        pen = QPen(ink, 1.8, cap=Qt.PenCapStyle.RoundCap, join=Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if self.kind == "mic":
            p.drawRoundedRect(QRectF(cx - 3.5, cy - 9, 7, 11.5), 3.5, 3.5)
            p.drawArc(QRectF(cx - 6.5, cy - 6, 13, 11), 180 * 16, 180 * 16)
            p.drawLine(QPointF(cx, cy + 5), QPointF(cx, cy + 8.5))
            p.drawLine(QPointF(cx - 3.5, cy + 8.5), QPointF(cx + 3.5, cy + 8.5))
        elif self.kind == "speaker":
            p.setBrush(QBrush(ink))
            p.drawPolygon(QPolygonF([QPointF(cx - 8, cy - 3), QPointF(cx - 4.5, cy - 3),
                                     QPointF(cx + 0.5, cy - 7.5), QPointF(cx + 0.5, cy + 7.5),
                                     QPointF(cx - 4.5, cy + 3), QPointF(cx - 8, cy + 3)]))
            p.setBrush(Qt.BrushStyle.NoBrush)
            if self.on:
                p.drawArc(QRectF(cx - 2.5, cy - 4.5, 9, 9), -50 * 16, 100 * 16)
                p.drawArc(QRectF(cx - 5, cy - 8, 16, 16), -50 * 16, 100 * 16)
        elif self.kind == "send":
            p.setBrush(QBrush(ink))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(QPolygonF([QPointF(cx - 7, cy - 7), QPointF(cx + 8, cy),
                                     QPointF(cx - 7, cy + 7), QPointF(cx - 3.5, cy)]))
        if not self.on and self.kind in ("mic", "speaker"):
            p.setPen(QPen(qcol(C.RED), 2, cap=Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(cx - 8, cy - 8), QPointF(cx + 8, cy + 8))


class Chip(QPushButton):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFont(_font(11))
        self.setFixedHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(f"""
            QPushButton {{ background: {C.SURFACE_2}; color: {C.TEXT_2};
                          border: 1px solid {C.BORDER}; border-radius: 15px; padding: 0 10px; }}
            QPushButton:hover {{ color: {C.TEXT}; border-color: {C.PRI_DIM}; background: {C.PRI_GHOST}; }}
        """)


# ── Conversation ─────────────────────────────────────────────────────────────

class Bubble(QWidget):
    """One message: a small caption ("JARVIS · 14:02") over a rounded bubble."""

    def __init__(self, text: str, mine: bool, parent=None):
        super().__init__(parent)
        self.mine, self.full = mine, text
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        cap = QLabel(f"{'YOU' if mine else 'JARVIS'}  ·  {time.strftime('%H:%M')}")
        cap.setFont(_font(8, QFont.Weight.Bold, mono=True, spacing=1.2))
        cap.setStyleSheet(f"color: {C.TEXT_3 if mine else C.PRI_DIM}; background: transparent;")
        cap.setAlignment(Qt.AlignmentFlag.AlignRight if mine else Qt.AlignmentFlag.AlignLeft)

        self.body = QLabel("" if not mine else text)
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.TextFormat.PlainText)
        self.body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body.setFont(_font(13))
        bg, bd = (C.USER_BG, C.USER_BD) if mine else (C.SURFACE_2, C.BORDER)
        accent = "" if mine else f"border-left: 2px solid {C.PRI};"
        self.body.setStyleSheet(f"""
            QLabel {{ color: {C.TEXT}; background: {bg}; border: 1px solid {bd}; {accent}
                     border-radius: 14px; padding: 10px 14px; }}
        """)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if mine:
            row.addStretch(1)
        row.addWidget(self.body)
        if not mine:
            row.addStretch(1)
        outer.addWidget(cap)
        outer.addLayout(row)

        self._shown = len(text) if mine else 0
        if not mine:
            self._typer = QTimer(self)
            self._typer.timeout.connect(self._type)
            self._typer.start(14)

    def _type(self):
        self._shown = min(len(self.full), self._shown + max(4, len(self.full) // 60))   # ≤ ~1 s
        self.body.setText(self.full[:self._shown])
        if self._shown >= len(self.full):
            self._typer.stop()

    def fit(self, max_width: int):
        doc = QTextDocument()
        doc.setDefaultFont(self.body.font())
        doc.setPlainText(self.full)
        natural = int(doc.idealWidth()) + 32          # + horizontal padding and border
        self.body.setFixedWidth(max(60, min(max_width, natural)))


class Note(QLabel):
    def __init__(self, text: str, error: bool = False, parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFont(_font(10.5))
        self.setStyleSheet(f"color: {C.RED if error else C.TEXT_3}; background: transparent; padding: 2px 8px;")


class ChatView(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(f"""
            QScrollArea {{ background: {C.SURFACE}; border: none; }}
            QScrollBar:vertical {{ background: transparent; width: 8px; margin: 4px 2px; }}
            QScrollBar::handle:vertical {{ background: {C.BORDER_HI}; border-radius: 3px; min-height: 30px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
        """)
        box = QWidget()
        box.setStyleSheet(f"background: {C.SURFACE};")
        self._lay = QVBoxLayout(box)
        self._lay.setContentsMargins(18, 16, 18, 16)
        self._lay.setSpacing(14)
        self._lay.addStretch(1)
        self.setWidget(box)
        self._bubbles: list[Bubble] = []
        self._stick = True
        bar = self.verticalScrollBar()
        bar.valueChanged.connect(lambda v: setattr(self, "_stick", v >= bar.maximum() - 40))
        bar.rangeChanged.connect(lambda _a, b: self._stick and bar.setValue(b))

    def _max_bubble(self) -> int:
        return int((self.viewport().width() - 36) * 0.88)

    def _add(self, w: QWidget):
        self._stick = True
        self._lay.addWidget(w)

    def add_message(self, text: str, mine: bool):
        b = Bubble(text, mine)
        b.fit(self._max_bubble())
        self._bubbles.append(b)
        self._add(b)

    def add_note(self, text: str, error: bool = False):
        self._add(Note(text, error))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        for b in self._bubbles:
            b.fit(self._max_bubble())


# ── Main window ──────────────────────────────────────────────────────────────

QUICK_COMMANDS = ["Сайн уу, Жарвис", "Цаг хэд болж байна?", "Цаг агаар ямар байна?",
                  "Онигоо яриач", "Юу чадах вэ?", "What can you do?"]


class MainWindow(QMainWindow):
    ev = pyqtSignal(str, object)

    def __init__(self, face_path: str):
        super().__init__()
        self.setWindowTitle("J.A.R.V.I.S — Mark XXXIX")
        self.setMinimumSize(_MIN_W, _MIN_H)
        self.resize(_DEFAULT_W, _DEFAULT_H)
        scr = QApplication.primaryScreen().availableGeometry()
        self.move((scr.width() - _DEFAULT_W) // 2, (scr.height() - _DEFAULT_H) // 2)

        self.on_submit: Callable[[str], None] | None = None
        self.on_stop: Callable[[], None] | None = None
        self.on_voice_toggled: Callable[[bool], None] | None = None
        self.muted = False
        self.voice_on = True
        self._speaking = self._busy = False
        self._listener = "LOADING"

        central = QWidget()
        central.setStyleSheet(f"background: {C.BG};")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.hud = HudCanvas(face_path)
        self.hud.clicked.connect(self._stop)
        self.vision = VisionView()
        self.vision.clicked.connect(self._stop)
        self._stack = QStackedWidget()
        self._stack.addWidget(self.hud)
        self._stack.addWidget(self.vision)
        body.addWidget(self._stack, stretch=1)
        body.addWidget(self._build_chat())
        root.addLayout(body, stretch=1)
        root.addWidget(self._build_footer())

        self.ev.connect(self._on_event)
        clock = QTimer(self)
        clock.timeout.connect(self._tick)
        clock.start(1000)
        self._tick()

        for key, fn in (("F4", self.toggle_mic), ("F5", self.toggle_voice),
                        ("F6", lambda: self.set_mode(1 - self._stack.currentIndex())),
                        ("Escape", self._stop), ("F11", self._toggle_fullscreen)):
            QShortcut(QKeySequence(key), self).activated.connect(fn)
        self._refresh_state()

    # layout
    def _build_header(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(60)
        w.setStyleSheet(f"background: {C.SURFACE}; border-bottom: 1px solid {C.BORDER};")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(22, 0, 22, 0)

        brand = QVBoxLayout()
        brand.setSpacing(0)
        title = QLabel("J.A.R.V.I.S")
        title.setFont(_font(17, QFont.Weight.Bold, mono=True, spacing=4))
        title.setStyleSheet(f"color: {C.PRI}; border: none;")
        sub = QLabel("MARK XXXIX  ·  DEMO")
        sub.setFont(_font(8, QFont.Weight.Bold, mono=True, spacing=2))
        sub.setStyleSheet(f"color: {C.TEXT_3}; border: none;")
        brand.addStretch()
        brand.addWidget(title)
        brand.addWidget(sub)
        brand.addStretch()
        lay.addLayout(brand)
        lay.addSpacing(28)
        lay.addWidget(self._build_mode_switch())
        lay.addStretch()

        self._pill = QLabel()
        self._pill.setFont(_font(9, QFont.Weight.Bold, mono=True, spacing=2))
        self._pill.setFixedHeight(30)
        lay.addWidget(self._pill)
        lay.addStretch()

        right = QVBoxLayout()
        right.setSpacing(0)
        self._clock = QLabel()
        self._clock.setFont(_font(17, QFont.Weight.DemiBold, mono=True))
        self._clock.setStyleSheet(f"color: {C.TEXT}; border: none;")
        self._clock.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._date = QLabel()
        self._date.setFont(_font(8, QFont.Weight.Bold, mono=True, spacing=1.5))
        self._date.setStyleSheet(f"color: {C.TEXT_3}; border: none;")
        self._date.setAlignment(Qt.AlignmentFlag.AlignRight)
        right.addStretch()
        right.addWidget(self._clock)
        right.addWidget(self._date)
        right.addStretch()
        lay.addLayout(right)
        return w

    def _build_mode_switch(self) -> QWidget:
        seg = QFrame()
        seg.setObjectName("seg")
        seg.setFixedHeight(34)
        seg.setStyleSheet(f"QFrame#seg {{ background: {C.SURFACE_2}; border: 1px solid {C.BORDER_HI}; border-radius: 16px; }}")
        row = QHBoxLayout(seg)
        row.setContentsMargins(3, 3, 3, 3)
        row.setSpacing(2)
        self._modes = QButtonGroup(self)
        for i, name in enumerate(("◉  REACTOR", "◎  VISION")):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setChecked(i == 0)
            b.setFixedHeight(26)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setFont(_font(8.5, QFont.Weight.Bold, mono=True, spacing=2))
            b.setToolTip("Switch view  (F6)")
            b.setStyleSheet(f"""
                QPushButton {{ color: {C.TEXT_3}; background: transparent; border: none;
                              border-radius: 13px; padding: 0 14px; }}
                QPushButton:hover {{ color: {C.TEXT}; }}
                QPushButton:checked {{ color: {C.BG}; background: {C.PRI}; }}
            """)
            self._modes.addButton(b, i)
            row.addWidget(b)
        self._modes.idClicked.connect(self.set_mode)
        return seg

    def set_mode(self, index: int):
        self._stack.setCurrentIndex(index)
        self._modes.button(index).setChecked(True)

    def _build_chat(self) -> QWidget:
        w = QFrame()
        w.setObjectName("chatPanel")
        w.setFixedWidth(_CHAT_W)
        w.setStyleSheet(f"QFrame#chatPanel {{ background: {C.SURFACE}; border-left: 1px solid {C.BORDER}; }}")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        head = QHBoxLayout()
        head.setContentsMargins(18, 14, 18, 6)
        cap = QLabel("CONVERSATION")
        cap.setFont(_font(8.5, QFont.Weight.Bold, mono=True, spacing=2))
        cap.setStyleSheet(f"color: {C.TEXT_3};")
        langs = QLabel("МОНГОЛ  ·  ENGLISH")
        langs.setFont(_font(8.5, QFont.Weight.Bold, mono=True, spacing=2))
        langs.setStyleSheet(f"color: {C.PRI_DIM};")
        head.addWidget(cap)
        head.addStretch()
        head.addWidget(langs)
        lay.addLayout(head)

        self.chat = ChatView()
        lay.addWidget(self.chat, stretch=1)

        chips = QGridLayout()
        chips.setContentsMargins(18, 6, 18, 10)
        chips.setHorizontalSpacing(8)
        chips.setVerticalSpacing(8)
        for i, text in enumerate(QUICK_COMMANDS):
            chip = Chip(text)
            chip.clicked.connect(lambda _, t=text: self._submit(t))
            chips.addWidget(chip, i // 2, i % 2)
        lay.addLayout(chips)

        bar = QFrame()
        bar.setObjectName("bar")
        bar.setFixedHeight(56)
        bar.setStyleSheet(f"""
            QFrame#bar {{ background: {C.SURFACE_2}; border: 1px solid {C.BORDER_HI}; border-radius: 28px; }}
        """)
        row = QHBoxLayout(bar)
        row.setContentsMargins(9, 8, 9, 8)
        row.setSpacing(8)
        self._mic_btn = IconButton("mic")
        self._mic_btn.accent = C.GREEN
        self._mic_btn.setToolTip("Microphone on/off  (F4)")
        self._mic_btn.clicked.connect(self.toggle_mic)
        self._input = QLineEdit()
        self._input.setPlaceholderText("Жарвист бичих…")
        self._input.setFont(_font(13))
        self._input.setStyleSheet(f"""
            QLineEdit {{ background: transparent; color: {C.TEXT}; border: none; padding: 0 6px;
                        selection-background-color: {C.PRI_DIM}; }}
        """)
        self._input.returnPressed.connect(lambda: self._submit(self._input.text()))
        self._voice_btn = IconButton("speaker")
        self._voice_btn.setToolTip("Spoken replies on/off  (F5)")
        self._voice_btn.clicked.connect(self.toggle_voice)
        send = IconButton("send")
        send.setToolTip("Send  (Enter)")
        send.clicked.connect(lambda: self._submit(self._input.text()))
        row.addWidget(self._mic_btn)
        row.addWidget(self._input, stretch=1)
        row.addWidget(self._voice_btn)
        row.addWidget(send)

        wrap = QHBoxLayout()
        wrap.setContentsMargins(18, 0, 18, 18)
        wrap.addWidget(bar)
        lay.addLayout(wrap)
        return w

    def _build_footer(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(28)
        w.setStyleSheet(f"background: {C.SURFACE}; border-top: 1px solid {C.BORDER};")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(22, 0, 22, 0)
        for text, align in (("F4 MIC   ·   F5 VOICE   ·   F6 VISION   ·   ESC STOP   ·   F11 FULLSCREEN", None),
                            ("BASED ON MARK XXXIX BY FATIHMAKES  ·  LOCAL DEMO", "right")):
            lbl = QLabel(text)
            lbl.setFont(_font(7.5, QFont.Weight.Bold, mono=True, spacing=1.2))
            lbl.setStyleSheet(f"color: {C.TEXT_3}; border: none;")
            if align:
                lay.addStretch()
            lay.addWidget(lbl)
        return w

    # behaviour
    def _submit(self, text: str):
        text = text.strip()
        if not text:
            return
        self._input.clear()
        self.chat.add_message(text, mine=True)
        if self.on_submit:
            self.on_submit(text)

    def _stop(self):
        if self.on_stop:
            self.on_stop()

    def toggle_mic(self):
        self.muted = not self.muted
        self._mic_btn.on = not self.muted
        self._mic_btn.update()
        self._refresh_state()

    def toggle_voice(self):
        self.voice_on = not self.voice_on
        self._voice_btn.on = self.hud.voice_on = self.voice_on
        self._voice_btn.update()
        if self.on_voice_toggled:
            self.on_voice_toggled(self.voice_on)

    def _toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _on_event(self, kind: str, data):
        if kind == "user":
            self.chat.add_message(data, mine=True)
        elif kind == "jarvis":
            self.chat.add_message(data, mine=False)
        elif kind == "note":
            self.chat.add_note(*data)
        elif kind == "listener":
            self._listener = data
            self._refresh_state()
        elif kind == "busy":
            self._busy = data
            self._refresh_state()
        elif kind == "speaking":
            self._speaking = data
            self._refresh_state()

    def _refresh_state(self):
        label, color = display_state(speaking=self._speaking, busy=self._busy,
                                     muted=self.muted, listener=self._listener)
        self.hud.label, self.hud.color = label, color
        self.vision.label, self.vision.color = label, color
        self.hud.speaking, self.hud.thinking = self._speaking, self._busy
        self.hud.listening = self._listener in ("READY", "HEARING") and not self.muted
        self._pill.setText(f"  ●  {label}  ")
        self._pill.setStyleSheet(f"""
            QLabel {{ color: {color}; background: {qcol(color, 26).name(QColor.NameFormat.HexArgb)};
                     border: 1px solid {qcol(color, 90).name(QColor.NameFormat.HexArgb)};
                     border-radius: 15px; padding: 0 10px; }}
        """)

    def _tick(self):
        self._clock.setText(time.strftime("%H:%M:%S"))
        self._date.setText(time.strftime("%a %d %b %Y").upper())


class _RootShim:
    def __init__(self, app: QApplication):
        self._app = app

    def mainloop(self):
        self._app.exec()


class JarvisUI:
    """Thread-safe facade used by main.py."""

    def __init__(self, face_path: str):
        self._app = QApplication.instance() or QApplication(sys.argv)
        self._app.setStyle("Fusion")
        self._win = MainWindow(face_path)
        self._win.show()
        self.root = _RootShim(self._app)

    # callbacks (set before mainloop)
    @property
    def on_submit(self):
        return self._win.on_submit

    @on_submit.setter
    def on_submit(self, cb):
        self._win.on_submit = cb

    @property
    def on_stop(self):
        return self._win.on_stop

    @on_stop.setter
    def on_stop(self, cb):
        self._win.on_stop = cb

    @property
    def on_voice_toggled(self):
        return self._win.on_voice_toggled

    @on_voice_toggled.setter
    def on_voice_toggled(self, cb):
        self._win.on_voice_toggled = cb

    @property
    def muted(self) -> bool:
        return self._win.muted

    # updates (any thread)
    def add_user_message(self, text: str):
        self._win.ev.emit("user", text)

    def add_reply(self, text: str):
        self._win.ev.emit("jarvis", text)

    def add_note(self, text: str, error: bool = False):
        self._win.ev.emit("note", (text, error))

    def set_listener_state(self, state: str):
        self._win.ev.emit("listener", state)

    def set_busy(self, busy: bool):
        self._win.ev.emit("busy", busy)

    def set_speaking(self, speaking: bool):
        self._win.ev.emit("speaking", speaking)

    def set_mic_level(self, level: float):
        self._win.hud.mic_level = level

    def write_log(self, text: str):
        self.add_note(text)
