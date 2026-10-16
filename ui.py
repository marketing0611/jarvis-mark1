"""
J.A.R.V.I.S — desktop interface (PyQt6).

Left: the animated arc-reactor HUD, live mic level and system stats.
Right: the conversation, quick-command chips and the input bar.

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
from collections import deque
from pathlib import Path
from typing import Callable

import psutil
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QFontDatabase, QKeySequence, QPainter, QPainterPath, QPen,
    QPixmap, QPolygonF, QRadialGradient, QShortcut, QTextDocument,
)
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

_OS = platform.system()

_DEFAULT_W, _DEFAULT_H = 1160, 740
_MIN_W, _MIN_H = 940, 620
_LEFT_W = 340


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

class HudCanvas(QWidget):
    clicked = pyqtSignal()

    def __init__(self, face_path: str, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.label, self.color = "STANDBY", C.PRI
        self.speaking = self.listening = self.thinking = False
        self.mic_level = 0.0
        self._levels: deque[float] = deque([0.0] * 40, maxlen=40)
        self._t = 0.0
        self._rings = [0.0, 120.0, 240.0]
        self._glow = 0.5
        self._pulses: list[float] = []
        self._face = self._load_face(face_path)
        tmr = QTimer(self)
        tmr.timeout.connect(self._step)
        tmr.start(33)                                   # 30 fps keeps CPU use low

    @staticmethod
    def _load_face(path: str) -> QPixmap | None:
        if not Path(path).exists():
            return None
        px = QPixmap(path)
        return px if not px.isNull() else None

    def mousePressEvent(self, _):
        self.clicked.emit()

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
        for i, d in enumerate((speed, -speed * 0.7, speed * 1.4)):
            self._rings[i] = (self._rings[i] + d) % 360

        lvl = (self.mic_level if self.listening else
               0.3 + 0.7 * random.random() if self.speaking else
               0.25 + 0.2 * math.sin(self._t * 8) if self.thinking else 0.04)
        self._levels.append(lvl)

        fw = min(self.width(), self.height())
        grow = 6.4 if self.speaking else 3.2
        self._pulses = [r + grow for r in self._pulses if r + grow < fw * 0.5]
        if (self.speaking or (self.listening and self.mic_level > 0.35)) and random.random() < 0.12:
            self._pulses.append(fw * 0.2)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), qcol(C.BG))
        W, H = self.width(), self.height()
        fw = min(W, H - 70)
        cx, cy = W / 2, (H - 70) / 2 + 6
        col = self.color
        g = self._glow

        # backdrop glow
        grad = QRadialGradient(QPointF(cx, cy), fw * 0.55)
        grad.setColorAt(0.0, qcol(col, int(70 * g)))
        grad.setColorAt(0.55, qcol(col, int(18 * g)))
        grad.setColorAt(1.0, qcol(col, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(QPointF(cx, cy), fw * 0.55, fw * 0.55)

        # expanding pulses
        for r in self._pulses:
            a = int(160 * (1 - r / (fw * 0.5)))
            p.setPen(QPen(qcol(col, max(a, 0)), 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, cy), r, r)

        # tick ring
        p.setPen(QPen(qcol(col, 90), 1))
        r_out, r_in = fw * 0.47, fw * 0.45
        for deg in range(0, 360, 6):
            rad = math.radians(deg)
            inner = r_in - (5 if deg % 30 == 0 else 0)
            p.drawLine(QPointF(cx + r_out * math.cos(rad), cy + r_out * math.sin(rad)),
                       QPointF(cx + inner * math.cos(rad), cy + inner * math.sin(rad)))

        # rotating arc segments
        for idx, (frac, width, arc, gap) in enumerate([(0.41, 3, 70, 50), (0.355, 2, 40, 20), (0.305, 1.5, 110, 70)]):
            rr = fw * frac
            rect = QRectF(cx - rr, cy - rr, rr * 2, rr * 2)
            p.setPen(QPen(qcol(col, int(120 + 120 * g) - idx * 30), width,
                          cap=Qt.PenCapStyle.RoundCap))
            a = self._rings[idx]
            while a < self._rings[idx] + 360:
                p.drawArc(rect, int(a * 16), int(arc * 16))
                a += arc + gap

        # core
        if self._face:
            size = int(fw * 0.5)
            clip = QPainterPath()
            clip.addEllipse(QPointF(cx, cy), size / 2, size / 2)
            p.setClipPath(clip)
            p.drawPixmap(int(cx - size / 2), int(cy - size / 2),
                         self._face.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                           Qt.TransformationMode.SmoothTransformation))
            p.setClipping(False)
        else:
            self._paint_reactor(p, cx, cy, fw * 0.25, col, g)

        # status + waveform
        base = H - 62
        p.setPen(QPen(qcol(col), 1))
        p.setFont(_font(10.5, QFont.Weight.Bold, mono=True, spacing=3))
        p.drawText(QRectF(0, base, W, 20), Qt.AlignmentFlag.AlignCenter, self.label)
        n, bw = len(self._levels), 5
        x0 = (W - n * bw) / 2
        for i, lv in enumerate(self._levels):
            h = 2 + lv * 22
            fade = 0.35 + 0.65 * (i / n)
            p.fillRect(QRectF(x0 + i * bw, base + 44 - h, bw - 2, h), qcol(col, int(220 * fade)))

    @staticmethod
    def _paint_reactor(p: QPainter, cx: float, cy: float, r: float, col: str, g: float):
        grad = QRadialGradient(QPointF(cx, cy), r)
        grad.setColorAt(0.0, qcol("#ffffff", 235))
        grad.setColorAt(0.25, qcol(col, 230))
        grad.setColorAt(0.7, qcol(col, int(60 + 60 * g)))
        grad.setColorAt(1.0, qcol(col, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(QPointF(cx, cy), r, r)
        # ten coil segments
        p.setPen(QPen(qcol(col, 200), 1.2))
        p.setBrush(QBrush(qcol(C.SURFACE, 200)))
        for k in range(10):
            a0 = math.radians(k * 36 + 4)
            a1 = math.radians(k * 36 + 32)
            r1, r2 = r * 0.62, r * 0.9
            pts = [QPointF(cx + r1 * math.cos(a0), cy + r1 * math.sin(a0)),
                   QPointF(cx + r2 * math.cos(a0), cy + r2 * math.sin(a0)),
                   QPointF(cx + r2 * math.cos(a1), cy + r2 * math.sin(a1)),
                   QPointF(cx + r1 * math.cos(a1), cy + r1 * math.sin(a1))]
            p.drawPolygon(QPolygonF(pts))
        p.setPen(QPen(qcol("#ffffff", 220), 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r * 0.42, r * 0.42)


# ── Small widgets ────────────────────────────────────────────────────────────

class StatTile(QWidget):
    def __init__(self, label: str, color: str, parent=None):
        super().__init__(parent)
        self._label, self._color = label, color
        self._value, self._text = 0.0, "--"
        self.setFixedHeight(54)

    def set_value(self, pct: float, text: str):
        self._value, self._text = max(0.0, min(100.0, pct)), text
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.setPen(QPen(qcol(C.BORDER), 1))
        p.setBrush(QBrush(qcol(C.SURFACE_2)))
        p.drawRoundedRect(QRectF(0.5, 0.5, W - 1, H - 1), 10, 10)
        p.setPen(qcol(C.TEXT_3))
        p.setFont(_font(8, QFont.Weight.Bold, mono=True, spacing=1.5))
        p.drawText(QRectF(12, 8, W - 24, 14), Qt.AlignmentFlag.AlignLeft, self._label)
        col = C.RED if self._value > 85 else C.AMBER if self._value > 65 else self._color
        p.setPen(qcol(C.TEXT))
        p.setFont(_font(13, QFont.Weight.DemiBold))
        p.drawText(QRectF(12, 20, W - 24, 20), Qt.AlignmentFlag.AlignLeft, self._text)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(qcol(C.SURFACE_3)))
        p.drawRoundedRect(QRectF(12, H - 10, W - 24, 3), 1.5, 1.5)
        p.setBrush(QBrush(qcol(col)))
        p.drawRoundedRect(QRectF(12, H - 10, (W - 24) * self._value / 100, 3), 1.5, 1.5)


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
        self.setFixedWidth(self.fontMetrics().horizontalAdvance(text) + 32)
        self.setStyleSheet(f"""
            QPushButton {{ background: {C.SURFACE_2}; color: {C.TEXT_2};
                          border: 1px solid {C.BORDER}; border-radius: 15px; padding: 0 14px; }}
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
        self._shown = min(len(self.full), self._shown + 4)
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
        self.setStyleSheet(f"color: {C.RED if error else C.TEXT_3}; background: transparent; padding: 2px 30px;")


class ChatView(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(f"""
            QScrollArea {{ background: {C.BG}; border: none; }}
            QScrollBar:vertical {{ background: transparent; width: 8px; margin: 4px 2px; }}
            QScrollBar::handle:vertical {{ background: {C.BORDER_HI}; border-radius: 3px; min-height: 30px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
        """)
        box = QWidget()
        box.setStyleSheet(f"background: {C.BG};")
        self._lay = QVBoxLayout(box)
        self._lay.setContentsMargins(24, 20, 24, 20)
        self._lay.setSpacing(14)
        self._lay.addStretch(1)
        self.setWidget(box)
        self._bubbles: list[Bubble] = []
        self._stick = True
        bar = self.verticalScrollBar()
        bar.valueChanged.connect(lambda v: setattr(self, "_stick", v >= bar.maximum() - 40))
        bar.rangeChanged.connect(lambda _a, b: self._stick and bar.setValue(b))

    def _max_bubble(self) -> int:
        return int((self.viewport().width() - 48) * 0.78)

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
        body.addWidget(self._build_left(face_path))
        body.addWidget(self._build_chat(), stretch=1)
        root.addLayout(body, stretch=1)
        root.addWidget(self._build_footer())

        self.ev.connect(self._on_event)
        clock = QTimer(self)
        clock.timeout.connect(self._tick)
        clock.start(1000)
        self._tick()

        for key, fn in (("F4", self.toggle_mic), ("F5", self.toggle_voice),
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

    def _build_left(self, face_path: str) -> QWidget:
        w = QWidget()
        w.setFixedWidth(_LEFT_W)
        w.setStyleSheet(f"background: {C.BG}; border-right: 1px solid {C.BORDER};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(18, 14, 18, 18)
        lay.setSpacing(12)
        self.hud = HudCanvas(face_path)
        self.hud.setStyleSheet("border: none;")
        self.hud.clicked.connect(self._stop)
        lay.addWidget(self.hud, stretch=1)

        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self._t_cpu = StatTile("CPU", C.PRI)
        self._t_mem = StatTile("MEMORY", C.GREEN)
        self._t_net = StatTile("NETWORK", C.AMBER)
        for t in (self._t_cpu, self._t_mem, self._t_net):
            tiles.addWidget(t)
        lay.addLayout(tiles)

        self._info = QLabel()
        self._info.setFont(_font(8.5, QFont.Weight.Bold, mono=True, spacing=1))
        self._info.setStyleSheet(f"color: {C.TEXT_3}; border: none;")
        self._info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._info)
        return w

    def _build_chat(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background: {C.BG};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        head = QHBoxLayout()
        head.setContentsMargins(24, 14, 24, 0)
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

        strip = QWidget()
        strip.setStyleSheet(f"background: {C.BG};")
        chips = QHBoxLayout(strip)
        chips.setContentsMargins(24, 4, 24, 10)
        chips.setSpacing(8)
        for text in QUICK_COMMANDS:
            chip = Chip(text)
            chip.clicked.connect(lambda _, t=text: self._submit(t))
            chips.addWidget(chip)
        chips.addStretch()
        scroller = QScrollArea()
        scroller.setWidget(strip)
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.Shape.NoFrame)
        scroller.setFixedHeight(46)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroller.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroller.setStyleSheet(f"background: {C.BG}; border: none;")
        lay.addWidget(scroller)

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
        self._input.setPlaceholderText("Жарвистай ярих эсвэл бичих…   Talk or type to JARVIS")
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
        wrap.setContentsMargins(24, 0, 24, 18)
        wrap.addWidget(bar)
        lay.addLayout(wrap)
        return w

    def _build_footer(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(28)
        w.setStyleSheet(f"background: {C.SURFACE}; border-top: 1px solid {C.BORDER};")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(22, 0, 22, 0)
        for text, align in (("F4 MIC   ·   F5 VOICE   ·   ESC STOP   ·   F11 FULLSCREEN", None),
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
        self._voice_btn.on = self.voice_on
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
        self._t_cpu.set_value(_metrics.cpu, f"{_metrics.cpu:.0f}%")
        self._t_mem.set_value(_metrics.mem, f"{_metrics.mem:.0f}%")
        net = _metrics.net
        self._t_net.set_value(min(100, net * 10), f"{net * 1024:.0f} KB/s" if net < 1 else f"{net:.1f} MB/s")
        bits = []
        b = psutil.sensors_battery()
        if b:
            bits.append(f"BAT {b.percent:.0f}%{'+' if b.power_plugged else ''}")
        up = time.time() - psutil.boot_time()
        bits.append(f"UP {int(up // 86400)}D {int(up % 86400 // 3600)}H" if up >= 86400
                    else f"UP {int(up // 3600)}H {int(up % 3600 // 60):02d}M")
        bits.append({"Darwin": "MACOS", "Windows": "WINDOWS"}.get(_OS, _OS.upper()))
        self._info.setText("  ·  ".join(bits))


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
