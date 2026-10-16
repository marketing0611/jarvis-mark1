"""
Things JARVIS can actually do on this computer — macOS first, best effort elsewhere.
None of them need an API key.
"""
from __future__ import annotations

import os
import platform
import subprocess
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import psutil
import requests

DRY_RUN = False            # tests set this so nothing is opened or changed

_OS = platform.system()

# wttr.in weather codes → (Mongolian, English)
_WEATHER_CODES = {
    (113,): ("Цэлмэг", "Clear skies"),
    (116,): ("Багавтар үүлтэй", "Partly cloudy"),
    (119,): ("Үүлэрхэг", "Cloudy"),
    (122,): ("Бүрхэг", "Overcast"),
    (143, 248, 260): ("Манантай", "Foggy"),
    (176, 263, 266, 293, 296, 353): ("Бага зэргийн бороотой", "Light rain"),
    (299, 302, 305, 308, 356, 359): ("Бороотой", "Rain"),
    (179, 227, 323, 326, 368): ("Бага зэргийн цастай", "Light snow"),
    (230, 329, 332, 335, 338, 371): ("Цастай", "Snow"),
    (182, 185, 281, 284, 311, 314, 317, 320, 350, 362, 365, 374, 377): ("Нойтон цастай", "Sleet"),
    (200, 386, 389, 392, 395): ("Аянга цахилгаантай", "Thunderstorms"),
}


def _osascript(script: str) -> str | None:
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def open_app(name: str) -> bool:
    if DRY_RUN:
        return True
    try:
        if _OS == "Darwin":
            return subprocess.run(["open", "-a", name], capture_output=True, timeout=10).returncode == 0
        if _OS == "Windows":
            os.startfile(name)                                   # type: ignore[attr-defined]
            return True
        subprocess.Popen([name.lower()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def open_url(url: str) -> bool:
    if DRY_RUN:
        return True
    return webbrowser.open(url)


def google(query: str) -> bool:
    return open_url("https://www.google.com/search?q=" + quote(query))


def youtube(query: str) -> bool:
    return open_url("https://www.youtube.com/results?search_query=" + quote(query))


def weather(city: str) -> dict | None:
    """Current conditions from wttr.in (free, no key). None if unreachable."""
    if DRY_RUN:
        return {"temp": -4, "feels": -9, "humidity": 60, "wind": 14, "mn": "Цэлмэг", "en": "Clear skies"}
    try:
        r = requests.get(f"https://wttr.in/{quote(city)}", params={"format": "j1"}, timeout=8)
        cur = r.json()["current_condition"][0]
    except Exception:
        return None
    code = int(cur.get("weatherCode", 0))
    mn, en = next((v for k, v in _WEATHER_CODES.items() if code in k),
                  (None, cur["weatherDesc"][0]["value"]))
    return {"temp": int(cur["temp_C"]), "feels": int(cur["FeelsLikeC"]),
            "humidity": int(cur["humidity"]), "wind": int(cur["windspeedKmph"]),
            "mn": mn, "en": en}


def battery() -> tuple[int, bool] | None:
    b = psutil.sensors_battery()
    return (round(b.percent), bool(b.power_plugged)) if b else None


def system_status() -> dict:
    return {
        "cpu":  round(psutil.cpu_percent(interval=0.3)),
        "mem":  round(psutil.virtual_memory().percent),
        "disk": round(psutil.disk_usage(str(Path.home())).percent),
    }


def get_volume() -> int | None:
    if _OS != "Darwin":
        return None
    out = _osascript("output volume of (get volume settings)")
    return int(out) if out and out.isdigit() else None


def set_volume(level: int) -> bool:
    level = max(0, min(100, level))
    if DRY_RUN:
        return True
    if _OS != "Darwin":
        return False
    return _osascript(f"set volume output volume {level} without output muted") is not None


def set_muted(muted: bool) -> bool:
    if DRY_RUN:
        return True
    if _OS != "Darwin":
        return False
    return _osascript(f"set volume {'with' if muted else 'without'} output muted") is not None


def screenshot() -> Path | None:
    path = Path.home() / "Desktop" / f"JARVIS {datetime.now():%Y-%m-%d %H.%M.%S}.png"
    if DRY_RUN:
        return path
    try:
        if _OS == "Darwin":
            subprocess.run(["screencapture", "-x", str(path)], timeout=10, check=True)
        else:
            import mss
            with mss.mss() as sct:
                sct.shot(output=str(path))
        return path if path.exists() and path.stat().st_size > 0 else None
    except Exception:
        return None


def lock_screen() -> bool:
    if DRY_RUN:
        return True
    if _OS == "Darwin":
        return subprocess.run(["pmset", "displaysleepnow"], timeout=5).returncode == 0
    if _OS == "Windows":
        return subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"]).returncode == 0
    return False


def notify(title: str, message: str) -> None:
    if DRY_RUN or _OS != "Darwin":
        return
    subprocess.run([
        "osascript",
        "-e", "on run argv",
        "-e", 'display notification (item 2 of argv) with title (item 1 of argv) sound name "Glass"',
        "-e", "end run",
        title, message,
    ], timeout=5)


def start_timer(seconds: float, on_done: Callable[[], None]) -> None:
    if DRY_RUN:
        return
    timer = threading.Timer(seconds, on_done)
    timer.daemon = True
    timer.start()
