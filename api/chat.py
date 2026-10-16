"""
Serverless endpoint for the web JARVIS (Vercel). It runs the same conversation
engine as the desktop app (core.brain); skills that would touch a computer become
actions for the browser instead (open a tab, start a timer). The function keeps no
state of its own: conversation state and memory travel with every request.

POST /api/chat  {"text": "...", "lang": "mn"|"en"|null, "state": {...}, "memory": {...},
                 "tz_offset": minutes east of UTC, "battery": {"level": 0-1, "charging": bool}|null,
                 "greeting": true to get the startup line instead}
→ {"replies": [{"text", "speech", "lang"}], "actions": [...], "state": {...}, "memory": {...}}
"""
from __future__ import annotations

import contextvars
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.brain as brain_mod                      # noqa: E402
from core import skills                             # noqa: E402
from core.brain import App, Brain, Reply, _State    # noqa: E402

# Desktop apps that have a reasonable web counterpart.
WEB_APPS = {
    "Google Chrome": "https://www.google.com", "Safari": "https://www.google.com",
    "Firefox": "https://www.google.com", "Spotify": "https://open.spotify.com",
    "Telegram": "https://web.telegram.org", "Discord": "https://discord.com/app",
    "Visual Studio Code": "https://vscode.dev", "Notes": "https://keep.google.com",
    "Calculator": "https://www.google.com/search?q=calculator", "Calendar": "https://calendar.google.com",
    "Music": "https://music.apple.com", "Photos": "https://photos.google.com",
    "Mail": "https://mail.google.com", "Maps": "https://maps.google.com",
    "App Store": "https://apps.apple.com",
}
DESKTOP_ONLY = {
    "mn": "Энэ үйлдэл зөвхөн desktop хувилбарт ажиллана, ноёнтоон — вэб хөтчөөс компьютерийг удирдах боломжгүй.",
    "en": "That one only works in the desktop app, sir — a web page can't control your computer.",
}


@dataclass
class _Session:
    tz_offset: int = 0
    battery: dict | None = None
    memory: dict = field(default_factory=dict)
    actions: list = field(default_factory=list)
    timers: list = field(default_factory=list)
    detecting: bool = False


_session: contextvars.ContextVar[_Session] = contextvars.ContextVar("session")


# ── core.skills, redirected to the browser ──
def _open(url: str) -> bool:
    if not _session.get().detecting:
        _session.get().actions.append({"type": "open", "url": url})
    return True


def _start_timer(seconds: float, on_done) -> None:
    if not _session.get().detecting:
        _session.get().timers.append((seconds, on_done))


def _battery():
    b = _session.get().battery
    return (round(b["level"] * 100), bool(b.get("charging"))) if b else None


def _server_status() -> dict:
    """JARVIS's own (server) load — used in small talk like "how are you"."""
    load = os.getloadavg()[0] / (os.cpu_count() or 1) if hasattr(os, "getloadavg") else 0.1
    return {"cpu": min(100, round(load * 100)), "mem": 0, "disk": round(
        100 * (1 - shutil.disk_usage("/tmp").free / shutil.disk_usage("/tmp").total))}


skills.open_url = _open
skills.open_app = lambda name: _open(WEB_APPS[name]) if name in WEB_APPS else False
skills.google = lambda q: _open("https://www.google.com/search?q=" + quote(q))
skills.youtube = lambda q: _open("https://www.youtube.com/results?search_query=" + quote(q))
skills.start_timer = _start_timer
skills.battery = _battery
skills.system_status = _server_status
skills.notify = lambda *a: None
skills.get_volume = lambda: None
skills.set_volume = lambda level: True
skills.set_muted = lambda muted: True
skills.screenshot = lambda: Path("screenshot.png")
skills.lock_screen = lambda: True


class _ClientClock(datetime):
    """`datetime.now()` in the visitor's time zone, so "what time is it" is right."""

    @classmethod
    def now(cls, tz=None):
        local = datetime.now(timezone.utc) + timedelta(minutes=_session.get().tz_offset)
        return local.replace(tzinfo=None)


brain_mod.datetime = _ClientClock
brain_mod.load_memory = lambda: _session.get().memory


def _update_memory(update: dict) -> dict:
    memory = _session.get().memory
    for category, items in update.items():
        memory.setdefault(category, {}).update(items)
    return memory


brain_mod.update_memory = _update_memory


class WebBrain(Brain):
    """The desktop brain, minus the things a browser can't do."""

    def _desktop_only(self, handler, msg, lang):
        session = _session.get()
        session.detecting = True
        try:
            matched = handler(msg, lang)
        finally:
            session.detecting = False
        return Reply(DESKTOP_ONLY[lang], lang) if matched else None

    def _volume(self, msg, lang):
        return self._desktop_only(super()._volume, msg, lang)

    def _screenshot(self, msg, lang):
        return self._desktop_only(super()._screenshot, msg, lang)

    def _lock(self, msg, lang):
        return self._desktop_only(super()._lock, msg, lang)

    def _system(self, msg, lang):
        return self._desktop_only(super()._system, msg, lang)

    def _battery(self, msg, lang):
        if _session.get().battery is None:
            return self._desktop_only(super()._battery, msg, lang)
        return super()._battery(msg, lang)

    def _launch(self, app: App, lang: str) -> Reply:
        if not app.is_url and app.target not in WEB_APPS:
            return Reply({"mn": f"{app.label}-г вэбээс нээх боломжгүй — desktop хувилбарт ажиллана.",
                          "en": f"I can't open {app.label} from a web page — that needs the desktop app."}[lang],
                         lang)
        return super()._launch(app, lang)


# ── state that travels with each request ──
def _encode(v):
    if isinstance(v, Reply):
        return {"$reply": asdict(v)}
    if isinstance(v, set):
        return {"$set": [_encode(x) for x in v]}
    if isinstance(v, tuple):
        return {"$tuple": [_encode(x) for x in v]}
    if isinstance(v, list):
        return [_encode(x) for x in v]
    if isinstance(v, dict):
        return {k: _encode(x) for k, x in v.items()}
    return v


def _decode(v):
    if isinstance(v, dict):
        if "$reply" in v:
            return Reply(**v["$reply"])
        if "$set" in v:
            return {_decode(x) for x in v["$set"]}
        if "$tuple" in v:
            return tuple(_decode(x) for x in v["$tuple"])
        return {k: _decode(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_decode(x) for x in v]
    return v


def _reply_json(r: Reply) -> dict:
    return {"text": r.text, "speech": r.speech, "lang": r.lang}


def handle(payload: dict) -> dict:
    """One conversational turn. Pure function of the payload — shared by Vercel and dev.py."""
    session = _Session(tz_offset=int(payload.get("tz_offset") or 0), battery=payload.get("battery"),
                       memory=payload.get("memory") or {})
    token = _session.set(session)
    try:
        timer_replies: list[Reply] = []
        brain = WebBrain(on_reply=timer_replies.append)
        saved = payload.get("state") or {}
        for f in fields(_State):
            if f.name in saved:
                try:
                    setattr(brain.s, f.name, _decode(saved[f.name]))
                except Exception:
                    pass

        if payload.get("greeting"):
            reply = brain.greeting()
        else:
            reply = brain.respond(str(payload.get("text", ""))[:500], payload.get("lang"))

        replies = [_reply_json(reply)] if reply and not reply.silent else []
        actions = list(session.actions)
        for seconds, on_done in session.timers:     # the browser counts down; we pre-write the alert
            timer_replies.clear()
            on_done()
            if timer_replies:
                actions.append({"type": "timer", "seconds": seconds, "reply": _reply_json(timer_replies[-1])})
        return {"replies": replies, "actions": actions, "silent": bool(reply and reply.silent),
                "state": {f.name: _encode(getattr(brain.s, f.name)) for f in fields(_State)},
                "memory": session.memory}
    finally:
        _session.reset(token)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            body, status = json.dumps(handle(payload), ensure_ascii=False).encode(), 200
        except Exception as e:
            body, status = json.dumps({"error": str(e)}).encode(), 500
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
