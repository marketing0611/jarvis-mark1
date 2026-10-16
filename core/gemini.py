"""
Gemini for JARVIS — answers anything the local brain has no rule for.

Uses Google's free Gemini API tier. The key comes from config/api_keys.json
({"gemini_api_key": "..."}) or the GEMINI_API_KEY environment variable; without
one, `available` is False and JARVIS stays in offline demo mode.
"""
from __future__ import annotations

import os
import re
import threading
import time
from collections import deque

from memory.config_manager import load_api_keys
from memory.memory_manager import format_memory_for_prompt, load_memory

# Tried in order. Each model has its own free-tier quota, so when one is used up
# (429) or retired (404) the next one takes over. "gemini_model" in
# config/api_keys.json goes first.
MODELS = ["gemini-3.5-flash-lite", "gemini-2.5-flash-lite", "gemini-2.5-flash"]
HISTORY_TURNS = 8                      # user/JARVIS exchanges sent along as context
TIMEOUT_MS = 20_000
COOLDOWN_S = 300                       # skip a model this long after it runs out of quota

_PERSONA = """You are J.A.R.V.I.S., Tony Stark's AI assistant, running on the user's computer.
Personality: calm, witty, loyal, very capable. Address the user as "{addr}".
Your replies are shown on screen AND read aloud, so:
- Keep them short: one to three sentences unless the user asks for detail.
- Plain text only: no markdown, no bullet symbols, no emoji.
- Reply in {language}.
The local system already handles the time, weather, opening apps, timers, volume and
screenshots; if asked to do something on the computer you cannot do, say so briefly.
Today is {today}."""

_LANGUAGES = {"mn": "Mongolian, written in Cyrillic script", "en": "English"}


def _api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY") or load_api_keys().get("gemini_api_key")
    return key.strip() if key and key.strip() else None


def _plain(text: str) -> str:
    """Strip the markdown Gemini sometimes adds anyway, so the speaker doesn't read it out."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"^\s*(#+|[-*•]|\d+\.)\s+", "", text, flags=re.M)
    text = re.sub(r"[*_`#]", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class Gemini:
    def __init__(self):
        self._key = _api_key()
        self._client = None
        self._history: deque[tuple[str, str]] = deque(maxlen=HISTORY_TURNS)
        self._lock = threading.Lock()
        custom = load_api_keys().get("gemini_model")
        self._models = ([custom] if custom else []) + [m for m in MODELS if m != custom]
        self._resting: dict[str, float] = {}          # model -> time its cooldown ends
        self.last_error: str | None = None
        self.problem: str | None = None               # "key" or "billing" when the account needs fixing

    @property
    def available(self) -> bool:
        return self._key is not None

    def note(self, user: str, reply: str) -> None:
        """Remember an exchange the local brain answered, so Gemini keeps the thread."""
        with self._lock:
            self._history.append((user, reply))

    def ask(self, text: str, lang: str, addr: str, today: str) -> str | None:
        """Gemini's answer in `lang`, or None if it is unavailable or every model failed."""
        if not self._key:
            return None
        from google import genai
        from google.genai import errors, types

        if self._client is None:
            self._client = genai.Client(api_key=self._key,
                                        http_options=types.HttpOptions(timeout=TIMEOUT_MS))

        system = _PERSONA.format(addr=addr, language=_LANGUAGES.get(lang, "English"), today=today)
        memory = format_memory_for_prompt(load_memory())
        if memory:
            system += "\n\nWhat you remember about the user:\n" + memory

        with self._lock:
            contents = []
            for user, reply in self._history:
                contents.append(types.Content(role="user", parts=[types.Part(text=user)]))
                contents.append(types.Content(role="model", parts=[types.Part(text=reply)]))
        contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
        config = types.GenerateContentConfig(system_instruction=system, temperature=0.7)

        now = time.monotonic()
        ready = [m for m in self._models if self._resting.get(m, 0) <= now]
        for model in ready + [m for m in self._models if m not in ready]:
            try:
                response = self._client.models.generate_content(
                    model=model, contents=contents, config=config)
            except errors.APIError as e:
                self.last_error = f"{model}: {e.code} {e.message or ''}".strip()
                print(f"[Gemini] ⚠️ {self.last_error}")
                if e.code == 404:                       # model retired: stop trying it
                    self._models.remove(model)
                    continue
                if e.code == 429:                       # out of free quota for now
                    self._resting[model] = time.monotonic() + COOLDOWN_S
                    continue
                if e.code in (500, 503):                # overloaded: try the next one
                    continue
                if e.code in (400, 401, 403) and "key" in (e.message or "").lower():
                    self.problem = "key"
                elif e.code == 402:                     # paid project with no credit left
                    self.problem = "billing"
                return None                             # account or request problem: other models won't help
            except Exception as e:                      # network down, timeout
                self.last_error = f"{type(e).__name__}: {e}"
                print(f"[Gemini] ⚠️ {self.last_error}")
                return None

            answer = _plain(response.text or "")
            if answer:
                self.last_error = self.problem = None
                self.note(text, answer)
                return answer
        return None
