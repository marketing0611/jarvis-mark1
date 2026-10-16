"""
Voice I/O for JARVIS — no API keys.

  Listener  microphone → voice-activity detection → local faster-whisper (Mongolian / English)
  Speaker   text → Microsoft Edge neural voices (free, needs internet), falling back to
            the offline macOS `say` voice for English when the network is unavailable
"""
from __future__ import annotations

import asyncio
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

SAMPLE_RATE   = 16000
FRAME_SAMPLES = 480                    # 30 ms frames
WHISPER_MODEL = "large-v3-turbo"       # ~1.6 GB, downloaded once; good Mongolian accuracy
LANGUAGES     = ("mn", "en")

VOICES = {"mn": "mn-MN-BataaNeural", "en": "en-GB-RyanNeural"}
OFFLINE_EN_VOICE = "Daniel"

_PREROLL_FRAMES = 10                   # audio kept from just before speech starts
_START_FRAMES   = 3                    # loud frames in a row that open an utterance
_END_SILENCE    = 25                   # quiet frames (~0.75 s) that close it
_MIN_SPEECH_SEC = 0.4
_MAX_SPEECH_SEC = 25.0
_ECHO_GUARD_SEC = 0.4                  # ignore the mic briefly after JARVIS stops talking

# Biases recognition toward the assistant's name and typical commands.
_PROMPTS = {
    "mn": "Жарвис аа, сайн байна уу? Цаг хэд болж байна? Хром нээгээч. Баярлалаа.",
    "en": "Hey Jarvis, what time is it? Open Chrome. Thank you.",
}

# Phrases whisper tends to "hear" in background noise.
_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks for watching!", "thanks for watching.", "you",
    "bye.", "subtitles by the amara.org community", "продолжение следует...",
}

_model = None
_model_lock = threading.Lock()


def _whisper():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        return _model


def transcribe(audio: np.ndarray) -> tuple[str, str]:
    """Transcribe mono float32 16 kHz audio as Mongolian or English. Returns (text, lang)."""
    model = _whisper()
    segments, info = model.transcribe(audio, beam_size=1, vad_filter=True,
                                      condition_on_previous_text=False)
    lang = info.language
    if lang not in LANGUAGES:
        # Whisper often mistakes Mongolian for Russian/Kazakh; pick the likelier of the two.
        probs = dict(info.all_language_probs or [])
        lang  = max(LANGUAGES, key=lambda code: probs.get(code, 0.0))
        segments, _ = model.transcribe(audio, language=lang, beam_size=1, vad_filter=True,
                                       condition_on_previous_text=False,
                                       initial_prompt=_PROMPTS[lang])
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, lang


class Listener:
    """
    Listens continuously and calls `on_utterance(text, lang)` for each finished phrase.

    `is_blocked()` is polled every frame — while it returns True (mic muted or
    JARVIS speaking) audio is discarded so JARVIS never transcribes itself.
    """

    def __init__(
        self,
        on_utterance: Callable[[str, str], None],
        on_state: Callable[[str], None],
        on_level: Callable[[float], None],
        is_blocked: Callable[[], bool],
    ):
        self._on_utterance = on_utterance
        self._on_state     = on_state
        self._on_level     = on_level
        self._is_blocked   = is_blocked
        self._frames: queue.Queue[np.ndarray] = queue.Queue()
        self._quiet_until  = 0.0

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="jarvis-listener").start()

    def _callback(self, indata, frames, time_info, status):
        self._frames.put(indata[:, 0].copy())

    def _run(self) -> None:
        self._on_state("LOADING")
        try:
            _whisper()
        except Exception as e:
            self._on_state(f"ERROR:Speech model failed to load — {e}")
            return
        try:
            import sounddevice as sd
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                    blocksize=FRAME_SAMPLES, callback=self._callback)
        except Exception as e:
            self._on_state(f"ERROR:Microphone unavailable — {e}")
            return
        with stream:
            self._on_state("READY")
            self._loop()

    def _loop(self) -> None:
        noise = 0.004
        preroll: deque[np.ndarray] = deque(maxlen=_PREROLL_FRAMES)
        speech: list[np.ndarray] = []
        loud_run = quiet_run = 0
        in_speech = False

        while True:
            frame = self._frames.get()
            rms   = float(np.sqrt(np.mean(frame * frame)))

            if self._is_blocked():
                self._quiet_until = time.time() + _ECHO_GUARD_SEC
            if time.time() < self._quiet_until:
                if in_speech:
                    self._on_state("READY")
                preroll.clear(); speech = []
                loud_run = quiet_run = 0
                in_speech = False
                self._on_level(0.0)
                continue

            threshold = max(noise * 3.2, 0.012)
            self._on_level(min(1.0, rms / (threshold * 4)))

            if not in_speech:
                preroll.append(frame)
                if rms > threshold:
                    loud_run += 1
                    if loud_run >= _START_FRAMES:
                        in_speech, speech, quiet_run = True, list(preroll), 0
                        self._on_state("HEARING")
                else:
                    loud_run = 0
                    noise = 0.95 * noise + 0.05 * rms
                continue

            speech.append(frame)
            quiet_run = quiet_run + 1 if rms < threshold * 0.7 else 0
            duration  = len(speech) * FRAME_SAMPLES / SAMPLE_RATE
            if quiet_run < _END_SILENCE and duration < _MAX_SPEECH_SEC:
                continue

            in_speech, loud_run = False, 0
            preroll.clear()
            voiced = duration - quiet_run * FRAME_SAMPLES / SAMPLE_RATE
            audio, speech = np.concatenate(speech), []
            if voiced < _MIN_SPEECH_SEC:
                self._on_state("READY")
                continue

            self._on_state("TRANSCRIBING")
            try:
                text, lang = transcribe(audio)
            except Exception as e:
                self._on_state(f"ERROR:Transcription failed — {e}")
                continue
            self._on_state("READY")
            text = text.strip()
            if len(text) > 1 and text.lower() not in _HALLUCINATIONS:
                self._on_utterance(text, lang)


# ── Text to speech ───────────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")


def _speakable(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_`#>|~•]+", "", text)
    text = text.replace("°C", " градус" if re.search(r"[а-яөү]", text, re.I) else " degrees")
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text)     # emoji
    return re.sub(r"\s+", " ", text).strip()


async def _synthesize(text: str, voice: str, path: str) -> None:
    import edge_tts
    await edge_tts.Communicate(text, voice, rate="+4%").save(path)


class Speaker:
    """
    Speaks replies sentence by sentence. Synthesis of the next sentence overlaps
    playback of the current one; `stop()` cuts everything off immediately.
    """

    def __init__(self, on_speaking: Callable[[bool], None]):
        self._on_speaking = on_speaking
        self._jobs:  queue.Queue = queue.Queue()      # (generation, sentence, lang)
        self._ready: queue.Queue = queue.Queue()      # (generation, path | None, sentence, lang)
        self._gen     = 0
        self._pending = 0
        self._lock    = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self.enabled  = True
        self._has_afplay = shutil.which("afplay") is not None
        self._has_say    = shutil.which("say") is not None
        threading.Thread(target=self._synth_loop, daemon=True, name="jarvis-tts").start()
        threading.Thread(target=self._play_loop, daemon=True, name="jarvis-audio").start()

    @property
    def speaking(self) -> bool:
        return self._pending > 0

    def say(self, text: str, lang: str) -> None:
        if not self.enabled:
            return
        sentences = [s for s in (_speakable(p) for p in _SENTENCE_END.split(text)) if s]
        if not sentences:
            return
        with self._lock:
            first = self._pending == 0
            self._pending += len(sentences)
            gen = self._gen
        if first:
            self._on_speaking(True)
        for sentence in sentences:
            self._jobs.put((gen, sentence, lang))

    def stop(self) -> None:
        with self._lock:
            self._gen += 1
            was_speaking, self._pending = self._pending > 0, 0
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        if was_speaking:
            self._on_speaking(False)

    def _synth_loop(self) -> None:
        while True:
            gen, sentence, lang = self._jobs.get()
            if gen != self._gen:
                continue
            path = None
            if self._has_afplay:
                fd, path = tempfile.mkstemp(suffix=".mp3", prefix="jarvis_")
                os.close(fd)
                try:
                    asyncio.run(asyncio.wait_for(_synthesize(sentence, VOICES[lang], path), 10))
                except Exception as e:
                    print(f"[Voice] ⚠️ Neural voice unavailable ({e}); using offline voice.")
                    os.unlink(path)
                    path = None
            self._ready.put((gen, path, sentence, lang))

    def _play_loop(self) -> None:
        while True:
            gen, path, sentence, lang = self._ready.get()
            if gen == self._gen:
                if path:
                    cmd = ["afplay", path]
                elif lang == "en" and self._has_say:
                    cmd = ["say", "-v", OFFLINE_EN_VOICE, "-r", "185", sentence]
                else:
                    cmd = None                          # no offline Mongolian voice
                if cmd:
                    with self._lock:
                        self._proc = subprocess.Popen(cmd)
                    self._proc.wait()
            if path:
                os.unlink(path)
            with self._lock:
                if gen != self._gen:
                    continue
                self._pending -= 1
                done = self._pending == 0
            if done:
                self._on_speaking(False)
