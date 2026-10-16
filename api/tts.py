"""
Serverless text-to-speech for the web JARVIS: Microsoft Edge's free neural voices
(Mongolian "Bataa", British English "Ryan"), returned as MP3.

POST /api/tts  {"text": "...", "lang": "mn"|"en"}  →  audio/mpeg
"""
from __future__ import annotations

import asyncio
import json
from http.server import BaseHTTPRequestHandler

VOICES = {"mn": "mn-MN-BataaNeural", "en": "en-GB-RyanNeural"}


async def _stream(text: str, voice: str) -> bytes:
    import edge_tts
    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, voice, rate="+4%").stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    return bytes(audio)


def synthesize(text: str, lang: str) -> bytes:
    return asyncio.run(asyncio.wait_for(_stream(text[:800], VOICES.get(lang, VOICES["en"])), 15))


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            audio = synthesize(str(payload.get("text", "")), payload.get("lang", "en"))
            if not audio:
                raise RuntimeError("no audio")
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(audio)
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
