"""
Run the web version of JARVIS locally:  python dev.py   →  http://localhost:3000

Serves web/ and the same /api/chat and /api/tts handlers that Vercel runs.
"""
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from api import chat, tts  # noqa: E402


class DevHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        try:
            if self.path == "/api/chat":
                body, ctype = json.dumps(chat.handle(payload), ensure_ascii=False).encode(), "application/json"
            elif self.path == "/api/tts":
                body, ctype = tts.synthesize(str(payload.get("text", "")), payload.get("lang", "en")), "audio/mpeg"
            else:
                self.send_error(404)
                return
            status = 200
        except Exception as e:
            body, ctype, status = json.dumps({"error": str(e)}).encode(), "application/json", 502
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    print(f"J.A.R.V.I.S web  →  http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), DevHandler).serve_forever()
