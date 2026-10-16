"""
J.A.R.V.I.S — Mark XXXIX, local demo.

Talk or type to JARVIS in Mongolian or English. Speech recognition and everyday
commands run on this computer; other questions go to Gemini's free tier when
config/api_keys.json has a "gemini_api_key". Spoken replies use Microsoft Edge's
free neural voices when online (offline: macOS `say`, English only).
"""
from core.brain import Brain, Reply
from core.voice import Listener, Speaker
from ui import JarvisUI


def main():
    ui = JarvisUI("face.png")
    speaker = Speaker(on_speaking=ui.set_speaking)

    def deliver(reply: Reply):
        if reply.silent:
            return
        ui.add_reply(reply.text)
        speaker.say(reply.speech or reply.text, reply.lang)

    brain = Brain(on_reply=deliver, on_busy=ui.set_busy)
    if brain.ai.available:
        ui.add_note("Gemini холбогдсон  ·  Gemini on")
    else:
        ui.add_note("Gemini түлхүүр алга — config/api_keys.json  ·  No Gemini key, demo mode")

    def typed(text: str):
        speaker.stop()                     # a new request interrupts whatever JARVIS was saying
        brain.submit(text)

    def heard(text: str, lang: str):
        ui.add_user_message(text)
        brain.submit(text, lang_hint=lang)

    voice_ready = False

    def listener_state(state: str):
        nonlocal voice_ready
        if state.startswith("ERROR:"):
            ui.add_note(state.removeprefix("ERROR:"), error=True)
            state = "ERROR"
        elif state == "READY" and not voice_ready:
            voice_ready = True
            ui.add_note("Микрофон бэлэн — шууд яриарай  ·  Mic ready, just talk")
        ui.set_listener_state(state)

    def voice_toggled(on: bool):
        speaker.enabled = on
        if not on:
            speaker.stop()

    ui.on_submit = typed
    ui.on_stop = speaker.stop
    ui.on_voice_toggled = voice_toggled

    Listener(on_utterance=heard, on_state=listener_state, on_level=ui.set_mic_level,
             is_blocked=lambda: ui.muted or speaker.speaking).start()

    deliver(brain.greeting())
    ui.root.mainloop()


if __name__ == "__main__":
    main()
