// Web JARVIS: chat, voice (browser speech recognition + neural TTS), the two HUD views.
import { C, Reactor } from "./hud.js";
import { Vision } from "./vision.js";

const $ = sel => document.querySelector(sel);
const store = {
  get(key, fallback) { try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ } },
};

const QUICK = ["Сайн уу, Жарвис", "Цаг хэд болж байна?", "Цаг агаар ямар байна?", "Онигоо яриач", "Юу чадах вэ?", "What can you do?"];
const PLACEHOLDER = "Жарвист бичих…";
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

const app = {
  state: store.get("jarvis.state", {}), memory: store.get("jarvis.memory", {}),
  voiceOn: store.get("jarvis.voice", true), sttLang: store.get("jarvis.sttLang", "mn"),
  busy: false, speaking: false, online: true, micOn: false, hearing: false,
  mode: "reactor", latency: null, battery: null, started: Date.now(),
  micLevel: 0, voiceLevel: 0,
};

const messages = $("#messages"), input = $("#input");
const reactor = new Reactor($("#reactor"));
const vision = new Vision($("#vision"));

// ── chat log ──
function addMessage(text, mine) {
  const el = document.createElement("div");
  el.className = `msg${mine ? " me" : ""}`;
  const cap = document.createElement("div");
  cap.className = "cap";
  cap.textContent = `${mine ? "YOU" : "JARVIS"}  ·  ${new Date().toTimeString().slice(0, 5)}`;
  const body = document.createElement("div");
  body.className = "body";
  el.append(cap, body);
  messages.append(el);
  if (mine) body.textContent = text;
  else {                                   // type it out, ~1 s even for long replies
    const step = Math.max(3, Math.ceil(text.length / 60));
    let shown = 0;
    const tick = () => {
      shown = Math.min(text.length, shown + step);
      body.textContent = text.slice(0, shown);
      messages.scrollTop = messages.scrollHeight;
      if (shown < text.length) requestAnimationFrame(tick);
    };
    tick();
  }
  messages.scrollTop = messages.scrollHeight;
  return el;
}

function addNote(text, error = false) {
  const el = document.createElement("div");
  el.className = `note${error ? " err" : ""}`;
  el.textContent = text;
  messages.append(el);
  messages.scrollTop = messages.scrollHeight;
}

function addLink(msgEl, url) {
  let links = msgEl.querySelector(".links");
  if (!links) { links = document.createElement("div"); links.className = "links"; msgEl.append(links); }
  const a = document.createElement("a");
  a.href = url; a.target = "_blank"; a.rel = "noopener";
  a.textContent = `${new URL(url).hostname.replace(/^www\./, "")} ↗`;
  links.append(a);
}

// ── state shown in the pill and the HUDs ──
function displayState() {
  if (!app.online) return ["OFFLINE", C.red];
  if (app.speaking) return ["SPEAKING", C.acc];
  if (app.busy) return ["THINKING", C.amber];
  if (app.micOn && app.hearing) return ["HEARING YOU", C.green];
  if (app.micOn) return ["LISTENING", C.green];
  return ["STANDBY", C.pri];
}

function refreshState() {
  const [label, color] = displayState();
  $("#pill-text").textContent = label;
  document.documentElement.style.setProperty("--state", color);
  for (const view of [reactor, vision]) { view.label = label; view.color = color; }
  reactor.speaking = app.speaking; reactor.thinking = app.busy; reactor.listening = app.micOn && !app.speaking;
  $("#mic").classList.toggle("live", app.micOn);
  $("#voice").classList.toggle("off", !app.voiceOn);
  $("#lang").textContent = app.sttLang.toUpperCase();
}

// ── talking to the brain (one request at a time) ──
let chain = Promise.resolve();

function ask(payload) {
  const run = async () => {
    const t0 = performance.now();
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...payload, state: app.state, memory: app.memory,
                             tz_offset: -new Date().getTimezoneOffset(), battery: app.battery }),
    });
    app.latency = performance.now() - t0;
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    app.state = data.state; app.memory = data.memory;
    store.set("jarvis.state", app.state); store.set("jarvis.memory", app.memory);
    return data;
  };
  const p = chain.then(run, run);
  chain = p.catch(() => {});
  return p;
}

function deliver(reply) {
  const el = addMessage(reply.text, false);
  speak(reply.speech || reply.text, reply.lang);
  return el;
}

async function submit(text, lang = null) {
  text = text.trim();
  if (!text) return;
  addMessage(text, true);
  stopSpeaking();
  app.busy = true; refreshState();
  try {
    const data = await ask({ text, lang });
    app.online = true;
    let last = null;
    for (const r of data.replies) last = deliver(r);
    for (const action of data.actions) {
      if (action.type === "open") {
        const w = window.open(action.url, "_blank");
        if (w) w.opener = null;
        if (last) addLink(last, action.url);
      } else if (action.type === "timer") {
        setTimeout(() => deliver(action.reply), action.seconds * 1000);
      }
    }
  } catch (e) {
    app.online = false;
    addNote("Сервертэй холбогдож чадсангүй — дахин оролдоно уу.  ·  Can't reach the server.", true);
  } finally {
    app.busy = false; refreshState();
  }
}

// ── speaking ──
const audio = new Audio();
let audioCtx = null, voiceAnalyser = null, speechGen = 0, speechQueue = Promise.resolve(), pending = 0;

function speakable(text) {
  const mn = /[а-яөү]/i.test(text);
  return text.replace(/https?:\/\/\S+/g, "").replace(/[*_`#>|~•]+/g, "")
    .replace(/°C/g, mn ? " градус" : " degrees").replace(/\p{Extended_Pictographic}/gu, "")
    .replace(/\s+/g, " ").trim();
}

function ensureAudioGraph() {
  if (audioCtx) return;
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    voiceAnalyser = audioCtx.createAnalyser();
    voiceAnalyser.fftSize = 512;
    audioCtx.createMediaElementSource(audio).connect(voiceAnalyser);
    voiceAnalyser.connect(audioCtx.destination);
  } catch { audioCtx = null; voiceAnalyser = null; }
}

function setSpeaking(on) {
  if (app.speaking === on) return;
  app.speaking = on;
  if (on) stopRecognition(); else if (app.micOn) setTimeout(startRecognition, 300);
  refreshState();
}

function speak(text, lang) {
  text = speakable(text);
  if (!app.voiceOn || !text) return;
  const gen = speechGen;
  pending++;
  setSpeaking(true);
  speechQueue = speechQueue.then(async () => {
    if (gen !== speechGen) return;
    try { await playNeural(text, lang, gen); }
    catch { if (gen === speechGen) await speakWithBrowser(text, lang); }
  }).finally(() => {
    if (gen !== speechGen) return;
    pending = Math.max(0, pending - 1);
    if (!pending) setSpeaking(false);
  });
}

async function playNeural(text, lang, gen) {
  const res = await fetch("/api/tts", { method: "POST", headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({ text, lang }) });
  if (!res.ok) throw new Error("tts");
  const url = URL.createObjectURL(await res.blob());
  if (gen !== speechGen) return;
  ensureAudioGraph();
  if (audioCtx?.state === "suspended") audioCtx.resume().catch(() => {});
  audio.src = url;
  try {
    await audio.play();
    await new Promise(resolve => { audio.onended = audio.onpause = resolve; });
  } finally { URL.revokeObjectURL(url); }
}

function speakWithBrowser(text, lang) {
  return new Promise(resolve => {
    const synth = window.speechSynthesis;
    if (!synth) return resolve();
    const voices = synth.getVoices();
    const voice = lang === "mn" ? voices.find(v => v.lang.startsWith("mn"))
      : voices.find(v => /en-GB/.test(v.lang)) || voices.find(v => v.lang.startsWith("en"));
    if (!voice) return resolve();                   // no Mongolian voice here: stay quiet
    const u = new SpeechSynthesisUtterance(text);
    u.voice = voice; u.lang = voice.lang; u.onend = u.onerror = resolve;
    synth.speak(u);
  });
}

function stopSpeaking() {
  speechGen++;
  pending = 0;
  audio.pause();
  window.speechSynthesis?.cancel();
  speechQueue = Promise.resolve();
  setSpeaking(false);
}

// ── listening (browser speech recognition; Chrome and Edge support Mongolian) ──
let rec = null, micStream = null, micAnalyser = null;

function startRecognition() {
  if (!SR || !app.micOn || app.speaking || rec) return;
  rec = new SR();
  rec.lang = app.sttLang === "mn" ? "mn-MN" : "en-US";
  rec.continuous = true;
  rec.interimResults = true;
  rec.onspeechstart = () => { app.hearing = true; refreshState(); };
  rec.onresult = e => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      if (r.isFinal) { input.placeholder = PLACEHOLDER; submit(r[0].transcript, app.sttLang); }
      else interim += r[0].transcript;
    }
    if (interim) input.placeholder = `🎙 ${interim}`;
  };
  rec.onerror = e => {
    if (e.error === "not-allowed" || e.error === "service-not-allowed") {
      app.micOn = false;
      addNote("Микрофоны зөвшөөрөл олгогдоогүй байна.  ·  Microphone access was blocked.", true);
    } else if (e.error === "language-not-supported") {
      addNote("Энэ хөтөч монгол яриа танихгүй байна — EN горимд шилжүүлэх эсвэл Chrome ашиглана уу.", true);
    }
  };
  rec.onend = () => {
    rec = null; app.hearing = false; input.placeholder = PLACEHOLDER; refreshState();
    if (app.micOn && !app.speaking) setTimeout(startRecognition, 250);
  };
  try { rec.start(); } catch { rec = null; }
}

function stopRecognition() {
  if (rec) { rec.onend = null; try { rec.abort(); } catch { /* already stopped */ } }
  rec = null; app.hearing = false; input.placeholder = PLACEHOLDER;
}

async function toggleMic() {
  if (!SR) {
    addNote("Таны хөтөч яриа таних боломжгүй — Chrome эсвэл Edge ашиглана уу.  ·  Use Chrome or Edge for voice.", true);
    return;
  }
  app.micOn = !app.micOn;
  if (app.micOn) {
    try {                                   // a level meter for the HUD visualiser
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      ensureAudioGraph();
      if (audioCtx) {
        micAnalyser = audioCtx.createAnalyser();
        micAnalyser.fftSize = 512;
        audioCtx.createMediaStreamSource(micStream).connect(micAnalyser);
        audioCtx.resume().catch(() => {});
      }
    } catch {
      app.micOn = false;
      addNote("Микрофоны зөвшөөрөл олгогдоогүй байна.  ·  Microphone access was blocked.", true);
    }
    startRecognition();
  } else {
    stopRecognition();
    micStream?.getTracks().forEach(t => t.stop());
    micStream = null; micAnalyser = null;
  }
  refreshState();
}

function level(analyser) {
  if (!analyser) return 0;
  const data = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(data);
  let sum = 0;
  for (const v of data) sum += ((v - 128) / 128) ** 2;
  return Math.min(1, Math.sqrt(sum / data.length) * 6);
}

// ── views ──
function setMode(mode) {
  app.mode = mode;
  document.querySelectorAll(".seg button").forEach(b => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
  $("#reactor").hidden = mode !== "reactor";
  $("#vision").hidden = mode !== "vision";
  if (mode === "vision") vision.start(); else vision.stop();
}

function platformName() {
  const p = (navigator.userAgentData?.platform || navigator.platform || "").toLowerCase();
  const ua = navigator.userAgent.toLowerCase();
  if (/iphone|ipad/.test(ua)) return "IOS";
  if (/android/.test(ua)) return "ANDROID";
  if (p.includes("mac")) return "MACOS";
  if (p.includes("win")) return "WINDOWS";
  return p ? p.toUpperCase().slice(0, 10) : "WEB";
}

function everySecond() {
  const now = new Date();
  $("#clock").textContent = now.toTimeString().slice(0, 8);
  $("#date").textContent = now.toLocaleDateString("en-GB", { weekday: "short", day: "2-digit", month: "short", year: "numeric" }).toUpperCase();
  const secs = Math.floor((Date.now() - app.started) / 1000);
  const net = navigator.connection?.effectiveType?.toUpperCase() || (navigator.onLine ? "ONLINE" : "OFFLINE");
  const bat = app.battery ? `${Math.round(app.battery.level * 100)}%${app.battery.charging ? " +" : ""}` : "—";
  reactor.readouts = {
    tl: [["CORE", app.online ? "ONLINE" : "OFFLINE"], ["MODE", "WEB DEMO"]],
    tr: [["LANGUAGE", "МОН · ENG"], ["VOICE", app.voiceOn ? "ON" : "OFF"]],
    bl: [["NETWORK", net], ["SESSION", `${String(Math.floor(secs / 60)).padStart(2, "0")}:${String(secs % 60).padStart(2, "0")}`]],
    br: [["BATTERY", bat], ["SYSTEM", platformName()]],
  };
  $("#g-net").textContent = net;
}

let last = performance.now(), ringTick = 0;
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  app.micLevel += (level(micAnalyser) - app.micLevel) * 0.4;
  app.voiceLevel += ((app.speaking ? Math.max(level(voiceAnalyser), 0.35 + 0.3 * Math.random()) : 0) - app.voiceLevel) * 0.4;
  const lat = app.latency;
  if (app.mode === "reactor") {
    reactor.level = app.speaking ? app.voiceLevel : app.micLevel;
    reactor.gauges = [[lat ? Math.min(100, lat / 10) : 0, lat ? `LAT ${Math.round(lat)}MS` : "LAT —"],
                      [app.micLevel * 100, `MIC ${Math.round(app.micLevel * 100)}%`]];
    reactor.step(dt); reactor.draw();
  } else {
    vision.step(dt); vision.draw();
    if ((ringTick += dt) > 0.15) {
      ringTick = 0;
      $("#ring-a").style.strokeDasharray = `${(lat ? Math.min(1, lat / 1000) : 0) * 201} 999`;
      $("#ring-b").style.strokeDasharray = `${app.micLevel * 138} 999`;
      $("#g-mic").textContent = `${Math.round(app.micLevel * 100)}%`;
      $("#g-lat").textContent = lat ? `${Math.round(lat)} MS` : "—";
    }
  }
  requestAnimationFrame(frame);
}

// ── wiring ──
for (const text of QUICK) {
  const b = document.createElement("button");
  b.type = "button"; b.textContent = text; b.title = text;
  b.onclick = () => submit(text);
  $("#chips").append(b);
}
$("#bar").addEventListener("submit", e => { e.preventDefault(); const t = input.value; input.value = ""; submit(t); });
$("#mic").onclick = toggleMic;
$("#voice").onclick = () => { app.voiceOn = !app.voiceOn; store.set("jarvis.voice", app.voiceOn); if (!app.voiceOn) stopSpeaking(); refreshState(); };
$("#lang").onclick = () => {
  app.sttLang = app.sttLang === "mn" ? "en" : "mn";
  store.set("jarvis.sttLang", app.sttLang);
  if (rec) { stopRecognition(); startRecognition(); }
  refreshState();
};
document.querySelectorAll(".seg button").forEach(b => { b.onclick = () => setMode(b.dataset.mode); });
$("#reactor").onclick = stopSpeaking;
$("#vision-canvas").onclick = () => (vision.status === "error" ? vision.start() : stopSpeaking());
document.addEventListener("keydown", e => {
  if (e.key === "Escape") stopSpeaking();
  if (!e.altKey) return;
  const act = { KeyM: toggleMic, KeyV: () => $("#voice").click(), KeyC: () => setMode(app.mode === "reactor" ? "vision" : "reactor") }[e.code];
  if (act) { e.preventDefault(); act(); }
});
if (!SR) { $("#mic").disabled = true; $("#mic").title = "Яриа таних: Chrome эсвэл Edge хэрэгтэй"; $("#lang").hidden = true; }
navigator.getBattery?.().then(b => {
  const update = () => { app.battery = { level: b.level, charging: b.charging }; };
  update(); b.onlevelchange = b.onchargingchange = update;
}).catch(() => {});
window.speechSynthesis?.getVoices();

refreshState();
everySecond();
setInterval(everySecond, 1000);
requestAnimationFrame(frame);

ask({ greeting: true })
  .then(data => { app.online = true; data.replies.forEach(deliver); })
  .catch(() => { app.online = false; addNote("Сервертэй холбогдож чадсангүй.  ·  Can't reach the JARVIS server.", true); })
  .finally(refreshState);
