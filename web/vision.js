// VISION view: the webcam seen from inside an Iron Man helmet, with an overlay that
// follows the face. Face tracking runs in the browser with MediaPipe (nothing is uploaded).
import { C, rgba, font, fitCanvas, paintReadouts, paintState, isCompact } from "./hud.js";

const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODEL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
const ICE = C.ice;
const TAP = matchMedia("(pointer: coarse)").matches ? "tap" : "click";

const TRACE = [
  m => `track.lock(eye='L', conf=${m.conf.toFixed(2)})`, () => "mesh.align(pts=9) ........ OK",
  () => `bio::scan → MATCH ${(96 + Math.random() * 3.5).toFixed(1)}%`, () => `thermal.read() → 36.${rnd(1, 9)}°C`,
  () => `pupil.dilation = 4.${rnd(1, 9)} mm`, m => `gaze.vec = (${sign(m.gx)}, ${sign(m.gy)})`,
  m => `frame #${String(m.frame).padStart(6, "0")} · ${m.fps.toFixed(0)} fps`, m => `dist ≈ ${m.dist.toFixed(0)} cm`,
  () => "jarvis.vision.sync() ✓", () => `neural.infer ${(2 + Math.random() * 2).toFixed(1)} ms`,
  () => "stark.protocol XXXIX ✓", () => `retina.map → 0x${hex(4)}`,
  m => `head.yaw ${sign(m.yaw, 1)}°  roll ${sign(m.roll, 1)}°`,
];
const IDLE_TRACE = [
  () => "scan.sweep() ........ no subject", () => `sensor.gain ${(1 + Math.random()).toFixed(2)}`,
  () => "jarvis.vision.sync() ✓", () => `neural.infer ${(2 + Math.random() * 2).toFixed(1)} ms`,
  () => `retina.map → 0x${hex(4)}`, () => "stark.protocol XXXIX ✓",
];
const GLYPH_SPOTS = [[0.30, 0.16], [0.55, 0.10], [0.74, 0.20], [0.14, 0.52], [0.88, 0.50], [0.50, 0.48],
  [0.22, 0.74], [0.80, 0.74], [0.50, 0.92], [0.40, 0.28], [0.66, 0.62]];

function rnd(a, b) { return a + Math.floor(Math.random() * (b - a + 1)); }
function hex(n) { return Math.floor(Math.random() * 16 ** n).toString(16).toUpperCase().padStart(n, "0"); }
function sign(v, d = 2) { return (v >= 0 ? "+" : "") + v.toFixed(d); }

/** Break text into lines no wider than maxW, at the context's current font. */
function wrap(ctx, text, maxW) {
  const lines = [];
  let line = "";
  for (const word of text.split(" ")) {
    const next = line ? `${line} ${word}` : word;
    if (line && ctx.measureText(next).width > maxW) { lines.push(line); line = word; } else line = next;
  }
  if (line) lines.push(line);
  return lines;
}

function visorPath(W, H) {
  const p = new Path2D(), P = (x, y) => [W * x, H * y];
  p.moveTo(...P(0.07, 0.12));
  p.bezierCurveTo(...P(0.25, 0.025), ...P(0.75, 0.025), ...P(0.93, 0.12));
  p.bezierCurveTo(...P(0.99, 0.30), ...P(0.995, 0.62), ...P(0.93, 0.84));
  p.bezierCurveTo(...P(0.86, 0.95), ...P(0.70, 0.985), ...P(0.5, 0.985));
  p.bezierCurveTo(...P(0.30, 0.985), ...P(0.14, 0.95), ...P(0.07, 0.84));
  p.bezierCurveTo(...P(0.005, 0.62), ...P(0.01, 0.30), ...P(0.07, 0.12));
  p.closePath();
  return p;
}

export class Vision {
  constructor(root) {
    this.root = root;
    this.video = root.querySelector("video");
    this.canvas = root.querySelector("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.trace = root.querySelector("#trace");
    this.hex = root.querySelector("#hex");
    this.label = "STANDBY"; this.color = C.pri;
    this.status = "off";                // off | starting | live | error
    this.error = "";
    this.detector = null; this.detectorName = "—";
    this.face = null; this.lostT = 0; this.t = 0; this.frame = 0;
    this.fps = 0; this._fpsN = 0; this._fpsT = performance.now();
    this.yaw = this.roll = this.pitch = 0;
    this.lines = { trace: [], hex: [] }; this.nextLine = 0;
    this.glyphs = GLYPH_SPOTS.map(() => "");
    this.pupil = 4;
    this.helmet = null; this.helmetKey = "";
    this.stream = null;
  }

  async start() {
    if (this.status === "live" || this.status === "starting") return;
    this.status = "starting"; this.error = "";
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: "user" }, audio: false });
      this.video.srcObject = this.stream;
      await this.video.play();
      this.status = "live";
    } catch (e) {
      this.status = "error";
      this.error = e && e.name === "NotAllowedError" ? "denied" : "unavailable";
      return;
    }
    if (!this.detector) this.loadDetector();
  }

  stop() {
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    this.stream = null; this.video.srcObject = null;
    this.status = "off"; this.face = null;
  }

  async loadDetector() {
    try {
      const { FilesetResolver, FaceDetector } = await import(`${MP}/vision_bundle.mjs`);
      const files = await FilesetResolver.forVisionTasks(`${MP}/wasm`);
      this.detector = await FaceDetector.createFromOptions(files, {
        baseOptions: { modelAssetPath: MODEL, delegate: "GPU" }, runningMode: "VIDEO", minDetectionConfidence: 0.6,
      });
      this.detectorName = "MEDIAPIPE";
    } catch (e) {
      this.detectorName = "UNAVAILABLE";
    }
  }

  // video pixels → screen pixels (cover fit, mirrored like a selfie camera)
  mapping(W, H) {
    const vw = this.video.videoWidth || 1280, vh = this.video.videoHeight || 720;
    const s = Math.max(W / vw, H / vh);
    return { s, ox: (W - vw * s) / 2, oy: (H - vh * s) / 2, vw, vh };
  }

  detect() {
    if (!this.detector || this.status !== "live" || this.video.readyState < 2) return;
    const now = performance.now();
    let result;
    try { result = this.detector.detectForVideo(this.video, now); } catch { return; }
    this.frame++; this._fpsN++;
    if (now - this._fpsT > 1000) { this.fps = this._fpsN * 1000 / (now - this._fpsT); this._fpsN = 0; this._fpsT = now; }
    const det = (result.detections || []).sort((a, b) => b.boundingBox.width - a.boundingBox.width)[0];
    if (!det) { this.lostT += 0.033; if (this.lostT > 0.4) this.face = null; return; }
    this.lostT = 0;
    const vw = this.video.videoWidth, vh = this.video.videoHeight, bb = det.boundingBox;
    const mx = x => vw - x;                                   // mirror horizontally
    const kp = det.keypoints.map(k => [mx(k.x * vw), k.y * vh]);
    const eyes = [kp[0], kp[1]].sort((a, b) => a[0] - b[0]); // screen-left eye first
    const next = {
      box: [mx(bb.originX + bb.width), bb.originY, bb.width, bb.height],
      le: eyes[0], re: eyes[1], nose: kp[2], mouth: kp[3],
      conf: det.categories?.[0]?.score ?? 0.9,
    };
    if (!this.face) { this.face = next; return; }
    const mix = (a, b) => a.map((v, i) => v + 0.5 * (b[i] - v));
    for (const k of ["box", "le", "re", "nose", "mouth"]) this.face[k] = mix(this.face[k], next[k]);
    this.face.conf = next.conf;
  }

  metrics() {
    const f = this.face, vw = this.video.videoWidth || 1280;
    if (!f) return { conf: 0, gx: 0, gy: 0, frame: this.frame, fps: this.fps, dist: 0, yaw: 0, roll: 0 };
    const [bx, by, bw, bh] = f.box;
    const dx = ((f.le[0] + f.re[0]) / 2 - (bx + bw / 2)) / bw;
    const dy = (f.nose[1] - (by + bh * 0.55)) / bh;
    const roll = Math.atan2(f.re[1] - f.le[1], f.re[0] - f.le[0]) * 180 / Math.PI;
    return { conf: f.conf, gx: dx * 2, gy: dy * 2, frame: this.frame, fps: this.fps,
             dist: 15 * vw * 0.9 / Math.max(bw, 1), yaw: dx * -60, roll };
  }

  step(dt) {
    this.t += dt;
    this.detect();
    const m = this.metrics();
    this.yaw += 0.2 * (m.yaw - this.yaw); this.roll += 0.2 * (m.roll - this.roll); this.pitch += 0.2 * (m.gy * 20 - this.pitch);
    if (this.t >= this.nextLine) {
      this.nextLine = this.t + 0.22;
      this.pupil = rnd(1, 9);
      const pool = this.face ? TRACE : IDLE_TRACE;
      this.lines.trace = [...this.lines.trace, pool[rnd(0, pool.length - 1)](m)].slice(-8);
      this.lines.hex = [...this.lines.hex, `0x${hex(4)}  ` + Array.from({ length: 6 }, () => hex(2)).join(" ")].slice(-8);
      this.trace.textContent = this.lines.trace.join("\n");
      this.hex.textContent = this.lines.hex.join("\n");
      this.glyphs = this.glyphs.map(g => Math.random() < 0.5
        ? [hex(4), Math.random().toFixed(3), `Δ${String(rnd(1, 99)).padStart(2, "0")}`, `#${hex(2)}`][rnd(0, 3)] : g);
    }
    // the side panels drift a little with the face
    const W = this.canvas.clientWidth;
    let drift = 0;
    if (this.face) {
      const { s, ox } = this.mapping(W, this.canvas.clientHeight);
      drift = (ox + (this.face.box[0] + this.face.box[2] / 2) * s - W / 2) * 0.03;
    }
    this.root.style.setProperty("--drift", `${drift.toFixed(1)}px`);
  }

  helmetLayer(W, H) {
    const dpr = window.devicePixelRatio || 1, key = `${W}x${H}@${dpr}`;
    if (this.helmet && this.helmetKey === key) return this.helmet;
    const off = document.createElement("canvas");
    off.width = W * dpr; off.height = H * dpr;
    const ctx = off.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.fillStyle = "rgba(0,0,0,0.13)";
    for (let y = 0; y < H; y += 3) ctx.fillRect(0, y, W, 1);
    const glare = ctx.createLinearGradient(0, 0, W * 0.45, H * 0.6);
    glare.addColorStop(0, "rgba(255,255,255,0.06)"); glare.addColorStop(0.5, "rgba(255,255,255,0.015)"); glare.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = glare; ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(W * 0.55, 0); ctx.lineTo(0, H * 0.75); ctx.fill();
    const visor = visorPath(W, H);
    const shade = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * 0.7);
    shade.addColorStop(0, "rgba(4,10,16,0.92)"); shade.addColorStop(1, "rgba(0,0,0,1)");
    const outside = new Path2D(); outside.rect(0, 0, W, H); outside.addPath(visor);
    ctx.fillStyle = shade; ctx.fill(outside, "evenodd");
    for (const [w, a] of [[9, 18], [4, 40], [1.6, 150]]) { ctx.lineWidth = w; ctx.strokeStyle = rgba(ICE, a); ctx.stroke(visor); }
    this.helmet = off; this.helmetKey = key;
    return off;
  }

  draw() {
    const ctx = this.ctx;
    const [W, H] = fitCanvas(this.canvas, ctx);
    if (!W || !H) return;
    ctx.clearRect(0, 0, W, H);
    const live = this.status === "live";
    const { s, ox, oy } = this.mapping(W, H);
    const f = this.face;

    // black out everything but a soft spotlight on the head
    let cx = W / 2, cy = H * 0.45, rx = W * 0.3, ry = H * 0.42, stops = [[0, 150], [1, 255]];
    if (f && live) {
      const [bx, by, bw, bh] = f.box;
      cx = ox + (bx + bw / 2) * s; cy = oy + (by + bh * 0.42) * s; rx = bw * s * 0.75; ry = bh * s * 0.92;
      stops = [[0, 0], [0.6, 0], [0.84, 205], [1, 255]];
    }
    if (!live) stops = [[0, 255], [1, 255]];
    ctx.save();
    ctx.translate(cx, cy); ctx.scale(rx, ry);
    const spot = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
    stops.forEach(([at, a]) => spot.addColorStop(at, `rgba(0,0,0,${a / 255})`));
    ctx.fillStyle = spot;
    ctx.fillRect(-cx / rx, -cy / ry, W / rx, H / ry);
    ctx.restore();
    ctx.fillStyle = rgba(ICE, 10); ctx.fillRect(0, 0, W, H);

    const compact = isCompact(W, H), stateY = H * 0.885 - (compact ? 30 : 40);
    ctx.drawImage(this.helmetLayer(W, H), 0, 0, W, H);
    this.paintHeading(ctx, W, H);
    this.paintHorizon(ctx, W, H);
    if (!live) this.paintPlaceholder(ctx, W, H, compact);
    else if (f) this.paintFace(ctx, f, s, ox, oy, stateY);
    else this.paintSearching(ctx, W, H, compact);

    const tracking = f && live ? "LOCKED" : live ? "SEARCHING" : "—";
    paintReadouts(ctx, W, H, {
      tl: [["VISION", live ? "ONLINE" : "OFFLINE"], ["TRACKING", tracking]],
      tr: [["FPS", this.fps.toFixed(0)], ["DETECTOR", this.detectorName]],
    }, W * 0.1, H * 0.12, { scale: compact ? 0.8 : 1 });
    if (live || !compact) paintState(ctx, W, stateY, this.label, this.color, compact);   // small screens: the notice needs the room
  }

  paintHeading(ctx, W, H) {
    const cx = W / 2, y = Math.max(H * 0.11, 34), half = W * 0.17, heading = ((47 + this.yaw * 1.5) % 360 + 360) % 360, ppd = half / 30;
    font(ctx, 9, { weight: 700, mono: true, spacing: 1 }); ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let d = Math.floor(heading) - 30; d <= heading + 30; d++) {
      if (d % 5) continue;
      const x = cx + (d - heading) * ppd, fade = Math.max(0, 1 - Math.abs(x - cx) / half);
      ctx.strokeStyle = ctx.fillStyle = rgba(ICE, 200 * fade); ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + (d % 15 === 0 ? 9 : 5)); ctx.stroke();
      if (d % 15 === 0) {
        const dd = ((d % 360) + 360) % 360;
        ctx.fillText({ 0: "N", 90: "E", 180: "S", 270: "W" }[dd] || String(dd).padStart(3, "0"), x, y + 12);
      }
    }
    ctx.fillStyle = rgba(ICE, 220);
    ctx.beginPath(); ctx.moveTo(cx - 5, y - 9); ctx.lineTo(cx + 5, y - 9); ctx.lineTo(cx, y - 2); ctx.fill();
    font(ctx, 11, { weight: 700, mono: true, spacing: 2 });
    ctx.fillText(`HDG ${String(Math.round(heading) % 360).padStart(3, "0")}`, cx, y - 28);
  }

  paintHorizon(ctx, W, H) {
    ctx.save();
    ctx.translate(W / 2, H * 0.5 + this.pitch * 3); ctx.rotate(this.roll * Math.PI / 180);
    const inner = W * 0.19, outer = W * 0.3, rung = 3.2 * Math.min(1, H / 520);
    ctx.strokeStyle = rgba(ICE, 150); ctx.lineWidth = 1.6;
    for (const s of [-1, 1]) {
      ctx.beginPath(); ctx.moveTo(s * outer, 0); ctx.lineTo(s * inner, 0); ctx.lineTo(s * inner, 8); ctx.stroke();
    }
    font(ctx, 8.5, { weight: 700, mono: true }); ctx.textAlign = "left"; ctx.textBaseline = "middle";
    for (const step of [-20, -10, 10, 20]) {
      const y = -step * rung;
      ctx.strokeStyle = ctx.fillStyle = rgba(ICE, 70); ctx.lineWidth = 1; ctx.setLineDash(step < 0 ? [4, 4] : []);
      for (const s of [-1, 1]) { ctx.beginPath(); ctx.moveTo(s * inner, y); ctx.lineTo(s * (inner + W * 0.05), y); ctx.stroke(); }
      ctx.fillText((step > 0 ? "+" : "") + step, outer - W * 0.06, y);
    }
    ctx.setLineDash([]);
    ctx.restore();
  }

  paintReticle(ctx, cx, cy, r, spin) {
    const rad = d => d * Math.PI / 180, TAU = Math.PI * 2;
    ctx.lineWidth = 1.5; ctx.strokeStyle = rgba(ICE, 120); ctx.lineCap = "butt";
    for (let k = 0; k < 24; k++) { const a = rad(spin + k * 15); ctx.beginPath(); ctx.arc(cx, cy, r * 1.7, a, a + rad(7)); ctx.stroke(); }
    ctx.lineWidth = 2.2; ctx.strokeStyle = rgba(ICE, 235); ctx.lineCap = "round";
    for (const base of [-spin * 1.6, -spin * 1.6 + 180]) { ctx.beginPath(); ctx.arc(cx, cy, r * 1.32, rad(base), rad(base + 80)); ctx.stroke(); }
    ctx.lineWidth = 1.4; ctx.strokeStyle = rgba(ICE, 215);
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, TAU); ctx.stroke();
    for (let deg = 0; deg < 360; deg += 15) {
      const a = rad(deg), inner = r * (deg % 45 === 0 ? 0.8 : 0.88);
      ctx.beginPath(); ctx.moveTo(cx + r * Math.cos(a), cy + r * Math.sin(a)); ctx.lineTo(cx + inner * Math.cos(a), cy + inner * Math.sin(a)); ctx.stroke();
    }
    ctx.beginPath(); ctx.arc(cx, cy, r * 0.42, 0, TAU); ctx.stroke();
    ctx.fillStyle = rgba(ICE, 235); ctx.beginPath(); ctx.arc(cx, cy, 2.2, 0, TAU); ctx.fill();
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      ctx.beginPath(); ctx.moveTo(cx + dx * r * 1.05, cy + dy * r * 1.05); ctx.lineTo(cx + dx * r * 1.5, cy + dy * r * 1.5); ctx.stroke();
    }
    ctx.fillStyle = rgba(ICE, 200);
    for (let k = 0; k < 4; k++) {
      const a = rad(spin * 0.5 + 45 + k * 90), c = Math.cos(a), sn = Math.sin(a);
      const tip = [cx + r * 1.85 * c, cy + r * 1.85 * sn], back = [cx + r * 2.1 * c, cy + r * 2.1 * sn], w = r * 0.12;
      ctx.beginPath(); ctx.moveTo(...tip); ctx.lineTo(back[0] - sn * w, back[1] + c * w); ctx.lineTo(back[0] + sn * w, back[1] - c * w); ctx.fill();
    }
  }

  paintSearching(ctx, W, H, compact) {
    const cx = W / 2, cy = H * (compact ? 0.37 : 0.45), r = Math.min(W, H) * (compact ? 0.1 : 0.14), y = cy + r * 2.15;
    this.paintReticle(ctx, cx, cy, r, this.t * 60);
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    font(ctx, compact ? 11 : 12, { weight: 700, mono: true, spacing: compact ? 2 : 4 }); ctx.fillStyle = rgba(ICE, 220);
    ctx.fillText(this.detector ? "SCANNING FOR SUBJECT" : "LOADING TRACKER", cx, y);
    font(ctx, compact ? 13 : 14); ctx.fillStyle = C.text2;
    ctx.fillText(this.detector ? "Царай хайж байна…" : "Царай танигч ачаалж байна…", cx, y + (compact ? 17 : 22));
  }

  paintPlaceholder(ctx, W, H, compact) {
    const cx = W / 2, cy = compact ? H * 0.36 : H / 2 - 30, r = Math.min(W, H) * (compact ? 0.09 : 0.12);
    this.paintReticle(ctx, cx, cy, r, this.t * 40);
    ctx.textAlign = "center"; ctx.textBaseline = "top"; font(ctx, compact ? 12.5 : 14); ctx.fillStyle = C.text2;
    const where = "Хөтчийн хаягийн мөрөн дэх камерын тэмдэг дээр дарж зөвшөөрөөд,";   // desktop browsers only
    let lines = this.status === "error"
      ? (this.error === "denied"
        ? ["Камерын зөвшөөрөл олгогдоогүй байна.", ...(compact ? [] : [where]), `${compact ? "Энд" : "энд"} дарж дахин оролдоно уу.  ·  Allow camera access, then ${TAP} to retry.`]
        : ["Камер олдсонгүй.  ·  No camera found.", "Энд дарж дахин оролдоно уу."])
      : ["Камер асааж байна…  ·  Starting camera…"];
    lines = lines.flatMap(l => wrap(ctx, l, W * 0.8));
    const lh = compact ? 17 : 22;
    lines.forEach((l, i) => ctx.fillText(l, cx, cy + r * 2.2 + i * lh));
  }

  paintFace(ctx, f, s, ox, oy, stateY) {
    const P = ([x, y]) => [ox + x * s, oy + y * s];
    const [bx, by, bw, bh] = f.box;
    const x0 = ox + bx * s, y0 = oy + by * s, w = bw * s, h = bh * s;
    const pad = w * 0.12, L = x0 - pad, T = y0 - pad * 1.3, R = x0 + w + pad, B = y0 + h + pad;
    const pulse = 0.75 + 0.25 * Math.sin(this.t * 5), arm = w * 0.18, m = this.metrics();

    ctx.strokeStyle = rgba(ICE, 235 * pulse); ctx.lineWidth = 2.2; ctx.lineCap = "square";
    for (const [cx, cy, sx, sy] of [[L, T, 1, 1], [R, T, -1, 1], [L, B, 1, -1], [R, B, -1, -1]]) {
      ctx.beginPath(); ctx.moveTo(cx + sx * arm, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + sy * arm); ctx.stroke();
    }
    font(ctx, 10.5, { weight: 700, mono: true, spacing: 2 }); ctx.textBaseline = "bottom"; ctx.fillStyle = ICE;
    ctx.textAlign = "left"; ctx.fillText("SUBJECT LOCKED", L, T - 6);
    ctx.textAlign = "right"; ctx.fillText(`CONF ${Math.round(m.conf * 100)}%`, R, T - 6);
    if (B + 24 < stateY - 6) {
      ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = C.text2;
      ctx.fillText(`DIST ≈ ${m.dist.toFixed(0)} CM   ·   ROLL ${sign(m.roll, 0)}°`, (L + R) / 2, B + 6);
    }

    const sy = T + ((this.t * 0.55) % 1) * (B - T);
    const grad = ctx.createLinearGradient(L, 0, R, 0);
    grad.addColorStop(0, rgba(ICE, 0)); grad.addColorStop(0.5, rgba(ICE, 160)); grad.addColorStop(1, rgba(ICE, 0));
    ctx.strokeStyle = grad; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(L, sy); ctx.lineTo(R, sy); ctx.stroke();

    const le = P(f.le), re = P(f.re), no = P(f.nose), mo = P(f.mouth);
    const ml = [mo[0] - w * 0.14, mo[1]], mr = [mo[0] + w * 0.14, mo[1]];
    const brow = [(le[0] + re[0]) / 2, y0 + h * 0.18], chin = [no[0], y0 + h * 0.98];
    const ckl = [x0 + w * 0.12, y0 + h * 0.58], ckr = [x0 + w * 0.88, y0 + h * 0.58];
    ctx.strokeStyle = rgba(ICE, 60); ctx.lineWidth = 1;
    for (const [a, b] of [[le, re], [le, no], [re, no], [no, ml], [no, mr], [ml, mr], [le, brow], [re, brow],
                          [le, ckl], [ckl, ml], [re, ckr], [ckr, mr], [ml, chin], [mr, chin]]) {
      ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
    }
    ctx.fillStyle = rgba(ICE, 210);
    for (const q of [le, re, no, ml, mr, brow, chin, ckl, ckr]) { ctx.beginPath(); ctx.arc(q[0], q[1], 2.3, 0, Math.PI * 2); ctx.fill(); }

    font(ctx, 9.5, { weight: 700, mono: true }); ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    GLYPH_SPOTS.forEach(([u, v], i) => {
      if (!this.glyphs[i]) return;
      ctx.fillStyle = rgba(ICE, 110 + 100 * Math.sin(this.t * 3 + i));
      ctx.fillText(this.glyphs[i], x0 + u * w, y0 + v * h);
    });

    const er = w * 0.15;
    this.paintReticle(ctx, le[0], le[1], er, this.t * 90);
    ctx.strokeStyle = rgba(ICE, 150); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(re[0], re[1], er * 0.55, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(re[0] - er * 0.8, re[1]); ctx.lineTo(re[0] - er * 0.6, re[1]);
    ctx.moveTo(re[0] + er * 0.6, re[1]); ctx.lineTo(re[0] + er * 0.8, re[1]); ctx.stroke();
    const k1 = [le[0] - er * 1.5, le[1] - er * 1.5], k2 = [k1[0] - er * 0.5, k1[1] - er * 0.5], k3 = [Math.max(8, k2[0] - 70), k2[1]];
    ctx.strokeStyle = rgba(ICE, 200);
    ctx.beginPath(); ctx.moveTo(...k1); ctx.lineTo(...k2); ctx.lineTo(...k3); ctx.stroke();
    font(ctx, 10.5, { weight: 700, mono: true, spacing: 1.5 }); ctx.textAlign = "right"; ctx.textBaseline = "bottom";
    const tx = Math.max(k2[0], 8 + ctx.measureText("OCULAR LOCK · L").width);   // a face close to a phone camera fills the frame
    ctx.fillStyle = ICE; ctx.fillText("OCULAR LOCK · L", tx, k3[1] - 20);
    ctx.fillStyle = C.text2; ctx.fillText(`PUPIL 4.${this.pupil} MM`, tx, k3[1] - 4);
  }
}
