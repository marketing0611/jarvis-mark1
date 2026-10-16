// Shared HUD painting + the arc-reactor view (a canvas port of the desktop HudCanvas).

export const C = {
  bg: "#04080c", surface: "#08121a", pri: "#3fd6ff", text: "#e4f5fc", text2: "#9fbccb",
  text3: "#58788a", acc: "#ff8a3d", amber: "#ffc857", green: "#3ef0a1", red: "#ff4d6d",
  ice: "#8fe3ff", warn: "#ff4a4a",
};

export const STATE_MN = {
  "SPEAKING": "Ярьж байна", "THINKING": "Бодож байна", "HEARING YOU": "Таныг сонсож байна",
  "LISTENING": "Сонсож байна", "STANDBY": "Бэлэн байна", "OFFLINE": "Сервертэй холбогдсонгүй",
};

const MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace';
const SANS = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';

export function rgba(hex, a = 255) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${Math.max(0, Math.min(255, a)) / 255})`;
}

export function font(ctx, size, { weight = 400, mono = false, spacing = 0 } = {}) {
  ctx.font = `${weight} ${size}px ${mono ? MONO : SANS}`;
  if ("letterSpacing" in ctx) ctx.letterSpacing = `${spacing}px`;
}

/** Fit a canvas to its CSS box at device resolution; returns [W, H] in CSS pixels. */
export function fitCanvas(canvas, ctx) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [W, H];
}

/** Two label/value pairs in each corner, with an L-shaped bracket. */
export function paintReadouts(ctx, W, H, readouts, mx = 26, my = 26) {
  for (const [corner, items] of Object.entries(readouts)) {
    const right = corner[1] === "r", top = corner[0] === "t";
    const y0 = top ? my : H - my - 2 * 42 + 6;
    const x = right ? W - mx : mx;
    const bx = right ? W - mx + 10 : mx - 10;
    const by = top ? y0 - 8 : y0 + 2 * 42 - 4;
    const sx = right ? -1 : 1, sy = top ? 1 : -1;
    ctx.strokeStyle = rgba(C.pri, 120); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(bx + sx * 16, by); ctx.lineTo(bx, by); ctx.lineTo(bx, by + sy * 16); ctx.stroke();
    ctx.textAlign = right ? "right" : "left"; ctx.textBaseline = "top";
    items.forEach(([label, value], i) => {
      const y = y0 + i * 42;
      font(ctx, 9.5, { weight: 700, mono: true, spacing: 2 }); ctx.fillStyle = C.text3; ctx.fillText(label, x, y);
      font(ctx, 15, { weight: 600, mono: true, spacing: 1 }); ctx.fillStyle = C.text; ctx.fillText(value, x, y + 14);
    });
  }
}

export function paintState(ctx, W, y, label, color) {
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  font(ctx, 22, { weight: 700, mono: true, spacing: 8 }); ctx.fillStyle = color; ctx.fillText(label, W / 2, y);
  font(ctx, 15); ctx.fillStyle = C.text2; ctx.fillText(STATE_MN[label] || "", W / 2, y + 32);
}

export class Reactor {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.label = "STANDBY"; this.color = C.pri;
    this.speaking = this.listening = this.thinking = false;
    this.level = 0;                       // live mic or voice level, 0..1
    this.gauges = [[0, "LAT —"], [0, "MIC 0%"]];
    this.readouts = {};
    this.bars = new Array(48).fill(0);
    this.rings = [0, 120, 240, 0];
    this.sweep = 0; this.glow = 0.5; this.t = 0; this.pulses = [];
    this.static = null; this.staticKey = "";
  }

  geometry(W, H) {
    const R = Math.max(110, Math.min((H - 130) / 2, (W - 80) / 2));
    return [W / 2, (H - 110) / 2 + 12, R];
  }

  step(dt) {
    this.t += dt;
    const f = dt / 0.033;
    const energy = this.speaking ? 0.75 + 0.25 * Math.random()
      : this.listening ? 0.35 + 0.65 * this.level
      : this.thinking ? 0.55 + 0.15 * Math.sin(this.t * 6)
      : 0.4 + 0.08 * Math.sin(this.t * 1.6);
    this.glow += (energy - this.glow) * 0.3;
    const speed = (this.thinking ? 4.4 : this.speaking ? 2.8 : 1.0) * f;
    [speed, -speed * 0.7, speed * 1.4, -0.4 * f].forEach((d, i) => { this.rings[i] = (this.rings[i] + d + 360) % 360; });
    this.sweep = (this.sweep + (this.thinking ? 7 : 3.5) * f) % 360;
    for (let i = 0; i < this.bars.length; i++) {
      const target = this.speaking ? this.level * (0.55 + 0.45 * Math.random()) * (0.6 + 0.4 * Math.sin(this.t * 9 + i * 0.5) ** 2)
        : this.listening ? this.level * (0.55 + 0.45 * Math.random())
        : this.thinking ? 0.2 + 0.15 * Math.sin(this.t * 6 + i * 0.35)
        : 0.05 + 0.03 * Math.sin(this.t * 2 + i * 0.25);
      this.bars[i] += (target - this.bars[i]) * 0.35;
    }
    const [, , R] = this.geometry(this.canvas.clientWidth, this.canvas.clientHeight);
    const grow = R * (this.speaking ? 0.014 : 0.007) * f;
    this.pulses = this.pulses.map(r => r + grow).filter(r => r < R * 0.95);
    if ((this.speaking || (this.listening && this.level > 0.35)) && Math.random() < 0.1) this.pulses.push(R * 0.32);
  }

  staticLayer(W, H) {
    const dpr = window.devicePixelRatio || 1;
    const key = `${W}x${H}@${dpr}`;
    if (this.static && this.staticKey === key) return this.static;
    const off = document.createElement("canvas");
    off.width = W * dpr; off.height = H * dpr;
    const ctx = off.getContext("2d");
    ctx.scale(dpr, dpr);
    const [cx, cy, R] = this.geometry(W, H);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = rgba(C.pri, 26);
    for (let x = cx % 28; x < W; x += 28) for (let y = cy % 28; y < H; y += 28) ctx.fillRect(x - 0.8, y - 0.8, 1.6, 1.6);
    ctx.strokeStyle = rgba(C.pri, 40); ctx.lineWidth = 1;
    for (const s of [-1, 1]) {
      ctx.beginPath(); ctx.moveTo(cx + s * R * 1.06, cy); ctx.lineTo(cx + s * (W / 2 - 16), cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx, cy + s * R * 1.06); ctx.lineTo(cx, cy + s * (cy - 8)); ctx.stroke();
    }
    ctx.strokeStyle = rgba(C.pri, 130); ctx.lineWidth = 1.5;
    const b = R * 1.1;
    for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
      const x = cx + sx * b, y = cy + sy * b;
      ctx.beginPath(); ctx.moveTo(x - sx * 22, y); ctx.lineTo(x, y); ctx.lineTo(x, y - sy * 22); ctx.stroke();
    }
    ctx.strokeStyle = rgba(C.pri, 80); ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = rgba(C.pri, 40); ctx.beginPath(); ctx.arc(cx, cy, R * 0.955, 0, Math.PI * 2); ctx.stroke();
    for (let deg = 0; deg < 360; deg += 2) {
      const [len, a] = deg % 30 === 0 ? [13, 170] : deg % 10 === 0 ? [8, 110] : [4, 60];
      const rad = deg * Math.PI / 180, sx = Math.sin(rad), sy = -Math.cos(rad);
      ctx.strokeStyle = rgba(C.pri, a); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx + R * sx, cy + R * sy); ctx.lineTo(cx + (R - len) * sx, cy + (R - len) * sy); ctx.stroke();
    }
    font(ctx, 9, { weight: 700, mono: true }); ctx.fillStyle = rgba(C.pri, 120);
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (let deg = 0; deg < 360; deg += 30) {
      const rad = deg * Math.PI / 180;
      ctx.fillText(String(deg).padStart(3, "0"), cx + (R - 26) * Math.sin(rad), cy - (R - 26) * Math.cos(rad));
    }
    this.static = off; this.staticKey = key;
    return off;
  }

  draw() {
    const ctx = this.ctx;
    const [W, H] = fitCanvas(this.canvas, ctx);
    if (!W || !H) return;
    ctx.drawImage(this.staticLayer(W, H), 0, 0, W, H);
    const [cx, cy, R] = this.geometry(W, H);
    const col = this.color, g = this.glow, TAU = Math.PI * 2, rad = d => d * Math.PI / 180;

    let grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 1.1);
    grad.addColorStop(0, rgba(col, 80 * g)); grad.addColorStop(0.45, rgba(col, 22 * g)); grad.addColorStop(1, rgba(col, 0));
    ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(cx, cy, R * 1.1, 0, TAU); ctx.fill();

    const cone = ctx.createConicGradient(rad(this.sweep - 90), cx, cy);    // trail behind the sweep line
    cone.addColorStop(0, rgba(col, 0)); cone.addColorStop(0.86, rgba(col, 0)); cone.addColorStop(1, rgba(col, 70 * g));
    ctx.fillStyle = cone; ctx.beginPath(); ctx.arc(cx, cy, R * 0.95, 0, TAU); ctx.fill();
    const a = rad(this.sweep - 90);
    ctx.strokeStyle = rgba(col, 140); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(cx + R * 0.32 * Math.cos(a), cy + R * 0.32 * Math.sin(a));
    ctx.lineTo(cx + R * 0.95 * Math.cos(a), cy + R * 0.95 * Math.sin(a)); ctx.stroke();

    for (const r of this.pulses) {
      ctx.strokeStyle = rgba(col, 150 * (1 - r / (R * 0.95))); ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, TAU); ctx.stroke();
    }

    ctx.strokeStyle = rgba(col, 110); ctx.lineWidth = 2.5; ctx.lineCap = "butt";
    for (let k = 0; k < 90; k++) {
      const s = rad(this.rings[3] + k * 4);
      ctx.beginPath(); ctx.arc(cx, cy, R * 0.9, s, s + rad(2)); ctx.stroke();
    }
    ctx.lineCap = "round";
    [[0.835, 3.5, 70, 50], [0.78, 2, 40, 20], [0.355, 1.5, 110, 70]].forEach(([frac, width, arc, gap], idx) => {
      ctx.strokeStyle = rgba(col, 120 + 120 * g - idx * 25); ctx.lineWidth = width;
      for (let s = this.rings[idx]; s < this.rings[idx] + 360; s += arc + gap) {
        ctx.beginPath(); ctx.arc(cx, cy, R * frac, rad(s), rad(s + arc)); ctx.stroke();
      }
    });

    // gauges with a gap at the bottom (start bottom-left, sweep clockwise over the top)
    ctx.lineCap = "butt";
    this.gauges.forEach(([value], i) => {
      const r = R * (i ? 0.66 : 0.705);
      ctx.lineWidth = 5;
      ctx.strokeStyle = rgba(col, 34); ctx.beginPath(); ctx.arc(cx, cy, r, rad(135), rad(405)); ctx.stroke();
      ctx.strokeStyle = rgba(col, i ? 165 : 235);
      ctx.beginPath(); ctx.arc(cx, cy, r, rad(135), rad(135 + 270 * Math.min(value, 100) / 100)); ctx.stroke();
    });
    font(ctx, 10.5, { weight: 700, mono: true, spacing: 1.5 }); ctx.fillStyle = C.text2; ctx.textBaseline = "top";
    ctx.textAlign = "right"; ctx.fillText(this.gauges[0][1], cx - R * 0.08, cy + R * 0.6);
    ctx.textAlign = "left"; ctx.fillText(this.gauges[1][1], cx + R * 0.08, cy + R * 0.6);

    // circular audio visualiser
    const n = 96, r0 = R * 0.405;
    ctx.lineWidth = Math.max(2, R * 0.012); ctx.lineCap = "round";
    for (let k = 0; k < n; k++) {
      const lvl = Math.min(1, this.bars[k < 48 ? k : n - 1 - k]);
      const ang = rad(k * 360 / n), sx = Math.sin(ang), sy = -Math.cos(ang), r1 = r0 + R * (0.02 + 0.19 * lvl);
      ctx.strokeStyle = rgba(col, 110 + 145 * lvl);
      ctx.beginPath(); ctx.moveTo(cx + r0 * sx, cy + r0 * sy); ctx.lineTo(cx + r1 * sx, cy + r1 * sy); ctx.stroke();
    }

    this.paintCore(ctx, cx, cy, R * 0.3, col, g, this.rings[2] * 0.25);
    paintState(ctx, W, cy + R + 18, this.label, col);
    paintReadouts(ctx, W, H, this.readouts);
  }

  paintCore(ctx, cx, cy, r, col, g, spin) {
    const TAU = Math.PI * 2;
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
    grad.addColorStop(0, "rgba(255,255,255,0.94)"); grad.addColorStop(0.22, rgba(col, 235));
    grad.addColorStop(0.7, rgba(col, 60 + 70 * g)); grad.addColorStop(1, rgba(col, 0));
    ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(cx, cy, r, 0, TAU); ctx.fill();
    ctx.lineWidth = 1.5; ctx.strokeStyle = rgba(col, 170); ctx.beginPath(); ctx.arc(cx, cy, r * 1.08, 0, TAU); ctx.stroke();
    ctx.lineWidth = 1; ctx.strokeStyle = rgba(col, 90); ctx.beginPath(); ctx.arc(cx, cy, r * 1.16, 0, TAU); ctx.stroke();
    ctx.lineWidth = 1.3; ctx.strokeStyle = rgba(col, 210); ctx.fillStyle = rgba(C.surface, 205); ctx.lineJoin = "miter";
    for (let k = 0; k < 10; k++) {
      const a0 = (spin + k * 36 + 4) * Math.PI / 180, a1 = (spin + k * 36 + 32) * Math.PI / 180;
      const r1 = r * 0.6, r2 = r * 0.92;
      ctx.beginPath();
      ctx.moveTo(cx + r1 * Math.cos(a0), cy + r1 * Math.sin(a0)); ctx.lineTo(cx + r2 * Math.cos(a0), cy + r2 * Math.sin(a0));
      ctx.lineTo(cx + r2 * Math.cos(a1), cy + r2 * Math.sin(a1)); ctx.lineTo(cx + r1 * Math.cos(a1), cy + r1 * Math.sin(a1));
      ctx.closePath(); ctx.fill(); ctx.stroke();
    }
    ctx.lineWidth = 2; ctx.strokeStyle = "rgba(255,255,255,0.88)"; ctx.beginPath(); ctx.arc(cx, cy, r * 0.42, 0, TAU); ctx.stroke();
    ctx.lineWidth = 1; ctx.strokeStyle = rgba(col, 200); ctx.beginPath(); ctx.arc(cx, cy, r * 0.5, 0, TAU); ctx.stroke();
  }
}
