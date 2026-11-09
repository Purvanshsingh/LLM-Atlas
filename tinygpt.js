/*
 * TinyGPT: a minimal GPT-style transformer with its own autograd, written for teaching.
 * Character-level, pre-LayerNorm, causal multi-head attention, GELU MLP, AdamW-style updates.
 * No dependencies. Works in the browser (window.TinyGPT) and in Node (module.exports).
 */
(function (root) {
  'use strict';
  const F = root && root.TINYGPT_FLOAT64 ? Float64Array : Float32Array;
  let OPS = [], RECORD = true;

  class Tensor {
    constructor(r, c, d) { this.r = r; this.c = c; this.d = d || new F(r * c); this.g = null; }
    grad() { return this.g || (this.g = new F(this.r * this.c)); }
  }
  const rec = fn => { if (RECORD) OPS.push(fn); };

  /* ---------- ops: each computes a forward value and records its backward ---------- */
  function matmul(a, w) {
    const n = a.r, k = a.c, m = w.c, o = new Tensor(n, m), A = a.d, W = w.d, O = o.d;
    for (let i = 0; i < n; i++) {
      const io = i * m, ia = i * k;
      for (let p = 0; p < k; p++) {
        const av = A[ia + p]; if (av === 0) continue;
        const wp = p * m;
        for (let j = 0; j < m; j++) O[io + j] += av * W[wp + j];
      }
    }
    rec(() => {
      if (!o.g) return;
      const G = o.g, dA = a.grad(), dW = w.grad();
      for (let i = 0; i < n; i++) {
        const io = i * m, ia = i * k;
        for (let p = 0; p < k; p++) {
          const av = A[ia + p], wp = p * m; let s = 0;
          for (let j = 0; j < m; j++) { const g = G[io + j]; s += g * W[wp + j]; dW[wp + j] += av * g; }
          dA[ia + p] += s;
        }
      }
    });
    return o;
  }

  function addBias(a, b) {
    const n = a.r, m = a.c, o = new Tensor(n, m), A = a.d, B = b.d, O = o.d;
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) O[i * m + j] = A[i * m + j] + B[j];
    rec(() => {
      if (!o.g) return;
      const G = o.g, dA = a.grad(), dB = b.grad();
      for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { const g = G[i * m + j]; dA[i * m + j] += g; dB[j] += g; }
    });
    return o;
  }

  function add(a, b) {
    const o = new Tensor(a.r, a.c), A = a.d, B = b.d, O = o.d;
    for (let i = 0; i < O.length; i++) O[i] = A[i] + B[i];
    rec(() => {
      if (!o.g) return;
      const G = o.g, dA = a.grad(), dB = b.grad();
      for (let i = 0; i < G.length; i++) { dA[i] += G[i]; dB[i] += G[i]; }
    });
    return o;
  }

  // token embedding + positional embedding, rows laid out as [batch * T]
  function embed(wte, wpe, ids, T) {
    const n = ids.length, d = wte.c, o = new Tensor(n, d), O = o.d, E = wte.d, P = wpe.d;
    for (let i = 0; i < n; i++) {
      const e = ids[i] * d, p = (i % T) * d;
      for (let j = 0; j < d; j++) O[i * d + j] = E[e + j] + P[p + j];
    }
    rec(() => {
      if (!o.g) return;
      const G = o.g, dE = wte.grad(), dP = wpe.grad();
      for (let i = 0; i < n; i++) {
        const e = ids[i] * d, p = (i % T) * d;
        for (let j = 0; j < d; j++) { const g = G[i * d + j]; dE[e + j] += g; dP[p + j] += g; }
      }
    });
    return o;
  }

  function layernorm(x, g, b) {
    const n = x.r, m = x.c, o = new Tensor(n, m), X = x.d, O = o.d, xh = new F(n * m), rs = new F(n), Gm = g.d, Bt = b.d;
    for (let i = 0; i < n; i++) {
      let mu = 0; for (let j = 0; j < m; j++) mu += X[i * m + j]; mu /= m;
      let v = 0; for (let j = 0; j < m; j++) { const t = X[i * m + j] - mu; v += t * t; } v /= m;
      const r = 1 / Math.sqrt(v + 1e-5); rs[i] = r;
      for (let j = 0; j < m; j++) { const h = (X[i * m + j] - mu) * r; xh[i * m + j] = h; O[i * m + j] = h * Gm[j] + Bt[j]; }
    }
    rec(() => {
      if (!o.g) return;
      const G = o.g, dX = x.grad(), dg = g.grad(), db = b.grad();
      for (let i = 0; i < n; i++) {
        let s1 = 0, s2 = 0;
        for (let j = 0; j < m; j++) { const k = i * m + j, dy = G[k], dh = dy * Gm[j]; dg[j] += dy * xh[k]; db[j] += dy; s1 += dh; s2 += dh * xh[k]; }
        s1 /= m; s2 /= m;
        for (let j = 0; j < m; j++) { const k = i * m + j; dX[k] += rs[i] * (G[k] * Gm[j] - s1 - xh[k] * s2); }
      }
    });
    return o;
  }

  const GC = Math.sqrt(2 / Math.PI);
  function gelu(x) {
    const o = new Tensor(x.r, x.c), X = x.d, O = o.d;
    for (let i = 0; i < X.length; i++) { const v = X[i]; O[i] = 0.5 * v * (1 + Math.tanh(GC * (v + 0.044715 * v * v * v))); }
    rec(() => {
      if (!o.g) return;
      const G = o.g, dX = x.grad();
      for (let i = 0; i < X.length; i++) {
        const v = X[i], t = Math.tanh(GC * (v + 0.044715 * v * v * v));
        dX[i] += G[i] * (0.5 * (1 + t) + 0.5 * v * (1 - t * t) * GC * (1 + 3 * 0.044715 * v * v));
      }
    });
    return o;
  }

  // causal multi-head self-attention on a fused [q | k | v] input of width 3d
  function attention(qkv, B, T, H) {
    const d = qkv.c / 3, hd = d / H, sc = 1 / Math.sqrt(hd), Q = qkv.d, W3 = 3 * d;
    const o = new Tensor(B * T, d), O = o.d, P = new F(B * H * T * T);
    for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) for (let i = 0; i < T; i++) {
      const qo = (b * T + i) * W3 + h * hd, pb = ((b * H + h) * T + i) * T;
      let mx = -Infinity;
      for (let j = 0; j <= i; j++) {
        const ko = (b * T + j) * W3 + d + h * hd; let s = 0;
        for (let e = 0; e < hd; e++) s += Q[qo + e] * Q[ko + e];
        s *= sc; P[pb + j] = s; if (s > mx) mx = s;
      }
      let sum = 0;
      for (let j = 0; j <= i; j++) { const e = Math.exp(P[pb + j] - mx); P[pb + j] = e; sum += e; }
      const oo = (b * T + i) * d + h * hd;
      for (let j = 0; j <= i; j++) {
        const p = (P[pb + j] /= sum), vo = (b * T + j) * W3 + 2 * d + h * hd;
        for (let e = 0; e < hd; e++) O[oo + e] += p * Q[vo + e];
      }
    }
    o.att = P;
    rec(() => {
      if (!o.g) return;
      const G = o.g, dQ = qkv.grad(), dp = new F(T);
      for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) for (let i = 0; i < T; i++) {
        const qo = (b * T + i) * W3 + h * hd, pb = ((b * H + h) * T + i) * T, oo = (b * T + i) * d + h * hd;
        let dot = 0;
        for (let j = 0; j <= i; j++) {
          const vo = (b * T + j) * W3 + 2 * d + h * hd, p = P[pb + j]; let s = 0;
          for (let e = 0; e < hd; e++) { const g = G[oo + e]; s += g * Q[vo + e]; dQ[vo + e] += p * g; }
          dp[j] = s; dot += p * s;
        }
        for (let j = 0; j <= i; j++) {
          const ds = P[pb + j] * (dp[j] - dot) * sc, ko = (b * T + j) * W3 + d + h * hd;
          for (let e = 0; e < hd; e++) { dQ[qo + e] += ds * Q[ko + e]; dQ[ko + e] += ds * Q[qo + e]; }
        }
      }
    });
    return o;
  }

  function crossEntropy(logits, targets) {
    const n = logits.r, V = logits.c, L = logits.d, probs = new F(n * V);
    let loss = 0;
    for (let i = 0; i < n; i++) {
      let mx = -Infinity; for (let j = 0; j < V; j++) mx = Math.max(mx, L[i * V + j]);
      let s = 0; for (let j = 0; j < V; j++) { const e = Math.exp(L[i * V + j] - mx); probs[i * V + j] = e; s += e; }
      for (let j = 0; j < V; j++) probs[i * V + j] /= s;
      loss -= Math.log(probs[i * V + targets[i]] + 1e-12);
    }
    return {
      loss: loss / n,
      backward() {
        const dL = logits.grad();
        for (let i = 0; i < n; i++) for (let j = 0; j < V; j++) {
          const k = i * V + j; dL[k] += (probs[k] - (j === targets[i] ? 1 : 0)) / n;
        }
        for (let k = OPS.length - 1; k >= 0; k--) OPS[k]();
      },
    };
  }

  /* ---------- model ---------- */
  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gaussian(seed) { const r = mulberry32(seed); return () => { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }; }

  function createModel(V, cfg) {
    const c = Object.assign({ d: 32, L: 2, H: 4, T: 24, seed: 42 }, cfg || {});
    const rnd = gaussian(c.seed), params = [], d = c.d;
    const P = (name, r, cc, std, fill) => {
      const t = new Tensor(r, cc); t.name = name;
      if (fill !== undefined) t.d.fill(fill); else for (let i = 0; i < t.d.length; i++) t.d[i] = rnd() * std;
      params.push(t); return t;
    };
    const m = { cfg: c, V, params, step: 0, t: 0 };
    m.wte = P('wte', V, d, 0.1); m.wpe = P('wpe', c.T, d, 0.1);
    m.layers = [];
    for (let l = 0; l < c.L; l++) m.layers.push({
      ln1g: P(`L${l}.attn.ln_g`, 1, d, 0, 1), ln1b: P(`L${l}.attn.ln_b`, 1, d, 0, 0),
      wqkv: P(`L${l}.attn.wqkv`, d, 3 * d, 1 / Math.sqrt(d)), bqkv: P(`L${l}.attn.bqkv`, 1, 3 * d, 0, 0),
      wo: P(`L${l}.attn.wo`, d, d, 0.5 / Math.sqrt(d)), bo: P(`L${l}.attn.bo`, 1, d, 0, 0),
      ln2g: P(`L${l}.mlp.ln_g`, 1, d, 0, 1), ln2b: P(`L${l}.mlp.ln_b`, 1, d, 0, 0),
      w1: P(`L${l}.mlp.w1`, d, 4 * d, 1 / Math.sqrt(d)), b1: P(`L${l}.mlp.b1`, 1, 4 * d, 0, 0),
      w2: P(`L${l}.mlp.w2`, 4 * d, d, 0.5 / Math.sqrt(4 * d)), b2: P(`L${l}.mlp.b2`, 1, d, 0, 0),
    });
    m.lnfg = P('head.ln_g', 1, d, 0, 1); m.lnfb = P('head.ln_b', 1, d, 0, 0);
    m.wout = P('head.wout', d, V, 0.02); m.bout = P('head.bout', 1, V, 0, 0);
    m.nParams = params.reduce((s, p) => s + p.d.length, 0);
    m.opt = params.map(p => ({ m: new F(p.d.length), v: new F(p.d.length) }));
    return m;
  }

  function forward(m, ids, B, T) {
    let x = embed(m.wte, m.wpe, ids, T);
    const atts = [];
    for (const l of m.layers) {
      const a = attention(addBias(matmul(layernorm(x, l.ln1g, l.ln1b), l.wqkv), l.bqkv), B, T, m.cfg.H);
      atts.push(a.att);
      x = add(x, addBias(matmul(a, l.wo), l.bo));
      x = add(x, addBias(matmul(gelu(addBias(matmul(layernorm(x, l.ln2g, l.ln2b), l.w1), l.b1)), l.w2), l.b2));
    }
    return { logits: addBias(matmul(layernorm(x, m.lnfg, m.lnfb), m.wout), m.bout), atts };
  }

  // loss and parameter gradients for one batch
  function lossAndGrad(m, ids, targets, B, T) {
    OPS = []; RECORD = true;
    const ce = crossEntropy(forward(m, ids, B, T).logits, targets);
    for (const p of m.params) p.grad().fill(0);
    ce.backward();
    OPS = [];
    return ce.loss;
  }
  function lossOnly(m, ids, targets, B, T) {
    RECORD = false;
    const l = crossEntropy(forward(m, ids, B, T).logits, targets).loss;
    RECORD = true; OPS = [];
    return l;
  }

  function batch(data, B, T, rng) {
    const ids = new Int32Array(B * T), tg = new Int32Array(B * T);
    for (let b = 0; b < B; b++) {
      const s = Math.floor(rng() * (data.length - T));
      for (let t = 0; t < T; t++) { ids[b * T + t] = data[s + t]; tg[b * T + t] = data[s + t + 1]; }
    }
    return { ids, tg };
  }

  function trainStep(m, data, opts) {
    const o = Object.assign({ lr: 3e-3, B: 12, clip: 1, rng: Math.random }, opts || {}), T = m.cfg.T;
    const { ids, tg } = batch(data, o.B, T, o.rng);
    const loss = lossAndGrad(m, ids, tg, o.B, T);
    let norm = 0;
    const gnorms = new Float32Array(m.params.length);
    m.params.forEach((p, k) => { let s = 0; for (let i = 0; i < p.g.length; i++) s += p.g[i] * p.g[i]; gnorms[k] = Math.sqrt(s); norm += s; });
    norm = Math.sqrt(norm);
    const scale = norm > o.clip ? o.clip / norm : 1, b1 = 0.9, b2 = 0.99;
    m.t++;
    const bc1 = 1 - Math.pow(b1, m.t), bc2 = 1 - Math.pow(b2, m.t);
    m.params.forEach((p, k) => {
      const s = m.opt[k], G = p.g, D = p.d;
      for (let i = 0; i < D.length; i++) {
        const g = G[i] * scale;
        s.m[i] = b1 * s.m[i] + (1 - b1) * g;
        s.v[i] = b2 * s.v[i] + (1 - b2) * g * g;
        D[i] -= o.lr * (s.m[i] / bc1) / (Math.sqrt(s.v[i] / bc2) + 1e-8);
      }
    });
    m.step++;
    return { loss, gradNorm: norm, gnorms };
  }

  function evalLoss(m, data, nBatches, rng) {
    const T = m.cfg.T, B = 8; let s = 0;
    for (let k = 0; k < (nBatches || 4); k++) { const { ids, tg } = batch(data, B, T, rng || Math.random); s += lossOnly(m, ids, tg, B, T); }
    return s / (nBatches || 4);
  }

  // next-token distribution after a context; also returns attention maps for that context
  function predict(m, ctx, temperature) {
    const T = Math.min(ctx.length, m.cfg.T), ids = Int32Array.from(ctx.slice(-T));
    RECORD = false;
    const { logits, atts } = forward(m, ids, 1, T);
    RECORD = true; OPS = [];
    const V = m.V, row = logits.d.subarray((T - 1) * V, T * V), tmp = temperature || 1;
    let mx = -Infinity; for (let j = 0; j < V; j++) mx = Math.max(mx, row[j] / tmp);
    const probs = new Float64Array(V); let s = 0;
    for (let j = 0; j < V; j++) { probs[j] = Math.exp(row[j] / tmp - mx); s += probs[j]; }
    for (let j = 0; j < V; j++) probs[j] /= s;
    return { probs, atts, T };
  }

  // Runs one context through the model and keeps every intermediate tensor, for visualisation.
  function trace(m, ctx) {
    const T = Math.min(ctx.length, m.cfg.T), ids = Int32Array.from(ctx.slice(-T)), d = m.cfg.d, H = m.cfg.H, hd = d / H;
    RECORD = false;
    const tok = new F(T * d), pos = new F(T * d);
    for (let i = 0; i < T; i++) for (let j = 0; j < d; j++) { tok[i * d + j] = m.wte.d[ids[i] * d + j]; pos[i * d + j] = m.wpe.d[i * d + j]; }
    let x = embed(m.wte, m.wpe, ids, T);
    const out = { ids: Array.from(ids), T, d, H, hd, V: m.V, tok, pos, x0: x.d, layers: [] };
    for (const l of m.layers) {
      const ln1 = layernorm(x, l.ln1g, l.ln1b);
      const qkv = addBias(matmul(ln1, l.wqkv), l.bqkv);
      const a = attention(qkv, 1, T, H);
      const q = new F(T * d), k = new F(T * d), v = new F(T * d);
      for (let i = 0; i < T; i++) for (let j = 0; j < d; j++) { q[i * d + j] = qkv.d[i * 3 * d + j]; k[i * d + j] = qkv.d[i * 3 * d + d + j]; v[i * d + j] = qkv.d[i * 3 * d + 2 * d + j]; }
      const scores = new F(H * T * T).fill(NaN);
      for (let h = 0; h < H; h++) for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) {
        let s = 0; for (let e = 0; e < hd; e++) s += q[i * d + h * hd + e] * k[j * d + h * hd + e];
        scores[(h * T + i) * T + j] = s / Math.sqrt(hd);
      }
      const proj = addBias(matmul(a, l.wo), l.bo), x1 = add(x, proj);
      const ln2 = layernorm(x1, l.ln2g, l.ln2b), hpre = addBias(matmul(ln2, l.w1), l.b1), hpost = gelu(hpre);
      const mlp = addBias(matmul(hpost, l.w2), l.b2), x2 = add(x1, mlp);
      out.layers.push({ ln1: ln1.d, q, k, v, scores, att: a.att, mix: a.d, proj: proj.d, x1: x1.d, ln2: ln2.d, hpre: hpre.d, hpost: hpost.d, mlp: mlp.d, x2: x2.d });
      x = x2;
    }
    const lnf = layernorm(x, m.lnfg, m.lnfb), logits = addBias(matmul(lnf, m.wout), m.bout);
    RECORD = true; OPS = [];
    out.lnf = lnf.d; out.logits = logits.d;
    return out;
  }

  const api = { createModel, trainStep, evalLoss, predict, trace, lossAndGrad, lossOnly, mulberry32 };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.TinyGPT = api;
})(typeof window !== 'undefined' ? window : globalThis);
