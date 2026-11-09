/* Runs TinyGPT training off the main thread and streams progress back to the page. */
importScripts('tinygpt.js');

let model = null, train = null, val = null, lr = 6e-3, running = false, timer = null, gEma = null;
const B = 8;

function flatParams() {
  const out = new Float32Array(model.nParams); let o = 0;
  for (const p of model.params) { out.set(p.d, o); o += p.d.length; }
  return out;
}

function tick() {
  if (!running) return;
  const losses = [], t0 = performance.now();
  let valLoss = null;
  while (performance.now() - t0 < 120) {
    const { loss, gnorms } = TinyGPT.trainStep(model, train, { lr, B });
    losses.push(loss);
    if (!gEma) gEma = Float32Array.from(gnorms);
    else for (let i = 0; i < gnorms.length; i++) gEma[i] = 0.8 * gEma[i] + 0.2 * gnorms[i];
    if (val && model.step % 50 === 0) valLoss = { step: model.step, loss: TinyGPT.evalLoss(model, val, 3) };
  }
  const params = flatParams();
  postMessage({ type: 'tick', step: model.step, losses, valLoss, params, gnorms: gEma }, [params.buffer]);
  timer = setTimeout(tick, 0);
}

onmessage = ({ data: msg }) => {
  if (msg.type === 'init') {
    clearTimeout(timer); running = false;
    model = TinyGPT.createModel(msg.V, msg.cfg); gEma = null;
    train = msg.train; val = msg.val; lr = msg.lr;
    const params = flatParams();
    postMessage({ type: 'ready', step: 0, params }, [params.buffer]);
  } else if (msg.type === 'run') { if (!running) { running = true; tick(); } }
  else if (msg.type === 'pause') { running = false; clearTimeout(timer); }
  else if (msg.type === 'lr') lr = msg.lr;
};
