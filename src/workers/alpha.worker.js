// ── Shared: Sobel gradient ─────────────────────────────────────────────────
function computeSobel(imgData, W, H) {
  const grad = new Float32Array(W * H);
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      let gxSq = 0, gySq = 0;
      for (let c = 0; c < 3; c++) {
        const v = (dy, dx) => imgData[((y + dy) * W + (x + dx)) * 4 + c];
        const gx = -v(-1,-1) - 2*v(0,-1) - v(1,-1) + v(-1,1) + 2*v(0,1) + v(1,1);
        const gy = -v(-1,-1) - 2*v(-1,0) - v(-1,1) + v(1,-1) + 2*v(1,0) + v(1,1);
        gxSq += gx * gx; gySq += gy * gy;
      }
      grad[y * W + x] = Math.sqrt(gxSq + gySq) / (3 * 255 * 8);
    }
  }
  return grad;
}

// ── Mode A: click-to-select (BFS + gradient barrier) ──────────────────────
// The user clicks a point; we grow the selection from there using
// color similarity and stop at high-gradient edges (object boundaries).
function clickSelect(imgData, W, H, clickX, clickY) {
  const cx = Math.max(0, Math.min(W - 1, clickX));
  const cy = Math.max(0, Math.min(H - 1, clickY));

  // Sample 9×9 neighborhood to get object color statistics
  const NR = 4;
  let sumR = 0, sumG = 0, sumB = 0, n = 0;
  const samples = [];
  for (let sy = Math.max(0, cy - NR); sy <= Math.min(H - 1, cy + NR); sy++) {
    for (let sx = Math.max(0, cx - NR); sx <= Math.min(W - 1, cx + NR); sx++) {
      const p = (sy * W + sx) * 4;
      const r = imgData[p], g = imgData[p+1], b = imgData[p+2];
      samples.push(r, g, b);
      sumR += r; sumG += g; sumB += b; n++;
    }
  }
  const mR = sumR / n, mG = sumG / n, mB = sumB / n;

  // Adaptive tolerance based on local variance (handles both uniform and textured objects)
  let varSum = 0;
  for (let i = 0; i < samples.length; i += 3) {
    varSum += (samples[i]-mR)**2 + (samples[i+1]-mG)**2 + (samples[i+2]-mB)**2;
  }
  const std = Math.sqrt(varSum / (n * 3));
  const tolerance = Math.max(22, Math.min(75, std * 2.8 + 24));

  const grad = computeSobel(imgData, W, H);
  const GRAD_CUT = 0.20;

  const result  = new Float32Array(W * H);
  const visited = new Uint8Array(W * H);
  const queue   = new Int32Array(W * H);
  let head = 0, tail = 0;

  const seedIdx = cy * W + cx;
  queue[tail++] = seedIdx;
  visited[seedIdx] = 1;
  result[seedIdx]  = 1;

  while (head < tail) {
    const idx = queue[head++];
    const y   = (idx / W) | 0;
    const x   = idx % W;

    // 4-connected neighbors
    const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    for (const [dy, dx] of dirs) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
      const ni = ny * W + nx;
      if (visited[ni]) continue;
      visited[ni] = 1;

      // Stop at strong edges
      if (grad[ni] > GRAD_CUT) continue;

      const p    = ni * 4;
      const r    = imgData[p], g = imgData[p+1], b = imgData[p+2];
      const dist = Math.sqrt((r-mR)**2 + (g-mG)**2 + (b-mB)**2);
      if (dist <= tolerance) {
        result[ni] = Math.max(0, 1 - dist / (tolerance * 1.25));
        queue[tail++] = ni;
      }
    }
  }

  return result;
}

// ── Mode B: edge-aware propagation from stroke mask ────────────────────────
// Used by the processing pipeline when the user painted rough brush strokes.
// The propagation starts from those strokes and stops at image edges.
function edgeAwareSelect(strokeData, imgData, W, H) {
  const grad = computeSobel(imgData, W, H);

  const alpha = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) alpha[i] = strokeData[i * 4 + 3] / 255;

  const BARRIER = 18;
  const PASSES  = 10;

  for (let pass = 0; pass < PASSES; pass++) {
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const i    = y * W + x;
        const cost = 1 + grad[i] * BARRIER;
        const nbrs = [(y-1)*W+x, y*W+(x-1), (y-1)*W+(x-1), (y-1)*W+(x+1)];
        for (const ni of nbrs) {
          const p = alpha[ni] / cost;
          if (p > alpha[i]) alpha[i] = p;
        }
      }
    }
    for (let y = H - 2; y >= 1; y--) {
      for (let x = W - 2; x >= 1; x--) {
        const i    = y * W + x;
        const cost = 1 + grad[i] * BARRIER;
        const nbrs = [(y+1)*W+x, y*W+(x+1), (y+1)*W+(x+1), (y+1)*W+(x-1)];
        for (const ni of nbrs) {
          const p = alpha[ni] / cost;
          if (p > alpha[i]) alpha[i] = p;
        }
      }
    }
  }

  return alpha;
}

function applyThreshold(alpha, cutoff = 0.07, feather = 0.18) {
  for (let i = 0; i < alpha.length; i++) {
    const a = alpha[i];
    if (a <= cutoff) { alpha[i] = 0; continue; }
    if (a < cutoff + feather) alpha[i] = (a - cutoff) / feather;
  }
  return alpha;
}

// ── Dispatch ───────────────────────────────────────────────────────────────
self.onmessage = function ({ data }) {
  if (data.type === 'clickSelect') {
    const imgData  = new Uint8ClampedArray(data.imgBuffer);
    const alphaMap = clickSelect(imgData, data.W, data.H, data.clickX, data.clickY);
    applyThreshold(alphaMap, 0.05, 0.15);
    self.postMessage({ alphaBuffer: alphaMap.buffer }, [alphaMap.buffer]);
    return;
  }

  // Stroke-based (pipeline)
  const strokeData = new Uint8ClampedArray(data.strokeBuffer);
  let alphaMap;
  if (data.imgBuffer) {
    const imgData = new Uint8ClampedArray(data.imgBuffer);
    alphaMap = edgeAwareSelect(strokeData, imgData, data.W, data.H);
  } else {
    alphaMap = new Float32Array(data.W * data.H);
    for (let i = 0; i < data.W * data.H; i++) alphaMap[i] = strokeData[i * 4 + 3] / 255;
  }
  applyThreshold(alphaMap);
  self.postMessage({ alphaBuffer: alphaMap.buffer }, [alphaMap.buffer]);
};
