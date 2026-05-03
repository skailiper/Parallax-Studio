// Edge-aware selection: starts from brush strokes and propagates outward,
// but strong image gradients (object edges) act as barriers — the selection
// naturally stops at real object boundaries, similar to DaVinci Magic Mask.

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
      grad[y * W + x] = Math.sqrt(gxSq + gySq) / (3 * 255 * 8); // 0–1
    }
  }
  return grad;
}

// Multi-pass raster scan approximation of geodesic distance with gradient barriers.
// BARRIER controls how hard it is to cross an edge:
//   high BARRIER → selection stops cleanly at object contours
//   low BARRIER  → selection bleeds across edges (closer to simple dilation)
function edgeAwareSelect(strokeData, imgData, W, H) {
  const grad = computeSobel(imgData, W, H);

  // Seed: all painted pixels start with full alpha
  const alpha = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    alpha[i] = strokeData[i * 4 + 3] / 255;
  }

  const BARRIER = 18;
  const PASSES  = 12;

  for (let pass = 0; pass < PASSES; pass++) {
    // Forward sweep (TL → BR)
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const i = y * W + x;
        const cost = 1 + grad[i] * BARRIER;
        const nbrs = [(y-1)*W+x, y*W+(x-1), (y-1)*W+(x-1), (y-1)*W+(x+1)];
        for (const ni of nbrs) {
          const prop = alpha[ni] / cost;
          if (prop > alpha[i]) alpha[i] = prop;
        }
      }
    }
    // Backward sweep (BR → TL)
    for (let y = H - 2; y >= 1; y--) {
      for (let x = W - 2; x >= 1; x--) {
        const i = y * W + x;
        const cost = 1 + grad[i] * BARRIER;
        const nbrs = [(y+1)*W+x, y*W+(x+1), (y+1)*W+(x+1), (y+1)*W+(x-1)];
        for (const ni of nbrs) {
          const prop = alpha[ni] / cost;
          if (prop > alpha[i]) alpha[i] = prop;
        }
      }
    }
  }

  return alpha;
}

// Soft threshold: values below cutoff become 0; transition zone gets smooth feather.
function applyThreshold(alpha, cutoff = 0.08, feather = 0.18) {
  for (let i = 0; i < alpha.length; i++) {
    const a = alpha[i];
    if (a <= cutoff) { alpha[i] = 0; continue; }
    if (a <= cutoff + feather) {
      alpha[i] = (a - cutoff) / feather;
    }
    // above feather zone: keep as-is (will be boosted in cutout step)
  }
  return alpha;
}

self.onmessage = function ({ data }) {
  const { strokeBuffer, imgBuffer, W, H } = data;
  const strokeData = new Uint8ClampedArray(strokeBuffer);

  let alphaMap;
  if (imgBuffer) {
    const imgData = new Uint8ClampedArray(imgBuffer);
    alphaMap = edgeAwareSelect(strokeData, imgData, W, H);
  } else {
    // Fallback: simple copy of stroke alpha if no image data
    alphaMap = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) alphaMap[i] = strokeData[i * 4 + 3] / 255;
  }

  applyThreshold(alphaMap);
  self.postMessage({ alphaBuffer: alphaMap.buffer }, [alphaMap.buffer]);
};
