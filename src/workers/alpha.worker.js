function expandStrokeToAlphaMap(strokeData, W, H, brushSize, expansionFactor) {
  const base = new Float32Array(W * H);
  for (let p = 3; p < strokeData.length; p += 4) base[(p - 3) / 4] = strokeData[p] / 255;
  const radius = brushSize * expansionFactor;
  const result = new Float32Array(W * H);
  const step = Math.max(1, Math.round(radius / 12));
  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      let maxA = 0;
      const x0 = Math.max(0, (px - radius) | 0), x1 = Math.min(W - 1, (px + radius) | 0);
      const y0 = Math.max(0, (py - radius) | 0), y1 = Math.min(H - 1, (py + radius) | 0);
      for (let ny = y0; ny <= y1; ny += step) {
        for (let nx = x0; nx <= x1; nx += step) {
          const dx = nx - px, dy = ny - py;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d <= radius) {
            const a = base[ny * W + nx] * Math.max(0, 1 - d / (radius * 1.25));
            if (a > maxA) maxA = a;
          }
        }
      }
      result[py * W + px] = maxA;
    }
  }
  return result;
}

function applyBBoxesToAlphaMap(alphaMap, bboxes, W, H) {
  for (const obj of bboxes) {
    const x1 = Math.round(obj.x1pct / 100 * W), y1 = Math.round(obj.y1pct / 100 * H);
    const x2 = Math.round(obj.x2pct / 100 * W), y2 = Math.round(obj.y2pct / 100 * H);
    const soft = Math.max(4, (obj.softness || 10) * 3);
    for (let py = y1; py <= y2; py++) {
      for (let px = x1; px <= x2; px++) {
        const minD = Math.min(py - y1, y2 - py, px - x1, x2 - px);
        const a = Math.min(1, minD / soft) * (obj.priority || 1.0);
        if (a > alphaMap[py * W + px]) alphaMap[py * W + px] = a;
      }
    }
  }
  return alphaMap;
}

self.onmessage = function ({ data }) {
  const { strokeBuffer, W, H, brushSize, expansionFactor, bboxes } = data;
  const strokeData = new Uint8ClampedArray(strokeBuffer);
  let alphaMap = expandStrokeToAlphaMap(strokeData, W, H, brushSize, expansionFactor);
  if (bboxes?.length) alphaMap = applyBBoxesToAlphaMap(alphaMap, bboxes, W, H);
  self.postMessage({ alphaBuffer: alphaMap.buffer }, [alphaMap.buffer]);
};
