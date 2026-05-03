export function canvasToJpeg(canvas, quality = 0.85) {
  return canvas.toDataURL('image/jpeg', quality).split(',')[1];
}
export function canvasToPng(canvas) {
  return canvas.toDataURL('image/png').split(',')[1];
}
export function createCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h; return c;
}
export function resizeToFit(img, maxPx = 900) {
  const scale = Math.min(1, maxPx / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.round(img.naturalWidth * scale), h = Math.round(img.naturalHeight * scale);
  const c = createCanvas(w, h);
  c.getContext('2d', { willReadFrequently: true }).drawImage(img, 0, 0, w, h);
  return { canvas: c, w, h, scale };
}
export function resizeToStability(img) {
  const maxPx = 1024;
  const scale = Math.min(1, maxPx / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.round(img.naturalWidth * scale / 64) * 64;
  const h = Math.round(img.naturalHeight * scale / 64) * 64;
  const c = createCanvas(w, h);
  c.getContext('2d', { willReadFrequently: true }).drawImage(img, 0, 0, w, h);
  return { canvas: c, w, h };
}

// GPU compositing — no pixel loops
export function buildSketchOverlay(img, maskCanvases, numLayers, COLORS, tw, th) {
  const c = createCanvas(tw, th);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, tw, th);
  for (let i = 0; i < numLayers; i++) {
    const tmp = createCanvas(tw, th);
    const tCtx = tmp.getContext('2d', { willReadFrequently: true });
    const [r, g, b] = COLORS[i].rgb;
    tCtx.fillStyle = `rgba(${r},${g},${b},0.78)`;
    tCtx.fillRect(0, 0, tw, th);
    tCtx.globalCompositeOperation = 'destination-in';
    tCtx.drawImage(maskCanvases[i], 0, 0, tw, th);
    tCtx.globalCompositeOperation = 'source-over';
    ctx.drawImage(tmp, 0, 0);
  }
  return c;
}

// GPU compositing — single getImageData only for final threshold
export function buildInpaintMask(maskCanvases, layerIdx, W, H) {
  const c = createCanvas(W, H);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, W, H);
  for (let j = 0; j < layerIdx; j++) {
    const tmp = createCanvas(W, H);
    const tCtx = tmp.getContext('2d', { willReadFrequently: true });
    tCtx.fillStyle = 'white';
    tCtx.fillRect(0, 0, W, H);
    tCtx.globalCompositeOperation = 'destination-in';
    tCtx.drawImage(maskCanvases[j], 0, 0, W, H);
    tCtx.globalCompositeOperation = 'source-over';
    ctx.drawImage(tmp, 0, 0);
  }
  const fc = createCanvas(W, H);
  const fCtx = fc.getContext('2d', { willReadFrequently: true });
  fCtx.filter = 'blur(10px)'; fCtx.drawImage(c, 0, 0); fCtx.filter = 'none';
  const fd = fCtx.getImageData(0, 0, W, H);
  for (let p = 0; p < fd.data.length; p += 4) {
    const v = fd.data[p] > 15 ? 255 : 0;
    fd.data[p] = v; fd.data[p + 1] = v; fd.data[p + 2] = v; fd.data[p + 3] = 255;
  }
  fCtx.putImageData(fd, 0, 0);
  return fc;
}
