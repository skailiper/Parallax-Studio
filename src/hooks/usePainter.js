import { useRef, useEffect, useCallback } from 'react';
import { createCanvas } from '../lib/canvas';
import { LAYER_COLORS } from './usePipeline';

export function usePainter({ numLayers, activeLayer, tool, brushSize, layerVis, showOrig }) {
  const canvasRef  = useRef(null);
  const maskRefs   = useRef([]);
  const imgEl      = useRef(null);
  const isPainting = useRef(false);
  const lastPt     = useRef(null);

  const initMasks = useCallback((img) => {
    imgEl.current = img;
    const W = img.naturalWidth, H = img.naturalHeight;
    maskRefs.current = Array.from({ length: 8 }, () => createCanvas(W, H));
    if (canvasRef.current) { canvasRef.current.width = W; canvasRef.current.height = H; }
  }, []);

  const renderComposite = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imgEl.current) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(imgEl.current, 0, 0, W, H);
    if (showOrig) return;
    for (let i = 0; i < numLayers; i++) {
      if (!layerVis[i]) continue;
      const mask = maskRefs.current[i]; if (!mask) continue;
      const tmp = createCanvas(W, H);
      const tCtx = tmp.getContext('2d');
      tCtx.drawImage(mask, 0, 0, W, H);
      const td = tCtx.getImageData(0, 0, W, H);
      const [r, g, b] = LAYER_COLORS[i].rgb;
      for (let p = 0; p < td.data.length; p += 4)
        if (td.data[p + 3] > 0) { td.data[p]=r; td.data[p+1]=g; td.data[p+2]=b; td.data[p+3]=Math.round(td.data[p+3]*.58); }
      tCtx.putImageData(td, 0, 0);
      ctx.drawImage(tmp, 0, 0);
    }
  }, [numLayers, layerVis, showOrig]);

  useEffect(() => { renderComposite(); }, [layerVis, showOrig, renderComposite]);

  function getPos(e) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: (cx - rect.left) * sx, y: (cy - rect.top) * sy };
  }

  function paintAt(x, y) {
    const mask = maskRefs.current[activeLayer]; if (!mask) return;
    const ctx = mask.getContext('2d');
    if (tool === 'brush') {
      const [r, g, b] = LAYER_COLORS[activeLayer].rgb;
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = `rgba(${r},${g},${b},0.92)`;
      ctx.beginPath(); ctx.arc(x, y, brushSize, 0, Math.PI * 2); ctx.fill();
    } else {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.fillStyle = 'rgba(0,0,0,1)';
      ctx.beginPath(); ctx.arc(x, y, brushSize * 1.5, 0, Math.PI * 2); ctx.fill();
      ctx.globalCompositeOperation = 'source-over';
    }
  }

  function onDown(e) { e.preventDefault(); isPainting.current = true; const pt = getPos(e); lastPt.current = pt; paintAt(pt.x, pt.y); renderComposite(); }
  function onMove(e) {
    e.preventDefault(); if (!isPainting.current) return;
    const pt = getPos(e);
    if (lastPt.current) {
      const dx = pt.x - lastPt.current.x, dy = pt.y - lastPt.current.y;
      const steps = Math.max(1, Math.floor(Math.sqrt(dx*dx + dy*dy) / (brushSize * .22)));
      for (let i = 1; i <= steps; i++) paintAt(lastPt.current.x + dx * (i/steps), lastPt.current.y + dy * (i/steps));
    }
    lastPt.current = pt; renderComposite();
  }
  function onUp() { isPainting.current = false; lastPt.current = null; }
  function clearLayer(i) { const m = maskRefs.current[i]; if (m) m.getContext('2d').clearRect(0, 0, m.width, m.height); renderComposite(); }
  function clearAll() { maskRefs.current.forEach(m => m?.getContext('2d').clearRect(0, 0, m.width, m.height)); renderComposite(); }

  return { canvasRef, maskRefs, imgEl, initMasks, renderComposite, onDown, onMove, onUp, clearLayer, clearAll };
}
