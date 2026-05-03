import { useRef, useEffect, useCallback } from 'react';
import { createCanvas } from '../lib/canvas';
import { LAYER_COLORS } from './usePipeline';

export function usePainter({ numLayers, activeLayer, tool, brushSize, layerVis, showOrig }) {
  const canvasRef   = useRef(null);
  const maskRefs    = useRef([]);
  const overlayRefs = useRef([]); // cached overlay canvases — reused every frame
  const imgEl       = useRef(null);
  const isPainting  = useRef(false);
  const lastPt      = useRef(null);
  const activeBtn   = useRef(0);   // 0=left(brush) 2=right(erase)
  const rafId       = useRef(null);

  // ── Render ────────────────────────────────────────────────────────────────
  const renderComposite = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imgEl.current) return;
    const img = imgEl.current;

    // Resize canvas to image dimensions if needed (handles late-mount race)
    if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;
    }

    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(img, 0, 0, W, H);
    if (showOrig) return;

    for (let i = 0; i < numLayers; i++) {
      if (!layerVis[i]) continue;
      const mask    = maskRefs.current[i];    if (!mask)    continue;
      const overlay = overlayRefs.current[i]; if (!overlay) continue;

      // GPU compositing — no pixel loops, no getImageData
      const oCtx = overlay.getContext('2d');
      oCtx.clearRect(0, 0, W, H);
      const [r, g, b] = LAYER_COLORS[i].rgb;
      oCtx.fillStyle = `rgba(${r},${g},${b},0.58)`;
      oCtx.fillRect(0, 0, W, H);
      oCtx.globalCompositeOperation = 'destination-in';
      oCtx.drawImage(mask, 0, 0, W, H);
      oCtx.globalCompositeOperation = 'source-over';

      ctx.drawImage(overlay, 0, 0);
    }
  }, [numLayers, layerVis, showOrig]);

  // Throttle renders to one per animation frame
  const scheduleRender = useCallback(() => {
    if (rafId.current !== null) return;
    rafId.current = requestAnimationFrame(() => {
      rafId.current = null;
      renderComposite();
    });
  }, [renderComposite]);

  // ── Init ──────────────────────────────────────────────────────────────────
  const initMasks = useCallback((img) => {
    imgEl.current = img;
    const W = img.naturalWidth, H = img.naturalHeight;
    maskRefs.current    = Array.from({ length: 8 }, () => createCanvas(W, H));
    overlayRefs.current = Array.from({ length: 8 }, () => createCanvas(W, H));
    if (canvasRef.current) {
      canvasRef.current.width  = W;
      canvasRef.current.height = H;
    }
    // PaintScreen may not be mounted yet — defer until after React commit
    setTimeout(() => renderComposite(), 0);
  }, [renderComposite]);

  useEffect(() => { renderComposite(); }, [layerVis, showOrig, renderComposite]);

  // ── Pointer helpers ───────────────────────────────────────────────────────
  function getPos(e) {
    const canvas = canvasRef.current;
    const rect   = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: (cx - rect.left) * sx, y: (cy - rect.top) * sy };
  }

  function effectiveTool(e) {
    if (e.touches) return tool;                // touch: use UI tool
    return activeBtn.current === 2 ? 'eraser' : tool;
  }

  function paintAt(x, y, t) {
    const mask = maskRefs.current[activeLayer]; if (!mask) return;
    const ctx  = mask.getContext('2d');
    if (t === 'brush') {
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

  // ── Event handlers ────────────────────────────────────────────────────────
  function onDown(e) {
    if (e.button === 1) return;   // middle mouse → panning (handled by scroll container)
    e.preventDefault();
    activeBtn.current  = e.button ?? 0;
    isPainting.current = true;
    const pt = getPos(e);
    lastPt.current = pt;
    paintAt(pt.x, pt.y, effectiveTool(e));
    scheduleRender();
  }

  function onMove(e) {
    e.preventDefault();
    if (!isPainting.current) return;
    const pt = getPos(e);
    if (lastPt.current) {
      const dx = pt.x - lastPt.current.x, dy = pt.y - lastPt.current.y;
      const steps = Math.max(1, Math.floor(Math.sqrt(dx * dx + dy * dy) / (brushSize * .22)));
      const t = effectiveTool(e);
      for (let i = 1; i <= steps; i++)
        paintAt(lastPt.current.x + dx * (i / steps), lastPt.current.y + dy * (i / steps), t);
    }
    lastPt.current = pt;
    scheduleRender();
  }

  function onUp(e) {
    if (e?.button === 1) return;
    isPainting.current = false;
    lastPt.current = null;
  }

  function clearLayer(i) {
    const m = maskRefs.current[i];
    if (m) m.getContext('2d').clearRect(0, 0, m.width, m.height);
    renderComposite();
  }

  function clearAll() {
    maskRefs.current.forEach(m => m?.getContext('2d').clearRect(0, 0, m.width, m.height));
    renderComposite();
  }

  return { canvasRef, maskRefs, imgEl, initMasks, renderComposite, onDown, onMove, onUp, clearLayer, clearAll };
}
