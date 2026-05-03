import { useRef, useEffect, useCallback, useState } from 'react';
import { createCanvas } from '../lib/canvas';
import { LAYER_COLORS } from './usePipeline';

const SELECT_MAX_PX = 2048;
const MAX_UNDO      = 20;

export function usePainter({ numLayers, activeLayer, tool, brushSize, layerVis, showOrig }) {
  const canvasRef    = useRef(null);
  const maskRefs     = useRef([]);
  const overlayRefs  = useRef([]);
  const imgEl        = useRef(null);
  const isPainting   = useRef(false);
  const lastPt       = useRef(null);
  const activeBtn    = useRef(0);
  const rafId        = useRef(null);
  const selectingRef = useRef(false);
  const undoStack    = useRef([]);          // [{ layerIndex, data: ImageData }]

  // Stable refs for window-level handlers (avoid stale closures in useEffect)
  const toolRef        = useRef(tool);
  const brushSizeRef   = useRef(brushSize);
  const activeLayerRef = useRef(activeLayer);
  useEffect(() => { toolRef.current = tool; },           [tool]);
  useEffect(() => { brushSizeRef.current = brushSize; }, [brushSize]);
  useEffect(() => { activeLayerRef.current = activeLayer; }, [activeLayer]);

  const [selecting, setSelecting] = useState(false);

  // ── Render ────────────────────────────────────────────────────────────────
  const renderComposite = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imgEl.current) return;
    const img = imgEl.current;
    if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;
    }
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(img, 0, 0, W, H);
    if (showOrig) return;

    for (let i = 0; i < numLayers; i++) {
      if (!layerVis[i]) continue;
      const mask    = maskRefs.current[i];    if (!mask)    continue;
      const overlay = overlayRefs.current[i]; if (!overlay) continue;
      const oCtx = overlay.getContext('2d', { willReadFrequently: true });
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

  const scheduleRender = useCallback(() => {
    if (rafId.current !== null) return;
    rafId.current = requestAnimationFrame(() => { rafId.current = null; renderComposite(); });
  }, [renderComposite]);

  // ── Init ──────────────────────────────────────────────────────────────────
  const initMasks = useCallback((img) => {
    imgEl.current = img;
    const W = img.naturalWidth, H = img.naturalHeight;
    maskRefs.current    = Array.from({ length: 8 }, () => createCanvas(W, H));
    overlayRefs.current = Array.from({ length: 8 }, () => createCanvas(W, H));
    undoStack.current   = [];
    if (canvasRef.current) { canvasRef.current.width = W; canvasRef.current.height = H; }
    setTimeout(() => renderComposite(), 0);
  }, [renderComposite]);

  useEffect(() => { renderComposite(); }, [layerVis, showOrig, renderComposite]);

  // ── Undo ──────────────────────────────────────────────────────────────────
  function saveUndoState(layerIndex) {
    const mask = maskRefs.current[layerIndex];
    if (!mask) return;
    const data = mask.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, mask.width, mask.height);
    undoStack.current.push({ layerIndex, data });
    if (undoStack.current.length > MAX_UNDO) undoStack.current.shift();
  }

  const undo = useCallback(() => {
    if (undoStack.current.length === 0) return;
    const { layerIndex, data } = undoStack.current.pop();
    const mask = maskRefs.current[layerIndex];
    if (!mask) return;
    mask.getContext('2d', { willReadFrequently: true }).putImageData(data, 0, 0);
    renderComposite();
  }, [renderComposite]);

  useEffect(() => {
    const onKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [undo]);

  // ── Coordinates ───────────────────────────────────────────────────────────
  function getPos(e) {
    const canvas = canvasRef.current;
    const rect   = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: (cx - rect.left) * sx, y: (cy - rect.top) * sy };
  }

  // ── Brush ─────────────────────────────────────────────────────────────────
  function effectiveTool(e) {
    if (e.button === 2 || activeBtn.current === 2) return 'eraser';
    return tool;
  }

  // paintAt reads brushSize from ref so window-level handlers stay fresh
  function paintAt(x, y, t, layer) {
    const mask = maskRefs.current[layer]; if (!mask) return;
    const ctx  = mask.getContext('2d', { willReadFrequently: true });
    const bs   = brushSizeRef.current;
    if (t === 'brush') {
      const [r, g, b] = LAYER_COLORS[layer].rgb;
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = `rgba(${r},${g},${b},0.92)`;
      ctx.beginPath(); ctx.arc(x, y, bs, 0, Math.PI * 2); ctx.fill();
    } else {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.fillStyle = 'rgba(0,0,0,1)';
      ctx.beginPath(); ctx.arc(x, y, bs * 1.5, 0, Math.PI * 2); ctx.fill();
      ctx.globalCompositeOperation = 'source-over';
    }
  }

  // ── Window-level mouse handlers (cross-boundary painting) ─────────────────
  useEffect(() => {
    function onWindowMove(e) {
      if (!isPainting.current || toolRef.current === 'selector') return;
      const canvas = canvasRef.current; if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      // Canvas's own onMouseMove handles events inside — only handle outside here
      if (e.clientX >= rect.left && e.clientX <= rect.right &&
          e.clientY >= rect.top  && e.clientY <= rect.bottom) return;
      const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
      const pt = { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy };
      const layer = activeLayerRef.current;
      if (lastPt.current) {
        const dx = pt.x - lastPt.current.x, dy = pt.y - lastPt.current.y;
        const bs = brushSizeRef.current;
        const steps = Math.max(1, Math.floor(Math.sqrt(dx*dx + dy*dy) / (bs * .22)));
        const t = activeBtn.current === 2 ? 'eraser' : toolRef.current;
        for (let i = 1; i <= steps; i++)
          paintAt(lastPt.current.x + dx*(i/steps), lastPt.current.y + dy*(i/steps), t, layer);
      }
      lastPt.current = pt;
      scheduleRender();
    }

    function onWindowUp(e) {
      if (e.button === 1) return;
      isPainting.current = false;
      lastPt.current     = null;
    }

    window.addEventListener('mousemove', onWindowMove);
    window.addEventListener('mouseup',  onWindowUp);
    return () => {
      window.removeEventListener('mousemove', onWindowMove);
      window.removeEventListener('mouseup',  onWindowUp);
    };
  }, [scheduleRender]);

  // ── Click-to-select ────────────────────────────────────────────────────────
  async function doClickSelect(posX, posY, eraseMode, layer) {
    if (selectingRef.current) return;
    selectingRef.current = true;
    setSelecting(true);
    try {
      const img = imgEl.current;
      if (!img) return;
      const W = img.naturalWidth, H = img.naturalHeight;

      const scale = Math.min(1, SELECT_MAX_PX / Math.max(W, H));
      const sW = Math.round(W * scale), sH = Math.round(H * scale);

      const tmpC = createCanvas(sW, sH);
      tmpC.getContext('2d', { willReadFrequently: true }).drawImage(img, 0, 0, sW, sH);
      const id  = tmpC.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, sW, sH);
      const cX  = Math.max(0, Math.min(sW - 1, Math.round(posX * scale)));
      const cY  = Math.max(0, Math.min(sH - 1, Math.round(posY * scale)));
      const buf = id.data.buffer.slice(0);

      const alphaSmall = await new Promise((resolve, reject) => {
        const w = new Worker(new URL('../workers/alpha.worker.js', import.meta.url));
        w.onmessage = ({ data }) => { w.terminate(); resolve(new Float32Array(data.alphaBuffer)); };
        w.onerror   = (e)        => { w.terminate(); reject(new Error(e.message)); };
        w.postMessage({ type: 'clickSelect', imgBuffer: buf, W: sW, H: sH, clickX: cX, clickY: cY }, [buf]);
      });

      const mask = maskRefs.current[layer];
      if (!mask) return;
      const mCtx  = mask.getContext('2d', { willReadFrequently: true });
      const [r, g, b] = LAYER_COLORS[layer].rgb;
      const aC    = createCanvas(sW, sH);
      const aCtx  = aC.getContext('2d', { willReadFrequently: true });
      const aImg  = new ImageData(sW, sH);
      for (let j = 0; j < alphaSmall.length; j++) {
        const v = Math.min(255, Math.round(Math.pow(alphaSmall[j], 0.55) * 255));
        aImg.data[j * 4]     = r;
        aImg.data[j * 4 + 1] = g;
        aImg.data[j * 4 + 2] = b;
        aImg.data[j * 4 + 3] = v;
      }
      aCtx.putImageData(aImg, 0, 0);

      if (eraseMode) {
        mCtx.globalCompositeOperation = 'destination-out';
        mCtx.drawImage(aC, 0, 0, W, H);
        mCtx.globalCompositeOperation = 'source-over';
      } else {
        mCtx.drawImage(aC, 0, 0, W, H);
      }
      scheduleRender();
    } catch (err) {
      console.error('click-select failed:', err);
    } finally {
      selectingRef.current = false;
      setSelecting(false);
    }
  }

  // ── Event handlers ─────────────────────────────────────────────────────────
  function onDown(e) {
    if (e.button === 1) return;
    e.preventDefault();
    activeBtn.current = e.button ?? 0;
    const pt = getPos(e);

    if (tool === 'selector') {
      const capturedLayer = activeLayer;
      saveUndoState(capturedLayer);
      doClickSelect(pt.x, pt.y, activeBtn.current === 2, capturedLayer);
      return;
    }

    saveUndoState(activeLayer);
    isPainting.current = true;
    lastPt.current     = pt;
    paintAt(pt.x, pt.y, effectiveTool(e), activeLayer);
    scheduleRender();
  }

  function onMove(e) {
    e.preventDefault();
    if (!isPainting.current || tool === 'selector') return;
    const pt = getPos(e);
    if (lastPt.current) {
      const dx = pt.x - lastPt.current.x, dy = pt.y - lastPt.current.y;
      const steps = Math.max(1, Math.floor(Math.sqrt(dx*dx + dy*dy) / (brushSize * .22)));
      const t     = effectiveTool(e);
      for (let i = 1; i <= steps; i++)
        paintAt(lastPt.current.x + dx*(i/steps), lastPt.current.y + dy*(i/steps), t, activeLayer);
    }
    lastPt.current = pt;
    scheduleRender();
  }

  function onUp(e) {
    if (e?.button === 1) return;
    isPainting.current = false;
    lastPt.current     = null;
  }

  // Resume painting when cursor re-enters canvas with button still held
  function onEnter(e) {
    if (toolRef.current === 'selector') return;
    if (e.buttons === 1 || e.buttons === 2) {
      activeBtn.current  = e.buttons === 2 ? 2 : 0;
      isPainting.current = true;
      lastPt.current     = null;
    }
  }

  function clearLayer(i) {
    saveUndoState(i);
    const m = maskRefs.current[i];
    if (m) m.getContext('2d', { willReadFrequently: true }).clearRect(0, 0, m.width, m.height);
    renderComposite();
  }

  function clearAll() {
    for (let i = 0; i < numLayers; i++) saveUndoState(i);
    maskRefs.current.forEach(m => m?.getContext('2d', { willReadFrequently: true }).clearRect(0, 0, m.width, m.height));
    renderComposite();
  }

  return { canvasRef, maskRefs, imgEl, initMasks, renderComposite, onDown, onMove, onUp, onEnter, clearLayer, clearAll, selecting };
}
