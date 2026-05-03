import { useState, useCallback } from 'react';
import { resizeToReplicate, buildInpaintMask, canvasToPng, createCanvas } from '../lib/canvas';
import { createProject, updateProjectStatus, saveProjectLayers } from '../lib/supabase';
import { getSessionId } from '../lib/session';
import { withRetry } from '../lib/retry';

export const LAYER_COLORS = [
  { hex: '#FF3D5A', name: 'Layer 1', rgb: [255, 61, 90]   },
  { hex: '#00E5FF', name: 'Layer 2', rgb: [0, 229, 255]   },
  { hex: '#AAFF00', name: 'Layer 3', rgb: [170, 255, 0]   },
  { hex: '#FF9500', name: 'Layer 4', rgb: [255, 149, 0]   },
  { hex: '#BF5FFF', name: 'Layer 5', rgb: [191, 95, 255]  },
  { hex: '#FFE600', name: 'Layer 6', rgb: [255, 230, 0]   },
  { hex: '#00FFB2', name: 'Layer 7', rgb: [0, 255, 178]   },
  { hex: '#FF6BCA', name: 'Layer 8', rgb: [255, 107, 202] },
];

async function callReplicate(body) {
  const res = await fetch('/api/replicate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Replicate error');
  return data;
}

function retryOpts(addLog, label) {
  return {
    attempts: 3,
    baseDelayMs: 2000,
    onRetry: (err, attempt) => addLog(`⟳ ${label} — tentativa ${attempt + 1}/3: ${err.message}`, 'warn'),
  };
}

// Edge-aware selection in a Web Worker — main thread stays free.
function runAlphaWorker({ strokeBuffer, imgBuffer, W, H }) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('../workers/alpha.worker.js', import.meta.url));
    worker.onmessage = ({ data }) => { worker.terminate(); resolve(new Float32Array(data.alphaBuffer)); };
    worker.onerror  = (e)        => { worker.terminate(); reject(new Error(e.message || 'Worker error')); };
    const transfers = [strokeBuffer];
    if (imgBuffer) transfers.push(imgBuffer);
    worker.postMessage({ strokeBuffer, imgBuffer, W, H }, transfers);
  });
}

export function usePipeline() {
  const [logs,     setLogs]     = useState([]);
  const [progress, setProgress] = useState(0);
  const [phase,    setPhase]    = useState('idle');
  const addLog = useCallback((msg, type = 'info') => setLogs(p => [...p, { msg, type, id: Date.now() + Math.random() }]), []);

  const run = useCallback(async ({ imgEl, maskRefs, numLayers, imgFile, useGenerativeAI = true }) => {
    setPhase('running'); setLogs([]); setProgress(0);
    const sessionId = getSessionId();
    const W = imgEl.current.naturalWidth, H = imgEl.current.naturalHeight;
    let projectId = null;
    try {
      const project = await createProject({ sessionId, numLayers, imageFilename: imgFile.name, imageSizeBytes: imgFile.size });
      projectId = project.id;
    } catch { addLog('⚠️ Supabase offline — continuando sem salvar', 'warn'); }

    // Processing resolution capped at 640px to keep worker fast
    const procScale = Math.min(1, 640 / Math.max(W, H));
    const pW = Math.round(W * procScale), pH = Math.round(H * procScale);

    try {
      // ── Step 1: Per-layer cutout via edge-aware canvas worker ───────────────
      addLog('✂️ Segmentando layers…', 'ai');
      const cutouts = [];

      for (let i = 0; i < numLayers; i++) {
        addLog(`✂️ Recortando Layer ${i + 1}…`, 'ai');

        const smallMask = createCanvas(pW, pH);
        smallMask.getContext('2d', { willReadFrequently: true }).drawImage(maskRefs[i], 0, 0, pW, pH);
        const sd = smallMask.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, pW, pH);

        // Skip unpainted layers
        let painted = false;
        for (let p = 3; p < sd.data.length; p += 4) if (sd.data[p] > 10) { painted = true; break; }
        if (!painted) { addLog(`⚪ Layer ${i + 1} sem pintura, pulando`, 'warn'); cutouts.push(null); continue; }

        // Edge-aware worker (Sobel gradient propagation)
        const smallImg = createCanvas(pW, pH);
        smallImg.getContext('2d', { willReadFrequently: true }).drawImage(imgEl.current, 0, 0, pW, pH);
        const imgSd = smallImg.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, pW, pH);
        const alphaSmall = await runAlphaWorker({
          strokeBuffer: sd.data.buffer.slice(0),
          imgBuffer:    imgSd.data.buffer.slice(0),
          W: pW, H: pH,
        });

        // Alpha → RGBA canvas
        const aCanvas = createCanvas(pW, pH);
        const aCtx    = aCanvas.getContext('2d', { willReadFrequently: true });
        const aImg    = new ImageData(pW, pH);
        for (let j = 0; j < alphaSmall.length; j++) {
          const v = Math.min(255, Math.round(Math.pow(alphaSmall[j], 0.6) * 255));
          aImg.data[j * 4]     = 255;
          aImg.data[j * 4 + 1] = 255;
          aImg.data[j * 4 + 2] = 255;
          aImg.data[j * 4 + 3] = v;
        }
        aCtx.putImageData(aImg, 0, 0);

        // GPU cutout at full resolution
        const cutoutCanvas = createCanvas(W, H);
        const cCtx = cutoutCanvas.getContext('2d', { willReadFrequently: true });
        cCtx.drawImage(imgEl.current, 0, 0);
        cCtx.globalCompositeOperation = 'destination-in';
        cCtx.drawImage(aCanvas, 0, 0, W, H);
        cCtx.globalCompositeOperation = 'source-over';

        cutouts.push({ index: i, cutoutCanvas });
        setProgress(Math.round(((i + 1) / numLayers) * 46));
        addLog(`   ✅ Layer ${i + 1} recortada`, 'success');
        await new Promise(r => setTimeout(r, 20));
      }

      setProgress(46);

      // ── Step 2: SDXL Inpainting via Replicate ──────────────────────────────
      const results = [];

      if (useGenerativeAI) {
        if (projectId) await updateProjectStatus(projectId, 'inpainting');
        addLog('🎨 SDXL Inpainting preenchendo fundo…', 'ai');

        const { canvas: inpaintCanvas } = resizeToReplicate(imgEl.current);
        const inpaintW = inpaintCanvas.width, inpaintH = inpaintCanvas.height;
        const inpaintB64 = canvasToPng(inpaintCanvas);

        for (let i = 0; i < cutouts.length; i++) {
          const co = cutouts[i];
          if (!co) { results.push(null); continue; }

          // Layer 1 (frontmost) — no inpainting needed
          if (i === 0) {
            addLog(`   ℹ️ Layer 1 é frontal — sem inpainting`, 'info');
            results.push({
              index: i, label: `Layer ${i + 1}`, color: LAYER_COLORS[i].hex, elements: [],
              cutoutDataURL: co.cutoutCanvas.toDataURL('image/png'),
              inpaintedDataURL: null, hasInpaint: false,
            });
            setProgress(46 + Math.round(((i + 1) / cutouts.length) * 46));
            continue;
          }

          addLog(`🖌️ SDXL: Layer ${i + 1}…`, 'ai');

          const mResized = buildInpaintMask(maskRefs, i, inpaintW, inpaintH);
          const md = mResized.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, inpaintW, inpaintH);
          let hasArea = false;
          for (let p = 0; p < md.data.length; p += 4) if (md.data[p] > 128) { hasArea = true; break; }

          let inpaintedDataURL = null;
          if (hasArea) {
            const prompt = [
              'Seamlessly inpaint the masked region to match the surrounding scene.',
              'Preserve exactly: photographic or artistic style, lighting direction and quality, dominant color palette,',
              'atmospheric depth, texture detail, grain, and level of sharpness.',
              'The filled area must be indistinguishable from the rest of the image.',
              'No new objects, no color casts, no visible seams, no style changes.',
              'Ultra-realistic, high detail, perfect edge continuity.',
            ].join(' ');
            try {
              const { imageBase64: resultB64 } = await withRetry(
                () => callReplicate({ type: 'inpaint', imageBase64: inpaintB64, maskBase64: canvasToPng(mResized), prompt }),
                retryOpts(addLog, `Inpainting L${i + 1}`),
              );
              inpaintedDataURL = `data:image/png;base64,${resultB64}`;
              addLog(`   ✅ Layer ${i + 1} preenchida`, 'success');
            } catch (e) { addLog(`   ⚠️ ${e.message}`, 'warn'); }
          } else { addLog(`   Layer ${i + 1}: sem área para preencher`, 'info'); }

          results.push({
            index: i, label: `Layer ${i + 1}`, color: LAYER_COLORS[i].hex, elements: [],
            cutoutDataURL: co.cutoutCanvas.toDataURL('image/png'),
            inpaintedDataURL, hasInpaint: !!inpaintedDataURL,
          });
          setProgress(46 + Math.round(((i + 1) / cutouts.length) * 46));
          await new Promise(r => setTimeout(r, 80));
        }
      } else {
        addLog('⚡ IA generativa desligada — exportando recortes…', 'info');
        for (let i = 0; i < cutouts.length; i++) {
          const co = cutouts[i];
          if (!co) { results.push(null); continue; }
          results.push({
            index: i, label: `Layer ${i + 1}`, color: LAYER_COLORS[i].hex, elements: [],
            cutoutDataURL: co.cutoutCanvas.toDataURL('image/png'),
            inpaintedDataURL: null, hasInpaint: false,
          });
          setProgress(46 + Math.round(((i + 1) / cutouts.length) * 54));
        }
      }

      const finalResults = results.filter(Boolean);
      if (projectId && finalResults.length) {
        try {
          await saveProjectLayers(projectId, finalResults);
          await updateProjectStatus(projectId, 'done', { layers_count: finalResults.length });
        } catch { addLog('⚠️ Falha ao salvar no banco', 'warn'); }
      }
      setProgress(100);
      addLog(`🎉 ${finalResults.length} layer${finalResults.length !== 1 ? 's' : ''} pronta${finalResults.length !== 1 ? 's' : ''}!`, 'success');
      setPhase('done');
      return finalResults;

    } catch (e) {
      addLog(`❌ Erro: ${e.message}`, 'error');
      if (projectId) await updateProjectStatus(projectId, 'error', { error_message: e.message }).catch(() => {});
      setPhase('error');
      return [];
    }
  }, [addLog]);

  return { run, logs, progress, phase, setPhase };
}
