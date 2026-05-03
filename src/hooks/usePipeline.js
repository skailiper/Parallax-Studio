import { useState, useCallback } from 'react';
import { resizeToFit, resizeToReplicate, buildInpaintMask, canvasToJpeg, canvasToPng, createCanvas } from '../lib/canvas';
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
  const res = await fetch('/api/replicate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
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

// Extract evenly-spaced foreground points from a painted mask for SAM2.
function extractSAM2Points(sd, pW, pH, tw, th, maxPoints = 6) {
  const painted = [];
  for (let y = 3; y < pH - 3; y += 4) {
    for (let x = 3; x < pW - 3; x += 4) {
      if (sd.data[(y * pW + x) * 4 + 3] > 64) {
        painted.push([Math.round(x * tw / pW), Math.round(y * th / pH)]);
      }
    }
  }
  if (painted.length === 0) return null;
  const stride = Math.max(1, Math.floor(painted.length / maxPoints));
  const pts = painted.filter((_, i) => i % stride === 0).slice(0, maxPoints);
  return { points: pts, pointLabels: pts.map(() => 1) };
}

// Decode a base64 PNG mask (from SAM2) into a Float32Array alpha at w×h.
function sam2MaskToAlpha(maskBase64, w, h) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = createCanvas(w, h);
      c.getContext('2d', { willReadFrequently: true }).drawImage(img, 0, 0, w, h);
      const id = c.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h);
      const alpha = new Float32Array(w * h);
      for (let i = 0; i < w * h; i++) alpha[i] = id.data[i * 4] / 255;
      resolve(alpha);
    };
    img.onerror = reject;
    img.src = `data:image/png;base64,${maskBase64}`;
  });
}

// Runs edge-aware selection in a Web Worker so the main thread stays free.
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

    // Processing resolution: cap at 640px to keep worker fast
    const MAX_PROC  = 640;
    const procScale = Math.min(1, MAX_PROC / Math.max(W, H));
    const pW = Math.round(W * procScale), pH = Math.round(H * procScale);

    // Thumbnail used for SAM2 point prompts
    const { canvas: thumb, w: tw, h: th } = resizeToFit(imgEl.current, 800);
    const origB64 = canvasToJpeg(thumb, 0.85);

    try {
      // ── Step 1: Per-layer cutout (SAM2 + edge-aware fallback) ──────────────
      addLog('✂️ Iniciando segmentação das layers…', 'ai');
      const cutouts = [];
      for (let i = 0; i < numLayers; i++) {
        addLog(`✂️ Recortando Layer ${i + 1}…`, 'ai');

        // Downscale mask for worker
        const smallMask = createCanvas(pW, pH);
        smallMask.getContext('2d', { willReadFrequently: true }).drawImage(maskRefs[i], 0, 0, pW, pH);
        const sd = smallMask.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, pW, pH);

        // Quick painted-check
        let painted = false;
        for (let p = 3; p < sd.data.length; p += 4) if (sd.data[p] > 10) { painted = true; break; }
        if (!painted) { addLog(`⚪ Layer ${i + 1} sem pintura, pulando`, 'warn'); cutouts.push(null); continue; }

        // Try SAM2 for precise segmentation
        let alphaSmall = null;
        let alphaW = pW, alphaH = pH;

        if (useGenerativeAI) {
          try {
            const sam2Pts = extractSAM2Points(sd, pW, pH, tw, th);
            if (sam2Pts) {
              addLog(`   🎯 SAM2 segmentando Layer ${i + 1}…`, 'ai');
              const { maskBase64 } = await withRetry(
                () => callReplicate({ type: 'segment', imageBase64: origB64, ...sam2Pts }),
                retryOpts(addLog, `SAM2 L${i + 1}`),
              );
              alphaSmall = await sam2MaskToAlpha(maskBase64, tw, th);
              alphaW = tw; alphaH = th;
              addLog(`   ✅ SAM2 concluído`, 'success');
            }
          } catch (e) {
            addLog(`   ⚠️ SAM2 falhou, usando método local: ${e.message}`, 'warn');
          }
        }

        if (!alphaSmall) {
          // Fallback: local edge-aware selection in Web Worker
          const smallImg = createCanvas(pW, pH);
          smallImg.getContext('2d', { willReadFrequently: true }).drawImage(imgEl.current, 0, 0, pW, pH);
          const imgSd = smallImg.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, pW, pH);
          const strokeBuffer = sd.data.buffer.slice(0);
          const imgBuffer    = imgSd.data.buffer.slice(0);
          alphaSmall = await runAlphaWorker({ strokeBuffer, imgBuffer, W: pW, H: pH });
        }

        // Convert Float32 alpha → RGBA canvas (GPU-upscaled to W×H later)
        const aCanvas = createCanvas(alphaW, alphaH);
        const aCtx    = aCanvas.getContext('2d', { willReadFrequently: true });
        const aImg    = new ImageData(alphaW, alphaH);
        for (let j = 0; j < alphaSmall.length; j++) {
          const v = Math.min(255, Math.round(Math.pow(alphaSmall[j], 0.6) * 255));
          aImg.data[j * 4]     = 255;
          aImg.data[j * 4 + 1] = 255;
          aImg.data[j * 4 + 2] = 255;
          aImg.data[j * 4 + 3] = v;
        }
        aCtx.putImageData(aImg, 0, 0);

        // GPU cutout: full-res image × upscaled alpha mask
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

      // ── Step 2: Inpainting (SDXL via Replicate) ────────────────────────────
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

          addLog(`🖌️ SDXL Inpainting: Layer ${i + 1}…`, 'ai');

          // Build inpaint mask at Replicate resolution
          const mResized = buildInpaintMask(maskRefs, i, inpaintW, inpaintH);
          const md = mResized.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, inpaintW, inpaintH);
          let hasArea = false;
          for (let p = 0; p < md.data.length; p += 4) if (md.data[p] > 128) { hasArea = true; break; }

          let inpaintedDataURL = null;
          if (hasArea) {
            const prompt = 'Seamless photorealistic background continuation. Match existing colors, lighting, and textures exactly. No new objects or subjects.';
            const negativePrompt = 'blurry, artifacts, low quality, watermark, text, new objects, people, different style';

            try {
              const { imageBase64: resultB64 } = await withRetry(() => callReplicate({
                type: 'inpaint',
                imageBase64: inpaintB64,
                maskBase64: canvasToPng(mResized),
                prompt,
                negativePrompt,
                strength: 0.60,
                steps: 30,
              }), retryOpts(addLog, `Inpainting L${i + 1}`));
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
        // No generative AI — transparent cutouts only
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
