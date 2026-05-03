import { useState, useCallback } from 'react';
import { resizeToFit, resizeToStability, buildSketchOverlay, buildInpaintMask, canvasToJpeg, canvasToPng, createCanvas } from '../lib/canvas';
import { createProject, updateProjectStatus, saveProjectLayers, logProcessingEvent } from '../lib/supabase';
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

async function callClaude(body) {
  const res = await fetch('/api/claude', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Claude error');
  return data.content.map(b => b.text || '').join('').replace(/```json|```/g, '').trim();
}

async function callStability(body) {
  const res = await fetch('/api/stability', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Stability error');
  return `data:image/png;base64,${data.imageBase64}`;
}

function retryOpts(addLog, label) {
  return {
    attempts: 3,
    baseDelayMs: 1000,
    onRetry: (err, attempt) => addLog(`⟳ ${label} — tentativa ${attempt + 1}/3: ${err.message}`, 'warn'),
  };
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

    try {
      // ── Step 1: Scene analysis ──────────────────────────────────────────────
      addLog('🔍 Claude analisando cena…', 'ai');
      const { canvas: thumb, w: tw, h: th } = resizeToFit(imgEl.current, 800);
      const origB64   = canvasToJpeg(thumb, 0.85);
      const sketchB64 = canvasToJpeg(buildSketchOverlay(imgEl.current, maskRefs, numLayers, LAYER_COLORS, tw, th), 0.85);

      let analysis = { imageStyle: 'photograph', scene: '', mood: '', layers: [] };
      try {
        const raw = await withRetry(() => callClaude({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 2000,
          messages: [{ role: 'user', content: [
            { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: origB64 } },
            { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: sketchB64 } },
            { type: 'text', text: `Image 1: the original scene. Image 2: rough brush strokes indicating ${numLayers} parallax layers (${LAYER_COLORS.slice(0, numLayers).map((c, i) => `L${i+1}=${c.hex}`).join(', ')}). L1=frontmost, L${numLayers}=background.

Analyze the scene thoroughly. Pay attention to:
- Overall mood, lighting, color palette, and atmosphere
- What each painted layer represents
- What background content would be VISIBLE behind each painted object (what is already partially visible at its edges)

Respond ONLY with valid JSON:
{"imageStyle":"photograph|painting|illustration","scene":"detailed scene description","mood":"precise lighting and atmosphere description — colors, temperature, brightness","palette":"dominant colors as hex codes e.g. #1a2b3c, #4d5e6f","layers":[{"index":0,"elements":["specific object names"],"depth":"foreground|midground|background","behindDescription":"what is ALREADY VISIBLE at the edges of this object in the original image — describe exact colors, textures, patterns seen there"}]}` },
          ]}],
        }), retryOpts(addLog, 'Análise de cena'));
        analysis = JSON.parse(raw);
        addLog(`✅ Cena: ${analysis.scene?.slice(0, 60)}`, 'success');
        if (projectId) await logProcessingEvent(projectId, 'scene_analyzed', { style: analysis.imageStyle });
      } catch { addLog('⚠️ Análise parcial — continuando', 'warn'); }
      setProgress(12);

      // ── Step 2: Per-layer cutout (edge-aware selection) ─────────────────────
      const cutouts = [];
      for (let i = 0; i < numLayers; i++) {
        const li = analysis.layers?.find(l => l.index === i) || {};
        addLog(`✂️ Recortando Layer ${i + 1}${li.elements?.length ? ` — ${li.elements.join(', ')}` : ''} …`, 'ai');

        // Downscale mask and image for worker
        const smallMask = createCanvas(pW, pH);
        smallMask.getContext('2d').drawImage(maskRefs[i], 0, 0, pW, pH);
        const sd = smallMask.getContext('2d').getImageData(0, 0, pW, pH);

        // Quick painted-check on small data
        let painted = false;
        for (let p = 3; p < sd.data.length; p += 4) if (sd.data[p] > 10) { painted = true; break; }
        if (!painted) { addLog(`⚪ Layer ${i + 1} sem pintura, pulando`, 'warn'); cutouts.push(null); continue; }

        // Downscale original image for gradient-based edge detection
        const smallImg = createCanvas(pW, pH);
        smallImg.getContext('2d').drawImage(imgEl.current, 0, 0, pW, pH);
        const imgSd = smallImg.getContext('2d').getImageData(0, 0, pW, pH);

        // Edge-aware selection in worker (main thread stays free)
        const strokeBuffer = sd.data.buffer.slice(0);
        const imgBuffer    = imgSd.data.buffer.slice(0);
        const alphaSmall = await runAlphaWorker({ strokeBuffer, imgBuffer, W: pW, H: pH });

        // Convert Float32 alpha → RGBA canvas at thumbnail scale
        const aCanvas = createCanvas(pW, pH);
        const aCtx    = aCanvas.getContext('2d');
        const aImg    = new ImageData(pW, pH);
        for (let j = 0; j < alphaSmall.length; j++) {
          // Boost: alpha values are 0–1 from propagation; compress to 0–255
          const v = Math.min(255, Math.round(Math.pow(alphaSmall[j], 0.6) * 255));
          aImg.data[j * 4]     = 255;
          aImg.data[j * 4 + 1] = 255;
          aImg.data[j * 4 + 2] = 255;
          aImg.data[j * 4 + 3] = v;
        }
        aCtx.putImageData(aImg, 0, 0);

        // GPU cutout: full-res image × upscaled alpha mask
        const cutoutCanvas = createCanvas(W, H);
        const cCtx = cutoutCanvas.getContext('2d');
        cCtx.drawImage(imgEl.current, 0, 0);
        cCtx.globalCompositeOperation = 'destination-in';
        cCtx.drawImage(aCanvas, 0, 0, W, H); // GPU upscale
        cCtx.globalCompositeOperation = 'source-over';

        cutouts.push({ index: i, cutoutCanvas, layerInfo: li });
        setProgress(12 + Math.round(((i + 1) / numLayers) * 34));
        addLog(`   ✅ Layer ${i + 1} recortada`, 'success');
        await new Promise(r => setTimeout(r, 20));
      }

      setProgress(46);

      // ── Step 3: Inpainting (Stability AI) ──────────────────────────────────
      const results = [];

      if (useGenerativeAI) {
        if (projectId) await updateProjectStatus(projectId, 'inpainting');
        addLog('🎨 Stability AI preenchendo fundo…', 'ai');

        const { canvas: stableCanvas } = resizeToStability(imgEl.current);
        const stableW = stableCanvas.width, stableH = stableCanvas.height;
        const stableB64 = canvasToPng(stableCanvas);

        for (let i = 0; i < cutouts.length; i++) {
          const co = cutouts[i];
          if (!co) { results.push(null); continue; }

          // Layer 1 (index 0) is the frontmost — no inpainting needed
          if (i === 0) {
            addLog(`   ℹ️ Layer 1 é frontal — sem inpainting`, 'info');
            results.push({
              index: i, label: `Layer ${i + 1}`, color: LAYER_COLORS[i].hex,
              elements: co.layerInfo?.elements || [],
              cutoutDataURL: co.cutoutCanvas.toDataURL('image/png'),
              inpaintedDataURL: null, hasInpaint: false,
            });
            setProgress(46 + Math.round(((i + 1) / cutouts.length) * 46));
            continue;
          }

          addLog(`🖌️ Stability AI: Layer ${i + 1}…`, 'ai');

          // Build inpaint mask at stability resolution
          const mResized = buildInpaintMask(maskRefs, i, stableW, stableH);
          const md = mResized.getContext('2d').getImageData(0, 0, stableW, stableH);
          let hasArea = false;
          for (let p = 0; p < md.data.length; p += 4) if (md.data[p] > 128) { hasArea = true; break; }

          let inpaintedDataURL = null;
          if (hasArea) {
            // Build a highly specific prompt from the scene analysis.
            // The goal: Stability AI should CONTINUE the existing background,
            // not invent new content. We describe exactly what's visible at the edges.
            const li = co.layerInfo;
            const behindDesc = li.behindDescription || '';
            const sceneCtx   = [analysis.scene, analysis.mood].filter(Boolean).join(', ');
            const palette     = analysis.palette ? `Color palette: ${analysis.palette}.` : '';

            const prompt = behindDesc
              ? `Seamlessly extend and fill: ${behindDesc}. Scene context: ${sceneCtx}. ${palette} Match existing colors, lighting, and texture exactly. No new objects. Photorealistic continuation only.`
              : `Seamless background fill matching this scene: ${sceneCtx}. ${palette} Continue existing background with same lighting, colors, and atmosphere. No new objects or subjects.`;

            const negativePrompt = [
              'interior', 'indoors', 'room', 'furniture', 'chair', 'table', 'wall decoration',
              'ceiling', 'floor', 'window', 'curtain', 'lamp',
              'different style', 'painting', 'cartoon', 'anime',
              'new objects', 'people', 'hallucination', 'artifacts',
              'blurry', 'low quality', 'watermark', 'text',
            ].join(', ');

            try {
              inpaintedDataURL = await withRetry(() => callStability({
                imageBase64: stableB64,
                maskBase64: canvasToPng(mResized),
                prompt,
                negativePrompt,
                strength: 0.60,
                steps: 30,
              }), retryOpts(addLog, `Inpainting L${i + 1}`));
              addLog(`   ✅ Layer ${i + 1} preenchida`, 'success');
            } catch (e) { addLog(`   ⚠️ ${e.message}`, 'warn'); }
          } else { addLog(`   Layer ${i + 1}: sem área para preencher`, 'info'); }

          results.push({
            index: i, label: `Layer ${i + 1}`, color: LAYER_COLORS[i].hex,
            elements: co.layerInfo?.elements || [],
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
            index: i, label: `Layer ${i + 1}`, color: LAYER_COLORS[i].hex,
            elements: co.layerInfo?.elements || [],
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
