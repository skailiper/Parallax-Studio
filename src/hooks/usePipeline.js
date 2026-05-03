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

// Runs heavy alpha computation + edge refinement in a Web Worker.
function runAlphaWorker({ strokeBuffer, imgBuffer, W, H, brushSize, expansionFactor, bboxes }) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('../workers/alpha.worker.js', import.meta.url));
    worker.onmessage = ({ data }) => { worker.terminate(); resolve(new Float32Array(data.alphaBuffer)); };
    worker.onerror  = (e)        => { worker.terminate(); reject(new Error(e.message || 'Worker error')); };
    const transfers = [strokeBuffer];
    if (imgBuffer) transfers.push(imgBuffer);
    worker.postMessage({ strokeBuffer, imgBuffer, W, H, brushSize, expansionFactor, bboxes }, transfers);
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

    // Thumbnail scale for alpha computation (avoids freeze on large images)
    const MAX_PROC = 640;
    const procScale = Math.min(1, MAX_PROC / Math.max(W, H));
    const pW = Math.round(W * procScale), pH = Math.round(H * procScale);

    try {
      addLog('🔍 Claude analisando cena e esboços…', 'ai');
      const { canvas: thumb, w: tw, h: th } = resizeToFit(imgEl.current, 800);
      const origB64   = canvasToJpeg(thumb, 0.85);
      const sketchB64 = canvasToJpeg(buildSketchOverlay(imgEl.current, maskRefs, numLayers, LAYER_COLORS, tw, th), 0.85);

      let analysis = null;
      try {
        const raw = await withRetry(() => callClaude({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 2000,
          messages: [{ role: 'user', content: [
            { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: origB64 } },
            { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: sketchB64 } },
            { type: 'text', text: `Image 1: original photo/illustration. Image 2: rough brush strokes marking ${numLayers} parallax layers (colors: ${LAYER_COLORS.slice(0, numLayers).map((c, i) => `L${i+1}=${c.hex}`).join(', ')}). L1=closest foreground, L${numLayers}=background.

Analyze the scene and identify each painted layer. Respond ONLY with valid JSON:
{"imageStyle":"photograph|painting|illustration","scene":"one-sentence description","mood":"lighting and atmosphere","layers":[{"index":0,"elements":["specific object names"],"depth":"foreground|midground|background"}]}` },
          ]}],
        }), retryOpts(addLog, 'Análise de cena'));
        analysis = JSON.parse(raw);
        addLog(`✅ Cena: ${analysis.scene?.slice(0, 60)}`, 'success');
        addLog(`🎨 Estilo: ${analysis.imageStyle}`, 'info');
        if (projectId) await logProcessingEvent(projectId, 'scene_analyzed', { style: analysis.imageStyle });
      } catch { addLog('⚠️ Análise parcial', 'warn'); analysis = { imageStyle: 'photograph', scene: '', mood: '', layers: [] }; }
      setProgress(12);

      const cutouts = [];
      for (let i = 0; i < numLayers; i++) {
        const li = analysis.layers?.find(l => l.index === i) || {};
        addLog(`✂️ Recortando Layer ${i + 1}${li.elements?.length ? ` — ${li.elements.join(', ')}` : ''} …`, 'ai');

        // Scale mask and image to thumbnail for fast processing
        const smallMask = createCanvas(pW, pH);
        smallMask.getContext('2d').drawImage(maskRefs[i], 0, 0, pW, pH);
        const sd = smallMask.getContext('2d').getImageData(0, 0, pW, pH);

        let painted = false;
        for (let p = 3; p < sd.data.length; p += 4) if (sd.data[p] > 10) { painted = true; break; }
        if (!painted) { addLog(`⚪ Layer ${i + 1} sem pintura, pulando`, 'warn'); cutouts.push(null); continue; }

        // Small original image for color-based edge refinement in worker
        const smallImg = createCanvas(pW, pH);
        smallImg.getContext('2d').drawImage(imgEl.current, 0, 0, pW, pH);
        const imgSd = smallImg.getContext('2d').getImageData(0, 0, pW, pH);

        // Thumbnail mask for Claude segmentation prompt
        const mt = createCanvas(tw, th); mt.getContext('2d').drawImage(maskRefs[i], 0, 0, tw, th);
        const mb64 = canvasToJpeg(mt, 0.75);

        let seg = null;
        try {
          // Ask Claude to: 1) give tight bboxes, 2) describe background at mask edges for inpainting
          const sr = await withRetry(() => callClaude({
            model: 'claude-sonnet-4-20250514',
            max_tokens: 1500,
            messages: [{ role: 'user', content: [
              { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: origB64 } },
              { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: mb64 } },
              { type: 'text', text: `Image 1: original (${W}×${H}px, style: ${analysis.imageStyle}).
Image 2: user's rough brush strokes marking Layer ${i + 1} (${(li.elements || []).join(', ') || 'unknown objects'}).

Task A — Tight segmentation: Identify the exact objects painted and return ONE tight bounding box per distinct object, wrapping the true object boundary visible in Image 1, NOT the brush stroke boundary. Coordinates as percentages of image dimensions.

Task B — Background analysis for inpainting: Look at the pixels in Image 1 that are DIRECTLY ADJACENT to (just outside) the painted region. What specific colors, textures, patterns, or content do you see there? Describe only what is ALREADY VISIBLE in the image at those border areas — this will be used to seamlessly extend that content inward. Do NOT describe what might be behind the object; describe the actual visible border content.

Respond ONLY with valid JSON:
{"objects":[{"label":"name","x1pct":0,"y1pct":0,"x2pct":100,"y2pct":100,"softness":10,"priority":1.0}],"expansionFactor":2.2,"inpaintPrompt":"[specific: extend the exact colors/textures/patterns visible at the edges — e.g. 'continue the muted blue-gray overcast sky visible at the top edge, matching existing grain and lighting']","negativePrompt":"no new objects, no hallucination, no mountains, no vegetation unless already present, match existing image exactly"}` },
            ]}],
          }), retryOpts(addLog, `Segmentação L${i + 1}`));
          seg = JSON.parse(sr);
          addLog(`   📐 ${seg.objects?.map(o => o.label).join(', ') || 'mapeado'}`, 'info');
        } catch { addLog(`   ⚠️ Segmentação base para L${i + 1}`, 'warn'); }

        // Heavy alpha expansion + color-based edge refinement in worker (thumbnail scale)
        const strokeBuffer = sd.data.buffer.slice(0);
        const imgBuffer    = imgSd.data.buffer.slice(0);
        const alphaSmall = await runAlphaWorker({
          strokeBuffer, imgBuffer,
          W: pW, H: pH,
          brushSize: Math.max(2, Math.round(36 * procScale)),
          expansionFactor: seg?.expansionFactor || 2.2,
          bboxes: seg?.objects || [],
        });

        // Convert Float32 alpha → RGBA canvas at thumbnail scale
        const aCanvas = createCanvas(pW, pH);
        const aCtx    = aCanvas.getContext('2d');
        const aImg    = new ImageData(pW, pH);
        for (let j = 0; j < alphaSmall.length; j++) {
          const v = Math.min(255, Math.round(alphaSmall[j] * 400));
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
        cCtx.drawImage(aCanvas, 0, 0, W, H);
        cCtx.globalCompositeOperation = 'source-over';

        cutouts.push({ index: i, cutoutCanvas, layerInfo: li, seg });
        setProgress(12 + Math.round(((i + 1) / numLayers) * 33));
        addLog(`   ✅ Layer ${i + 1} recortada`, 'success');
        await new Promise(r => setTimeout(r, 20));
      }

      setProgress(46);

      // ── Inpainting (Stability AI) ─────────────────────────────────────────
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
          addLog(`🖌️ Stability AI: Layer ${i + 1}…`, 'ai');

          // Build inpaint mask at stability resolution (avoids full-res processing)
          const mResized = buildInpaintMask(maskRefs, i, stableW, stableH);
          const md = mResized.getContext('2d').getImageData(0, 0, stableW, stableH);
          let hasArea = false;
          for (let p = 0; p < md.data.length; p += 4) if (md.data[p] > 128) { hasArea = true; break; }

          let inpaintedDataURL = null;
          if (hasArea) {
            // Use the per-layer inpaint prompt from segmentation (describes actual border content)
            // Fall back to scene description only as a last resort
            const prompt = co.seg?.inpaintPrompt
              || `seamless continuation of the existing ${analysis.scene} background, matching exact colors and lighting already present, photorealistic`;
            const negativePrompt = co.seg?.negativePrompt
              || 'new objects, hallucination, different style, artifacts, blur, watermark, text';
            try {
              inpaintedDataURL = await withRetry(() => callStability({
                imageBase64: stableB64,
                maskBase64: canvasToPng(mResized),
                prompt,
                negativePrompt,
                steps: 35,
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
        // No generative AI — export transparent cutouts only
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
