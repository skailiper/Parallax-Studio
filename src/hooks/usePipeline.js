import { useState, useCallback } from 'react';
import { resizeToFit, resizeToStability, buildSketchOverlay, buildInpaintMask, buildCutout, expandStrokeToAlphaMap, applyBBoxesToAlphaMap, canvasToJpeg, canvasToPng, createCanvas } from '../lib/canvas';
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

export function usePipeline() {
  const [logs,     setLogs]     = useState([]);
  const [progress, setProgress] = useState(0);
  const [phase,    setPhase]    = useState('idle');
  const addLog = useCallback((msg, type = 'info') => setLogs(p => [...p, { msg, type, id: Date.now() + Math.random() }]), []);

  const run = useCallback(async ({ imgEl, maskRefs, numLayers, imgFile }) => {
    setPhase('running'); setLogs([]); setProgress(0);
    const sessionId = getSessionId();
    const W = imgEl.current.naturalWidth, H = imgEl.current.naturalHeight;
    let projectId = null;
    try {
      const project = await createProject({ sessionId, numLayers, imageFilename: imgFile.name, imageSizeBytes: imgFile.size });
      projectId = project.id;
    } catch { addLog('⚠️ Supabase offline — continuando sem salvar', 'warn'); }

    try {
      addLog('🔍 Claude analisando cena e esboços…', 'ai');
      const { canvas: thumb, w: tw, h: th } = resizeToFit(imgEl.current, 800);
      const origB64   = canvasToJpeg(thumb, 0.85);
      const sketchB64 = canvasToJpeg(buildSketchOverlay(imgEl.current, maskRefs, numLayers, LAYER_COLORS, tw, th), 0.85);

      let analysis = null;
      try {
        const raw = await withRetry(() => callClaude({ model: 'claude-sonnet-4-20250514', max_tokens: 3000, messages: [{ role: 'user', content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: origB64 } },
          { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: sketchB64 } },
          { type: 'text', text: `Image 1: original. Image 2: colored brush strokes showing ${numLayers} parallax layers. Colors: ${LAYER_COLORS.slice(0, numLayers).map((c, i) => `L${i+1}=${c.hex}`).join(',')}. L0=foreground, L${numLayers-1}=background. Respond ONLY valid JSON: {"imageStyle":"photograph|painting|illustration","scene":"description","mood":"lighting","layers":[{"index":0,"elements":["objects"],"depth":"foreground","inpaintPrompt":"what appears BEHIND after removal, same style/lighting/atmosphere, very detailed","negativePrompt":"avoid","segBoxes":[{"label":"obj","x1pct":0,"y1pct":0,"x2pct":100,"y2pct":100,"softness":10,"priority":1.0}]}]}` },
        ]}] }), retryOpts(addLog, 'Análise de cena'));
        analysis = JSON.parse(raw);
        addLog(`✅ Cena: ${analysis.scene?.slice(0, 60)}…`, 'success');
        addLog(`🎨 Estilo: ${analysis.imageStyle}`, 'info');
        if (projectId) await logProcessingEvent(projectId, 'scene_analyzed', { style: analysis.imageStyle });
      } catch { addLog('⚠️ Análise parcial', 'warn'); analysis = { imageStyle: 'photograph', scene: '', mood: '', layers: [] }; }
      setProgress(12);

      const cutouts = [];
      for (let i = 0; i < numLayers; i++) {
        const li = analysis.layers?.find(l => l.index === i) || {};
        addLog(`✂️ Recortando Layer ${i+1}${li.elements?.length ? ` — ${li.elements.join(', ')}` : ''} …`, 'ai');
        const sd = maskRefs[i].getContext('2d').getImageData(0, 0, W, H);
        let painted = false;
        for (let p = 3; p < sd.data.length; p += 4) if (sd.data[p] > 10) { painted = true; break; }
        if (!painted) { addLog(`⚪ Layer ${i+1} sem pintura, pulando`, 'warn'); cutouts.push(null); continue; }

        const mt = createCanvas(tw, th); mt.getContext('2d').drawImage(maskRefs[i], 0, 0, tw, th);
        const mb64 = canvasToJpeg(mt, 0.75);
        let seg = null;
        try {
          const sr = await withRetry(() => callClaude({ model: 'claude-sonnet-4-20250514', max_tokens: 1200, messages: [{ role: 'user', content: [
            { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: origB64 } },
            { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: mb64 } },
            { type: 'text', text: `Img1: original ${W}x${H}. Img2: brush for L${i+1}. Elements:${(li.elements || []).join(',')}. Style:${analysis.imageStyle}. Respond ONLY JSON: {"objects":[{"label":"name","x1pct":0,"y1pct":0,"x2pct":100,"y2pct":100,"softness":12,"priority":1.0}],"featherPx":14,"expansionFactor":2.4}` },
          ]}] }), retryOpts(addLog, `Segmentação L${i+1}`));
          seg = JSON.parse(sr);
          addLog(`   📐 ${seg.objects?.map(o => o.label).join(', ') || 'mapeado'}`, 'info');
        } catch { addLog(`   ⚠️ Segmentação base para L${i+1}`, 'warn'); }

        let alphaMap = expandStrokeToAlphaMap(sd, W, H, 36, seg?.expansionFactor || 2.3);
        if (seg?.objects?.length) alphaMap = applyBBoxesToAlphaMap(alphaMap, seg.objects, W, H);
        cutouts.push({ index: i, cutoutCanvas: buildCutout(imgEl.current, W, H, alphaMap), layerInfo: li });
        setProgress(12 + Math.round(((i + 1) / numLayers) * 33));
        addLog(`   ✅ Layer ${i+1} recortada`, 'success');
        await new Promise(r => setTimeout(r, 20));
      }

      setProgress(46);
      if (projectId) await updateProjectStatus(projectId, 'inpainting');
      addLog('🎨 Stability AI preenchendo áreas removidas…', 'ai');

      const { canvas: stableCanvas } = resizeToStability(imgEl.current);
      const stableW = stableCanvas.width, stableH = stableCanvas.height;
      const stableB64 = canvasToPng(stableCanvas);
      const results = [];

      for (let i = 0; i < cutouts.length; i++) {
        const co = cutouts[i];
        if (!co) { results.push(null); continue; }
        addLog(`🖌️ Stability AI: Layer ${i+1}…`, 'ai');
        const mFull = buildInpaintMask(maskRefs, i, W, H);
        const mResized = createCanvas(stableW, stableH);
        mResized.getContext('2d').drawImage(mFull, 0, 0, stableW, stableH);
        const md = mResized.getContext('2d').getImageData(0, 0, stableW, stableH);
        let hasArea = false;
        for (let p = 0; p < md.data.length; p += 4) if (md.data[p] > 128) { hasArea = true; break; }
        let inpaintedDataURL = null;
        if (hasArea) {
          try {
            inpaintedDataURL = await withRetry(() => callStability({
              imageBase64: stableB64,
              maskBase64: canvasToPng(mResized),
              prompt: co.layerInfo.inpaintPrompt || `${analysis.scene}, ${analysis.mood}, seamless, highly detailed`,
              negativePrompt: co.layerInfo.negativePrompt || 'blurry, low quality, artifacts, watermark',
              steps: 40,
            }), retryOpts(addLog, `Inpainting L${i+1}`));
            addLog(`   ✅ Layer ${i+1} preenchida`, 'success');
          } catch (e) { addLog(`   ⚠️ ${e.message}`, 'warn'); }
        } else { addLog(`   Layer ${i+1}: sem área para preencher`, 'info'); }

        results.push({
          index: i, label: `Layer ${i+1}`, color: LAYER_COLORS[i].hex,
          elements: co.layerInfo?.elements || [], cutoutDataURL: co.cutoutCanvas.toDataURL('image/png'),
          inpaintedDataURL, hasInpaint: !!inpaintedDataURL,
        });
        setProgress(46 + Math.round(((i + 1) / numLayers) * 46));
        await new Promise(r => setTimeout(r, 80));
      }

      const finalResults = results.filter(Boolean);
      if (projectId && finalResults.length) {
        try {
          await saveProjectLayers(projectId, finalResults);
          await updateProjectStatus(projectId, 'done', { layers_count: finalResults.length });
        } catch { addLog('⚠️ Falha ao salvar no banco', 'warn'); }
      }
      setProgress(100);
      addLog(`🎉 ${finalResults.length} layers prontas!`, 'success');
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
