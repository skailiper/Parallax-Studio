import { useState } from 'react';
import { UploadScreen }     from './components/UploadScreen';
import { PaintScreen }      from './components/PaintScreen';
import { ProcessingScreen } from './components/ProcessingScreen';
import { ExportScreen }     from './components/ExportScreen';
import { usePainter }       from './hooks/usePainter';
import { usePipeline }      from './hooks/usePipeline';
import { trackUsage }       from './lib/supabase';
import { getSessionId }     from './lib/session';
import styles from './App.module.css';

export default function App() {
  const [screen,       setScreen]       = useState('upload');
  const [imgFile,      setImgFile]      = useState(null);
  const [numLayers,    setNumLayers]    = useState(3);
  const [activeLayer,  setActiveLayer]  = useState(0);
  const [tool,         setTool]         = useState('selector');
  const [brushSize,    setBrushSize]    = useState(36);
  const [layerVis,     setLayerVis]     = useState(Array(8).fill(true));
  const [showOrig,     setShowOrig]     = useState(false);
  const [zoom,         setZoom]         = useState(1);
  const [exportLayers,     setExportLayers]     = useState([]);
  const [useGenerativeAI, setUseGenerativeAI] = useState(true);

  const { run, logs, progress, phase, setPhase } = usePipeline();
  const painter = usePainter({ numLayers, activeLayer, tool, brushSize, layerVis, showOrig });

  function handleFile(file, img) {
    setImgFile(file);
    painter.initMasks(img);
    setScreen('paint');
    trackUsage({ sessionId: getSessionId(), action: 'image_loaded', meta: { filename: file.name, size: file.size } });
  }

  async function handleProcess() {
    setScreen('processing');
    setPhase('running');
    const results = await run({ imgEl: painter.imgEl, maskRefs: painter.maskRefs.current, numLayers, imgFile, useGenerativeAI });
    if (results.length > 0) { setExportLayers(results); setScreen('export'); }
    else setScreen('paint');
  }

  function handleEdit() { setScreen('paint'); setPhase('idle'); }

  function handleNew() {
    setScreen('upload'); setImgFile(null); setActiveLayer(0);
    setExportLayers([]); setPhase('idle'); setZoom(1); painter.clearAll();
  }

  return (
    <div className={styles.root}>
      {screen === 'upload'     && <UploadScreen numLayers={numLayers} setNumLayers={setNumLayers} onFile={handleFile} />}
      {screen === 'paint'      && <PaintScreen imgFile={imgFile} imgEl={painter.imgEl} numLayers={numLayers} activeLayer={activeLayer} setActiveLayer={setActiveLayer} tool={tool} setTool={setTool} brushSize={brushSize} setBrushSize={setBrushSize} layerVis={layerVis} setLayerVis={setLayerVis} showOrig={showOrig} setShowOrig={setShowOrig} zoom={zoom} setZoom={setZoom} canvasRef={painter.canvasRef} onDown={painter.onDown} onMove={painter.onMove} onUp={painter.onUp} onEnter={painter.onEnter} clearLayer={painter.clearLayer} onProcess={handleProcess} useGenerativeAI={useGenerativeAI} setUseGenerativeAI={setUseGenerativeAI} selecting={painter.selecting} />}
      {screen === 'processing' && <ProcessingScreen logs={logs} progress={progress} />}
      {screen === 'export'     && <ExportScreen layers={exportLayers} onEdit={handleEdit} onNew={handleNew} />}
    </div>
  );
}
