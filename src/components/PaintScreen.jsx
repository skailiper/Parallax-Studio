import { useRef, useEffect } from 'react';
import { LAYER_COLORS } from '../hooks/usePipeline';
import styles from './PaintScreen.module.css';

export function PaintScreen({
  imgFile, imgEl, numLayers, activeLayer, setActiveLayer,
  tool, setTool, brushSize, setBrushSize,
  layerVis, setLayerVis, showOrig, setShowOrig,
  zoom, setZoom, canvasRef,
  onDown, onMove, onUp, clearLayer, onProcess,
  useGenerativeAI, setUseGenerativeAI,
  selecting,
}) {
  const col       = LAYER_COLORS[activeLayer];
  const sz        = brushSize;
  const scrollRef = useRef(null);
  const panRef    = useRef(null);

  // ── Cursor SVG ────────────────────────────────────────────────────────────
  let cursor;
  if (tool === 'selector') {
    cursor = selecting ? 'wait' : 'crosshair';
  } else if (tool === 'brush') {
    const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='${sz*2}' height='${sz*2}'><circle cx='${sz}' cy='${sz}' r='${sz-1}' fill='${encodeURIComponent(col.hex)}' fill-opacity='.28' stroke='white' stroke-width='1.5'/></svg>`;
    cursor = `url("data:image/svg+xml,${svg}") ${sz} ${sz}, crosshair`;
  } else {
    const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='${sz*3}' height='${sz*3}'><circle cx='${sz*1.5}' cy='${sz*1.5}' r='${sz*1.5-1}' fill='none' stroke='%23ff5f57' stroke-width='1.5' stroke-dasharray='4,3'/></svg>`;
    cursor = `url("data:image/svg+xml,${svg}") ${sz*1.5} ${sz*1.5}, crosshair`;
  }

  // ── Scroll-wheel → pan (non-passive so preventDefault works) ─────────────
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onWheel = (e) => {
      e.preventDefault();
      el.scrollLeft += e.deltaX;
      el.scrollTop  += e.deltaY;
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  // ── Middle-mouse-button drag → pan ────────────────────────────────────────
  function onScrollPointerDown(e) {
    if (e.button !== 1) return;
    e.preventDefault();
    panRef.current = {
      x: e.clientX, y: e.clientY,
      sl: scrollRef.current.scrollLeft,
      st: scrollRef.current.scrollTop,
    };
    scrollRef.current.setPointerCapture(e.pointerId);
  }

  function onScrollPointerMove(e) {
    if (!panRef.current) return;
    const dx = e.clientX - panRef.current.x;
    const dy = e.clientY - panRef.current.y;
    scrollRef.current.scrollLeft = panRef.current.sl - dx;
    scrollRef.current.scrollTop  = panRef.current.st - dy;
  }

  function onScrollPointerUp(e) {
    if (e.button === 1) panRef.current = null;
  }

  return (
    <div className={styles.root}>
      <aside className={styles.sidebar}>
        <div className={styles.sideScroll}>
          <div className={styles.logoRow}>
            <svg width="18" height="18" viewBox="0 0 28 28" fill="none">
              <rect x="2"  y="2"  width="11" height="11" rx="2" fill="#5eead4" opacity=".9"/>
              <rect x="15" y="2"  width="11" height="11" rx="2" fill="#5eead4" opacity=".55"/>
              <rect x="2"  y="15" width="11" height="11" rx="2" fill="#5eead4" opacity=".35"/>
              <rect x="15" y="15" width="11" height="11" rx="2" fill="#5eead4" opacity=".15"/>
            </svg>
            <span className={styles.logoText}>PARALLAX</span>
          </div>

          <div className={styles.divider} />

          <div className={styles.section}>
            <div className={styles.sLabel}>LAYERS</div>
            {Array.from({ length: numLayers }).map((_, i) => (
              <div
                key={i}
                className={styles.layerRow}
                style={activeLayer === i ? { background: LAYER_COLORS[i].hex + '16', border: `1px solid ${LAYER_COLORS[i].hex}44` } : undefined}
                onClick={() => setActiveLayer(i)}
              >
                <div className={styles.layerDot} style={{ background: LAYER_COLORS[i].hex }} />
                <span className={styles.layerName} style={{ color: activeLayer === i ? '#fff' : '#666' }}>Layer {i+1}</span>
                <button className={styles.iconBtn} onClick={e => { e.stopPropagation(); setLayerVis(v => { const n=[...v]; n[i]=!n[i]; return n; }); }}>
                  {layerVis[i] ? '◉' : '○'}
                </button>
                <button className={styles.iconBtn} style={{ opacity: 0.45 }} onClick={e => { e.stopPropagation(); clearLayer(i); }}>✕</button>
              </div>
            ))}
          </div>

          <div className={styles.divider} />

          <div className={styles.section}>
            <div className={styles.sLabel}>FERRAMENTA</div>
            <div className={styles.toolRow}>
              {[['selector','◎','Seletor'],['brush','✦','Pincel'],['eraser','◻','Borracha']].map(([id,ico,lbl]) => (
                <button key={id} className={`${styles.toolBtn} ${tool===id ? styles.toolBtnOn : ''}`} onClick={() => setTool(id)}>
                  <span className={styles.toolIco}>{ico}</span>
                  <span className={styles.toolLbl}>{lbl}</span>
                </button>
              ))}
            </div>
            {tool === 'selector'
              ? <div className={styles.shortcutHint}><span>{selecting ? '⏳ Selecionando…' : '🖱 Esq: selecionar · Dir: apagar'}</span></div>
              : <div className={styles.shortcutHint}><span>🖱 Esq: pintar · Dir: apagar · Scroll: mover</span></div>
            }
          </div>

          <div className={styles.section}>
            <div className={styles.sLabel}>TAMANHO — {brushSize}px</div>
            <input type="range" min={4} max={120} value={brushSize} onChange={e => setBrushSize(+e.target.value)} className={styles.range} />
            <div className={styles.sizeBtns}>
              {[10,28,56,100].map(s => (
                <button key={s} className={`${styles.sizeBtn} ${brushSize===s ? styles.sizeBtnOn : ''}`} onClick={() => setBrushSize(s)}>{s}</button>
              ))}
            </div>
          </div>

          <div className={styles.divider} />

          {/* ── Generative AI toggle ── */}
          <div className={styles.toggleRow}>
            <div className={styles.toggleLabelGroup}>
              <span className={`${styles.toggleLabel} ${useGenerativeAI ? styles.toggleLabelOn : ''}`}>
                IA Generativa
              </span>
              <span
                className={styles.toggleInfo}
                title="Usa Stability AI para preencher automaticamente o fundo por trás de cada camada recortada, recriando a continuidade do que já existe na imagem. Desligado: gera apenas recortes transparentes, sem geração de conteúdo — mais rápido."
              >ⓘ</span>
            </div>
            <button
              className={`${styles.toggle} ${useGenerativeAI ? styles.toggleOn : ''}`}
              onClick={() => setUseGenerativeAI(v => !v)}
              aria-label={useGenerativeAI ? 'Desligar IA generativa' : 'Ligar IA generativa'}
            >
              <div className={styles.toggleKnob} />
            </button>
          </div>

          <div className={styles.divider} />

          <button className={styles.origBtn}
            onMouseDown={() => setShowOrig(true)} onMouseUp={() => setShowOrig(false)}
            onTouchStart={() => setShowOrig(true)} onTouchEnd={() => setShowOrig(false)}>
            ◎ Segurar: ver original
          </button>

          <div className={styles.hint}>
            <div className={styles.hintTitle}>💡 Dica</div>
            <div className={styles.hintText}>Pinceladas grossas estão ótimas. A IA entende o contexto e faz o recorte fino.</div>
          </div>
        </div>

        <div className={styles.sideBtm}>
          <button className={styles.processBtn} onClick={onProcess}>⚡ Processar com IA</button>
        </div>
      </aside>

      <div className={styles.canvasArea}>
        <div className={styles.topBar}>
          <span className={styles.fileMeta}>{imgFile?.name} · {imgEl.current?.naturalWidth}×{imgEl.current?.naturalHeight}px</span>
          <div className={styles.zoomRow}>
            <button className={styles.zBtn} onClick={() => setZoom(z => Math.max(0.1, +(z-.1).toFixed(2)))}>−</button>
            <span className={styles.zVal}>{Math.round(zoom*100)}%</span>
            <button className={styles.zBtn} onClick={() => setZoom(z => Math.min(6, +(z+.1).toFixed(2)))}>+</button>
            <button className={styles.zBtn} onClick={() => setZoom(1)}>⊡</button>
          </div>
          <div className={styles.activeBadge} style={{ borderColor: col.hex + '66' }}>
            <div className={styles.activeDot} style={{ background: col.hex }} />
            <span style={{ color: col.hex }}>{col.name}</span>
          </div>
        </div>

        {/* scroll container — handles wheel-pan and middle-mouse-pan */}
        <div
          ref={scrollRef}
          className={styles.canvasScroll}
          onPointerDown={onScrollPointerDown}
          onPointerMove={onScrollPointerMove}
          onPointerUp={onScrollPointerUp}
        >
          <div className={styles.canvasWrap} style={{ transform: `scale(${zoom})` }}>
            <canvas
              ref={canvasRef}
              style={{ display: 'block', cursor, touchAction: 'none', userSelect: 'none' }}
              onMouseDown={onDown}
              onMouseMove={onMove}
              onMouseUp={onUp}
              onMouseLeave={onUp}
              onContextMenu={e => e.preventDefault()}
              onTouchStart={onDown}
              onTouchMove={onMove}
              onTouchEnd={onUp}
            />
          </div>
        </div>

        <div className={styles.legend}>
          {Array.from({ length: numLayers }).map((_, i) => (
            <div key={i} className={styles.legendItem} style={{ opacity: activeLayer===i ? 1 : 0.3 }} onClick={() => setActiveLayer(i)}>
              <div className={styles.legendDot} style={{ background: LAYER_COLORS[i].hex }} />L{i+1}
            </div>
          ))}
          <span className={styles.legendSpacer} />
          <span className={styles.legendHint}>Esq: pintar · Dir: apagar · Scroll: mover</span>
        </div>
      </div>
    </div>
  );
}
