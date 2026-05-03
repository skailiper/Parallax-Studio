import { useState } from 'react';
import { trackUsage } from '../lib/supabase';
import { getSessionId } from '../lib/session';
import styles from './ExportScreen.module.css';

const TIPS = [
  { label: 'Layer frontal',  speed: '×1.8 – ×2.0', color: '#f87171' },
  { label: 'Layers do meio', speed: '×0.8 – ×1.2', color: '#fbbf24' },
  { label: 'Fundo',          speed: '×0.3 – ×0.5', color: '#5eead4' },
];

function Checker() {
  return <div className={styles.checker} />;
}

function ThumbCard({ label, src, dlLabel, onDl, green }) {
  return (
    <div className={styles.card}>
      <div className={`${styles.cardLabel} ${green ? styles.cardLabelGen : ''}`}>{label}</div>
      <div className={styles.thumbWrap}>
        <Checker />
        <img src={src} alt={label} className={styles.thumb} />
      </div>
      <div className={styles.thumbMeta}>
        <span className={styles.thumbLabel}>{dlLabel}</span>
        <button className={`${styles.dlBtn} ${green ? styles.dlBtnGreen : ''}`} onClick={onDl}>⬇ PNG</button>
      </div>
    </div>
  );
}

export function ExportScreen({ layers, onEdit, onNew }) {
  const [downloading, setDownloading] = useState(false);

  function dl(layer, type = 'cutout') {
    const a = document.createElement('a');
    a.href     = type === 'cutout' ? layer.cutoutDataURL : layer.inpaintedDataURL;
    a.download = `parallax-L${layer.index + 1}-${type}.png`;
    a.click();
    trackUsage({ sessionId: getSessionId(), action: 'download_layer', meta: { layerIndex: layer.index, type } });
  }

  async function dlAll() {
    setDownloading(true);
    layers.forEach((l, i) => {
      setTimeout(() => dl(l, 'cutout'), i * 280);
      if (l.hasInpaint) setTimeout(() => dl(l, 'inpainted'), i * 280 + 140);
    });
    await trackUsage({ sessionId: getSessionId(), action: 'download_all', meta: { count: layers.length } });
    setTimeout(() => setDownloading(false), layers.length * 300 + 500);
  }

  return (
    <div className={styles.page}>
      <div className={styles.topBar}>
        <div className={styles.logoRow}>
          <svg width="20" height="20" viewBox="0 0 28 28" fill="none">
            <rect x="2"  y="2"  width="11" height="11" rx="2" fill="#5eead4" opacity=".9"/>
            <rect x="15" y="2"  width="11" height="11" rx="2" fill="#5eead4" opacity=".55"/>
            <rect x="2"  y="15" width="11" height="11" rx="2" fill="#5eead4" opacity=".35"/>
            <rect x="15" y="15" width="11" height="11" rx="2" fill="#5eead4" opacity=".15"/>
          </svg>
          <div>
            <div className={styles.logoName}>PARALLAX STUDIO</div>
            <div className={styles.logoSub}>{layers.length} layer{layers.length !== 1 ? 's' : ''} pronta{layers.length !== 1 ? 's' : ''}</div>
          </div>
        </div>
        <div className={styles.btnRow}>
          <button className={styles.ghostBtn} onClick={onEdit}>← Editar</button>
          <button className={styles.ghostBtn} onClick={onNew}>Nova imagem</button>
          <button className={styles.dlAllBtn} onClick={dlAll} disabled={downloading}>
            {downloading ? 'Baixando…' : '⬇ Baixar todas'}
          </button>
        </div>
      </div>

      <div className={styles.inner}>
        {layers.map(l => (
          <div key={l.index} className={styles.layerGroup}>
            {/* Layer header */}
            <div className={styles.layerHeader}>
              <div className={styles.layerDot} style={{ background: l.color }} />
              <span className={styles.layerTitle} style={{ color: l.color }}>Layer {l.index + 1}</span>
              {l.elements?.length > 0 && (
                <span className={styles.layerElements}>{l.elements.slice(0, 4).join(' · ')}</span>
              )}
              <div className={styles.layerSep} />
            </div>

            {/* Cards: transparent cutout + generated (side by side) */}
            <div className={styles.layerCards}>
              <ThumbCard
                label="Recorte transparente"
                src={l.cutoutDataURL}
                dlLabel="Sem fundo"
                onDl={() => dl(l, 'cutout')}
                green={false}
              />
              {l.hasInpaint && (
                <ThumbCard
                  label="Com fundo gerado por IA"
                  src={l.inpaintedDataURL}
                  dlLabel="Com fundo"
                  onDl={() => dl(l, 'inpainted')}
                  green={true}
                />
              )}
            </div>
          </div>
        ))}

        <div className={styles.tipsBlock}>
          <div className={styles.tipsTitle}>Velocidade sugerida no parallax</div>
          <div className={styles.tipsRow}>
            {TIPS.map(t => (
              <div key={t.label} className={styles.tipItem}>
                <span className={styles.tipSpeed} style={{ color: t.color }}>{t.speed}</span>
                <span className={styles.tipLabel}>{t.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
