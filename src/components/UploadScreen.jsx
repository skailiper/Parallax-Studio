import { useRef, useState } from 'react';
import { LAYER_COLORS } from '../hooks/usePipeline';
import styles from './UploadScreen.module.css';

export function UploadScreen({ numLayers, setNumLayers, onFile }) {
  const fileRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function handleFile(file) {
    if (!file?.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => onFile(file, img);
    img.src = url;
  }

  return (
    <div className={styles.page}>
      <div className={styles.card}>
        <div className={styles.logoRow}>
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
            <rect x="2"  y="2"  width="11" height="11" rx="2" fill="#5eead4" opacity=".9"/>
            <rect x="15" y="2"  width="11" height="11" rx="2" fill="#5eead4" opacity=".55"/>
            <rect x="2"  y="15" width="11" height="11" rx="2" fill="#5eead4" opacity=".35"/>
            <rect x="15" y="15" width="11" height="11" rx="2" fill="#5eead4" opacity=".15"/>
          </svg>
          <div>
            <div className={styles.logoName}>PARALLAX STUDIO</div>
            <div className={styles.logoSub}>Pinte. A IA recorta e preenche.</div>
          </div>
        </div>

        <div className={styles.headline}>
          <h1 className={styles.h1}>
            Separação de layers<br />
            <span className={styles.accent}>com IA generativa</span>
          </h1>
          <p className={styles.sub}>
            Carregue qualquer imagem, pinte cada plano com uma cor e deixe a IA fazer o recorte preciso e o preenchimento de fundo.
          </p>
        </div>

        <div className={styles.layerBlock}>
          <div className={styles.layerBlockLabel}>Quantas layers no seu parallax?</div>
          <div className={styles.numRow}>
            {[2,3,4,5,6,7,8].map(n => (
              <button key={n} className={`${styles.numBtn} ${numLayers === n ? styles.numBtnOn : ''}`} onClick={() => setNumLayers(n)}>{n}</button>
            ))}
          </div>
          <div className={styles.swatchRow}>
            {Array.from({ length: numLayers }).map((_, i) => (
              <div key={i} className={styles.swatchItem}>
                <div className={styles.swatch} style={{ background: LAYER_COLORS[i].hex }} />
                <span className={styles.swatchLabel}>L{i+1}</span>
              </div>
            ))}
          </div>
        </div>

        <div
          className={`${styles.drop} ${dragging ? styles.dropOn : ''}`}
          onClick={() => fileRef.current.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
        >
          <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={e => handleFile(e.target.files[0])} />
          <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
            <circle cx="20" cy="20" r="19" stroke="#5eead420" strokeWidth="1.5"/>
            <path d="M20 13v14M13 20h14" stroke="#5eead460" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <div className={styles.dropTitle}>Arraste ou clique para carregar</div>
          <div className={styles.dropSub}>Foto · Pintura · Ilustração · Qualquer estilo</div>
        </div>

        <div className={styles.steps}>
          {[
            ['01', 'Pinte os planos',    'Dê pinceladas grossas em cada profundidade da imagem'],
            ['02', 'Claude recorta',     'IA detecta bordas reais e separa com precisão de pixel'],
            ['03', 'Replicate preenche', 'Áreas vazias são geradas no mesmo estilo da imagem original'],
          ].map(([n, title, desc]) => (
            <div key={n} className={styles.step}>
              <span className={styles.stepN}>{n}</span>
              <div>
                <div className={styles.stepTitle}>{title}</div>
                <div className={styles.stepDesc}>{desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
