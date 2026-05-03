import { useEffect, useRef } from 'react';
import styles from './ProcessingScreen.module.css';

const STAGES = [
  { label: 'Segmentação das layers',        min: 0,  max: 46  },
  { label: 'Flux Fill Pro: preenchimento',  min: 46, max: 94  },
  { label: 'Finalização',                  min: 94, max: 100 },
];

const LOG_CLASS = {
  success: styles.logSuccess,
  ai:      styles.logAi,
  warn:    styles.logWarn,
  error:   styles.logError,
  info:    styles.logInfo,
};

export function ProcessingScreen({ logs, progress }) {
  const logsEndRef = useRef(null);
  useEffect(() => { logsEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [logs]);
  const cur = STAGES.findIndex(s => progress >= s.min && progress < s.max);

  return (
    <div className={styles.page}>
      <div className={styles.card}>
        <div className={styles.logoAnim}>
          <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
            <circle cx="24" cy="24" r="22" stroke="#5eead415" strokeWidth="2"/>
            <circle cx="24" cy="24" r="22" stroke="#5eead4" strokeWidth="2"
              strokeDasharray="138.2" strokeDashoffset={138.2 * (1 - progress / 100)}
              strokeLinecap="round" transform="rotate(-90 24 24)"
              style={{ transition: 'stroke-dashoffset .5s ease' }}/>
          </svg>
          <div className={styles.logoInner}>
            <svg width="22" height="22" viewBox="0 0 28 28" fill="none">
              <rect x="2"  y="2"  width="11" height="11" rx="2" fill="#5eead4" opacity=".9"/>
              <rect x="15" y="2"  width="11" height="11" rx="2" fill="#5eead4" opacity=".55"/>
              <rect x="2"  y="15" width="11" height="11" rx="2" fill="#5eead4" opacity=".35"/>
              <rect x="15" y="15" width="11" height="11" rx="2" fill="#5eead4" opacity=".15"/>
            </svg>
          </div>
        </div>

        <div className={styles.title}>Processando</div>

        <div className={styles.barWrap}>
          <div className={styles.barFill} style={{ width: `${progress}%` }} />
        </div>
        <div className={styles.progNum}>{progress}%</div>

        <div className={styles.stages}>
          {STAGES.map((st, i) => {
            const done = progress >= st.max, active = i === cur;
            return (
              <div key={i} className={styles.stage} style={{ opacity: done || active ? 1 : 0.22 }}>
                <div className={`${styles.stageDot} ${done ? styles.stageDotDone : active ? styles.stageDotActive : ''}`}>
                  {done
                    ? <svg width="10" height="10" viewBox="0 0 10 10"><path d="M2 5l2.5 2.5L8 3" stroke="#020e0c" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
                    : active ? <div className={styles.stagePulse} /> : null}
                </div>
                <span className={styles.stageLabel} style={{ color: done ? '#5eead4' : active ? '#dde1ec' : '#333' }}>{st.label}</span>
              </div>
            );
          })}
        </div>

        <div className={styles.logBox}>
          {logs.map(l => (
            <div key={l.id} className={LOG_CLASS[l.type] || styles.logInfo}>{l.msg}</div>
          ))}
          <div ref={logsEndRef} />
        </div>
      </div>
    </div>
  );
}
