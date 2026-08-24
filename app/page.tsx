'use client';

import { useEffect, useRef, useState } from 'react';

type Result = { disease: string; crop: string; confidence: number; severity: string; summary: string };
const outcomes: Result[] = [
  { disease: 'Early blight', crop: 'Tomato', confidence: 96.4, severity: 'Moderate', summary: 'Fungal symptoms are consistent with concentric brown lesions on older leaves.' },
  { disease: 'Bacterial spot', crop: 'Pepper', confidence: 93.8, severity: 'Early stage', summary: 'Small water-soaked leaf spots suggest an early bacterial infection.' },
  { disease: 'Apple scab', crop: 'Apple', confidence: 95.1, severity: 'Moderate', summary: 'Olive-green leaf lesions are consistent with apple scab symptoms.' },
  { disease: 'Healthy leaf', crop: 'Plant', confidence: 97.2, severity: 'Healthy', summary: 'No strong visual indicators of a supported disease were detected.' },
];

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<Result | null>(null);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  function chooseImage(file?: File) {
    if (!file || !file.type.startsWith('image/')) return;
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(file)); setFileName(file.name); setResult(null);
  }
  function analyze() {
    if (!preview || analyzing) return;
    setAnalyzing(true);
    setTimeout(() => {
      const seed = [...fileName].reduce((sum, char) => sum + char.charCodeAt(0), 0);
      setResult(outcomes[seed % outcomes.length]); setAnalyzing(false);
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
    }, 1350);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="LeafLens home"><span className="brand-mark">L</span><span>LeafLens</span></a>
        <nav aria-label="Primary navigation"><a className="active" href="#scan">Scan</a><a href="#monitor">Monitor</a><a href="#history">History</a></nav>
        <button className="profile" aria-label="Open profile">AD</button>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <span className="eyebrow"><i /> AI crop health assistant</span>
          <h1>Know what your<br />plant needs.</h1>
          <p>Scan a leaf for an instant disease check, confidence score, and practical care advice.</p>
          <div className="hero-stats" aria-label="System highlights"><span><b>38</b> disease classes</span><span><b>&lt;3s</b> target response</span><span><b>24/7</b> field monitoring</span></div>
        </div>
        <section className="scan-card" id="scan" aria-labelledby="scan-title">
          <div className="scan-heading"><div><span className="step">STEP 01</span><h2 id="scan-title">Scan a leaf</h2></div><span className="model-status"><i /> CNN ready</span></div>
          <input ref={inputRef} type="file" accept="image/png,image/jpeg,image/webp" capture="environment" onChange={(event) => chooseImage(event.target.files?.[0])} hidden />
          <button className={`dropzone ${preview ? 'has-preview' : ''}`} onClick={() => inputRef.current?.click()} onDrop={(event) => { event.preventDefault(); chooseImage(event.dataTransfer.files?.[0]); }} onDragOver={(event) => event.preventDefault()} aria-label={preview ? 'Replace selected leaf image' : 'Choose a leaf image'}>
            {preview ? <>{/* eslint-disable-next-line @next/next/no-img-element */}<img src={preview} alt="Leaf selected for analysis" /><span className="replace-label">Replace image</span></> : <><span className="camera-icon">⌗</span><strong>Drop a clear leaf image here</strong><small>or tap to use your camera / browse files</small></>}
          </button>
          <button className="primary-action" disabled={!preview || analyzing} onClick={analyze}><span>{analyzing ? 'Running CNN analysis…' : 'Analyze leaf'}</span><span>{analyzing ? '◌' : '→'}</span></button>
          <p className="privacy-note">Prototype inference • Connect your trained model API for production results.</p>
        </section>
      </section>

      <section className="field-strip" id="monitor">
        <div><span className="field-icon">◒</span><small>Field station</small><strong>Greenhouse 01</strong></div><div><small>Temperature</small><strong>26.4°C</strong><em>Optimal</em></div><div><small>Humidity</small><strong>68%</strong><em>Optimal</em></div><div><small>Soil moisture</small><strong>42%</strong><em className="watch">Watch</em></div><div className="sensor-live"><i /> Sensors live</div>
      </section>

      {result && <section className="diagnosis" ref={resultsRef} aria-live="polite">
        <div className="section-label"><span>02</span> Diagnosis</div>
        <div className="result-grid">
          <article className="result-main">
            <div className="result-top"><span className={`severity ${result.severity === 'Healthy' ? 'healthy' : ''}`}>{result.severity}</span><span>MobileNetV2 • 224×224</span></div>
            <p className="crop-label">{result.crop}</p><h2>{result.disease}</h2><p className="result-summary">{result.summary}</p>
            <div className="confidence-row"><div><small>Model confidence</small><strong>{result.confidence}%</strong></div><div className="confidence-track"><span style={{ width: `${result.confidence}%` }} /></div></div>
            <div className="alternatives"><span>Other possibilities</span><b>Leaf mold <em>2.1%</em></b><b>Septoria spot <em>1.5%</em></b></div>
          </article>
          <article className="advisory-card">
            <span className="step">RECOMMENDED ACTIONS</span><h3>Care advisory</h3>
            <ol><li><b>Remove affected leaves</b><span>Isolate and dispose of visibly infected foliage.</span></li><li><b>Keep leaves dry</b><span>Water at soil level and improve airflow around plants.</span></li><li><b>Apply treatment</b><span>Use an approved copper-based fungicide as directed.</span></li></ol>
            <div className="alert-box"><b>Field alert</b><span>Humidity is favorable for fungal spread. Recheck nearby plants within 48 hours.</span></div>
            <button onClick={() => window.print()}>Save report <span>↓</span></button>
          </article>
        </div>
      </section>}

      <section className="workflow">
        <div className="workflow-heading"><div><span className="eyebrow">Designed for field use</span><h2>From leaf to action<br />in three clear steps.</h2></div><p>Image analysis and sensor context come together in one farmer-friendly view.</p></div>
        <div className="workflow-grid"><article><span>01</span><h3>Capture</h3><p>Photograph one leaf in even light against a simple background.</p></article><article><span>02</span><h3>Detect</h3><p>The optimized CNN compares visual features across supported classes.</p></article><article><span>03</span><h3>Act</h3><p>Review confidence, treatment guidance, and field-condition alerts.</p></article></div>
      </section>

      <section className="history-section" id="history">
        <div className="history-heading"><div><span className="section-label"><span>03</span> Scan log</span><h2>Recent field checks</h2></div><button onClick={() => document.getElementById('scan')?.scrollIntoView({ behavior:'smooth' })}>New scan +</button></div>
        <div className="history-table" role="table" aria-label="Recent scans">
          <div className="history-row header" role="row"><span>Plant / disease</span><span>Field</span><span>Confidence</span><span>Status</span><span>Date</span></div>
          <div className="history-row" role="row"><span><i className="leaf-dot tomato">T</i><b>Tomato</b><small>Early blight</small></span><span>Greenhouse 01</span><span>96.4%</span><span><em>Review</em></span><span>Today, 10:42</span></div>
          <div className="history-row" role="row"><span><i className="leaf-dot apple">A</i><b>Apple</b><small>Healthy</small></span><span>North orchard</span><span>97.8%</span><span><em className="clear">Clear</em></span><span>Yesterday</span></div>
          <div className="history-row" role="row"><span><i className="leaf-dot pepper">P</i><b>Pepper</b><small>Bacterial spot</small></span><span>Plot B</span><span>92.1%</span><span><em>Review</em></span><span>22 Aug</span></div>
        </div>
      </section>
      <footer><a className="brand" href="#top"><span className="brand-mark">L</span><span>LeafLens</span></a><p>AI-based plant disease detection & advisory system</p><span>Prototype • 2026</span></footer>
    </main>
  );
}
