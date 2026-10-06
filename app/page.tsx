'use client';

import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

type Advisory = { crop: string; condition: string; summary: string; actions: string[]; urgency: string };
type Labels = { classes: string[]; display_names: Record<string, string>; image_size: number[] };
type Result = { disease: string; crop: string; confidence: number; severity: string; summary: string; actions: string[]; isHealthy: boolean; alternatives: { label: string; confidence: number }[] };
type LeafCheck = { valid: boolean; message?: string };

function validateLeafImage(image: HTMLImageElement): LeafCheck {
  if (image.naturalWidth < 96 || image.naturalHeight < 96) {
    return { valid: false, message: 'This image is too small. Upload a clear, close-up photo of one leaf.' };
  }

  const size = 96;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) return { valid: false, message: 'The image could not be checked. Please try another photo.' };
  context.drawImage(image, 0, 0, size, size);
  const pixels = context.getImageData(0, 0, size, size).data;
  const vegetation = new Uint8Array(size * size);
  let vegetationPixels = 0;
  let centerVegetationPixels = 0;
  let centerPixels = 0;
  let darkOrBrightPixels = 0;

  for (let index = 0; index < size * size; index += 1) {
    const offset = index * 4;
    const red = pixels[offset];
    const green = pixels[offset + 1];
    const blue = pixels[offset + 2];
    const brightness = (red + green + blue) / 3;
    if (brightness < 18 || brightness > 247) darkOrBrightPixels += 1;

    // Leaves in this tomato model range from muted green to yellow-green.
    // Excess-green is more stable than a fixed hue under indoor/field lighting.
    const excessGreen = 2 * green - red - blue;
    const isVegetation = green > 28 && green >= blue * 1.03 && excessGreen > 12 && green - red > -8;
    if (isVegetation) {
      vegetation[index] = 1;
      vegetationPixels += 1;
    }

    const x = index % size;
    const y = Math.floor(index / size);
    if (x >= 24 && x < 72 && y >= 24 && y < 72) {
      centerPixels += 1;
      if (isVegetation) centerVegetationPixels += 1;
    }
  }

  if (darkOrBrightPixels / (size * size) > 0.72) {
    return { valid: false, message: 'The photo is too dark or overexposed. Retake it in even light.' };
  }

  // Find the largest connected vegetation region. This prevents a few green
  // pixels in an unrelated image from being mistaken for a leaf.
  const visited = new Uint8Array(size * size);
  let largestRegion = 0;
  let largestWidth = 0;
  let largestHeight = 0;
  const queue = new Int32Array(size * size);
  for (let start = 0; start < vegetation.length; start += 1) {
    if (!vegetation[start] || visited[start]) continue;
    let head = 0;
    let tail = 0;
    let minX = size;
    let maxX = 0;
    let minY = size;
    let maxY = 0;
    visited[start] = 1;
    queue[tail++] = start;
    while (head < tail) {
      const current = queue[head++];
      const x = current % size;
      const y = Math.floor(current / size);
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      const neighbors = [current - 1, current + 1, current - size, current + size];
      for (const neighbor of neighbors) {
        if (neighbor < 0 || neighbor >= vegetation.length || visited[neighbor] || !vegetation[neighbor]) continue;
        const neighborX = neighbor % size;
        if (Math.abs(neighborX - x) > 1) continue;
        visited[neighbor] = 1;
        queue[tail++] = neighbor;
      }
    }
    if (tail > largestRegion) {
      largestRegion = tail;
      largestWidth = maxX - minX + 1;
      largestHeight = maxY - minY + 1;
    }
  }

  const vegetationRatio = vegetationPixels / (size * size);
  const centerRatio = centerVegetationPixels / centerPixels;
  const connectedRatio = vegetationPixels ? largestRegion / vegetationPixels : 0;
  const hasLeafRegion = vegetationRatio >= 0.1 && centerRatio >= 0.08 && connectedRatio >= 0.45
    && largestWidth >= 18 && largestHeight >= 18;

  if (!hasLeafRegion) {
    return { valid: false, message: 'No clear leaf was detected. Upload a close-up photo containing one leaf.' };
  }
  return { valid: true };
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const modelRef = useRef<tf.LayersModel | null>(null);
  const resultsRef = useRef<HTMLElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [modelState, setModelState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [labels, setLabels] = useState<Labels | null>(null);
  const [advisories, setAdvisories] = useState<Record<string, Advisory>>({});

  useEffect(() => {
    let active = true;
    Promise.all([
      tf.loadLayersModel('/model/model.json'),
      fetch('/model/labels.json').then((response) => response.json() as Promise<Labels>),
      fetch('/model/advisories.json').then((response) => response.json() as Promise<Record<string, Advisory>>),
    ])
      .then(([model, modelLabels, modelAdvisories]) => { if (active) { modelRef.current = model; setLabels(modelLabels); setAdvisories(modelAdvisories); setModelState('ready'); } })
      .catch(() => { if (active) setModelState('error'); });
    return () => { active = false; modelRef.current?.dispose(); };
  }, []);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  function chooseImage(file?: File) {
    if (!file || !file.type.startsWith('image/')) return;
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(file)); setResult(null); setValidationError(null);
  }
  async function analyze() {
    if (!preview || analyzing || !modelRef.current || !imageRef.current || !labels) return;
    setAnalyzing(true);
    try {
      const leafCheck = validateLeafImage(imageRef.current);
      if (!leafCheck.valid) {
        setResult(null);
        setValidationError(leafCheck.message ?? 'This does not appear to be a leaf photo.');
        return;
      }
      const probabilities = tf.tidy(() => {
        const input = tf.browser.fromPixels(imageRef.current!).resizeBilinear([labels.image_size[0], labels.image_size[1]]).toFloat().expandDims(0);
        const prediction = modelRef.current!.predict(input) as tf.Tensor;
        return Array.from(prediction.dataSync());
      });
      const ranked = probabilities.map((confidence, index) => ({ index, confidence })).sort((a, b) => b.confidence - a.confidence);
      const best = ranked[0];
      const predictionMargin = best.confidence - ranked[1].confidence;
      if (best.confidence < 0.25 || predictionMargin < 0.02) {
        setResult(null);
        setValidationError('This photo is outside the model’s reliable range. Use one clear leaf on a plain background.');
        return;
      }
      const rawLabel = labels.classes[best.index];
      const advisory = advisories[rawLabel];
      const isHealthy = advisory?.condition === 'Healthy';
      setValidationError(null);
      setResult({
        disease: advisory?.condition ?? labels.display_names[rawLabel] ?? rawLabel,
        crop: advisory?.crop ?? 'Plant',
        severity: advisory?.urgency ?? 'Review', summary: advisory?.summary ?? 'Review this result with a local crop adviser.',
        actions: advisory?.actions ?? ['Capture another clear image.', 'Inspect nearby plants for similar symptoms.', 'Confirm the diagnosis with a local agriculture extension service.'],
        isHealthy, confidence: best.confidence * 100,
        alternatives: ranked.slice(1, 4).map(({ index, confidence }) => ({ label: labels.display_names[labels.classes[index]] ?? labels.classes[index], confidence: confidence * 100 })),
      });
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
    } finally { setAnalyzing(false); }
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
          <div className="hero-stats" aria-label="System highlights"><span><b>38</b> crop conditions</span><span><b>54K</b> training images</span><span><b>Local</b> browser inference</span></div>
        </div>
        <section className="scan-card" id="scan" aria-labelledby="scan-title">
          <div className="scan-heading"><div><span className="step">STEP 01</span><h2 id="scan-title">Scan a leaf</h2></div><span className={`model-status ${modelState === 'error' ? 'model-error' : ''}`}><i /> {modelState === 'loading' ? 'CNN loading' : modelState === 'ready' ? 'CNN ready' : 'CNN unavailable'}</span></div>
          <input ref={inputRef} type="file" accept="image/png,image/jpeg,image/webp" capture="environment" onChange={(event) => chooseImage(event.target.files?.[0])} hidden />
          <button className={`dropzone ${preview ? 'has-preview' : ''}`} onClick={() => inputRef.current?.click()} onDrop={(event) => { event.preventDefault(); chooseImage(event.dataTransfer.files?.[0]); }} onDragOver={(event) => event.preventDefault()} aria-label={preview ? 'Replace selected leaf image' : 'Choose a leaf image'}>
            {preview ? <>{/* eslint-disable-next-line @next/next/no-img-element */}<img ref={imageRef} src={preview} alt="Leaf selected for analysis" /><span className="replace-label">Replace image</span></> : <><span className="camera-icon">⌗</span><strong>Drop a clear leaf image here</strong><small>or tap to use your camera / browse files</small></>}
          </button>
          <button className="primary-action" disabled={!preview || analyzing || modelState !== 'ready'} onClick={analyze}><span>{analyzing ? 'Running CNN analysis…' : 'Analyze leaf'}</span><span>{analyzing ? '◌' : '→'}</span></button>
          {validationError && <div className="validation-error" role="alert"><b>Image not accepted</b><span>{validationError}</span></div>}
          <p className="privacy-note">Trained CNN • 54,305 PlantVillage images • Images stay in your browser.</p>
        </section>
      </section>

      <section className="field-strip" id="monitor">
        <div><span className="field-icon">◒</span><small>Field station</small><strong>Greenhouse 01</strong></div><div><small>Temperature</small><strong>26.4°C</strong><em>Optimal</em></div><div><small>Humidity</small><strong>68%</strong><em>Optimal</em></div><div><small>Soil moisture</small><strong>42%</strong><em className="watch">Watch</em></div><div className="sensor-live"><i /> Sensors live</div>
      </section>

      {result && <section className="diagnosis" ref={resultsRef} aria-live="polite">
        <div className="section-label"><span>02</span> Diagnosis</div>
        <div className="result-grid">
          <article className="result-main">
            <div className="result-top"><span className={`severity ${result.isHealthy ? 'healthy' : ''}`}>{result.severity}</span><span>LeafLens CNN • 64×64</span></div>
            <p className="crop-label">{result.crop}</p><h2>{result.disease}</h2><p className="result-summary">{result.summary}</p>
            <div className="confidence-row"><div><small>Model confidence</small><strong>{result.confidence.toFixed(1)}%</strong></div><div className="confidence-track"><span style={{ width: `${Math.min(result.confidence, 100)}%` }} /></div></div>
            <div className="alternatives"><span>Other possibilities</span>{result.alternatives.map((item) => <b key={item.label}>{item.label} <em>{item.confidence.toFixed(1)}%</em></b>)}</div>
          </article>
          <article className="advisory-card">
            <span className="step">RECOMMENDED ACTIONS</span><h3>Care advisory</h3>
            <ol>{result.actions.map((action, index) => <li key={action}><b>Step {index + 1}</b><span>{action}</span></li>)}</ol>
            <div className="alert-box"><b>{result.isHealthy ? 'Field note' : 'Field alert'}</b><span>{result.isHealthy ? 'Continue monitoring; this is still an image-based prediction.' : 'Confirm the diagnosis locally before applying any crop-protection product.'}</span></div>
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
