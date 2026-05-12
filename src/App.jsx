import { useState, useRef, useCallback, useEffect } from "react";
import * as ort from "onnxruntime-web";
import "./App.css";

// ============================================================
// CROP SCREEN COMPONENT
// ============================================================
function CropScreen({ imageUrl, onConfirm, onCancel }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const [drag, setDrag] = useState(null);      // { x, y } start point
  const [box, setBox] = useState(null);         // { x, y, w, h } in canvas coords
  const [isDragging, setIsDragging] = useState(false);

  // Draw image + overlay on every box change
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    if (box) {
      // dim everything outside box
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      // clear inside box
      ctx.clearRect(box.x, box.y, box.w, box.h);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      // redraw dim excluding box
      ctx.save();
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.beginPath();
      ctx.rect(0, 0, canvas.width, canvas.height);
      ctx.rect(box.x, box.y, box.w, box.h);
      ctx.fill("evenodd");
      ctx.restore();
      // border
      ctx.strokeStyle = "#C8A96E";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.strokeRect(box.x, box.y, box.w, box.h);
      ctx.setLineDash([]);
      // corner handles
      [[box.x, box.y],[box.x+box.w, box.y],[box.x, box.y+box.h],[box.x+box.w, box.y+box.h]].forEach(([cx,cy]) => {
        ctx.fillStyle = "#C8A96E";
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, Math.PI*2);
        ctx.fill();
      });
    }
  }, [box, imageUrl]);

  const getPos = (e, canvas) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const src = e.touches ? e.touches[0] : e;
    return {
      x: (src.clientX - rect.left) * scaleX,
      y: (src.clientY - rect.top) * scaleY,
    };
  };

  const onStart = (e) => {
    e.preventDefault();
    const pos = getPos(e, canvasRef.current);
    setDrag(pos);
    setIsDragging(true);
    setBox(null);
  };

  const onMove = (e) => {
    e.preventDefault();
    if (!isDragging || !drag) return;
    const pos = getPos(e, canvasRef.current);
    setBox({
      x: Math.min(drag.x, pos.x),
      y: Math.min(drag.y, pos.y),
      w: Math.abs(pos.x - drag.x),
      h: Math.abs(pos.y - drag.y),
    });
  };

  const onEnd = () => setIsDragging(false);

  const handleConfirm = () => {
    if (!box || box.w < 10 || box.h < 10) return;
    const canvas = canvasRef.current;
    const img = imgRef.current;
    // Scale box from canvas coords → real image coords
    const scaleX = img.naturalWidth / canvas.width;
    const scaleY = img.naturalHeight / canvas.height;
    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = Math.round(box.w * scaleX);
    cropCanvas.height = Math.round(box.h * scaleY);
    cropCanvas.getContext("2d").drawImage(
      img,
      Math.round(box.x * scaleX), Math.round(box.y * scaleY),
      cropCanvas.width, cropCanvas.height,
      0, 0, cropCanvas.width, cropCanvas.height
    );
    cropCanvas.toBlob((blob) => onConfirm(URL.createObjectURL(blob), cropCanvas), "image/jpeg", 0.92);
  };

  const skipCrop = () => onConfirm(imageUrl, imgRef.current);

  return (
    <div className="screen">
      <div className="crop-header">
        <h2 className="crop-title">Crop the Rock</h2>
        <p className="crop-hint">Drag to select only the mineral specimen — exclude hands &amp; background</p>
      </div>
      <div className="crop-canvas-wrap">
        <img
          ref={imgRef}
          src={imageUrl}
          alt=""
          style={{ display: "none" }}
          onLoad={() => {
            const img = imgRef.current;
            const canvas = canvasRef.current;
            // Fit canvas to image aspect ratio, max 420px wide
            const maxW = 420;
            const ratio = img.naturalHeight / img.naturalWidth;
            canvas.width = maxW;
            canvas.height = Math.round(maxW * ratio);
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          }}
        />
        <canvas
          ref={canvasRef}
          className="crop-canvas"
          onMouseDown={onStart} onMouseMove={onMove} onMouseUp={onEnd}
          onTouchStart={onStart} onTouchMove={onMove} onTouchEnd={onEnd}
        />
        {!box && (
          <div className="crop-guide-label">Drag here to select the rock area</div>
        )}
      </div>
      <div className="btn-group">
        <button className="btn-primary" onClick={handleConfirm} disabled={!box || box.w < 10}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12"/></svg>
          Analyze Cropped Region
        </button>
        <button className="btn-secondary" onClick={skipCrop}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/></svg>
          Skip — Use Full Image
        </button>
        <button className="btn-secondary" onClick={onCancel}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.5"/></svg>
          Retake
        </button>
      </div>
    </div>
  );
}

// ============================================================
// PROPERTY-BASED Q&A QUESTIONS
// ============================================================
const PROPERTY_QUESTIONS = {
  acid_reaction: {
    question: "Put a drop of vinegar or acid on the surface. What happens?",
    options: ["Nothing happens", "Slight fizzing", "Strong fizzing / bubbling"],
    values: ["None", "Weak fizz", "Strong fizz"],
  },
  special_property: {
    question: "Does it stick to a magnet?",
    options: ["Yes, it's attracted to magnet", "No reaction"],
    values: ["Magnetic", "not magnetic"],
  },
  hardness_testable: {
    question: "Try to scratch it with your fingernail. What happens?",
    options: [
      "Scratches easily (soft)",
      "Scratches with a coin but not fingernail",
      "Scratches with a knife blade",
      "Cannot be scratched — very hard",
    ],
    values: [
      "Scratches with fingernail",
      "Scratches with copper coin",
      "Scratches with steel knife",
      "Cannot be scratched easily",
    ],
  },
  luster: {
    question: "How does the surface shine?",
    options: ["Shiny like metal", "Glassy / vitreous", "Dull / earthy", "Pearly or silky"],
    values: ["Metallic", "Glassy", "Dull", "Pearly"],
  },
  streak_color: {
    question: "Scratch the mineral on rough concrete or tile. What color is the powder?",
    options: ["White or colorless", "Black", "Greenish black", "Red/brown", "Yellow"],
    values: ["White", "Black", "Greenish black", "Red", "Yellow"],
  },
};

const PROPERTY_PRIORITY = ["acid_reaction", "special_property", "hardness_testable", "luster", "streak_color"];
const CONFIDENCE_THRESHOLD = 0.35;
const NOT_MINERAL_LABEL = "not_mineral";

// ============================================================
// ONNX INFERENCE
// ============================================================
let onnxSession = null;

async function runInference(imageElement) {
  try {
    if (!onnxSession) {
      onnxSession = await ort.InferenceSession.create("/model/litholens_model.onnx");
    }
    const canvas = document.createElement("canvas");
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(imageElement, 0, 0, 224, 224);
    const imageData = ctx.getImageData(0, 0, 224, 224);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const input = new Float32Array(3 * 224 * 224);
    for (let i = 0; i < 224 * 224; i++) {
      input[i] = (imageData.data[i * 4] / 255 - mean[0]) / std[0];
      input[224 * 224 + i] = (imageData.data[i * 4 + 1] / 255 - mean[1]) / std[1];
      input[2 * 224 * 224 + i] = (imageData.data[i * 4 + 2] / 255 - mean[2]) / std[2];
    }
    const tensor = new ort.Tensor("float32", input, [1, 3, 224, 224]);
    // Try both common input names
    let results;
    try {
      results = await onnxSession.run({ input: tensor });
    } catch {
      results = await onnxSession.run({ image: tensor });
    }
    const outputKey = Object.keys(results)[0];
    const logits = results[outputKey].data;
    const maxLogit = Math.max(...logits);
    const exp = Array.from(logits).map((x) => Math.exp(x - maxLogit));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map((x) => x / sum);
  } catch (err) {
    console.error("ONNX inference error:", err);
    return null;
  }
}

function getDemoPredictions(count) {
  const idx = Math.floor(Math.random() * Math.max(1, count - 1));
  return Array.from({ length: count }, (_, i) => (i === idx ? 0.55 : Math.random() * 0.08));
}

// ============================================================
// Q&A HELPERS
// ============================================================
function findBestQuestion(m1Key, m2Key, mineralsDB) {
  const m1 = mineralsDB[m1Key], m2 = mineralsDB[m2Key];
  if (!m1 || !m2) return null;
  for (const prop of PROPERTY_PRIORITY) {
    const v1 = (m1[prop] || "").toLowerCase(), v2 = (m2[prop] || "").toLowerCase();
    if (v1 !== v2 && v1 && v2 && PROPERTY_QUESTIONS[prop]) return { property: prop };
  }
  return null;
}

function filterByAnswer(candidates, property, answerValue, mineralsDB) {
  return candidates.filter((key) => {
    const m = mineralsDB[key];
    if (!m) return true;
    const val = (m[property] || "").toLowerCase();
    return val.includes(answerValue.toLowerCase()) || answerValue.toLowerCase().includes(val);
  });
}

// ============================================================
// MAIN APP
// ============================================================
export default function LithoLens() {
  const [screen, setScreen] = useState("home");
  const [capturedImage, setCapturedImage] = useState(null);   // raw original
  const [croppedImageUrl, setCroppedImageUrl] = useState(null); // what was sent to model
  const [predictions, setPredictions] = useState([]);
  const [topCandidates, setTopCandidates] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [finalResult, setFinalResult] = useState(null);
  const [qaHistory, setQaHistory] = useState([]);
  const [classNames, setClassNames] = useState([]);
  const [mineralsDB, setMineralsDB] = useState({});
  const [notMineralIndex, setNotMineralIndex] = useState(-1);
  const [dataLoaded, setDataLoaded] = useState(false);
  const fileInputRef = useRef(null);

  // Load class names + minerals DB on mount
  useEffect(() => {
    async function loadData() {
      try {
        const cnRes = await fetch("/model/class_names.json");
        const cnData = await cnRes.json();
        setClassNames(cnData.classes || []);
        setNotMineralIndex(cnData.not_mineral_index ?? -1);

        try {
          const dbRes = await fetch("/model/minerals_db.json");
          if (dbRes.ok) setMineralsDB(await dbRes.json());
        } catch { /* minerals_db not yet available */ }
      } catch (e) {
        console.warn("Could not load class_names.json, using fallback", e);
        setClassNames(Object.keys(mineralsDB));
      }
      setDataLoaded(true);
    }
    loadData();
  }, []);

  // Step 1: File selected → go to crop screen
  const handleImageCapture = useCallback((file) => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setCapturedImage(url);
    setCroppedImageUrl(null);
    setScreen("crop");
  }, []);

  // Step 2: Crop confirmed → run model on cropped region
  const runAnalysis = useCallback(async (croppedUrl, imgElement) => {
    setCroppedImageUrl(croppedUrl);
    setScreen("analyzing");

    const doInference = async (img) => {
      await new Promise((r) => setTimeout(r, 1200));
      let probs = await runInference(img);
      const names = classNames.length > 0 ? classNames : Object.keys(mineralsDB);
      if (!probs) probs = getDemoPredictions(names.length);

      const indexed = probs.map((p, i) => ({ key: names[i] || `class_${i}`, prob: p, idx: i }));
      const sorted = indexed.sort((a, b) => b.prob - a.prob);

      const topPred = sorted[0];
      const isNotMineral = topPred.key === NOT_MINERAL_LABEL || topPred.idx === notMineralIndex;
      const maxConf = topPred.prob;
      const entropy = -probs.reduce((s, p) => s + (p > 0.001 ? p * Math.log(p) : 0), 0);
      const normalizedEntropy = entropy / Math.log(probs.length);

      if (isNotMineral || (maxConf < CONFIDENCE_THRESHOLD && normalizedEntropy > 0.85)) {
        setScreen("not_rock");
        return;
      }
      const mineralPreds = sorted.filter((p) => p.key !== NOT_MINERAL_LABEL && p.idx !== notMineralIndex);
      const top3 = mineralPreds.slice(0, 3);
      setPredictions(top3);
      setTopCandidates(top3.map((t) => t.key));
      setScreen("results");
    };

    // If imgElement is already a canvas/img element use it directly
    if (imgElement && imgElement.tagName) {
      await doInference(imgElement);
    } else {
      const img = new Image();
      img.onload = () => doInference(img);
      img.src = croppedUrl;
    }
  }, [classNames, notMineralIndex, mineralsDB]);

  const handleConfidencePath = (top3) => {
    if (top3[0]?.prob >= 0.75) {
      setFinalResult(mineralsDB[top3[0].key] || { name_english: top3[0].key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase()) });
      setScreen("final");
    } else {
      startQA(top3.map((t) => t.key));
    }
  };

  const startQA = (candidates) => {
    if (candidates.length <= 1) {
      const key = candidates[0] || topCandidates[0];
      setFinalResult(mineralsDB[key] || { name_english: key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase()) });
      setScreen("final");
      return;
    }
    const q = findBestQuestion(candidates[0], candidates[1], mineralsDB);
    if (!q) {
      setFinalResult(mineralsDB[candidates[0]] || { name_english: candidates[0].replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase()) });
      setScreen("final");
      return;
    }
    setCurrentQuestion({ ...PROPERTY_QUESTIONS[q.property], property: q.property, candidates });
    setScreen("qa");
  };

  const handleAnswer = (answerValue) => {
    if (!currentQuestion) return;
    const { property, candidates } = currentQuestion;
    setQaHistory((prev) => [...prev, { question: currentQuestion.question, answer: answerValue }]);
    const remaining = filterByAnswer(candidates, property, answerValue, mineralsDB);
    const next = remaining.length > 0 ? remaining : candidates.slice(0, 1);
    if (next.length === 1 || qaHistory.length >= 3) {
      setFinalResult(mineralsDB[next[0]] || { name_english: next[0].replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase()) });
      setScreen("final");
    } else startQA(next);
  };

  const reset = () => {
    setCapturedImage(null); setCroppedImageUrl(null); setPredictions([]); setTopCandidates([]);
    setCurrentQuestion(null); setFinalResult(null); setQaHistory([]);
    setScreen("home");
  };

  const getMineralDisplayName = (key) => {
    const m = mineralsDB[key];
    if (m?.name_english) return m.name_english;
    return key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
  };

  // ============================================================
  // SCREENS
  // ============================================================

  if (!dataLoaded) {
    return (
      <div className="app">
        <div className="loading-screen">
          <div className="spinner" />
          <div className="loading-text">Loading LithoLens...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <div className="topbar">
        <span className="topbar-name">
          <img src="/litholens-logo.png" alt="LithoLens" className="topbar-logo" />
          LithoLens
        </span>
        {screen !== "home" && (
          <button className="topbar-back" onClick={reset} aria-label="Close">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        )}
      </div>
      <div className="content">
        {/* HOME */}
        {screen === "home" && (
          <div className="screen">
            <div className="logo-area">
              <div className="logo">
                <img src="/litholens-logo.png" alt="LithoLens Logo" className="logo-img" />
              </div>
              <h1 className="title">LithoLens</h1>
              <p className="subtitle">AI Rock &amp; Mineral Identification</p>
              <p className="subtitle-ar">التعرف على الصخور والمعادن بالذكاء الاصطناعي</p>
            </div>
            <div className="feature-row">
              {[
                { icon: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>, label: "Scan" },
                { icon: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/><path d="M7 8h2M15 8h2M11 11h2"/></svg>, label: "AI Analysis" },
                { icon: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>, label: "Smart Q&A" },
                { icon: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12"/></svg>, label: "Result" },
              ].map((f, i) => (
                <div key={i} className="feature-chip"><span className="chip-icon">{f.icon}</span>{f.label}</div>
              ))}
            </div>
            <div className="btn-group">
              <button className="btn-primary" onClick={() => fileInputRef.current?.click()}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
                Take Photo
              </button>
              <button className="btn-primary btn-upload" onClick={() => fileInputRef.current?.click()}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                Upload Image
              </button>
            </div>
            <input ref={fileInputRef} type="file" accept="image/*" capture="environment" style={{ display: "none" }} onChange={(e) => handleImageCapture(e.target.files[0])} />
            <p className="hint">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" style={{verticalAlign:"middle",marginRight:4}}><path d="M1 6v16l7-4 8 4 7-4V2l-7 4-8-4-7 4z"/><path d="M8 2v16M16 6v16"/></svg>
              Works offline · No internet needed
            </p>
          </div>
        )}

        {/* CROP */}
        {screen === "crop" && (
          <CropScreen
            imageUrl={capturedImage}
            onConfirm={(url, el) => runAnalysis(url, el)}
            onCancel={reset}
          />
        )}

        {/* ANALYZING */}
        {screen === "analyzing" && (
          <div className="screen">
            {croppedImageUrl && <img src={croppedImageUrl} alt="captured" className="preview-img" />}
            <div className="analyze-card">
              <div className="spinner" />
              <h2 className="analyze-text">Analyzing sample...</h2>
              <p className="analyze-sub">AI model running on your device</p>
              <div className="progress-bar"><div className="progress-fill" /></div>
            </div>
          </div>
        )}

        {/* NOT A ROCK */}
        {screen === "not_rock" && (
          <div className="screen">
            {croppedImageUrl && <img src={croppedImageUrl} alt="captured" className="preview-img" />}
            <div className="not-rock-card">
              <div className="not-rock-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
              </div>
              <h2 className="not-rock-title">Not a Mineral</h2>
              <p className="not-rock-text">This doesn't appear to be a rock or mineral sample. Try again with a real specimen.</p>
              <div className="tips-list">
                <div className="tip-item"><span className="tip-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 3L3 21"/><path d="M21 21L3 3"/></svg></span>Get closer to the sample</div>
                <div className="tip-item"><span className="tip-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/></svg></span>Use good, even lighting</div>
                <div className="tip-item"><span className="tip-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><rect x="3" y="3" width="18" height="18" rx="2"/></svg></span>Place on a plain background</div>
                <div className="tip-item"><span className="tip-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></span>Show a fresh surface or fracture</div>
              </div>
            </div>
            <button className="btn-primary" onClick={reset}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
              Try Again
            </button>
          </div>
        )}

        {/* RESULTS */}
        {screen === "results" && (
          <div className="screen">
            <h2 className="section-title">Top Predictions</h2>
            {croppedImageUrl && <img src={croppedImageUrl} alt="captured" className="thumb-img" />}
            <div className="pred-list">
              {predictions.map((pred, i) => {
                const m = mineralsDB[pred.key];
                const conf = Math.round(pred.prob * 100);
                const color = conf >= 75 ? "var(--success)" : conf >= 50 ? "var(--warning)" : "var(--danger)";
                return (
                  <div key={i} className="pred-card">
                    <div className="pred-rank">{i + 1}</div>
                    <div className="pred-info">
                      <div className="pred-name">
                        {getMineralDisplayName(pred.key)}
                        <span className="pred-ar"> · {m?.name_arabic || ""}</span>
                      </div>
                      <div className="pred-cat">{m?.category || ""}</div>
                      <div className="conf-bar">
                        <div className="conf-fill" style={{ width: `${conf}%`, background: color }} />
                      </div>
                    </div>
                    <div className="conf-label" style={{ color }}>{conf}%</div>
                  </div>
                );
              })}
            </div>
            <div className="btn-group">
              {predictions[0]?.prob >= 0.75 ? (
                <button className="btn-primary" onClick={() => handleConfidencePath(predictions)}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12"/></svg>
                  Confirm — {getMineralDisplayName(predictions[0]?.key)}
                </button>
              ) : (
                <button className="btn-primary" onClick={() => handleConfidencePath(predictions)}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                  Answer Questions to Narrow Down
                </button>
              )}
              <button className="btn-secondary" onClick={reset}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.5"/></svg>
                Scan Again
              </button>
            </div>
          </div>
        )}

        {/* Q&A */}
        {screen === "qa" && (
          <div className="screen">
            <div className="qa-header">
              <div className="qa-step">Step {qaHistory.length + 1}</div>
              <h2 className="qa-title">Let's narrow it down</h2>
              <p className="qa-subtitle">
                Comparing: {currentQuestion?.candidates?.slice(0, 2).map((k) => getMineralDisplayName(k)).join(" vs ")}
              </p>
            </div>
            <div className="question-card">
              <p className="question-text">{currentQuestion?.question}</p>
            </div>
            <div className="options-list">
              {currentQuestion?.options?.map((opt, i) => (
                <button key={i} className="option-btn" onClick={() => handleAnswer(currentQuestion.values[i])}>{opt}</button>
              ))}
            </div>
            {qaHistory.length > 0 && (
              <div className="history-area">
                <p className="history-label">Previous answers:</p>
                {qaHistory.map((h, i) => <p key={i} className="history-item"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" style={{marginRight:6,verticalAlign:"middle",color:"var(--green)"}}><polyline points="20 6 9 17 4 12"/></svg>{h.answer}</p>)}
              </div>
            )}
          </div>
        )}

        {/* FINAL RESULT */}
        {screen === "final" && (() => {
          const m = finalResult || {};
          return (
            <div className="screen">
              <div className="result-badge">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" style={{marginRight:5,verticalAlign:"middle"}}><polyline points="20 6 9 17 4 12"/></svg>
                Identified
              </div>
              <div className="result-card">
                <div className="result-logo">
                  <img src="/litholens-logo.png" alt="mineral" className="result-logo-img" />
                </div>
                <h1 className="result-name">{m.name_english || "Unknown"}</h1>
                <p className="result-arabic">{m.name_arabic || ""}</p>
                {m.category && <div className="result-category">{m.category}</div>}
                <div className="prop-grid">
                  {[
                    [<svg key="h" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>, "Hardness", m.hardness_moh ? `${m.hardness_moh} Mohs` : ""],
                    [<svg key="f" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M18 8h1a4 4 0 0 1 0 8h-1"/><path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z"/><line x1="6" y1="1" x2="6" y2="4"/><line x1="10" y1="1" x2="10" y2="4"/><line x1="14" y1="1" x2="14" y2="4"/></svg>, "Field Test", m.hardness_testable || ""],
                    [<svg key="l" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>, "Luster", m.luster || ""],
                    [<svg key="s" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>, "Streak", m.streak_color || ""],
                    [<svg key="c" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/></svg>, "Cleavage", m.cleavage || ""],
                    [<svg key="a" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v11m0 0H5m4 0h10m0-11v11m0 0H5"/></svg>, "Acid Test", m.acid_reaction || ""],
                  ].filter(([, , v]) => v).map(([icon, label, value], i) => (
                    <div key={i} className="prop-item">
                      <div className="prop-label">{icon} {label}</div>
                      <div className="prop-value">{value}</div>
                    </div>
                  ))}
                </div>
                {m.special_property && <div className="special-prop">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" style={{marginRight:5,verticalAlign:"middle"}}><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
                  {m.special_property}
                </div>}
                {m.description_for_ai && <p className="description">{m.description_for_ai}</p>}
                {m.common_locations && <p className="locations">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" style={{marginRight:4,verticalAlign:"middle"}}><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
                  {m.common_locations}
                </p>}
              </div>
              <button className="btn-primary" onClick={reset}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.5"/></svg>
                Identify Another Sample
              </button>
            </div>
          );
        })()}
      </div>
    </div>
  );
}
