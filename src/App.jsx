import { useState, useRef, useCallback, useEffect } from "react";
import * as ort from "onnxruntime-web";
import "./App.css";
import { t, getPropertyQuestions } from "./i18n/strings.js";
import { buildConfidenceBullets } from "./lib/confidenceExplain.js";
import {
  saveSpecimenRecord,
  listSpecimens,
  deleteSpecimenRecord,
  clearAllSpecimens,
  resizeImageToDataUrl,
} from "./lib/specimenHistory.js";
import { renderSpecimenCardPng } from "./lib/exportCard.js";

const CONFIDENCE_THRESHOLD = 0.35;
const HIGH_CONFIDENCE_THRESHOLD = 0.75;
const TEMPERATURE = 2.0;
const NOT_MINERAL_LABEL = "not_mineral";

const PROPERTY_PRIORITY = ["acid_reaction", "special_property", "hardness_testable", "luster", "streak_color"];

// ============================================================
// ONNX SESSIONS (cached)
// ============================================================
let mineralSession = null;
let mobilenetSession = null;
let mobilenetLabels = null;
let manMadeIndices = null;

async function getMineralSession() {
  if (!mineralSession) {
    mineralSession = await ort.InferenceSession.create("/model/litholens_model.onnx");
  }
  return mineralSession;
}

async function getMobilenetSession() {
  if (!mobilenetSession) {
    mobilenetSession = await ort.InferenceSession.create("/model/mobilenetv2-7.onnx");
  }
  return mobilenetSession;
}

async function loadMobilenetMetadata() {
  if (!mobilenetLabels) {
    const res = await fetch("/model/imagenet_class_index.json");
    const raw = await res.json();
    mobilenetLabels = {};
    for (const [k, v] of Object.entries(raw)) {
      mobilenetLabels[parseInt(k)] = v[1].replace(/_/g, " ");
    }
  }
  if (!manMadeIndices) {
    const res = await fetch("/model/imagenet_man_made_indices.json");
    manMadeIndices = new Set(await res.json());
  }
  return { labels: mobilenetLabels, manMade: manMadeIndices };
}

// ============================================================
// IMAGE PREPROCESSING (canvas-based OOD heuristics)
// ============================================================
function analyzeImageNaturalness(imageElement) {
  const canvas = document.createElement("canvas");
  const size = 224;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imageElement, 0, 0, size, size);
  const data = ctx.getImageData(0, 0, size, size).data;

  let totalSaturation = 0;
  let totalBrightness = 0;
  let edgeSum = 0;
  const pixelCount = size * size;

  for (let i = 0; i < pixelCount; i++) {
    const idx = i * 4;
    const r = data[idx], g = data[idx + 1], b = data[idx + 2];
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const brightness = (r + g + b) / 3;
    totalBrightness += brightness;
    if (max > 0) totalSaturation += ((max - min) / max) * 100;
  }

  const avgSaturation = totalSaturation / pixelCount;
  const avgBrightness = totalBrightness / pixelCount;

  const grayCanvas = document.createElement("canvas");
  grayCanvas.width = size;
  grayCanvas.height = size;
  const gCtx = grayCanvas.getContext("2d");
  gCtx.drawImage(imageElement, 0, 0, size, size);
  const grayData = gCtx.getImageData(0, 0, size, size).data;

  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const idx = (y * size + x) * 4;
      const gx = Math.abs(grayData[idx] - grayData[(y * size + (x + 1)) * 4])
               + Math.abs(grayData[((y - 1) * size + x) * 4] - grayData[((y + 1) * size + x) * 4]);
      edgeSum += gx;
    }
  }

  const avgEdge = edgeSum / ((size - 2) * (size - 2));
  const edgeScore = Math.min(avgEdge / 40, 1);
  const satScore = avgSaturation / 100;
  const brightScore = avgBrightness / 255;

  const unnaturalColorCount = countUnnaturalColors(data, pixelCount);
  const colorScore = unnaturalColorCount / pixelCount;

  return {
    avgSaturation,
    avgBrightness,
    avgEdge,
    edgeScore,
    satScore,
    brightScore,
    colorScore,
    isNatural: satScore < 0.55 && edgeScore < 0.6 && colorScore < 0.3,
    naturalness: 1 - (satScore * 0.4 + edgeScore * 0.35 + colorScore * 0.25),
  };
}

function countUnnaturalColors(data, pixelCount) {
  let count = 0;
  for (let i = 0; i < Math.min(pixelCount, 5000); i += 5) {
    const idx = i * 4;
    const r = data[idx], g = data[idx + 1], b = data[idx + 2];
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const sat = max > 0 ? (max - min) / max : 0;
    const isPure = (r > 240 && g > 240 && b > 240) || (r < 15 && g < 15 && b < 15);
    const isPrimary = (r > 200 && g < 80 && b < 80) || (r < 80 && g > 200 && b < 80) || (r < 80 && g < 80 && b > 200);
    const isNeon = sat > 0.85 && max > 200;
    if (isPure || isPrimary || isNeon) count++;
  }
  return count;
}

// ============================================================
// MODEL INFERENCE HELPERS
// ============================================================
function preprocessImage(imageElement, size) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imageElement, 0, 0, size, size);
  return ctx.getImageData(0, 0, size, size);
}

function imageDataToTensor(imageData, size, mean, std) {
  const input = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    input[i] = (imageData.data[i * 4] / 255 - mean[0]) / std[0];
    input[size * size + i] = (imageData.data[i * 4 + 1] / 255 - mean[1]) / std[1];
    input[2 * size * size + i] = (imageData.data[i * 4 + 2] / 255 - mean[2]) / std[2];
  }
  return input;
}

function softmax(logits, temperature) {
  const t = temperature || 1.0;
  const scaled = logits.map((x) => x / t);
  const maxVal = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

function computeEntropy(probs) {
  const eps = 1e-10;
  return -probs.reduce((s, p) => s + (p > eps ? p * Math.log(p) : 0), 0);
}

async function runMineralInference(imageElement) {
  try {
    const session = await getMineralSession();
    const imageData = preprocessImage(imageElement, 224);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const input = imageDataToTensor(imageData, 224, mean, std);
    const tensor = new ort.Tensor("float32", input, [1, 3, 224, 224]);

    let results;
    try {
      results = await session.run({ input: tensor });
    } catch {
      results = await session.run({ image: tensor });
    }

    const outputKey = Object.keys(results)[0];
    const logits = results[outputKey].data;
    return Array.from(logits);
  } catch (err) {
    console.error("Mineral inference error:", err);
    return null;
  }
}

async function runMobilenetInference(imageElement) {
  try {
    const session = await getMobilenetSession();
    const imageData = preprocessImage(imageElement, 224);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const input = imageDataToTensor(imageData, 224, mean, std);
    const tensor = new ort.Tensor("float32", input, [1, 3, 224, 224]);

    let results;
    try {
      results = await session.run({ input: tensor });
    } catch {
      results = await session.run({ data: tensor });
    }

    const outputKey = Object.keys(results)[0];
    return Array.from(results[outputKey].data);
  } catch (err) {
    console.error("MobileNet inference error:", err);
    return null;
  }
}

// ============================================================
// OOD DETECTION
// ============================================================
function isLikelyManMade(imageElement, mineralLogits, mobilenetLogits) {
  const imageAnalysis = analyzeImageNaturalness(imageElement);
  const analysisFlags = [];

  if (!imageAnalysis.isNatural) {
    analysisFlags.push(`img: nat=${imageAnalysis.naturalness.toFixed(2)} sat=${imageAnalysis.avgSaturation.toFixed(1)} edge=${imageAnalysis.avgEdge.toFixed(1)}`);
  }

  const mineralProbs = softmax(mineralLogits);
  const mineralProbsScaled = softmax(mineralLogits, TEMPERATURE);
  const maxConf = Math.max(...mineralProbs);
  const entropy = computeEntropy(mineralProbsScaled);
  const normEntropy = entropy / Math.log(mineralProbs.length);

  const sorted = mineralProbs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
  const gap = sorted[0].p - sorted[1].p;

  let modelFlags = "ent=" + normEntropy.toFixed(2) + " conf=" + maxConf.toFixed(3) + " gap=" + gap.toFixed(3);

  let mobilenetFlags = "";
  let manMadeConfidence = 0;

  if (mobilenetLogits) {
    const mbProbs = softmax(mobilenetLogits);
    const mbProbsScaled = softmax(mobilenetLogits, TEMPERATURE);
    const mbMaxConf = Math.max(...mbProbs);
    const mbEntropy = computeEntropy(mbProbsScaled);
    const mbNormEntropy = mbEntropy / Math.log(mbProbs.length);

    const mbSorted = mbProbs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);

    if (manMadeIndices && manMadeIndices.size > 0) {
      manMadeConfidence = mbSorted
        .filter((x) => manMadeIndices.has(x.i))
        .reduce((s, x) => s + x.p, 0);
    }

    const top1ManMade = manMadeIndices && manMadeIndices.has(mbSorted[0].i);
    const mbLabel = mobilenetLabels ? mobilenetLabels[mbSorted[0].i] || "?" : "?";

    mobilenetFlags = ` mb: top=${mbLabel} conf=${mbMaxConf.toFixed(3)} ent=${mbNormEntropy.toFixed(2)} manMade=${(manMadeConfidence * 100).toFixed(0)}% 1stMM=${top1ManMade}`;
  }

  const isManMade = (
    (normEntropy < 0.4 && maxConf < 0.6) ||
    (normEntropy < 0.3 && gap < 0.05) ||
    manMadeConfidence > 0.7 ||
    (!imageAnalysis.isNatural && normEntropy < 0.5)
  );

  return {
    isManMade,
    confidence: isManMade ? Math.max(0.5, 1 - imageAnalysis.naturalness) : Math.min(0.5, imageAnalysis.naturalness),
    reason: `[${analysisFlags.join("|")} ${modelFlags}${mobilenetFlags}]`,
    imageAnalysis,
    normEntropy,
    maxConf,
    gap,
    manMadeConfidence,
  };
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
    if (v1 !== v2 && v1 && v2 && PROPERTY_PRIORITY.includes(prop)) return { property: prop };
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
// CROP SCREEN COMPONENT
// ============================================================
function CropScreen({ imageUrl, onConfirm, onCancel, lang }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const [drag, setDrag] = useState(null);
  const [box, setBox] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    if (box) {
      ctx.save();
      ctx.fillStyle = "rgba(2,3,8,0.62)";
      ctx.beginPath();
      ctx.rect(0, 0, canvas.width, canvas.height);
      ctx.rect(box.x, box.y, box.w, box.h);
      ctx.fill("evenodd");
      ctx.restore();

      ctx.strokeStyle = "#2ec4b6";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.strokeRect(box.x, box.y, box.w, box.h);
      ctx.setLineDash([]);

      [[box.x, box.y],[box.x+box.w, box.y],[box.x, box.y+box.h],[box.x+box.w, box.y+box.h]].forEach(([cx,cy]) => {
        ctx.fillStyle = "#4df0e0";
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
        <h2 className="crop-title">{t(lang, "crop_title")}</h2>
        <p className="crop-hint">{t(lang, "crop_hint")}</p>
      </div>
      <div
        className="crop-canvas-wrap"
        role="application"
        aria-label={t(lang, "crop_title")}
        aria-describedby="crop-instructions"
      >
        <p id="crop-instructions" className="visually-hidden">
          {t(lang, "crop_guide")}
        </p>
        <img
          ref={imgRef}
          src={imageUrl}
          alt=""
          style={{ display: "none" }}
          onLoad={() => {
            const img = imgRef.current;
            const canvas = canvasRef.current;
            const maxW = Math.min(window.innerWidth - 40, 480);
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
        <div className="crop-grid-overlay" aria-hidden="true" />
        {!box && (
          <div className="crop-guide-label" aria-hidden="true">{t(lang, "crop_guide")}</div>
        )}
      </div>
      <div className="btn-group">
        <button className="btn-primary" onClick={handleConfirm} disabled={!box || box.w < 10} aria-label={t(lang, "crop_analyze")}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12"/></svg>
          {t(lang, "crop_analyze")}
        </button>
        <button className="btn-secondary" onClick={skipCrop} aria-label={t(lang, "crop_skip")}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/></svg>
          {t(lang, "crop_skip")}
        </button>
        <button className="btn-secondary" onClick={onCancel} aria-label={t(lang, "crop_retake")}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.5"/></svg>
          {t(lang, "crop_retake")}
        </button>
      </div>
    </div>
  );
}

// ============================================================
// MAIN APP
// ============================================================
export default function LithoLens() {
  const [screen, setScreen] = useState("home");
  const [capturedImage, setCapturedImage] = useState(null);
  const [croppedImageUrl, setCroppedImageUrl] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [topCandidates, setTopCandidates] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [finalResult, setFinalResult] = useState(null);
  const [qaHistory, setQaHistory] = useState([]);
  const [classNames, setClassNames] = useState([]);
  const [mineralsDB, setMineralsDB] = useState({});
  const [notMineralIndex, setNotMineralIndex] = useState(-1);
  const [dataLoaded, setDataLoaded] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);
  const [oodInfo, setOodInfo] = useState(null);
  const [language, setLanguage] = useState(() => {
    try {
      const v = localStorage.getItem("litholens-lang");
      return v === "ar" ? "ar" : "en";
    } catch {
      return "en";
    }
  });
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("litholens-theme") === "light" ? "light" : "dark";
    } catch {
      return "dark";
    }
  });
  const [analysisMeta, setAnalysisMeta] = useState(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyItems, setHistoryItems] = useState([]);
  const [finalMineralKey, setFinalMineralKey] = useState(null);
  const lastTopProbRef = useRef(0);
  const cameraInputRef = useRef(null);
  const uploadInputRef = useRef(null);

  useEffect(() => {
    async function loadData() {
      try {
        const cnRes = await fetch("/model/class_names.json");
        const cnData = await cnRes.json();
        setClassNames(cnData.classes || []);
        setNotMineralIndex(cnData.not_mineral_index ?? -1);

        await loadMobilenetMetadata();

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
  }, [mineralsDB]);

  const handleImageCapture = useCallback((file) => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setCapturedImage(url);
    setCroppedImageUrl(null);
    setOodInfo(null);
    setAnalysisMeta(null);
    setScreen("crop");
  }, []);

  const runAnalysis = useCallback(async (croppedUrl, imgElement) => {
    setCroppedImageUrl(croppedUrl);
    setModelLoading(true);
    setAnalysisMeta(null);
    setScreen("analyzing");

    const doInference = async (img) => {
      await new Promise((r) => setTimeout(r, 800));

      const names = classNames.length > 0 ? classNames : Object.keys(mineralsDB);
      const mineralLogits = await runMineralInference(img);
      const probs = mineralLogits
        ? softmax(mineralLogits)
        : getDemoPredictions(names.length);

      let mobilenetLogits = null;
      try {
        mobilenetLogits = await runMobilenetInference(img);
      } catch (e) {
        console.warn("MobileNet inference failed, skipping:", e);
      }

      const isNotMineral = probs.length <= notMineralIndex + 1
        ? false
        : probs[notMineralIndex] > Math.max(...probs.filter((_, i) => i !== notMineralIndex));

      const sorted = probs.map((p, i) => ({ key: names[i] || `class_${i}`, prob: p, idx: i }))
        .sort((a, b) => b.prob - a.prob);

      const entropy = computeEntropy(probs);
      const normEntropy = entropy / Math.log(probs.length);
      const maxConf = sorted[0].prob;

      const ood = mobilenetLogits && mineralLogits
        ? isLikelyManMade(img, mineralLogits, mobilenetLogits)
        : {
            isManMade: normEntropy < 0.3 && maxConf < 0.6,
            confidence: normEntropy < 0.3 ? 0.7 : 0.3,
            reason: `[heuristic ent=${normEntropy.toFixed(2)}]`,
          };
      setOodInfo(ood);

      const isOOD = ood.isManMade || isNotMineral || (maxConf < CONFIDENCE_THRESHOLD && normEntropy > 0.85);

      if (isOOD) {
        setModelLoading(false);
        setAnalysisMeta(null);
        setScreen("not_rock");
        return;
      }

      const mineralPreds = sorted.filter((p) => p.key !== NOT_MINERAL_LABEL && p.idx !== notMineralIndex);
      const top3 = mineralPreds.slice(0, 3);
      const gap = (top3[0]?.prob ?? 0) - (top3[1]?.prob ?? 0);
      setAnalysisMeta({ normEntropy, gap, top1Prob: top3[0]?.prob ?? 0 });
      lastTopProbRef.current = top3[0]?.prob ?? 0;
      setPredictions(top3);
      setTopCandidates(top3.map((t) => t.key));
      setModelLoading(false);
      setScreen("results");
    };

    if (imgElement && imgElement.tagName) {
      await doInference(imgElement);
    } else {
      const img = new Image();
      img.onload = () => doInference(img);
      img.src = croppedUrl;
    }
  }, [classNames, notMineralIndex, mineralsDB]);

  const handleConfidencePath = (top3) => {
    if (top3[0]?.prob >= HIGH_CONFIDENCE_THRESHOLD) {
      setFinalMineralKey(top3[0].key);
      setFinalResult(mineralsDB[top3[0].key] || {
        name_english: top3[0].key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())
      });
      setScreen("final");
    } else {
      startQA(top3.map((t) => t.key));
    }
  };

  const startQA = (candidates) => {
    if (candidates.length <= 1) {
      const key = candidates[0] || topCandidates[0];
      setFinalMineralKey(key);
      setFinalResult(mineralsDB[key] || {
        name_english: key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())
      });
      setScreen("final");
      return;
    }
    const q = findBestQuestion(candidates[0], candidates[1], mineralsDB);
    if (!q) {
      setFinalMineralKey(candidates[0]);
      setFinalResult(mineralsDB[candidates[0]] || {
        name_english: candidates[0].replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())
      });
      setScreen("final");
      return;
    }
    setCurrentQuestion({ ...getPropertyQuestions(language)[q.property], property: q.property, candidates });
    setScreen("qa");
  };

  const handleAnswer = (answerValue) => {
    if (!currentQuestion) return;
    const { property, candidates } = currentQuestion;
    setQaHistory((prev) => [...prev, { question: currentQuestion.question, answer: answerValue }]);
    const remaining = filterByAnswer(candidates, property, answerValue, mineralsDB);
    const next = remaining.length > 0 ? remaining : candidates.slice(0, 1);
    if (next.length === 1 || qaHistory.length >= 3) {
      setFinalMineralKey(next[0]);
      setFinalResult(mineralsDB[next[0]] || {
        name_english: next[0].replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())
      });
      setScreen("final");
    } else startQA(next);
  };

  const reset = () => {
    setCapturedImage(null); setCroppedImageUrl(null); setPredictions([]); setTopCandidates([]);
    setCurrentQuestion(null); setFinalResult(null); setQaHistory([]); setOodInfo(null);
    setAnalysisMeta(null); setFinalMineralKey(null);
    setScreen("home");
  };

  const getMineralDisplayName = (key) => {
    const m = mineralsDB[key];
    if (m?.name_english) return m.name_english;
    return key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
  };

  useEffect(() => {
    try {
      localStorage.setItem("litholens-lang", language);
    } catch { /* ignore */ }
  }, [language]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("litholens-theme", theme);
    } catch { /* ignore */ }
    const meta = document.getElementById("theme-color-meta");
    if (meta) {
      meta.setAttribute("content", theme === "light" ? "#ebe3d7" : "#05070d");
    }
  }, [theme]);

  const refreshHistory = useCallback(async () => {
    try {
      setHistoryItems(await listSpecimens());
    } catch (e) {
      console.warn(e);
    }
  }, []);

  const onPickFile = (e) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    handleImageCapture(f);
  };

  useEffect(() => {
    if (screen !== "final" || !finalResult || !croppedImageUrl) return;
    const timer = setTimeout(() => {
      (async () => {
        try {
          const thumbDataUrl = await resizeImageToDataUrl(croppedImageUrl, 220);
          await saveSpecimenRecord({
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
            createdAt: Date.now(),
            thumbDataUrl,
            nameEn: finalResult.name_english || "",
            nameAr: finalResult.name_arabic || "",
            mineralKey: finalMineralKey || "",
            topProb: lastTopProbRef.current,
          });
          setHistoryItems(await listSpecimens());
        } catch (err) {
          console.warn("Specimen history save failed", err);
        }
      })();
    }, 400);
    return () => clearTimeout(timer);
  }, [screen, finalResult, croppedImageUrl, finalMineralKey]);

  const downloadResultCard = async () => {
    const m = finalResult || {};
    let thumb = "";
    try {
      if (croppedImageUrl) thumb = await resizeImageToDataUrl(croppedImageUrl, 320);
    } catch { /* ignore */ }
    const blob = await renderSpecimenCardPng({
      titleEn: m.name_english || t(language, "unknown"),
      titleAr: m.name_arabic || "",
      category: m.category || "",
      thumbDataUrl: thumb || undefined,
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `litholens-${(m.name_english || "specimen").replace(/\s+/g, "-")}.png`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const shareResultCard = async () => {
    const m = finalResult || {};
    let thumb = "";
    try {
      if (croppedImageUrl) thumb = await resizeImageToDataUrl(croppedImageUrl, 320);
    } catch { /* ignore */ }
    const blob = await renderSpecimenCardPng({
      titleEn: m.name_english || t(language, "unknown"),
      titleAr: m.name_arabic || "",
      category: m.category || "",
      thumbDataUrl: thumb || undefined,
    });
    const file = new File([blob], "litholens-specimen.png", { type: "image/png" });
    if (navigator.share && navigator.canShare?.({ files: [file] })) {
      try {
        await navigator.share({ files: [file], title: "LithoLens", text: m.name_english || "" });
      } catch (e) {
        if ((e && e.name) !== "AbortError") console.warn(e);
      }
    } else {
      window.alert(t(language, "export_unsupported"));
    }
  };

  const printResultCard = () => window.print();

  if (!dataLoaded) {
    return (
      <div className="app">
        <div className="loading-screen">
          <div className="loading-brand">
            <img src="/litholens-logo.png" alt="" className="loading-logo" aria-hidden="true" />
            <div className="spinner" />
          </div>
          <div className="loading-text">{t(language, "loading_text")}</div>
          <p className="loading-tagline">{t(language, "loading_tagline")}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <div className="topbar">
        <span className="topbar-name">
          <img src="/litholens-logo.png" alt="" className="topbar-logo" aria-hidden="true" />
          LithoLens
        </span>
        <div className="topbar-actions">
          <button
            type="button"
            className="topbar-icon-btn"
            onClick={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}
            aria-label={t(language, theme === "dark" ? "theme_switch_light" : "theme_switch_dark")}
            aria-pressed={theme === "light"}
          >
            {theme === "dark" ? (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                <circle cx="12" cy="12" r="4" />
                <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
              </svg>
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
              </svg>
            )}
          </button>
          <div className="lang-toggle" role="group" aria-label="Language">
            <button
              type="button"
              className={`lang-toggle__btn${language === "en" ? " lang-toggle__btn--active" : ""}`}
              onClick={() => setLanguage("en")}
              aria-pressed={language === "en"}
            >
              EN
            </button>
            <button
              type="button"
              className={`lang-toggle__btn${language === "ar" ? " lang-toggle__btn--active" : ""}`}
              onClick={() => setLanguage("ar")}
              aria-pressed={language === "ar"}
            >
              عربي
            </button>
          </div>
          <button
            type="button"
            className="topbar-icon-btn"
            onClick={() => {
              setHistoryOpen(true);
              void refreshHistory();
            }}
            aria-label={t(language, "topbar_history")}
            aria-haspopup="dialog"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
          </button>
          {screen !== "home" && (
            <button type="button" className="topbar-back" onClick={reset} aria-label={t(language, "topbar_close")}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          )}
        </div>
      </div>

      {historyOpen && (
        <div className="modal-root" role="presentation">
          <button type="button" className="modal-backdrop" aria-label={t(language, "history_close")} onClick={() => setHistoryOpen(false)} />
          <div className="modal-panel" role="dialog" aria-modal="true" aria-labelledby="history-dialog-title">
            <div className="modal-header">
              <h2 id="history-dialog-title" className="modal-title">{t(language, "history_title")}</h2>
              <button type="button" className="modal-close" onClick={() => setHistoryOpen(false)} aria-label={t(language, "history_close")}>×</button>
            </div>
            <p className="modal-note">{t(language, "history_saved_local")}</p>
            {historyItems.length > 0 && (
              <button type="button" className="btn-secondary btn-compact" onClick={async () => { await clearAllSpecimens(); refreshHistory(); }}>
                {t(language, "history_clear")}
              </button>
            )}
            <ul className="history-list">
              {historyItems.length === 0 ? (
                <li className="history-empty">{t(language, "history_empty")}</li>
              ) : (
                historyItems.map((h) => (
                  <li key={h.id} className="history-row">
                    <img src={h.thumbDataUrl} alt="" className="history-thumb" width="56" height="56" />
                    <div className="history-meta">
                      <div className="history-name">{h.nameEn}</div>
                      {h.nameAr && <div className="history-ar">{h.nameAr}</div>}
                      <div className="history-date">{new Date(h.createdAt).toLocaleString(language === "ar" ? "ar" : undefined)}</div>
                    </div>
                    <button
                      type="button"
                      className="btn-icon-danger"
                      aria-label={t(language, "history_delete")}
                      onClick={async () => { await deleteSpecimenRecord(h.id); refreshHistory(); }}
                    >
                      ×
                    </button>
                  </li>
                ))
              )}
            </ul>
          </div>
        </div>
      )}

      <main className="content" id="main-content" dir={language === "ar" ? "rtl" : "ltr"} lang={language}>
        {screen === "home" && (
          <div className="screen screen--home">
            <header className="hero">
              <p className="hero-kicker">{t(language, "hero_kicker")}</p>
              <div className="hero-mark">
                <img src="/litholens-logo.png" alt="" className="hero-logo" aria-hidden="true" />
              </div>
              <h1 className="title">{t(language, "hero_title")}</h1>
              <p className="subtitle">{t(language, "hero_subtitle")}</p>
              {language === "en" && (
                <p className="subtitle-ar" dir="rtl">التعرف على الصخور والمعادن بالذكاء الاصطناعي</p>
              )}
            </header>
            <ol className="pipeline" aria-label={t(language, "pipeline_label")}>
              <li className="pipeline__item">
                <span className="pipeline__idx" aria-hidden="true">1</span>
                <div className="pipeline__body">
                  <strong>{t(language, "pipeline_1_title")}</strong>
                  {t(language, "pipeline_1_body")}
                </div>
              </li>
              <li className="pipeline__item">
                <span className="pipeline__idx" aria-hidden="true">2</span>
                <div className="pipeline__body">
                  <strong>{t(language, "pipeline_2_title")}</strong>
                  {t(language, "pipeline_2_body")}
                </div>
              </li>
              <li className="pipeline__item">
                <span className="pipeline__idx" aria-hidden="true">3</span>
                <div className="pipeline__body">
                  <strong>{t(language, "pipeline_3_title")}</strong>
                  {t(language, "pipeline_3_body")}
                </div>
              </li>
            </ol>
            <div className="btn-group">
              <div className="cta-row">
                <button type="button" className="btn-primary" onClick={() => cameraInputRef.current?.click()} aria-label={t(language, "btn_camera")}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
                  {t(language, "btn_camera")}
                </button>
                <button type="button" className="btn-secondary btn-upload" onClick={() => uploadInputRef.current?.click()} aria-label={t(language, "btn_upload")}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                  {t(language, "btn_upload")}
                </button>
              </div>
              <input ref={cameraInputRef} type="file" accept="image/*" capture="environment" style={{ display: "none" }} onChange={onPickFile} />
              <input ref={uploadInputRef} type="file" accept="image/*" style={{ display: "none" }} onChange={onPickFile} />
              <p className="hint">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M1 6v16l7-4 8 4 7-4V2l-7 4-8-4-7 4z"/><path d="M8 2v16M16 6v16"/></svg>
                {t(language, "hint_offline")}
              </p>
            </div>
          </div>
        )}

          {screen === "crop" && (
            <CropScreen
              imageUrl={capturedImage}
              onConfirm={(url, el) => runAnalysis(url, el)}
              onCancel={reset}
              lang={language}
            />
          )}

          {screen === "analyzing" && (
            <div className="screen">
              {croppedImageUrl && (
                <img src={croppedImageUrl} alt={t(language, "preview_alt")} className="preview-img" />
              )}
              <div
                className="analyze-card"
                role="status"
                aria-live="polite"
                aria-busy={modelLoading ? "true" : "false"}
              >
                <div className="spinner" aria-hidden="true" />
                <h2 className="analyze-text">
                  {modelLoading ? t(language, "analyze_loading_title") : t(language, "analyze_title")}
                </h2>
                <p className="analyze-sub">
                  {modelLoading ? t(language, "analyze_loading_sub") : t(language, "analyze_sub")}
                </p>
                <div className="progress-bar" aria-hidden="true"><div className="progress-fill" /></div>
              </div>
            </div>
          )}

          {screen === "not_rock" && (
            <div className="screen">
              {croppedImageUrl && (
                <img src={croppedImageUrl} alt="" className="preview-img" />
              )}
              <div className="not-rock-card">
                <div className="not-rock-icon" aria-hidden="true">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                </div>
                <h2 className="not-rock-title">{t(language, "not_rock_title")}</h2>
                <p className="not-rock-text">
                  {oodInfo?.isManMade ? t(language, "not_rock_manmade") : t(language, "not_rock_generic")}
                </p>
                <div className="tips-list">
                  <div className="tip-item"><span className="tip-icon" aria-hidden="true"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 3L3 21"/><path d="M21 21L3 3"/></svg></span>{t(language, "tip_closer")}</div>
                  <div className="tip-item"><span className="tip-icon" aria-hidden="true"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/></svg></span>{t(language, "tip_light")}</div>
                  <div className="tip-item"><span className="tip-icon" aria-hidden="true"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><rect x="3" y="3" width="18" height="18" rx="2"/></svg></span>{t(language, "tip_bg")}</div>
                  <div className="tip-item"><span className="tip-icon" aria-hidden="true"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></span>{t(language, "tip_surface")}</div>
                </div>
              </div>
              <button type="button" className="btn-primary" onClick={reset} aria-label={t(language, "try_again")}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
                {t(language, "try_again")}
              </button>
            </div>
          )}

          {screen === "results" && (
            <div className="screen screen--results">
              <div className="results-panel">
                <div className="results-panel__header">
                  <div className="results-panel__titles">
                    <p className="eyebrow">{t(language, "results_eyebrow")}</p>
                    <h2 className="section-title">{t(language, "results_title")}</h2>
                  </div>
                  {croppedImageUrl && (
                    <div className="thumb-frame">
                      <img src={croppedImageUrl} alt="" className="thumb-img" />
                    </div>
                  )}
                </div>
              </div>
              {analysisMeta && (
                <div className="conf-explainer">
                  <h3 className="conf-explainer__title">{t(language, "conf_expl_title")}</h3>
                  <ul className="conf-explainer__list">
                    {buildConfidenceBullets(language, analysisMeta).map((line, i) => (
                      <li key={i}>{line}</li>
                    ))}
                  </ul>
                </div>
              )}
              <div className="pred-list">
                {predictions.map((pred, i) => {
                  const m = mineralsDB[pred.key];
                  const conf = Math.round(pred.prob * 100);
                  const color = conf >= 75 ? "var(--success)" : conf >= 50 ? "var(--warning)" : "var(--danger)";
                  return (
                    <div key={i} className={`pred-card${i === 0 ? " pred-card--lead" : ""}`}>
                      <div className="pred-rank">{i + 1}</div>
                      <div className="pred-info">
                        <div className="pred-name">
                          {getMineralDisplayName(pred.key)}
                          <span className="pred-ar"> · {m?.name_arabic || ""}</span>
                        </div>
                        <div className="pred-cat">{m?.category || ""}</div>
                        <div className="conf-bar" role="progressbar" aria-valuenow={conf} aria-valuemin={0} aria-valuemax={100} aria-label={`${getMineralDisplayName(pred.key)} ${conf}%`}>
                          <div className="conf-fill" style={{ width: `${conf}%`, background: color }} />
                        </div>
                      </div>
                      <div className="conf-label" style={{ color }}>{conf}%</div>
                    </div>
                  );
                })}
              </div>
              <div className="btn-group">
                {predictions[0]?.prob >= HIGH_CONFIDENCE_THRESHOLD ? (
                  <button type="button" className="btn-primary" onClick={() => handleConfidencePath(predictions)} aria-label={`${t(language, "btn_confirm")} ${getMineralDisplayName(predictions[0]?.key)}`}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>
                    {t(language, "btn_confirm")} — {getMineralDisplayName(predictions[0]?.key)}
                  </button>
                ) : (
                  <button type="button" className="btn-primary" onClick={() => handleConfidencePath(predictions)} aria-label={t(language, "btn_qa")}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                    {t(language, "btn_qa")}
                  </button>
                )}
                <button type="button" className="btn-secondary" onClick={reset} aria-label={t(language, "scan_again")}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.5"/></svg>
                  {t(language, "scan_again")}
                </button>
              </div>
            </div>
          )}

          {screen === "qa" && (
            <div className="screen">
              <div className="qa-header">
                <div className="qa-step">{t(language, "qa_step")} {qaHistory.length + 1}</div>
                <h2 className="qa-title">{t(language, "qa_title")}</h2>
                <p className="qa-subtitle">
                  {t(language, "qa_compare")}: {currentQuestion?.candidates?.slice(0, 2).map((k) => getMineralDisplayName(k)).join(" vs ")}
                </p>
              </div>
              <div className="question-card">
                <p className="question-text">{currentQuestion?.question}</p>
              </div>
              <div className="options-list">
                {currentQuestion?.options?.map((opt, i) => (
                  <button key={i} type="button" className="option-btn" onClick={() => handleAnswer(currentQuestion.values[i])} aria-label={opt}>{opt}</button>
                ))}
              </div>
              {qaHistory.length > 0 && (
                <div className="history-area">
                  <p className="history-label">{t(language, "qa_prev")}</p>
                  {qaHistory.map((h, i) => (
                    <p key={i} className="history-item">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                      {h.answer}
                    </p>
                  ))}
                </div>
              )}
            </div>
          )}

          {screen === "final" && (() => {
            const m = finalResult || {};
            return (
              <div className="screen">
                <div className="result-badge">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true" style={{ marginRight: 5, verticalAlign: "middle" }}><polyline points="20 6 9 17 4 12"/></svg>
                  {t(language, "final_badge")}
                </div>
                <div id="print-specimen-root" className="print-specimen-root">
                  <div className="result-card">
                    <div className="result-logo">
                      <img src="/litholens-logo.png" alt="" className="result-logo-img" aria-hidden="true" />
                    </div>
                    <h1 className="result-name">{m.name_english || t(language, "unknown")}</h1>
                    <p className="result-arabic">{m.name_arabic || ""}</p>
                    {m.category && <div className="result-category">{m.category}</div>}
                    <div className="prop-grid">
                      {[
                        [<svg key="h" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>, t(language, "prop_hardness"), m.hardness_moh ? `${m.hardness_moh} Mohs` : ""],
                        [<svg key="f" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M18 8h1a4 4 0 0 1 0 8h-1"/><path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z"/><line x1="6" y1="1" x2="6" y2="4"/><line x1="10" y1="1" x2="10" y2="4"/><line x1="14" y1="1" x2="14" y2="4"/></svg>, t(language, "prop_field"), m.hardness_testable || ""],
                        [<svg key="l" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>, t(language, "prop_luster"), m.luster || ""],
                        [<svg key="s" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>, t(language, "prop_streak"), m.streak_color || ""],
                        [<svg key="cl" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/></svg>, t(language, "prop_cleavage"), m.cleavage || ""],
                        [<svg key="a" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v11m0 0H5m4 0h10m0-11v11m0 0H5"/></svg>, t(language, "prop_acid"), m.acid_reaction || ""],
                      ].filter(([, , v]) => v).map(([icon, label, value], i) => (
                        <div key={i} className="prop-item">
                          <div className="prop-label">{icon} {label}</div>
                          <div className="prop-value">{value}</div>
                        </div>
                      ))}
                    </div>
                    {m.special_property && <div className="special-prop">
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true" style={{ marginRight: 5, verticalAlign: "middle" }}><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
                      {m.special_property}
                    </div>}
                    {m.description_for_ai && <p className="description">{m.description_for_ai}</p>}
                    {m.common_locations && <p className="locations">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true" style={{ marginRight: 4, verticalAlign: "middle" }}><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
                      {m.common_locations}
                    </p>}
                  </div>
                </div>
                <div className="export-row">
                  <button type="button" className="btn-secondary" onClick={() => downloadResultCard()}>{t(language, "export_download")}</button>
                  <button type="button" className="btn-secondary" onClick={() => shareResultCard()}>{t(language, "export_share")}</button>
                  <button type="button" className="btn-secondary" onClick={printResultCard}>{t(language, "export_print")}</button>
                </div>
                <button type="button" className="btn-primary" onClick={reset} aria-label={t(language, "final_another")}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.5"/></svg>
                  {t(language, "final_another")}
                </button>
              </div>
            );
          })()}
      </main>
    </div>
    );
  }
