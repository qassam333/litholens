# 🪨 LithoLens

**Identify rocks and minerals instantly from a photo — entirely in your browser, with no server required.**

LithoLens uses a custom-trained neural network (exported to ONNX) that runs **on-device via ONNX Runtime Web (WASM)**. Upload or photograph a specimen and get ranked predictions with confidence scores. When results are ambiguous, an interactive Q&A narrows down candidates using real mineralogical properties (hardness, luster, streak, acid reaction). Everything — inference, history, export — works **offline after the first load**, making it practical for field use where connectivity is unreliable.

> **Hackathon MVP** — built for demo and judge review. Identifications are probabilistic and intended for education and field exploration, not laboratory-grade mineralogy.

---

## 🚀 Live Demo

https://litholens.vercel.app/

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **On-device AI inference** | Custom mineral classifier runs fully in the browser via ONNX Runtime Web — no API calls, no backend |
| **Offline-capable** | After the first load, the app and models are cached; you can identify specimens with no internet connection |
| **Smart Q&A disambiguation** | When top predictions are close, the app asks short field-style questions to narrow candidates |
| **Specimen history** | Past identifications are saved locally (IndexedDB) and exportable as PNG cards |
| **English / Arabic UI** | Full localization with Arabic mineral names where available |
| **Dark & light themes** | Adapts to user preference |

---

## 🧠 How It Works

1. User provides an image (camera or file upload); optional crop focuses on the specimen
2. **`litholens_model.onnx`** — a custom-trained ONNX classifier — produces a probability distribution over mineral classes defined in `class_names.json`
3. Top predictions are shown with confidence context and explanatory notes
4. If the top scores are too close to call, **Q&A mode** filters candidates using properties from `minerals_db.json`
5. A final **specimen card** summarizes key properties; the user can save it to history or export as PNG

---

## 🛠️ Tech Stack

| Layer | Choice |
|---|---|
| UI | React 19 + Vite |
| AI Inference | [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (WASM) — runs the custom-trained model in-browser |
| Styling | CSS design tokens + component CSS |
| Storage | IndexedDB (specimen history) |
| Localization | Custom i18n (English / Arabic) |

---

## 📁 Repository Structure

```
public/
  model/
    litholens_model.onnx        # Main mineral classifier (required)
    class_names.json            # Class labels + not_mineral index
    minerals_db.json            # Per-mineral properties for display & Q&A
    mineral_arabic_names.json   # Arabic name glossary
    mobilenetv2-7.onnx          # Optional: out-of-distribution helper
src/
  App.jsx                       # Main app: screens, ONNX sessions, Q&A flow
  App.css / index.css
  main.jsx
  i18n/strings.js               # UI copy (EN / AR)
  lib/
    specimenHistory.js          # IndexedDB history helpers
    exportCard.js               # PNG specimen card renderer
    confidenceExplain.js        # Human-readable confidence bullets
scripts/
  csv_to_minerals_db.py         # Converts UTF-8 CSV → minerals_db.json
```

---

## ▶️ Running Locally

**Requirements:** Node.js 20+ and npm

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Open the URL Vite prints — typically `http://localhost:5173`

```bash
# Production build
npm run build

# Preview production build locally
npm run preview

# Lint
npm run lint
```

> **Model files:** ONNX and JSON assets live under `public/model/` and are served from `/model/...` at runtime. They are included in the repo.

---

## 🏋️ Model & Data

The classifier was trained on a labeled geological dataset (`geology_dataset.xlsx`), then exported to ONNX format (`litholens_model.onnx`). Class labels are stored in `class_names.json` and must match the model's output dimension.

**Rebuilding `minerals_db.json` from CSV:**

```bash
python3 scripts/csv_to_minerals_db.py \
  --csv "path/to/your_minerals.csv" \
  --merge-db public/model/minerals_db.json \
  --out public/model/minerals_db.json
```

---

## 🙏 Acknowledgments

Built with [Vite](https://vitejs.dev/), [React](https://react.dev/), and [ONNX Runtime Web](https://onnxruntime.ai/).
