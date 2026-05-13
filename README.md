# LithoLens

**Browser-based, on-device identification of rocks and minerals** from a single photo. LithoLens ranks likely species with a small neural classifier, optionally narrows candidates through short field-style questions grounded in a local mineral database, and presents results in a structured specimen view. After the app and models have been **cached from an initial load**, you can continue identifying specimens **without a network connection**—a practical fit for **field work** where signal is spotty or unavailable.

> **Status:** **Hackathon MVP**—built for **team development**, **demo**, and **judge review**, not as a certified analytical instrument. Identifications are **probabilistic** and intended for **education and field exploration**, not laboratory-grade mineralogy.  
> **Distribution:** This repository is **not open source**. Source, assets, and branding are **proprietary** and shared only among the **team** and **judges** unless the authors explicitly publish otherwise.

---

## Highlights

| Area | What you get today |
|------|---------------------|
| **Field / offline** | Classification runs **in the browser** with **ONNX Runtime Web** (WASM)—no live server round-trip for inference. After models and scripts are cached, you can keep identifying specimens **with no cellular or Wi‑Fi connection**, which matters on remote transects and in dead zones. |
| **Workflow** | Separate **camera** and **gallery / file** entry points, optional **crop**, ranked predictions, and **Q&A** when top confidence is ambiguous. |
| **Rejection** | Heuristic handling for **non-specimen** or **unlikely geological** inputs when auxiliary models and metadata are present. |
| **Localization** | **English / Arabic** UI strings; mineral names follow `minerals_db.json` where Arabic labels exist. |
| **Experience** | **Dark and light** themes, **local specimen history** (IndexedDB), **confidence notes**, and **PNG export / share / print** for the final card. |

---

## How it works (conceptual)

1. The user provides an image; the app may crop to the specimen region.
2. A **mineral classifier** (`litholens_model.onnx`) produces a probability distribution over classes defined in `class_names.json`.
3. If scores suggest a plausible mineral and pass basic sanity checks, the UI shows **top predictions** with confidence context.
4. If discrimination is needed, **Q&A** uses properties in `minerals_db.json` (e.g. hardness, luster, streak, acid reaction) to filter candidates.
5. A **final specimen sheet** summarizes key properties; the user may save a history entry or export a card locally.

Auxiliary **MobileNet v2** weights and ImageNet-derived metadata, when available, support **out-of-distribution** cues; they are optional and degrade gracefully if files are missing.

---

## Tech stack

| Layer | Choice |
|--------|--------|
| UI | React 19, Vite 8 |
| Inference | [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (WASM) |
| Styling | CSS design tokens (`index.css`) and component layout (`App.css`) |

---

## Repository layout

```text
public/
  litholens-logo.png
  model/
    litholens_model.onnx      # Main classifier (required for real predictions)
    class_names.json          # Class list + not_mineral index
    minerals_db.json          # Per-mineral facts for UI + Q&A
    mineral_arabic_names.json # Optional glossary for CSV → JSON tooling
    mobilenetv2-7.onnx        # Optional OOD / man-made helper
    imagenet_*.json           # Optional MobileNet metadata
src/
  App.jsx                     # Screens, ONNX sessions, crop canvas, Q&A flow
  App.css
  index.css
  main.jsx
  i18n/strings.js             # UI copy (EN / AR)
  lib/
    specimenHistory.js       # IndexedDB history helpers
    exportCard.js              # PNG specimen card rendering
    confidenceExplain.js       # Human-readable confidence bullets
scripts/
  csv_to_minerals_db.py       # UTF-8 CSV → minerals_db.json (preserves Arabic)
```

---

## Getting started

**Requirements:** Node.js 20+ (or an LTS version compatible with Vite 8) and npm.

```bash
npm install
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`).

**Production build**

```bash
npm run build
npm run preview   # optional: verify dist/ locally
```

**Lint**

```bash
npm run lint
```

---

## Models and data

Place ONNX and JSON assets under `public/model/` so they are served from `/model/...` at runtime.

| File | Role |
|------|------|
| `class_names.json` | Must align with the output dimension of `litholens_model.onnx`. Includes a **not-mineral** (or equivalent) slot used in rejection logic. |
| `minerals_db.json` | Object keyed by **the same classifier labels** (except synthetic keys). Drives display names, Arabic names where present, categories, and Q&A property matching. |

**Rebuilding `minerals_db.json` from CSV** (UTF-8 export, e.g. “CSV UTF-8” from Excel):

```bash
python3 scripts/csv_to_minerals_db.py \
  --csv "path/to/your_minerals.csv" \
  --merge-db public/model/minerals_db.json \
  --out public/model/minerals_db.json
```

Column expectations and legacy key mapping (e.g. `credit` ← `creedit`) are documented in `scripts/csv_to_minerals_db.py`.

---

## ONNX Runtime Web (operational notes)

- The Vite build splits **vendor** and **ort** chunks; WASM binaries load at runtime and dominate first-load size.
- The first inference may **warm up** the session; the UI surfaces a loading state during model preparation.
- For **offline** demos after first load, ship the full `dist/` output together with `public/model/` and ensure the host serves `.wasm` and `.onnx` with sensible caching headers if repeat visits matter.

---

## Client preferences (MVP)

| Key | Purpose |
|-----|---------|
| `litholens-lang` | `en` or `ar` for UI language. |
| `litholens-theme` | `dark` or `light` for color theme. |

---

## Future work

The items below are **internal roadmap ideas** for the team—not commitments or a public feature backlog.

- **Progressive Web App** — `manifest.json`, installability, and cache-first static assets (and optionally cached models) after first successful load.
- **Richer history** — Open a saved identification from history, attach user notes, and export or compare entries.
- **Themed export** — Match PNG specimen cards to the active light/dark theme (export is currently a fixed dark-branded layout).
- **Trust and safety copy** — Short, prominent disclaimer that the tool is educational; optional “clear cached data” control for shared devices.
- **Deeper accessibility** — Focus management and traps in dialogs, broader `aria-*` coverage, and audited keyboard paths through crop and Q&A.
- **Comparison mode** — Side-by-side property tables for two minerals or two saved runs.
- **User feedback loop** — Optional, consent-based correction capture (“actual mineral was X”) for future dataset or model improvements.
- **Engineering hardening** — Vitest / Playwright smoke tests, error boundaries around WASM session creation, and documented training → ONNX export steps (opset, input name, resolution, normalization).
- **Desktop polish** — Drag-and-drop import, keyboard shortcuts, and adaptive layout for large crop canvases.

If the product later ships more broadly, prioritize **in-browser inference** and **offline-friendly deployment** unless requirements change.

---

## License and distribution

**All rights reserved.** LithoLens (including this source tree, bundled or referenced model artifacts where applicable, and branding) is **proprietary** and **not licensed for public use, redistribution, or modification**. Access is limited to the **authoring team** and **hackathon judges** for evaluation unless a separate written agreement says otherwise.

A `LICENSE` file may be added later for **internal or sponsor paperwork**; unless that file explicitly grants open-source terms, nothing in this repository should be read as permission to copy or republish the work.

---

## Acknowledgments

Built with [Vite](https://vitejs.dev/), [React](https://react.dev/), and [ONNX Runtime Web](https://onnxruntime.ai/).
