const DB_NAME = "litholens-specimen-history";
const STORE = "specimens";
const VERSION = 1;

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, VERSION);
    req.onerror = () => reject(req.error);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const os = db.createObjectStore(STORE, { keyPath: "id" });
        os.createIndex("createdAt", "createdAt", { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
  });
}

/**
 * @param {string} imageUrl object URL or http URL
 * @param {number} maxEdge
 * @returns {Promise<string>} data URL (jpeg)
 */
export function resizeImageToDataUrl(imageUrl, maxEdge = 220) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      const scale = Math.min(1, maxEdge / Math.max(w, h));
      const cw = Math.round(w * scale);
      const ch = Math.round(h * scale);
      const canvas = document.createElement("canvas");
      canvas.width = cw;
      canvas.height = ch;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("no canvas"));
        return;
      }
      ctx.drawImage(img, 0, 0, cw, ch);
      resolve(canvas.toDataURL("image/jpeg", 0.82));
    };
    img.onerror = () => reject(new Error("image load failed"));
    img.src = imageUrl;
  });
}

/**
 * @param {object} rec
 * @param {string} rec.id
 * @param {number} rec.createdAt
 * @param {string} rec.thumbDataUrl
 * @param {string} rec.nameEn
 * @param {string} rec.nameAr
 * @param {string} rec.mineralKey
 * @param {number} [rec.topProb]
 */
export async function saveSpecimenRecord(rec) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.objectStore(STORE).put(rec);
  });
}

/** @returns {Promise<Array<{id:string,createdAt:number,thumbDataUrl:string,nameEn:string,nameAr:string,mineralKey:string,topProb?:number}>>} */
export async function listSpecimens() {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).getAll();
    req.onsuccess = () => {
      const rows = req.result || [];
      rows.sort((a, b) => b.createdAt - a.createdAt);
      resolve(rows);
    };
    req.onerror = () => reject(req.error);
  });
}

export async function deleteSpecimenRecord(id) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.objectStore(STORE).delete(id);
  });
}

export async function clearAllSpecimens() {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.objectStore(STORE).clear();
  });
}
