/**
 * @param {object} opts
 * @param {string} opts.titleEn
 * @param {string} opts.titleAr
 * @param {string} [opts.category]
 * @param {string} [opts.thumbDataUrl]
 * @returns {Promise<Blob>}
 */
export function renderSpecimenCardPng({ titleEn, titleAr, category, thumbDataUrl }) {
  return new Promise((resolve) => {
    const W = 720;
    const H = 900;
    const canvas = document.createElement("canvas");
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      resolve(new Blob());
      return;
    }

    const g = ctx.createLinearGradient(0, 0, W, H);
    g.addColorStop(0, "#0d111a");
    g.addColorStop(1, "#05070d");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "rgba(46,196,182,0.35)";
    ctx.lineWidth = 2;
    ctx.strokeRect(24, 24, W - 48, H - 48);

    let y = 72;

    const drawTextBlock = () => {
      ctx.fillStyle = "#eef1f7";
      ctx.font = "600 34px Outfit, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(titleEn, W / 2, y);
      y += 48;
      if (titleAr) {
        ctx.font = "500 28px IBM Plex Sans, system-ui, sans-serif";
        ctx.fillStyle = "#2ec4b6";
        ctx.fillText(titleAr, W / 2, y);
        y += 44;
      }
      if (category) {
        ctx.font = "14px IBM Plex Sans, system-ui, sans-serif";
        ctx.fillStyle = "#8b7cf6";
        ctx.fillText(category, W / 2, y);
        y += 36;
      }
      ctx.font = "13px IBM Plex Sans, system-ui, sans-serif";
      ctx.fillStyle = "#7d8699";
      ctx.fillText("LithoLens · on-device identification", W / 2, H - 80);
    };

    const finish = () => {
      canvas.toBlob((blob) => resolve(blob || new Blob()), "image/png");
    };

    if (thumbDataUrl) {
      const im = new Image();
      im.onload = () => {
        const tw = 200;
        const th = 200;
        const tx = (W - tw) / 2;
        ctx.save();
        ctx.beginPath();
        ctx.rect(tx, y, tw, th);
        ctx.clip();
        ctx.drawImage(im, tx, y, tw, th);
        ctx.restore();
        ctx.strokeStyle = "rgba(255,255,255,0.12)";
        ctx.lineWidth = 1;
        ctx.strokeRect(tx, y, tw, th);
        y += th + 40;
        drawTextBlock();
        finish();
      };
      im.onerror = () => {
        drawTextBlock();
        finish();
      };
      im.src = thumbDataUrl;
    } else {
      drawTextBlock();
      finish();
    }
  });
}
