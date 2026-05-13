import { t } from "../i18n/strings.js";

/**
 * @param {import('../i18n/strings.js').Lang} lang
 * @param {{ normEntropy: number, gap: number, top1Prob: number }} m
 * @returns {string[]}
 */
export function buildConfidenceBullets(lang, m) {
  const lines = [];
  const { normEntropy, gap, top1Prob } = m;

  if (normEntropy >= 0.55) lines.push(t(lang, "conf_expl_entropy_high"));
  else lines.push(t(lang, "conf_expl_entropy_low"));

  if (gap < 0.08) lines.push(t(lang, "conf_expl_gap_tight"));
  else lines.push(t(lang, "conf_expl_gap_wide"));

  if (top1Prob < 0.55) lines.push(t(lang, "conf_expl_conf_low"));
  else lines.push(t(lang, "conf_expl_conf_high"));

  lines.push(t(lang, "conf_expl_tip"));
  return lines;
}
