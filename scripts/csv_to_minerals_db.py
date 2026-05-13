#!/usr/bin/env python3
"""
Convert minerals CSV → public/model/minerals_db.json with UTF-8 Arabic preserved.

Usage:
  python3 scripts/csv_to_minerals_db.py \\
    --csv "minerals_template(minerals_template).csv" \\
    --class-names public/model/class_names.json \\
    --arabic-names public/model/mineral_arabic_names.json \\
    --merge-db public/model/minerals_db.json \\
    --out public/model/minerals_db.json

Encoding: always read/write UTF-8 (use utf-8-sig if Excel added BOM).

If your CSV has correct Arabic but Excel broke it, re-export as:
  “CSV UTF-8 (Comma delimited) (*.csv)”
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# Old DB keys that differ from class_names.json labels
LEGACY_DB_KEY = {
    "credit": "creedit",
    "almandine": "almandine garnet",
    "grossular": "grossular garnet",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sniff_encoding(path: Path) -> str:
    bom = path.read_bytes()[:4]
    if bom.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def load_csv_rows(path: Path) -> dict[str, dict[str, str]]:
    enc = sniff_encoding(path)
    text = path.read_text(encoding=enc, errors="replace")
    reader = csv.DictReader(text.splitlines())
    out: dict[str, dict[str, str]] = {}
    for row in reader:
        label = (row.get("label") or row.get("Label") or "").strip()
        if not label:
            continue
        out[label] = {k: (v or "").strip() if isinstance(v, str) else v for k, v in row.items()}
    return out


def is_placeholder_arabic(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return True
    if "?" in s and re.fullmatch(r"[\s\?,]+", s.replace("?", "")) is None:
        # mostly question marks = mojibake placeholder
        if s.count("?") >= max(3, len(s) // 3):
            return True
    return False


def acid_sentence(ar: str) -> str:
    t = (ar or "").strip().lower()
    if not t:
        return "لم يُذكر تفاعل واضح مع الأحماض المخففة في البيانات."
    if t in ("no", "none", "لا", "0"):
        return "لا يظهر عادةً تفاعلاً ملحوظاً مع حمض الهيدروكلوريك المخفف."
    if "yes" in t or "fizz" in t or "نعم" in t or "تفاعل" in t:
        return "قد يظهر تفاعلاً مع الأحماض المخففة حسب التركيب الكيميائي."
    return f"ملاحظات التحميض: {ar.strip()}."


def build_description_ar(
    name_ar: str,
    name_en: str,
    fields: dict[str, str],
) -> str:
    parts: list[str] = [
        f"هذا معدن {name_ar} ({name_en}).",
    ]
    hm = fields.get("hardness_moh") or fields.get("hardness_mohs")
    if hm:
        parts.append(f"صلادة موس تقريباً: {hm}.")
    lu = fields.get("luster")
    if lu:
        parts.append(f"اللمعان: {lu.strip()}.")
    st = fields.get("streak_color")
    if st:
        parts.append(f"لون الخط: {st.strip()}.")
    cl = fields.get("cleavage")
    if cl:
        parts.append(f"الانفصام: {cl.strip()}.")
    fr = fields.get("fracture")
    if fr:
        parts.append(f"الكسر: {fr.strip()}.")
    parts.append(acid_sentence(fields.get("acid_reaction", "")))
    loc = fields.get("common_locations")
    if loc:
        parts.append(f"أماكن شائعة: {loc.strip()}.")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--class-names", type=Path, default=Path("public/model/class_names.json"))
    ap.add_argument("--arabic-names", type=Path, default=Path("public/model/mineral_arabic_names.json"))
    ap.add_argument("--merge-db", type=Path, default=None, help="Existing minerals_db.json for fallback fields")
    ap.add_argument("--out", type=Path, default=Path("public/model/minerals_db.json"))
    args = ap.parse_args()

    classes = read_json(args.class_names)["classes"]
    labels = [c for c in classes if c != "not_mineral"]

    arabic_names = read_json(args.arabic_names)
    old_db = read_json(args.merge_db) if args.merge_db and args.merge_db.exists() else {}

    csv_by_label = load_csv_rows(args.csv)

    out: dict[str, dict[str, str]] = {}
    for label in labels:
        row = dict(csv_by_label.get(label, {}))
        legacy = LEGACY_DB_KEY.get(label, label)
        old = dict(old_db.get(legacy) or old_db.get(label) or {})

        name_en = (row.get("name_english") or old.get("name_english") or label.replace("_", " ").title()).strip()

        name_ar = (arabic_names.get(label) or "").strip()
        csv_ar = (row.get("name_arabic") or "").strip()
        if name_ar and is_placeholder_arabic(name_ar):
            name_ar = ""
        if not is_placeholder_arabic(csv_ar):
            name_ar = csv_ar or name_ar
        if not name_ar:
            name_ar = name_en

        category = (row.get("category") or old.get("category") or "Mineral").strip()
        if category:
            category = category[0].upper() + category[1:]

        merged = {
            "name_english": name_en,
            "name_arabic": name_ar,
            "category": category,
            "hardness_moh": (row.get("hardness_moh") or row.get("hardness_mohs") or old.get("hardness_moh") or "").strip(),
            "hardness_testable": (row.get("hardness_testable") or old.get("hardness_testable") or "").strip(),
            "luster": (row.get("luster") or old.get("luster") or "").strip(),
            "streak_color": (row.get("streak_color") or old.get("streak_color") or "").strip(),
            "cleavage": (row.get("cleavage") or old.get("cleavage") or "").strip(),
            "fracture": (row.get("fracture") or old.get("fracture") or "").strip(),
            "acid_reaction": (row.get("acid_reaction") or old.get("acid_reaction") or "").strip(),
            "special_property": (row.get("special_property") or old.get("special_property") or "").strip(),
            "common_locations": (row.get("common_locations") or old.get("common_locations") or "").strip(),
            "primary_color": (row.get("primary_color") or old.get("primary_color") or "").strip(),
        }

        merged["description_for_ai"] = build_description_ar(name_ar, name_en, merged)

        out[label] = merged

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(out)} entries → {args.out}")


if __name__ == "__main__":
    main()
