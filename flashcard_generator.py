#!/usr/bin/env python3
"""
AI Flashcard Generator (Anki-ready)
----------------------------------
Converts study notes or PDFs into concise Q/A flashcards and exports to CSV for Anki import.
- Input: .txt or .pdf
- Optional PII redaction before sending to the LLM
- Uses OpenAI Chat Completions to generate Q/A pairs in Hebrew or English
- Exports: flashcards.csv (Front,Back) and flashcards.md (preview)
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from pypdf import PdfReader

from openai import OpenAI

# ----------------------------
# Helpers: I/O and redaction
# ----------------------------
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3}[-.\s]?\d{4,6}")
URL_RE   = re.compile(r"https?://\S+|www\.\S+")

def read_text(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n\n".join(pages)
    raise ValueError("Only .txt and .pdf are supported for now.")

def redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = URL_RE.sub("<URL>", text)
    def _phone_sub(m):
        s = m.group(0)
        digits = re.sub(r"\D", "", s)
        return "<PHONE>" if len(digits) >= 7 else s
    return PHONE_RE.sub(_phone_sub, text)

# ----------------------------
# LLM call
# ----------------------------
def make_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Create .env with your key.")
    return OpenAI(api_key=api_key)

def build_prompt(lang: str, topic_text: str, count: int) -> str:
    lang_name = "Hebrew" if lang.lower().startswith("he") else "English"
    return f"""
You are a teaching assistant generating compact flashcards in {lang_name}.
Create exactly {count} Q/A cards from the study material below.

Rules:
- Keep questions short and specific. Avoid vagueness.
- Answers must be concise (1–3 lines). No fluff.
- Prefer factual recall, definitions, and key comparisons.
- Use plain text only. No markdown in answers.
- If the text lacks info for a card, skip it—do NOT invent facts.

Output JSON with this exact shape:
[
  {{"q": "...", "a": "..."}},
  ...
]

STUDY MATERIAL:
\"\"\"{topic_text}\"\"\"
"""

def generate_cards(client: OpenAI, lang: str, text: str, count: int, model: str, temperature: float) -> List[Dict[str,str]]:
    prompt = build_prompt(lang, text, count)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You generate high-quality Q/A flashcards for efficient studying."},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content.strip()
    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1:
        raise RuntimeError("Model did not return JSON array. Response was:\n" + content)
    data = json.loads(content[start:end+1])
    cards = []
    for item in data:
        q = (item.get("q") or "").strip()
        a = (item.get("a") or "").strip()
        if q and a:
            cards.append({"q": q, "a": a})
    if not cards:
        raise RuntimeError("No valid cards parsed from model output.")
    return cards[:count]

# ----------------------------
# Export
# ----------------------------
def save_csv(cards: List[Dict[str,str]], path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Front","Back"])
        for c in cards:
            w.writerow([c["q"], c["a"]])

def save_md(cards: List[Dict[str,str]], path: Path):
    lines = ["# Flashcards Preview", ""]
    for i,c in enumerate(cards, 1):
        lines.append(f"**{i}. Q:** {c['q']}")
        lines.append(f"**A:** {c['a']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate Anki-ready flashcards from text/PDF.")
    ap.add_argument("--input", required=True, help="Path to .txt or .pdf")
    ap.add_argument("--lang", choices=["he","en"], default="he", help="Language of cards")
    ap.add_argument("--count", type=int, default=20, help="Number of cards to generate")
    ap.add_argument("--model", default="gpt-3.5-turbo", help="LLM model")
    ap.add_argument("--redact", action="store_true", help="Redact PII before sending to LLM")
    ap.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    ap.add_argument("--outdir", default="flashcards_out", help="Output directory")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_text(in_path)
    if args.redact:
        raw = redact_pii(raw)

    client = make_client()
    cards = generate_cards(client, args.lang, raw, args.count, args.model, args.temperature)

    csv_path = out_dir / "flashcards.csv"
    md_path  = out_dir / "flashcards.md"
    save_csv(cards, csv_path)
    save_md(cards, md_path)

    print("✅ Done!")
    print("CSV:", csv_path)
    print("MD :", md_path)
    print(f"Cards generated: {len(cards)}")

if __name__ == "__main__":
    main()
