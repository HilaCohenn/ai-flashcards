# AI Flashcard Generator (Anki-ready)

### Quickstart
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt

# Replace in .env with your key
# OPENAI_API_KEY=sk-...

# Generate 20 Hebrew cards from a PDF, with PII redaction
python flashcard_generator.py --input notes.pdf --lang he --count 20 --model gpt-3.5-turbo --redact
