# Chatbot with NLP (Python + Flask + NLTK/Transformers)

A production-style starter project for a web chatbot using Flask on the backend and optional NLP through:
- **NLTK + TF‑IDF + LinearSVC** for lightweight intent classification + templated replies
- **🤗 Transformers (DialoGPT-small)** for generative fallback replies (optional; can be toggled)

## Features
- Clean Flask API: `GET /` for UI, `POST /api/chat` for messages
- Simple HTML/CSS front-end with fetch-based chat interface
- NLTK preprocessing (tokenization, lemmatization, stopword removal)
- Intent dataset in `data/intents.json` (easy to extend)
- Trains a small TF‑IDF + LinearSVC model at startup (fast)
- Confidence threshold + fallback to Transformers (optional)
- Per-session conversation memory (cookie-based session id)

## Quickstart

### 1) Create & activate venv (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

> Note: First run may download NLTK data and the DialoGPT model if Transformers fallback is enabled.

### 3) Run
```bash
# Option A: Classic Flask dev server
python app.py

# Option B: With gunicorn (prod-ish)
gunicorn -w 2 -k gthread -t 120 -b 0.0.0.0:8000 app:app
```

Then open http://127.0.0.1:5000/ (or 8000 if using gunicorn).

### 4) Toggle Transformers (optional)
By default, the bot uses only NLTK + TF‑IDF. To enable the generative fallback (DialoGPT-small):
```bash
export USE_TRANSFORMERS=1   # Windows (PowerShell): $env:USE_TRANSFORMERS = "1"
python app.py
```

## Project structure
```
flask_nlp_chatbot/
├── app.py
├── nlp_pipeline.py
├── requirements.txt
├── README.md
├── data/
│   └── intents.json
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Extending
- Add new intents/utterances in `data/intents.json`; restart to retrain.
- Replace DialoGPT with any other HF model in `nlp_pipeline.py` if desired.
- Convert the simple UI into a SPA or integrate with your app easily (API-first).

## License
MIT
