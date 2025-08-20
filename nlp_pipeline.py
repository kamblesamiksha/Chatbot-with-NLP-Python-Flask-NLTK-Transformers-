import os
import re
import random
import json
import uuid
from dataclasses import dataclass
from typing import List, Dict, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Try to ensure NLTK data is present (safe if already downloaded)
def _ensure_nltk_data():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

_ensure_nltk_data()

lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    tokens = [t for t in s.split() if t not in STOPWORDS]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)

@dataclass
class Intent:
    tag: str
    patterns: List[str]
    responses: List[str]

class IntentClassifier:
    def __init__(self, intents_path: str, threshold: float = 0.45):
        self.intents_path = intents_path
        self.threshold = threshold
        self.intents: List[Intent] = []
        self.pipeline: Pipeline = None
        self._tag_to_responses: Dict[str, List[str]] = {}

    def load_intents(self):
        with open(self.intents_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.intents = [Intent(**it) for it in data.get('intents', [])]
        self._tag_to_responses = {it.tag: it.responses for it in self.intents}

    def train(self):
        self.load_intents()
        X, y = [], []
        for it in self.intents:
            for p in it.patterns:
                X.append(normalize_text(p))
                y.append(it.tag)
        if not X:
            raise ValueError("No training data found in intents.json")
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=1)),
            ('clf', LinearSVC())
        ])
        self.pipeline.fit(X, y)

    def predict(self, user_text: str) -> Tuple[str, float]:
        norm = normalize_text(user_text)
        # LinearSVC doesn't expose probabilities; use decision function magnitude heuristic
        dec = self.pipeline.decision_function([norm])
        if dec.ndim == 1:  # binary
            scores = [dec[0], -dec[0]]
        else:
            scores = dec[0]
        labels = self.pipeline.classes_
        max_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        top_label = labels[max_idx]
        # Scale margin to [0,1] via sigmoid-ish heuristic
        import math
        margin = float(scores[max_idx])
        confidence = 1.0 / (1.0 + math.exp(-margin / 2.0))
        return top_label, confidence

    def get_response(self, tag: str) -> str:
        resps = self._tag_to_responses.get(tag) or self._tag_to_responses.get('fallback', ["I'm not sure I understood."])
        return random.choice(resps)

# Optional: Transformers fallback (DialoGPT)
class TransformerResponder:
    def __init__(self):
        self.enabled = os.getenv('USE_TRANSFORMERS', '0') == '1'
        self.generator = None
        self.chat_history_ids = {}

        if self.enabled:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model_name = os.getenv('TRANSFORMER_MODEL', 'microsoft/DialoGPT-small')
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
            except Exception as e:
                print(f"[WARN] Failed to load transformers model: {e}. Disabling fallback.")
                self.enabled = False

    def reply(self, session_id: str, user_text: str) -> str:
        if not self.enabled:
            return ""
        try:
            from transformers import AutoTokenizer
            tokenizer = self.tokenizer
            model = self.model

            input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors='pt')
            prev = self.chat_history_ids.get(session_id)
            if prev is not None:
                bot_input_ids = torch.cat([prev, input_ids], dim=-1)
            else:
                bot_input_ids = input_ids

            import torch
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=200,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            self.chat_history_ids[session_id] = chat_history_ids
            response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            return ""

def new_session_id() -> str:
    return uuid.uuid4().hex
