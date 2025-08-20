import os
from flask import Flask, request, jsonify, render_template, make_response
from nlp_pipeline import IntentClassifier, TransformerResponder, new_session_id

app = Flask(__name__)

# Initialize NLP components
classifier = IntentClassifier(intents_path="data/intents.json", threshold=0.45)
classifier.train()
transformer = TransformerResponder()

@app.route("/", methods=["GET"])
def index():
    resp = make_response(render_template("index.html"))
    if not request.cookies.get("session_id"):
        resp.set_cookie("session_id", new_session_id(), httponly=True, samesite="Lax")
    return resp

@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    tag, conf = classifier.predict(user_text)
    if conf >= classifier.threshold and tag != 'fallback':
        reply = classifier.get_response(tag)
        source = f"intent:{tag} ({conf:.2f})"
    else:
        # Fallback to Transformers if enabled
        session_id = request.cookies.get("session_id") or new_session_id()
        reply = transformer.reply(session_id, user_text) if transformer.enabled else ""
        if not reply:
            reply = classifier.get_response('fallback')
            source = f"fallback:intents ({conf:.2f})"
        else:
            source = "fallback:transformer"

    return jsonify({"ok": True, "reply": reply, "source": source})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
