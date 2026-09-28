from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from groq import Groq

app = Flask(__name__)
MODEL_NAME = "openai/gpt-oss-20b"

load_dotenv()
api_key = os.getenv("GROQAI_api_key") or os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key) if api_key else None

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    if not client:
        return jsonify({"error": "Groq API key is not configured. Please set GROQAI_api_key in your environment variables."}), 500

    question = (request.form.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please provide a question."}), 400

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Act like a helpful personal assistant"},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=512
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({"response": answer}), 200
    except Exception as exc:
        return jsonify({"error": f"AI request failed: {exc}"}), 500

@app.route("/summarize", methods=["POST"])
def summarize():
    if not client:
        return jsonify({"error": "Groq API key is not configured. Please set GROQAI_api_key in your environment variables."}), 500

    email_text = (request.form.get("email") or "").strip()
    if not email_text:
        return jsonify({"error": "Please provide email text to summarize."}), 400

    prompt = f"summarize the following email in 2-3 sentences: {email_text}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Act like an expert email assistant"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=512
        )
        summary = response.choices[0].message.content.strip()
        return jsonify({"response": summary}), 200
    except Exception as exc:
        return jsonify({"error": f"AI request failed: {exc}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

