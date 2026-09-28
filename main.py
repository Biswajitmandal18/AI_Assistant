from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

app = Flask(__name__)
MODEL_NAME = "openai/gpt-oss-20b"

load_dotenv()
api_key = os.getenv("GROQAI_api_key")

client = Groq(api_key=api_key)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")

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
    email_text = request.form.get("email")
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
    app.run(debug=True)
