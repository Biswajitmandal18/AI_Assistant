import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env if present
load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"

st.set_page_config(
    page_title="AI Personal Assistant",
    page_icon="🤖",
    layout="centered"
)

# Custom header
st.title("🤖 AI Personal Assistant")
st.markdown("Powered by Groq & LLaMA / OSS models")

# Retrieve API key from Streamlit secrets, environment variables, or sidebar input
api_key = None
if "GROQAI_api_key" in st.secrets:
    api_key = st.secrets["GROQAI_api_key"]
elif "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = os.getenv("GROQAI_api_key") or os.getenv("GROQ_API_KEY")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown(f"**Model:** `{MODEL_NAME}`")
    
    if not api_key:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Enter your Groq API key or set GROQAI_api_key in Streamlit secrets / .env"
        )
    else:
        st.success("API Key detected")

# Initialize Groq client
client = None
if api_key:
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        st.sidebar.error(f"Error initializing Groq client: {e}")

# Main Tabs
tab1, tab2 = st.tabs(["💬 Ask Anything", "✉️ Summarize Email"])

with tab1:
    st.subheader("Ask Anything")
    question = st.text_input("Enter your question:", placeholder="e.g. Explain quantum computing in simple terms")
    ask_button = st.button("Ask Assistant", key="ask_btn", type="primary")

    if ask_button:
        if not api_key or not client:
            st.error("Please configure your Groq API key in the sidebar or environment.")
        elif not question.strip():
            st.warning("Please enter a question before submitting.")
        else:
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "Act like a helpful personal assistant"},
                            {"role": "user", "content": question.strip()}
                        ],
                        temperature=0.7,
                        max_tokens=512
                    )
                    answer = response.choices[0].message.content.strip()
                    st.success("Response:")
                    st.markdown(answer)
                except Exception as exc:
                    st.error(f"AI request failed: {exc}")

with tab2:
    st.subheader("Summarize Email")
    email_text = st.text_area("Paste your email content here:", height=180, placeholder="Paste email content here...")
    summarize_button = st.button("Summarize Email", key="sum_btn", type="primary")

    if summarize_button:
        if not api_key or not client:
            st.error("Please configure your Groq API key in the sidebar or environment.")
        elif not email_text.strip():
            st.warning("Please paste email content before submitting.")
        else:
            with st.spinner("Summarizing..."):
                try:
                    prompt = f"summarize the following email in 2-3 sentences: {email_text.strip()}"
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
                    st.success("Summary:")
                    st.markdown(summary)
                except Exception as exc:
                    st.error(f"AI request failed: {exc}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 14px;'>Made By Biswajit</p>", unsafe_allow_html=True)

