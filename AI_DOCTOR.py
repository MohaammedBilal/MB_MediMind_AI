import streamlit as st
from rag_engine import *
import cohere
import os
import time

# === Setup ===
st.set_page_config(page_title="AI Medical Chatbot", layout="centered")
st.markdown("<h1 style='text-align:center;'>📲 MB MediMind AI</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color:#38bdf8; font-size:24px;'><i>🩺 Transforming healthcare through AI-powered clarity.</i></p>", unsafe_allow_html=True)
st.markdown("""
<style>
.chat-box {
    background-color: #1e293b;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
    font-family: 'Segoe UI', sans-serif;
    overflow-y: auto;
    max-height: 600px;
}
.user-msg, .bot-msg {
    padding: 0.75rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    line-height: 1.5;
    overflow-y: auto;
    max-height: 250px;
    white-space: pre-wrap;
    word-wrap: break-word;
    display: block;
    animation: fadeIn 0.5s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.typing-dots {
    font-size: 16px;
    color: #4ade80;
    text-align: center;
    margin-top: 10px;
    animation: blink 1.5s infinite;
}
@keyframes blink {
    0%   { opacity: 0.2; }
    20%  { opacity: 1; }
    100% { opacity: 0.2; }
}
.user-msg {
    background-color: #334155;
    color: #facc15;
    font-weight: 500;
}
.bot-msg {
    background-color: #0f172a;
    color: #4ade80;
}
input[type="text"] {
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
    display: block;
}
</style>
""", unsafe_allow_html=True)

# === Cohere Setup ===
co = cohere.Client("")  

# === Load or cache components ===
@st.cache_resource
def setup():
    df, embeddings = load_preprocessed_data()
    embedder = load_models()
    index = build_index_from_embeddings(embeddings)
    return df, embedder, index

df, embedder, index = setup()

# === Session State ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "generating" not in st.session_state:
    st.session_state.generating = False

# === Chat Interface ===
st.markdown("<hr style='height:4px; background-color:#ffffff; border:none; margin:1rem 0;'>", unsafe_allow_html=True)
st.markdown("### 👨‍⚕️ Talk to our AI Doctor")
user_query = st.text_input("", placeholder="Ask your health-related question here...", label_visibility="collapsed")
submit = st.button("Ask Doctor")

if submit and user_query:
    st.session_state.generating = True
    with st.spinner("🤖 AI Doctor is typing..."):
        log_file = "rag_chat_log.csv"
        answer = generate_answer_rag(
        user_query,
        embedder,
        df,
        index,
        co,
        st.session_state.chat_history,
        log_file,
        k=3
    )
    st.session_state.generating = False

# === Display conversation ===
if st.session_state.chat_history:
    with st.container():
        for turn in reversed(st.session_state.chat_history):
            st.markdown(f"<div class='user-msg'>🧑‍💬\n{turn['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='bot-msg'>🤖\n{turn['bot']}</div>", unsafe_allow_html=True)

# === Footer ===
st.markdown("""
---
<p style='text-align: center; color:#38bdf8; font-size: 20px;'>"Dedicated to real-world medical understanding and compassionate care".</p>
<p style='text-align: center; color:#38bdf8; font-size: 20px;'>Developed by <i><strong><span style='color:#ffffff;'> "MOHAMMED BILAL"</span></strong></i></p>
""", unsafe_allow_html=True)