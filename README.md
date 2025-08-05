# MB MediMind AI

An AI-powered medical assistant chatbot that provides intelligent, empathetic, and well-structured health advice. It uses **RAG (Retrieval-Augmented Generation)** to combine semantic search and language generation for accurate and context-rich medical responses.

---

## ✨ Key Features ✨

*   **Semantic Search with FAISS:** Efficient retrieval of relevant medical responses using precomputed sentence embeddings.
*   **Contextual Answer Generation:** Combines retrieved data with Cohere's Command R+ model to generate personalized, patient-friendly medical advice.
*   **Real-Time Q&A Flow:** Chat interface mimics a consultation, showing previous turns and structured answers.
*   **Logs & History:** Every question-answer pair is logged with a timestamp in a CSV for future review or audit.
*   **Modern Chat Interface with Streamlit:** Sleek UI with dark mode, message boxes, scrollable chat, and bottom-positioned input field (like mobile chat apps).

---

## 🛠️ Tech Stack 🛠️

*   **Streamlit:** For building the chat UI
*   **SentenceTransformers:** For semantic vector encoding
*   **FAISS:** For efficient vector similarity search
*   **Cohere:** For response generation using `command-r-plus`
*   **Transformers (optional):** Summarization capability
*   **Python (v3.9+)**

---

##  Models Info : 

*   **Embedding Model:** `all-MiniLM-L6-v2`
*   **LLM Generator:** `Cohere command-r-plus`
*   **Retriever:** `FAISS Flat Index` with cosine similarity

---

## 📁 Project Structure 📁


## Disclaimer::
This is an AI prototype built for educational and research purposes. It does not replace professional medical advice or diagnosis.


## Eembedding model file and datasets (original & processed used in app ) are also available.
