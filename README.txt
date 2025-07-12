# "" MB MediMind AI ""

""MB MediMind AI"" is a real-world AI-powered medical assistant chatbot that provides intelligent, empathetic, and well-structured health advice. It uses **RAG (Retrieval-Augmented Generation)** to combine semantic search (with FAISS & Sentence-BERT) and language generation (via Cohere Command R+) for accurate and context-rich medical responses.

## Key Features::

**Semantic Search with FAISS**  
Efficient retrieval of relevant medical responses using precomputed sentence embeddings.

**Contextual Answer Generation**  
Combines retrieved data with Cohere's Command R+ model to generate personalized, patient-friendly medical advice.

**Real-Time Q&A Flow**  
Chat interface mimics a consultation, showing previous turns and structured answers.

**Logs & History**  
Every question-answer pair is logged with a timestamp in a CSV for future review or audit.

**Modern Chat Interface with Streamlit**  
Sleek UI with dark mode, message boxes, scrollable chat, and bottom-positioned input field (like mobile chat apps).

## Tech Stack::

- [Streamlit] ------->>> for building the chat UI
- [SentenceTransformers] -------- >>>for semantic vector encoding
- [FAISS] ------->>> for efficient vector similarity search
- [Cohere] ------->>> for response generation using `command-r-plus`
- [Transformers](optional) ------->>>summarization capability
- Python (v3.9+)

## Models  Info::
Embedding Model: all-MiniLM-L6-v2

LLM Generator: Cohere command-r-plus

Retriever: FAISS Flat Index with cosine similarity


## Project Structure::

   """MB MediMind AI/

	--->app.py # Streamlit interface
	--->rag_engine.py # RAG logic (retrieval + generation)
	---> embeddings.npy # Precomputed vector embeddings
	---> processed.csv # Cleaned medical dataset
	---> rag_chat_log.csv # Chat logs (auto-generated)
	---> requirements.txt # Project dependencies
	---> README.md # Project description



## Disclaimer::
This is an AI prototype built for educational and research purposes. It does not replace professional medical advice or diagnosis.


##  embedding model file and datasets (original & processed used in app ) will be provided on demand.
