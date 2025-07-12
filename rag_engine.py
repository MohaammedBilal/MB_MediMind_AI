import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from datetime import datetime
import csv

# === Load and preprocess dataset ===
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df['query_text'] = df['Description'].astype(str).str.strip() + " - " + df['Patient'].astype(str).str.strip()
    df.drop_duplicates(subset='query_text', inplace=True)
    df['query_text'] = df['query_text'].apply(clean_text)
    df['Doctor'] = df['Doctor'].apply(clean_text)
    df.reset_index(drop=True, inplace=True)
    return df

def clean_text(text):
    text = str(text)
    text = re.sub(r'Q[.]?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bHi doctor\b[:,]? ?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# === Load preprocessed data ===
def load_preprocessed_data():
    df = pd.read_csv("processed.csv")
    embeddings = np.load("embeddings.npy")
    return df, embeddings

# === Load models ===
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
  #  summarizer = pipeline("summarization", model="google/flan-t5-small", tokenizer="google/flan-t5-small")
    return embedder #summarizer

# IN THE UPPER FUNCTION, YOU CAN UNCOMMENT THE LINE BELOW TO USE THE SUMMARIZER. I COMMENTED IT BECAUSE RUNNING APP LOCALLY TAKES A LOT OF TIME TO LOAD THE MODEL DUE TO ITS SIZE.

# === Build FAISS index ===
def build_index_from_embeddings(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


# === Retrieve top-k answers ===
def retrieve_top_k_answers(user_query, k, embedder, df, index):
    query_embedding = embedder.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [df.iloc[i]['Doctor'] for i in indices[0]]

# === Summarize top-k answers ===

# SIMMILARLY, YOU CAN UNCOMMENT THE FUNCTION BELOW TO USE THE SUMMARIZER. I COMMENTED IT BECAUSE RUNNING APP LOCALLY TAKES A LOT OF TIME TO LOAD THE MODEL DUE TO ITS SIZE.
# MODEL WORKS EXCEPTIONALLY WELL FOR LONG CONTEXTS AS USED IN JUPYTER NOTEBOOK (AI_DOCTOR.ipynb) ON COLAB, BUT IT TAKES A LOT OF TIME TO LOAD THE MODEL .
# def summarize_context_answers(context_list, summarizer):
#     joined_context = "\n".join(context_list)
#     if len(joined_context.split()) > 500:
#         joined_context = " ".join(joined_context.split()[:500])
#     summary = summarizer(joined_context, max_length=200, min_length=60, do_sample=False)[0]['summary_text']
#     return summary

# === Generate final response using Cohere ===
def generate_answer_rag(user_query, embedder, df, index, co, chat_history, log_file="rag_chat_log.csv", k=3):

# def generate_answer_rag(user_query, embedder, summarizer, df, index, co, chat_history, log_file, k=3):
    context_answers = retrieve_top_k_answers(user_query, k, embedder, df, index)
    #context_block = summarize_context_answers(context_answers, summarizer)
    # UNCOMMENT THE UPPER LINE TO USE THE SUMMARIZER.
    context_block = "\n".join(context_answers)
    memory_turns = chat_history[-3:]
    memory_block = ""
    for turn in memory_turns:
        memory_block += f"\nUser: {turn['user']}\nBot: {turn['bot']}\n"

    prompt = f"""You are a highly experienced and empathetic medical assistant with deep knowledge in general medicine, diagnostics, and patient care.

I need clear, accurate, and easy-to-understand information about the following medical question. Please use the provided context to guide your answer. Your response should not contain the points format untill very importantly needed ,use paragraph format and be structured in a clear and informative manner.
- Use the context provided to inform your answer.

- Your answer should be structured and easy to read. and use less points format untill important.
- Output limit of my model is 1000 tokens, so be concise but informative.
- Use simple language and avoid medical jargon unless necessary. If you use any medical terms, explain
- Explain the possible causes.
- Recommend appropriate actions or steps I can take.
- Clearly state when I should consider seeing a doctor.
- Avoid medical jargon unless necessary. If you use any, explain it in simple terms.
- If there are any warning signs or urgent symptoms related to the issue, highlight them clearly.

Respond with care and empathy and structure your answer using the following sections:
 Possible Causes:  
 Recommended Actions:  
 When to See a Doctor:
Chat History:
{memory_block}
Summarized Context (from medical data):
{context_block}
Current Question:
{user_query}
Answer:"""

    response = co.generate(
    model='command-r-plus',
    prompt=prompt,
    max_tokens=1000,  
    temperature=0.5,
    stop_sequences=None  
    )


    final_answer = response.generations[0].text.strip()
    chat_history.append({"user": user_query, "bot": final_answer})

    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().isoformat(),
            user_query,
            " || ".join(context_answers),
            context_block,
            final_answer
        ])

    return final_answer
