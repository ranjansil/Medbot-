from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
import sqlite3
import uuid
from datetime import datetime
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

print("=" * 50, flush=True)
print("STARTING APPLICATION...", flush=True)
print("=" * 50, flush=True)

app = Flask(__name__)

print("Loading environment variables...", flush=True)
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
INDEX_NAME = "medical"
DB_PATH = os.path.join(os.path.dirname(__file__), "chat_history.db")

print(f"Pinecone API Key loaded: {bool(PINECONE_API_KEY)}", flush=True)
print(f"HF Token loaded: {bool(os.environ.get('HUGGINGFACEHUB_API_TOKEN'))}", flush=True)


# ── Database ─────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations (
        id          TEXT PRIMARY KEY,
        title       TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        role            TEXT NOT NULL,
        content         TEXT NOT NULL,
        timestamp       TEXT NOT NULL,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    )''')
    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── LLM / RAG setup ──────────────────────────────────────────────────────────
print("Step 1: Loading embeddings model...", flush=True)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Step 2: Connecting to Pinecone...", flush=True)
docsearch = PineconeVectorStore.from_existing_index(INDEX_NAME, embeddings)
print("Step 3: Setting up LLM...", flush=True)

# ChatPromptTemplate works with ChatOpenAI (returns messages, not raw strings)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical assistant. Give specific, actionable answers using the context provided. List steps or ranges clearly. Only add 'consult a doctor' as a brief note at the end, never as the main answer."),
    ("human", "Context:\n{context}\n\n{history}Question: {question}"),
])

# Uses HuggingFace router (OpenAI-compatible) — no local model needed
llm = ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get('HUGGINGFACEHUB_API_TOKEN'),
    max_tokens=512,
    temperature=0.5,
)

print("✅ All initialization complete!", flush=True)

# k=3: retrieve 3 chunks for richer context
retriever = docsearch.as_retriever(search_kwargs={'k': 3})

def format_docs(docs):
    """Clean retrieved chunks: fix PDF hyphenation artifacts and strip Q/A labels."""
    parts = []
    for doc in docs:
        text = doc.page_content.strip()
        # Fix PDF line-break hyphenation (e.g. "indiabetic" → "in diabetic")
        # Pattern: lowercase letter immediately followed by lowercase (no space) after a line break
        text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)  # camelCase seams
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)            # hyphenated line breaks
        # Strip Q&A template labels only when chunk starts with them
        if text.startswith("Question:"):
            text = re.sub(r'^Question:.*?Helpful [Aa]nswer:\s*', '', text, flags=re.DOTALL)
        text = re.sub(r'^Helpful [Aa]nswer:\s*', '', text)
        text = text.strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)

def clean_output(text):
    """Strip any leaked prompt structure or mid-generation conversation restarts."""
    # Hard cut at plaintext conversation continuation markers
    cut_markers = [
        '\nUser:', '\n\nUser:', '\nAssistant:', '\n\nAssistant:',
        '\nMedical Context:', '\n\nMedical Context:',
        '\nContext:', '\n\nContext:',
    ]
    for marker in cut_markers:
        if marker in text:
            text = text[:text.index(marker)]

    # Strip if the model repeated the prompt header at the very start
    for artifact in ['Medical Context:', 'Context:', 'Information:']:
        if text.strip().startswith(artifact):
            # Drop the echoed header line and keep the rest
            newline = text.find('\n')
            text = text[newline:].strip() if newline != -1 else ''

    return text.strip()

def build_rag_input(input_dict):
    question = input_dict["question"]
    docs = retriever.invoke(question)
    context = format_docs(docs)
    print(f"\n[RAG] Query: {question!r}")
    print(f"[RAG] Retrieved {len(docs)} docs, context length: {len(context)} chars")
    if context:
        print(f"[RAG] Context preview: {context[:200]!r}")
    else:
        print("[RAG] WARNING: empty context after format_docs!")
    return {"context": context or "No specific context available.", "history": input_dict.get("history", ""), "question": question}

rag_chain = RunnableLambda(build_rag_input) | prompt | llm | StrOutputParser()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/conversations", methods=["POST"])
def new_conversation():
    conv_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    conn = get_db()
    conn.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (conv_id, "New Conversation", now, now)
    )
    conn.commit()
    conn.close()
    return jsonify({"id": conv_id, "title": "New Conversation", "created_at": now, "updated_at": now})


@app.route("/api/conversations/<conv_id>", methods=["GET"])
def get_conversation(conv_id):
    conn = get_db()
    conv = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    if not conv:
        conn.close()
        return jsonify({"error": "Not found"}), 404
    messages = conn.execute(
        "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conv_id,)
    ).fetchall()
    conn.close()
    return jsonify({"conversation": dict(conv), "messages": [dict(m) for m in messages]})


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    conv_id = request.form.get("conversation_id", "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400

    now = datetime.now().isoformat()
    timestamp = datetime.now().strftime("%H:%M")
    conn = get_db()

    if not conv_id:
        # Auto-create conversation from first message
        conv_id = str(uuid.uuid4())
        title = msg[:50] + ("..." if len(msg) > 50 else "")
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, now, now)
        )
    else:
        conv = conn.execute("SELECT id, title FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        if not conv:
            conn.close()
            return jsonify({"error": "Conversation not found"}), 404
        # Auto-title on first real message
        if conv["title"] == "New Conversation":
            title = msg[:50] + ("..." if len(msg) > 50 else "")
            conn.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                         (title, now, conv_id))
        else:
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))

    # Last 3 pairs (6 rows) for conversation memory
    history_rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT 6",
        (conv_id,)
    ).fetchall()
    history_rows = list(reversed(history_rows))

    # Persist user message
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (conv_id, "user", msg, timestamp)
    )
    conn.commit()
    conn.close()

    # Build history string for prompt
    history_str = ""
    if history_rows:
        lines = []
        for row in history_rows:
            prefix = "Patient" if row["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {row['content']}")
        history_str = "Previous conversation:\n" + "\n".join(lines) + "\n\n"

    # Run RAG chain with memory
    result = rag_chain.invoke({"question": msg, "history": history_str})
    print(f"[LLM] Raw output ({len(result)} chars): {result[:300]!r}")

    # Strip any special tokens that slipped through, then clean structural leakage
    for stop in ['</s>', '<|user|>', '<|system|>', '<|assistant|>']:
        if stop in result:
            result = result[:result.index(stop)]
    result = clean_output(result)
    print(f"[LLM] Cleaned output ({len(result)} chars): {result[:200]!r}")

    if not result:
        print("[LLM] WARNING: empty result after cleaning")
        result = "I'm sorry, I wasn't able to generate a response. Please try rephrasing your question."

    # Persist bot message
    conn2 = get_db()
    conn2.execute(
        "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (conv_id, "bot", result, timestamp)
    )
    conn2.commit()
    conn2.close()

    return jsonify({"response": result, "conversation_id": conv_id})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    print(f"Step 5: Starting Flask server on port {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
