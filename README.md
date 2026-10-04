# Medbot-
A smart medical chatbot for day to day basic health requirements 
# 🏥 MEDBOT

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-00B388?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Llama%203.1-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![HF Spaces](https://img.shields.io/badge/HF%20Spaces-Deployed-FF6B35?style=for-the-badge&logo=huggingface&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An AI-powered medical assistant that answers health questions using Retrieval-Augmented Generation (RAG), with a full chat history sidebar, conversation memory, and a modern UI.**

[Features](#-features) • [Tech Stack](#-tech-stack) • [Installation](#-installation--setup) • [API Endpoints](#-api-endpoints) • [Deployment](#-deployment)

</div>

---

## 📸 Screenshots

| Chat Interface | Chat History Sidebar |
|---|---|
| ![Chat Interface](screenshots/chat.png)| ![Sidebar_Interface](screenshots/sidebar.png)|

---

## ✨ Features

- 🤖 **AI Medical Q&A** — Ask any medical question and get specific, actionable answers
- 💬 **Chat History Sidebar** — Browse and resume previous conversations (like Claude / ChatGPT)
- ➕ **New Chat Button** — Start fresh conversations anytime; previous chats are auto-saved
- 🧠 **Conversation Memory** — Bot remembers context within each session for natural follow-ups
- 🔍 **RAG Pipeline** — Retrieves relevant medical knowledge from Pinecone before generating answers
- 📝 **Markdown Rendering** — Responses display with bold text, numbered lists, and headers
- 🗄️ **Persistent History** — All chats stored in local SQLite database across restarts
- 📱 **Responsive UI** — Collapsible sidebar works on desktop and mobile
- 🚀 **Cloud Inference** — No local GPU needed; uses HuggingFace Inference API

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10+, Flask 3.0 |
| **Containerization** | Docker (HuggingFace Spaces ready, port 7860) |
| **LLM** | Llama 3.1 8B Instruct via HuggingFace Inference API |
| **RAG Framework** | LangChain 0.2 |
| **Vector Database** | Pinecone (index: `medical`) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Chat Storage** | SQLite (local, zero-config) |
| **Frontend** | HTML5, CSS3, JavaScript, jQuery, marked.js, DOMPurify |

---

## 📂 Project Structure

```
GENAI-Smart-Medical-Chatbot/
├── Dockerfile               # Docker image — used for HuggingFace Spaces deployment
├── app.py                   # Flask app — routes, RAG chain, SQLite logic
├── store_index.py           # Script to embed documents and populate Pinecone
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
├── .env                     # Environment variables (not committed)
├── chat_history.db          # SQLite database (auto-created at runtime)
│
├── data/
│   └── blood_sugar_guide.txt  # Curated medical knowledge (blood sugar management)
│
├── src/
│   ├── helper.py            # PDF loading and text splitting utilities
│   └── prompt.py            # Legacy prompt template
│
├── model/
│   └── instructions.txt     # Model download instructions
│
├── templates/
│   └── chat.html            # Full frontend (sidebar + chat UI)
│
├── static/                  # Static assets (CSS, images)
└── research/                # Notebooks and experiments
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9+
- A [HuggingFace](https://huggingface.co/settings/tokens) account (free) with an API token
- A [Pinecone](https://www.pinecone.io/) account (free tier) with an API key

### 1. Clone the repository

```bash
git clone https://github.com/ranjansil/Medbot.git
cd GENAI-Smart-Medical-Chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create your `.env` file

```bash
cp .env.example .env   # or create manually
```

Add the following to `.env`:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 5. Populate the Pinecone index (first time only)

Place any medical PDF files in the `data/` folder, then run:

```bash
python store_index.py
```

> The curated `data/blood_sugar_guide.txt` is already indexed. Only re-run this when adding new documents.

### 6. Run the application

```bash
python app.py
```

Open your browser and visit:

```
http://localhost:7860
```

#### Or run with Docker

```bash
docker build -t medical-chatbot .
docker run -p 7860:7860 \
  -e HUGGINGFACEHUB_API_TOKEN=your_token \
  -e PINECONE_API_KEY=your_key \
  medical-chatbot
```

---

## 🔑 Environment Variables

| Variable | Description | Where to get it |
|---|---|---|
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace API token for Llama 3.1 inference | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `PINECONE_API_KEY` | Pinecone vector database API key | [app.pinecone.io](https://app.pinecone.io/) |

> ⚠️ Never commit your `.env` file. It is listed in `.gitignore`.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main chat interface |
| `POST` | `/get` | Send a message; returns AI response and `conversation_id` |
| `GET` | `/api/conversations` | List all saved conversations |
| `POST` | `/api/conversations` | Create a new blank conversation |
| `GET` | `/api/conversations/<id>` | Get a conversation with its full message history |
| `DELETE` | `/api/conversations/<id>` | Delete a conversation and all its messages |

### Example: Send a message

```bash
curl -X POST http://localhost:7860/get \
  -F "msg=What is my target blood sugar range?" \
  -F "conversation_id=optional-uuid-here"
```

```json
{
  "response": "For most adults with diabetes, target blood sugar levels are:\n- **Before meals:** 80–130 mg/dL\n- **After meals (2hr):** < 180 mg/dL",
  "conversation_id": "abc123-..."
}
```

---

## 🧠 How It Works

```
User Message
     │
     ▼
┌─────────────────────┐
│  Pinecone Retriever │  ← Finds top-3 relevant medical document chunks
└─────────────────────┘
     │ context
     ▼
┌─────────────────────┐
│   Prompt Builder    │  ← Injects context + last 3 conversation turns
└─────────────────────┘
     │ formatted prompt
     ▼
┌─────────────────────┐
│  Llama 3.1 8B via  │  ← HuggingFace Inference API (no local GPU needed)
│   HF Router API    │
└─────────────────────┘
     │ answer
     ▼
┌─────────────────────┐
│  SQLite + Response  │  ← Saves to DB, renders markdown, returns to UI
└─────────────────────┘
```

---

## 🚀 Deployment

This app requires **no local model download** — inference runs entirely in the cloud via the HuggingFace API.

### ⭐ HuggingFace Spaces (Recommended)

The project includes a `Dockerfile` configured for HF Spaces (port 7860).

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Docker** as the Space SDK
3. Link your GitHub repo **or** clone the Space and push manually:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push space main
   ```
4. Add secrets in **Settings → Variables and Secrets**:
   | Secret | Value |
   |---|---|
   | `HUGGINGFACEHUB_API_TOKEN` | Your HF token |
   | `PINECONE_API_KEY` | Your Pinecone key |
5. The Space will build the Docker image and go live automatically at:
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   ```

> ⚠️ SQLite chat history will reset on each Space restart. For persistent history, replace SQLite with a hosted database (PostgreSQL, MongoDB Atlas, etc.)

### 🐳 Docker (Self-hosted)

```bash
docker build -t medical-chatbot .
docker run -p 7860:7860 \
  -e HUGGINGFACEHUB_API_TOKEN=your_token \
  -e PINECONE_API_KEY=your_key \
  medical-chatbot
```

### Render / Railway

1. Connect your GitHub repo
2. Set start command: `python app.py`
3. Add `HUGGINGFACEHUB_API_TOKEN` and `PINECONE_API_KEY` as environment variables
4. Set port to `7860`

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please make sure your code runs without errors before submitting.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Aruneeth Sil**



---

<div align="center">

⭐ **If you find this project useful, please give it a star!** ⭐

</div>
