# 🧠 Conversation Intelligence - RAG Chatbot

> A full end-to-end Retrieval-Augmented Generation (RAG) system built on 11,000 real conversations (~192K messages), capable of answering questions about conversation history, habits, and personality traits.

---

## ✨ Features

- 🔍 **Semantic Search** - TF-IDF + LSA embeddings for fully offline retrieval (no API needed for indexing)
- 🗂️ **Topic Detection** - Sliding-window cosine similarity to detect conversation topic boundaries
- 👤 **Persona Extraction** - Automatically builds a personality profile from message patterns
- 💬 **RAG Chatbot** - FastAPI backend with a clean single-page UI
- ⚡ **Any LLM Backend** - Works with any OpenAI-compatible API (Groq, OpenAI, xAI, etc.)

---

## 🏗️ Architecture

### Part 1 — RAG Pipeline

Messages are processed **chronologically** using a sliding window of 10 messages. Each window is embedded via TF-IDF + LSA (SVD, 150 dims). When cosine similarity between consecutive windows drops below `0.30`, a **topic boundary** is declared.

This produces two types of checkpoints:

| Checkpoint | Description |
|---|---|
| `TopicCheckpoint` | Groups messages by detected topic with an extractive summary |
| `MessageCheckpoint` | Every 100 messages gets a fixed summary (topic-independent) |

**At query time:**
1. Query is embedded using the same fitted TF-IDF+LSA model
2. Top-3 topic summaries retrieved via cosine similarity
3. Top-5 raw message chunks retrieved via cosine similarity
4. Both passed as context to the LLM

### Part 2 — Persona Extraction

Extracted from User 1's messages using:
- Regex pattern matching for habits, facts, preferences, relationships
- Keyword frequency analysis for interests
- Linguistic signals for personality traits
- Statistical analysis for communication style

Output → `data/persona.json`

### Part 3 — Chatbot UI

FastAPI backend + single-page HTML/JS UI with a collapsible **"Retrieved Context"** drawer showing exactly which topics and chunks were used to generate each answer.

---

## 📊 Stats

| Metric | Value |
|---|---|
| Total messages | ~191,000 |
| Conversations | 11,000 |
| Topic checkpoints | ~5,700 |
| 100-msg checkpoints | ~1,900 |
| Retrieval chunks | ~19,000 |
| Build time | ~30 seconds |
| Embedding method | TF-IDF + LSA (fully offline) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- An API key from any OpenAI-compatible LLM provider (e.g. [Groq](https://console.groq.com), [OpenAI](https://platform.openai.com), [xAI](https://console.x.ai))
- Your `conversations.csv` file (one conversation per row, lines formatted as `User 1: ...` and `User 2: ...`)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/RAG-system.git
cd RAG-system
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Prepare Your Data

Create a `data/` folder one level **above** the project directory and place your `messages.json` inside it:

```
Desktop/
├── RAG-system/        ← project code
└── data/
    └── messages.json  ← your data goes here
```

Your `messages.json` should follow this format:

```json
[
  { "global_id": 0, "conv_id": 1, "speaker": "User 1", "text": "Hello!" },
  { "global_id": 1, "conv_id": 1, "speaker": "User 2", "text": "Hi there!" }
]
```

> 💡 If you have a `conversations.csv`, you can convert it to this format using the provided `convert_csv.py` script (see below).

### Step 4 — Configure Your LLM Provider

Open `app.py` and update the API endpoint and key to match your chosen LLM provider:

```python
# Groq example
resp = await client.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}"},
    json={"model": "llama-3.3-70b-versatile", ...}
)
```

```python
# OpenAI example
resp = await client.post(
    "https://api.openai.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"},
    json={"model": "gpt-4o-mini", ...}
)
```

Then set your API key in your terminal:

```bash
# Windows
set GROQ_API_KEY=your-api-key-here

# macOS / Linux
export GROQ_API_KEY=your-api-key-here
```

### Step 5 — Build the Index *(one-time)*

```bash
python build_index.py
```

This generates `data/rag_index.pkl` and `data/persona.json`. Takes ~30 seconds.

### Step 6 — Start the Server

```bash
python -m uvicorn app:app --reload
```

### Step 7 — Open the App

Visit → [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🔄 Converting conversations.csv → messages.json

If your data is in CSV format (one conversation per row), run:

```bash
python convert_csv.py
```

This will read `conversations.csv` from your project folder and write `messages.json` to `../data/messages.json`.

---

## 📁 Project Structure

```
RAG-system/
├── app.py              # FastAPI server & chat endpoint
├── rag_pipeline.py     # Topic detection, embedding, retrieval
├── persona.py          # Persona extraction logic
├── build_index.py      # One-time index builder script
├── convert_csv.py      # CSV → messages.json converter
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker support
└── static/
    └── index.html      # Frontend UI
```

```
data/                   # Lives outside the project folder
├── messages.json       # Parsed conversation data (you provide)
├── rag_index.pkl       # Auto-generated by build_index.py
└── persona.json        # Auto-generated by build_index.py
```

---

## 🐳 Docker

```bash
docker build -t rag-chatbot .
docker run -p 8000:7860 -e GROQ_API_KEY=your-key-here rag-chatbot
```

---

## 📝 Notes

- The `data/` folder is excluded from the repo via `.gitignore` since it contains large generated files
- All embeddings are computed **offline** — no external model downloads required
- The system works with any dataset following the `messages.json` schema above

---

## 📄 License

MIT License
