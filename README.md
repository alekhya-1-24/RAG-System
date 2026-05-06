# 🧠 Conversation Intelligence - RAG Chatbot

## 🎥 Demo Video

[Watch Demonstration Video](https://1drv.ms/v/c/b3767b8c5967eaba/IQAZn3yGGr9-T4tWmdBEfE25AQGL1-gHWb6DMMY4LppILdI?e=CRTfPa)

---

## 📌 Overview

A Retrieval-Augmented Generation (RAG) chatbot built using 11,000 real conversations (~192K messages). The system can answer questions about conversation history, habits, interests, and personality traits using semantic retrieval.

---

## ✨ Features

✅ Semantic Search using TF-IDF + LSA
✅ Offline Embedding Pipeline
✅ Topic Detection
✅ Persona Extraction
✅ FastAPI Chatbot Backend
✅ Single Page UI
✅ Retrieved Context Viewer
✅ Supports Groq / OpenAI / xAI APIs

---

## 🏗️ Architecture

### RAG Pipeline

1. Messages processed chronologically
2. Sliding window of 10 messages
3. TF-IDF + LSA embeddings generated
4. Cosine similarity used for topic detection
5. Topic checkpoints and message checkpoints created
6. Relevant chunks retrieved during querying
7. Retrieved context passed to LLM

### Persona Extraction

Extracts:

* Habits
* Interests
* Preferences
* Personality traits
* Communication style

### Chatbot UI

Built using:

* FastAPI
* HTML
* CSS
* JavaScript

---

## 📊 Statistics

| Metric              | Value   |
| ------------------- | ------- |
| Total Messages      | ~191K   |
| Conversations       | 11K     |
| Topic Checkpoints   | ~5700   |
| Message Checkpoints | ~1900   |
| Retrieval Chunks    | ~19000  |
| Build Time          | ~30 sec |

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/RAG-system.git
cd RAG-system
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Create Data Folder

```bash
Desktop/
├── RAG-system/
└── data/
    └── messages.json
```

### 4. messages.json Format

```json
[
  {
    "global_id": 0,
    "conv_id": 1,
    "speaker": "User 1",
    "text": "Hello"
  }
]
```

---

## ⚙️ Configure API

### Groq Example

```python
resp = await client.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}"},
    json={"model": "llama-3.3-70b-versatile"}
)
```

### Set API Key

#### Windows

```bash
set GROQ_API_KEY=your-api-key
```

#### Linux / macOS

```bash
export GROQ_API_KEY=your-api-key
```

---

## 🛠️ Build Index

```bash
python build_index.py
```

Generated Files:

```bash
/data/rag_index.pkl
/data/persona.json
```

---

## ▶️ Run Application

```bash
python -m uvicorn app:app --reload
```

Open:

```bash
http://127.0.0.1:8000
```

---

## 🔄 CSV to JSON Conversion

```bash
python convert_csv.py
```

Converts:

```bash
conversations.csv → messages.json
```

---

## 📁 Project Structure

```bash
RAG-system/
├── app.py
├── rag_pipeline.py
├── persona.py
├── build_index.py
├── convert_csv.py
├── requirements.txt
├── Dockerfile
└── static/
    └── index.html
```

### Data Folder

```bash
data/
├── messages.json
├── rag_index.pkl
└── persona.json
```

---

## 🐳 Docker

### Build Image

```bash
docker build -t rag-chatbot .
```

### Run Container

```bash
docker run -p 8000:7860 -e GROQ_API_KEY=your-key rag-chatbot
```

---

## 📝 Notes

* Fully offline embeddings
* No external embedding model required
* Works with OpenAI-compatible APIs
* data/ folder excluded using .gitignore

---

## 🔮 Future Improvements

* Vector Database Support
* Multi-user Memory
* Better Ranking System
* Streaming Responses
* Authentication

---

## 📄 License

MIT License
