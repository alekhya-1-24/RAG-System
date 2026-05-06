---
title: Conversation RAG Chatbot
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# Conversation Intelligence — RAG Chatbot

A full end-to-end RAG system built on 11,000 conversations (~192K messages).

## Architecture

### Part 1 — RAG System with Checkpoints

**Topic Detection (chronological sliding window)**

Messages are processed strictly in chronological order. A sliding window of 10 messages is moved across the full message stream. Each window is represented by the mean of its TF-IDF + LSA embeddings, normalized to unit length. When the cosine similarity between consecutive windows drops below the threshold (0.30), a **topic boundary** is declared and a new `TopicCheckpoint` begins.

This produces:
- `Topic N → messages start–end → extractive summary`

No lookahead, no shuffling — purely chronological.

**100-Message Checkpoints** (independent of topics):
- Every 100 messages in order gets a `MessageCheckpoint` with an extractive summary.

**Embeddings:** TF-IDF (60k vocab, bigrams) + Latent Semantic Analysis (SVD, 150 dims) — fully offline, no external model downloads needed.

**Retrieval at query time:**
1. Embed the query with the same fitted TF-IDF+LSA model.
2. Cosine similarity against all `TopicCheckpoint` embeddings → top-3 topic summaries.
3. Cosine similarity against all 10-message `MessageChunk` embeddings → top-5 raw chunks.
4. Both are passed as context to Claude for answer generation.

### Part 2 — Persona Extraction

Extracted from User 1's ~98K messages using:
- **Regex pattern matching** for habits, facts, preferences, relationships
- **Keyword frequency analysis** for interests
- **Linguistic signals** for personality traits (humor, empathy, curiosity, etc.)
- **Statistical analysis** for communication style (message length, emoji rate, etc.)

Output: structured JSON stored in `data/persona.json`.

### Part 3 — Chatbot

FastAPI backend + single-page HTML/JS UI. The chatbot:
1. Retrieves relevant topic summaries + message chunks via semantic search
2. Passes them as context to Claude Sonnet
3. Returns the answer with a collapsible "retrieved context" drawer showing which topics/chunks were used

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your conversations.csv in data/
# 3. Build the index (one-time, ~30 seconds)
python build_index.py

# 4. Start the server
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

# 5. Open http://localhost:8000
```

## Stats

| Metric | Value |
|---|---|
| Total messages | 191,839 |
| Conversations | 11,000 |
| Topic checkpoints | 5,789 |
| 100-msg checkpoints | 1,919 |
| Retrieval chunks | 19,184 |
| Build time | ~26 seconds |
| Embedding | TF-IDF + LSA (offline) |
