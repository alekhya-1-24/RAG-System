import json, os, sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR   = Path(__file__).parent.parent / "data"
STATIC_DIR = Path(__file__).parent.parent / "static"

app = FastAPI(title="Conversation RAG Chatbot")

_rag     = None
_persona: Dict[str, Any] = {}


def _get_rag():
    global _rag
    if _rag is None:
        from rag_pipeline import RAGPipeline
        _rag = RAGPipeline()
        _rag.load()
    return _rag


def _get_persona():
    global _persona
    if not _persona:
        from persona import PersonaExtractor
        _persona = PersonaExtractor.load()
    return _persona


def _build_system_prompt(persona: Dict) -> str:
    traits    = persona.get("personality_traits", {}).get("dominant_traits", [])
    comm      = persona.get("communication_style", {})
    interests = persona.get("inferred_interests", [])
    habits    = persona.get("habits", {})
    facts     = persona.get("personal_facts", {})
    return f"""You are an intelligent assistant that deeply analysed a large dataset of conversations.
You have built a persona profile and a RAG system over those conversations.

## User Persona
- Dominant personality traits: {', '.join(traits) or 'unknown'}
- Communication: avg {comm.get('avg_message_length_words','?')} words/msg, emoji usage {comm.get('emoji_usage_pct',0)}%
- Top interests: {', '.join(interests[:6]) or 'unknown'}
- Detected habits: {json.dumps(habits, ensure_ascii=False)[:500]}
- Personal facts: {json.dumps(facts, ensure_ascii=False)[:500]}

Answer the user's question using the retrieved context (topic summaries + message chunks).
Be specific and ground answers in conversation evidence.
For character/habit questions, use the persona data above.
Keep answers to 3-6 sentences unless more detail is explicitly requested.
"""


def _build_user_msg(query: str, ctx: Dict) -> str:
    topics = "\n".join(
        f"[Topic {t['topic_id']} | msgs {t['msgs']} | score {t['score']}]: {t['summary']}"
        for t in ctx["topic_summaries"]
    )
    chunks = "\n".join(
        f"[Chunk {c['chunk_id']} | msgs {c['msgs']} | score {c['score']}]: {c['text']}"
        for c in ctx["message_chunks"]
    )
    return f"## Retrieved Topic Summaries\n{topics}\n\n## Retrieved Chunks\n{chunks}\n\n## Question\n{query}\n\nAnswer based on the context above."


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/chat")
async def chat(request: Request):
    body  = await request.json()
    query = body.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "empty query"}, 400)

    rag     = _get_rag()
    persona = _get_persona()
    ctx     = rag.retrieve(query, top_topics=3, top_chunks=5)

    import httpx
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}",
            },
            json={
                "model":      "llama-3.3-70b-versatile",
                "max_tokens": 600,
                "messages":   [
                    {"role": "system", "content": _build_system_prompt(persona)},
                    {"role": "user",   "content": _build_user_msg(query, ctx)},
                ],
            },
        )
    try:
        data = resp.json()
    except Exception:
        data = {}
    if "choices" in data:
        answer = data["choices"][0]["message"]["content"]
    elif "error" in data:
        answer = f"API Error: {data['error']}"
    else:
        answer = f"Raw response ({resp.status_code}): {resp.text[:500]}"

    return JSONResponse({
        "answer": answer,
        "context": {
            "topics_used":     [t["topic_id"] for t in ctx["topic_summaries"]],
            "chunks_used":     [c["chunk_id"]  for c in ctx["message_chunks"]],
            "topic_summaries": ctx["topic_summaries"],
            "message_chunks":  ctx["message_chunks"],
        },
    })


@app.get("/persona")
async def get_persona():
    return JSONResponse(_get_persona())


@app.get("/stats")
async def get_stats():
    rag = _get_rag()
    return JSONResponse({
        "total_messages":      len(rag.messages),
        "topic_checkpoints":   len(rag.topic_checkpoints),
        "message_checkpoints": len(rag.msg_checkpoints),
        "retrieval_chunks":    len(rag.chunks),
        "topic_detection": {
            "method":    "sliding-window cosine similarity (TF-IDF + LSA)",
            "window":    10,
            "threshold": 0.30,
        },
    })


@app.get("/topics")
async def get_topics():
    rag = _get_rag()
    return JSONResponse({
        "total":  len(rag.topic_checkpoints),
        "topics": [
            {
                "topic_id":      tc.topic_id,
                "start_msg":     tc.start_msg,
                "end_msg":       tc.end_msg,
                "message_count": len(tc.messages),
                "summary":       tc.summary,
            }
            for tc in rag.topic_checkpoints[:100]
        ],
    })
