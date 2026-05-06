"""
RAG Pipeline with chronological topic-change detection and 100-message checkpoints.

Embedding: TF-IDF + Latent Semantic Analysis (SVD) — fully offline, no external API.
This gives genuine semantic similarity (synonyms, related concepts) beyond keyword matching.

Topic Detection Strategy:
- Embed each message with the fitted LSA model.
- Slide a window of WINDOW_SIZE messages; compute the mean vector.
- Compare consecutive window vectors with cosine similarity.
- When similarity drops below TOPIC_SIM_THRESHOLD → topic boundary.
- Purely chronological: no lookahead.
"""

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CACHE_FILE    = DATA_DIR / "rag_index.pkl"
MESSAGES_FILE = DATA_DIR / "messages.json"

# ── tuneable ──────────────────────────────────────────────────────────────────
WINDOW_SIZE          = 10     # messages per sliding window
TOPIC_SIM_THRESHOLD  = 0.30   # cosine sim below this → new topic
MSG_CHECKPOINT_EVERY = 100    # independent of topics
CHUNK_SIZE           = 10     # messages per retrieval chunk
LSA_COMPONENTS       = 150    # dimensionality of LSA space
MAX_FEATURES         = 60000  # TF-IDF vocabulary size
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Message:
    global_id: int
    conv_id: int
    speaker: str
    text: str


@dataclass
class TopicCheckpoint:
    topic_id: int
    start_msg: int
    end_msg: int
    messages: List[Message]
    summary: str = ""
    embedding: Optional[np.ndarray] = None


@dataclass
class MessageCheckpoint:
    checkpoint_id: int
    start_msg: int
    end_msg: int
    messages: List[Message]
    summary: str = ""
    embedding: Optional[np.ndarray] = None


@dataclass
class MessageChunk:
    chunk_id: int
    start_msg: int
    end_msg: int
    text: str
    embedding: Optional[np.ndarray] = None


class RAGPipeline:
    def __init__(self):
        self.messages: List[Message] = []
        self.topic_checkpoints: List[TopicCheckpoint] = []
        self.msg_checkpoints: List[MessageCheckpoint] = []
        self.chunks: List[MessageChunk] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._svd: Optional[TruncatedSVD] = None
        self._all_vecs: Optional[np.ndarray] = None

    # ── load messages ─────────────────────────────────────────────────────────

    def load_messages(self):
        with open(MESSAGES_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        self.messages = [Message(**r) for r in raw]
        print(f"[RAG] Loaded {len(self.messages):,} messages.")

    # ── embed all messages with TF-IDF + LSA ─────────────────────────────────

    def _fit_and_embed_all(self):
        print("[RAG] Fitting TF-IDF + LSA model on all messages…")
        texts = [m.text for m in self.messages]

        self._vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        tfidf_matrix = self._vectorizer.fit_transform(texts)
        print(f"  TF-IDF matrix: {tfidf_matrix.shape}")

        n_components = min(LSA_COMPONENTS, tfidf_matrix.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        lsa_matrix = self._svd.fit_transform(tfidf_matrix)
        self._all_vecs = normalize(lsa_matrix)
        print(f"  LSA embeddings: {self._all_vecs.shape}")

    def _embed_query(self, query: str) -> np.ndarray:
        tfidf = self._vectorizer.transform([query])
        lsa   = self._svd.transform(tfidf)
        return normalize(lsa)[0]

    def _window_mean(self, start: int, end: int) -> np.ndarray:
        vecs = self._all_vecs[start:end]
        mean = vecs.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else mean

    # ── Part 1a: topic checkpoints ────────────────────────────────────────────

    def build_topic_checkpoints(self):
        print("[RAG] Detecting topic boundaries (sliding window cosine similarity)…")
        n = len(self.messages)
        boundaries = [0]

        prev_vec = None
        for i in range(0, n - WINDOW_SIZE, WINDOW_SIZE):
            win_vec = self._window_mean(i, i + WINDOW_SIZE)
            if prev_vec is not None:
                sim = float(np.dot(win_vec, prev_vec))
                if sim < TOPIC_SIM_THRESHOLD:
                    boundaries.append(i)
            prev_vec = win_vec

        boundaries.append(n)
        print(f"  {len(boundaries)-1} topic segments detected.")

        for t_idx in range(len(boundaries) - 1):
            start, end = boundaries[t_idx], boundaries[t_idx + 1]
            seg_msgs = self.messages[start:end]
            seg_emb  = self._window_mean(start, end)
            self.topic_checkpoints.append(TopicCheckpoint(
                topic_id=t_idx,
                start_msg=self.messages[start].global_id,
                end_msg=self.messages[end - 1].global_id,
                messages=seg_msgs,
                embedding=seg_emb,
            ))

    # ── Part 1b: 100-message checkpoints ─────────────────────────────────────

    def build_message_checkpoints(self):
        print("[RAG] Building 100-message checkpoints…")
        n = len(self.messages)
        for cp_id, start in enumerate(range(0, n, MSG_CHECKPOINT_EVERY)):
            end = min(start + MSG_CHECKPOINT_EVERY, n)
            self.msg_checkpoints.append(MessageCheckpoint(
                checkpoint_id=cp_id,
                start_msg=self.messages[start].global_id,
                end_msg=self.messages[end - 1].global_id,
                messages=self.messages[start:end],
                embedding=self._window_mean(start, end),
            ))
        print(f"  {len(self.msg_checkpoints)} message checkpoints built.")

    # ── Retrieval chunks ──────────────────────────────────────────────────────

    def build_chunks(self):
        print("[RAG] Building retrieval chunks…")
        n = len(self.messages)
        for cid, start in enumerate(range(0, n, CHUNK_SIZE)):
            end = min(start + CHUNK_SIZE, n)
            seg  = self.messages[start:end]
            text = " ".join(f"{m.speaker}: {m.text}" for m in seg)
            self.chunks.append(MessageChunk(
                chunk_id=cid,
                start_msg=self.messages[start].global_id,
                end_msg=self.messages[end - 1].global_id,
                text=text,
                embedding=self._window_mean(start, end),
            ))
        print(f"  {len(self.chunks):,} retrieval chunks built.")

    # ── Summarisation (extractive, no API) ────────────────────────────────────

    _STOPWORDS = {
        "i","you","he","she","it","we","they","me","him","her","us","them",
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "that","this","is","are","was","were","be","been","being","have","has",
        "had","do","does","did","will","would","can","could","should","may",
        "might","shall","am","not","no","so","if","as","by","from","up","about",
        "my","your","his","its","our","their","what","how","when","where","who",
        "which","just","like","really","very","also","too","yeah","yes","oh",
        "okay","ok","hi","hey","hello","thanks","thank","great","good","nice",
        "that's","it's","i'm","don't","you're","i've","i'll","didn't","isn't",
    }

    def _summarise(self, msgs: List[Message], max_sents: int = 4) -> str:
        if not msgs:
            return ""
        freq: Dict[str, int] = {}
        for m in msgs:
            for w in re.findall(r"[a-z']+", m.text.lower()):
                if w not in self._STOPWORDS and len(w) > 2:
                    freq[w] = freq.get(w, 0) + 1

        scored = []
        for m in msgs:
            words = re.findall(r"[a-z']+", m.text.lower())
            score = sum(freq.get(w, 0) for w in words if w not in self._STOPWORDS)
            scored.append((score, m))

        scored.sort(key=lambda x: -x[0])
        top = sorted(scored[:max_sents], key=lambda x: x[1].global_id)
        return " | ".join(f"{m.speaker}: {m.text}" for _, m in top)

    def build_summaries(self):
        print("[RAG] Building summaries…")
        for tc in self.topic_checkpoints:
            tc.summary = self._summarise(tc.messages, 4)
        for mc in self.msg_checkpoints:
            mc.summary = self._summarise(mc.messages, 3)
        print("  Done.")

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self):
        print(f"[RAG] Saving index to {CACHE_FILE}…")
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({
                "topic_checkpoints": self.topic_checkpoints,
                "msg_checkpoints":   self.msg_checkpoints,
                "chunks":            self.chunks,
                "messages":          self.messages,
                "vectorizer":        self._vectorizer,
                "svd":               self._svd,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("[RAG] Saved.")

    def load(self):
        print(f"[RAG] Loading index from {CACHE_FILE}…")
        with open(CACHE_FILE, "rb") as f:
            p = pickle.load(f)
        self.topic_checkpoints = p["topic_checkpoints"]
        self.msg_checkpoints   = p["msg_checkpoints"]
        self.chunks            = p["chunks"]
        self.messages          = p["messages"]
        self._vectorizer       = p["vectorizer"]
        self._svd              = p["svd"]
        # Rebuild _all_vecs from stored embeddings (not re-computed)
        print(f"[RAG] Loaded: {len(self.topic_checkpoints)} topics, "
              f"{len(self.msg_checkpoints)} msg-checkpoints, "
              f"{len(self.chunks):,} chunks.")

    def build_all(self):
        self.load_messages()
        self._fit_and_embed_all()
        self.build_topic_checkpoints()
        self.build_message_checkpoints()
        self.build_chunks()
        self.build_summaries()
        self.save()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _top_k(self, q_vec: np.ndarray, items, k: int):
        vecs = np.array([
            item.embedding if item.embedding is not None else np.zeros(q_vec.shape)
            for item in items
        ])
        sims = vecs @ q_vec
        top_idx = np.argsort(sims)[::-1][:k]
        return [(i, float(sims[i])) for i in top_idx]

    def retrieve(self, query: str, top_topics: int = 3, top_chunks: int = 5) -> Dict[str, Any]:
        q_vec = self._embed_query(query)

        topic_hits = self._top_k(q_vec, self.topic_checkpoints, top_topics)
        retrieved_topics = [
            {
                "topic_id": self.topic_checkpoints[i].topic_id,
                "msgs": f"{self.topic_checkpoints[i].start_msg}–{self.topic_checkpoints[i].end_msg}",
                "summary": self.topic_checkpoints[i].summary,
                "score": round(s, 3),
            }
            for i, s in topic_hits
        ]

        chunk_hits = self._top_k(q_vec, self.chunks, top_chunks)
        retrieved_chunks = [
            {
                "chunk_id": self.chunks[i].chunk_id,
                "msgs": f"{self.chunks[i].start_msg}–{self.chunks[i].end_msg}",
                "text": self.chunks[i].text[:500],
                "score": round(s, 3),
            }
            for i, s in chunk_hits
        ]

        return {
            "query": query,
            "topic_summaries": retrieved_topics,
            "message_chunks":  retrieved_chunks,
        }
