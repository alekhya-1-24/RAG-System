#!/usr/bin/env python3
"""
Build the RAG index and extract persona.
Run this ONCE before starting the API server.

Usage:
    python build_index.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_pipeline import RAGPipeline, CACHE_FILE
from persona import PersonaExtractor, PERSONA_FILE


def main():
    print("=" * 60)
    print("  RAG Chatbot — Index Builder")
    print("=" * 60)

    # ── Step 1: Build RAG index ───────────────────────────────────
    if CACHE_FILE.exists():
        resp = input(f"\n[?] RAG index already exists at {CACHE_FILE}.\n    Rebuild? (y/N): ").strip().lower()
        if resp != "y":
            print("Skipping RAG build.")
        else:
            _build_rag()
    else:
        _build_rag()

    # ── Step 2: Extract persona ───────────────────────────────────
    if PERSONA_FILE.exists():
        resp = input(f"\n[?] Persona already exists at {PERSONA_FILE}.\n    Rebuild? (y/N): ").strip().lower()
        if resp != "y":
            print("Skipping persona extraction.")
            return
    _build_persona()

    print("\n✅ All done! Start the server with:")
    print("   uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload")


def _build_rag():
    pipeline = RAGPipeline()
    pipeline.build_all()
    print(f"\n✅ RAG index saved → {CACHE_FILE}")


def _build_persona():
    # Load messages (already parsed)
    import json
    from rag_pipeline import MESSAGES_FILE, Message

    with open(MESSAGES_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    messages = [Message(**r) for r in raw]

    extractor = PersonaExtractor(messages)
    persona = extractor.extract()

    print("\n── Persona Summary ──────────────────────────────────────")
    import json as _json
    print(_json.dumps(persona, indent=2, ensure_ascii=False)[:2000])
    print(f"\n✅ Persona saved → {PERSONA_FILE}")


if __name__ == "__main__":
    main()
