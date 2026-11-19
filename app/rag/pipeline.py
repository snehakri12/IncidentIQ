"""RAG pipeline over historical incidents.

When ``RAG_USE_CHROMADB`` is enabled and an ``OPENAI_API_KEY`` is present, the
corpus is chunked, embedded with OpenAI's ``text-embedding-3-small``, and
indexed in a persistent ChromaDB collection for semantic similarity search.
Without credentials the retriever falls back to keyword scoring over the same
local synthetic corpus, so the service always returns grounded evidence.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_incidents.json"

LOCAL_SOURCE = "local synthetic incident corpus"
VECTOR_SOURCE = "chromadb semantic search (text-embedding-3-small)"


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> list[str]:
    """Split text into overlapping word windows for embedding."""
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        overlap = max(chunk_size - 1, 0)
    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def _incident_text(incident: dict) -> str:
    return "\n".join(
        part
        for part in (
            incident.get("summary", ""),
            incident.get("details", ""),
            " ".join(incident.get("tags", [])),
            f"Resolution: {incident.get('resolution', '')}",
        )
        if part
    )


class IncidentRetriever:
    """Retrieve historical incidents relevant to a new failure."""

    def __init__(self, config: dict) -> None:
        settings = config.get("retrieval", {})
        self._settings = settings
        self._top_k = int(settings.get("top_k", 3))
        self._incidents: list[dict] = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
        self._collection = None
        self._embeddings = None
        if settings.get("use_chromadb") and os.getenv("OPENAI_API_KEY"):
            try:
                self._build_vector_index()
            except Exception:  # pragma: no cover - depends on optional services
                logger.warning(
                    "ChromaDB index unavailable; falling back to local keyword retrieval",
                    exc_info=True,
                )
                self._collection = None

    # ------------------------------------------------------------------ vector
    def _build_vector_index(self) -> None:
        import chromadb
        from langchain_openai import OpenAIEmbeddings

        self._embeddings = OpenAIEmbeddings(
            model=self._settings.get("embedding_model", "text-embedding-3-small")
        )
        client = chromadb.PersistentClient(
            path=self._settings.get("persist_directory", "chroma_data")
        )
        collection = client.get_or_create_collection(
            self._settings.get("collection", "incidents")
        )
        if collection.count() == 0:
            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict] = []
            for incident in self._incidents:
                chunks = chunk_text(
                    _incident_text(incident),
                    int(self._settings.get("chunk_size", 80)),
                    int(self._settings.get("chunk_overlap", 20)),
                )
                for index, chunk in enumerate(chunks):
                    ids.append(f"{incident['id']}-{index}")
                    documents.append(chunk)
                    metadatas.append({"incident_id": incident["id"]})
            if documents:
                vectors = self._embeddings.embed_documents(documents)
                collection.add(
                    ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors
                )
                logger.info("Indexed %d chunks into ChromaDB", len(documents))
        self._collection = collection

    def _vector_search(self, query: str) -> list[dict]:
        query_vector = self._embeddings.embed_query(query)
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(self._top_k * 4, max(self._collection.count(), 1)),
        )
        ordered_ids: list[str] = []
        for metadata in (result.get("metadatas") or [[]])[0]:
            incident_id = (metadata or {}).get("incident_id")
            if incident_id and incident_id not in ordered_ids:
                ordered_ids.append(incident_id)
        by_id = {incident["id"]: incident for incident in self._incidents}
        return [by_id[item] for item in ordered_ids if item in by_id][: self._top_k]

    # ----------------------------------------------------------------- keyword
    def _keyword_search(self, query: str) -> list[dict]:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))

        def score(incident: dict) -> int:
            incident_tokens = set(re.findall(r"[a-z0-9]+", _incident_text(incident).lower()))
            return len(query_tokens & incident_tokens)

        ranked = sorted(self._incidents, key=score, reverse=True)
        matched = [incident for incident in ranked if score(incident) > 0]
        return (matched or self._incidents)[: self._top_k]

    # ------------------------------------------------------------------ public
    def search(self, query: str) -> tuple[list[dict], str]:
        """Return (incidents, source label) for the given query."""
        if self._collection is not None:
            try:
                hits = self._vector_search(query)
                if hits:
                    return hits, VECTOR_SOURCE
            except Exception:  # pragma: no cover - depends on optional services
                logger.warning("Vector search failed; using keyword fallback", exc_info=True)
        return self._keyword_search(query), LOCAL_SOURCE
