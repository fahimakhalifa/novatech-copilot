"""Vector store setup: dense + sparse (BM25) embeddings in one Qdrant collection."""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

from app.config import get_settings


@lru_cache
def dense_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=get_settings().embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache
def sparse_embeddings() -> FastEmbedSparse:
    # BM25 as sparse vectors: exact-token matching for SKUs, order numbers, names.
    return FastEmbedSparse(model_name="Qdrant/bm25")


def _connection_kwargs() -> dict:
    s = get_settings()
    return {"url": s.qdrant_url} if s.qdrant_url else {"path": s.qdrant_path}


def build_store(documents, ids) -> QdrantVectorStore:
    """Create (or recreate) the collection from chunks. Used by the ingest script."""
    return QdrantVectorStore.from_documents(
        documents,
        ids=ids,
        embedding=dense_embeddings(),
        sparse_embedding=sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        collection_name=get_settings().qdrant_collection,
        force_recreate=True,
        **_connection_kwargs(),
    )


@lru_cache
def get_store() -> QdrantVectorStore:
    """Open the existing collection. Cached: one client per process."""
    return QdrantVectorStore.from_existing_collection(
        embedding=dense_embeddings(),
        sparse_embedding=sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        collection_name=get_settings().qdrant_collection,
        **_connection_kwargs(),
    )
