"""Add, replace, remove and list single documents in the live index (used by the API)."""
from pathlib import Path

from qdrant_client import models

from app.config import get_settings
from app.rag.loader import chunk_document, parse_markdown
from app.rag.vectorstore import get_store

CATEGORIES = ("customer", "hr", "internal", "products")


def docs_root() -> Path:
    return get_settings().docs_dir.resolve()


def _delete_points(source: str) -> None:
    store = get_store()
    store.client.delete(
        collection_name=store.collection_name,
        points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source))])),
    )


def index_file(path: Path) -> list:
    """(Re)index one document: remove its old chunks, then add the new ones."""
    chunks = chunk_document(parse_markdown(path, docs_root()))
    _delete_points(path.relative_to(docs_root()).as_posix())
    if chunks:
        get_store().add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return chunks


def remove_file(path: Path) -> None:
    _delete_points(path.relative_to(docs_root()).as_posix())
    path.unlink()


def list_documents() -> list[dict]:
    root = docs_root()
    result = []
    for file in sorted(root.rglob("*.md")):
        doc = parse_markdown(file, root)
        chunks = chunk_document(doc)
        result.append({
            "path": doc.path,
            "title": doc.meta.get("title", file.stem),
            "category": doc.path.split("/")[0],
            "access": doc.meta["access"],
            "last_updated": str(doc.meta.get("last_updated", "")),
            "chunks": len(chunks),
            "quarantined": [{"section": c.metadata["section"], "reason": c.metadata["quarantine_reason"]}
                            for c in chunks if c.metadata["quarantined"]],
        })
    return result
