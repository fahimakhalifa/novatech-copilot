"""Index company documents into Qdrant."""
from collections import Counter

from app.config import get_settings
from app.rag.loader import load_and_chunk
from app.rag.vectorstore import build_store


def main() -> None:
    docs_dir = get_settings().docs_dir
    chunks = load_and_chunk(docs_dir)
    store = build_store(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    store.client.close()

    by_access = Counter(c.metadata["access"] for c in chunks)
    n_docs = len({c.metadata["source"] for c in chunks})
    print(f"Indexed {len(chunks)} chunks from {n_docs} documents "
          f"({by_access['public']} public, {by_access['internal']} internal).")
    for c in chunks:
        if c.metadata["quarantined"]:
            print(f"  QUARANTINED {c.metadata['source']} > {c.metadata['section']} "
                  f"({c.metadata['quarantine_reason']})")


if __name__ == "__main__":
    main()
