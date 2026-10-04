"""Try retrieval from the terminal, no API key needed."""
import argparse

from app.db.models import Role
from app.rag.retriever import retrieve
from app.rag.vectorstore import get_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--role", default="customer", choices=[r.value for r in Role])
    args = parser.parse_args()

    result = retrieve(args.query, Role(args.role))
    print(f"\nrole={args.role}  sufficient_evidence={result.sufficient}  "
          f"top_relevance={result.top_relevance:.3f}\n")
    for i, c in enumerate(result.chunks, 1):
        preview = c.text.split("\n\n", 1)[-1][:160].replace("\n", " ")
        print(f"{i}. [{c.relevance:.3f}] {c.citation}\n   {preview}...\n")

    get_store().client.close()


if __name__ == "__main__":
    main()
