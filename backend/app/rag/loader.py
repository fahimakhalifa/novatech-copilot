"""Load company documents and split them into retrievable chunks."""
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.rag.scanner import scan

MAX_CHUNK_CHARS = 1500

_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "doc_heading"), ("##", "section")],
    strip_headers=True,  # we re-add a cleaner header ourselves below
)
_size_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_CHARS, chunk_overlap=150)


@dataclass
class ParsedDoc:
    path: str          # relative to the docs folder, e.g. "hr/vacation-policy.md"
    meta: dict         # frontmatter: title, category, access, ...
    body: str          # Markdown without the frontmatter


def parse_markdown(file: Path, docs_root: Path) -> ParsedDoc:
    """Split '---' YAML frontmatter from the Markdown body."""
    text = file.read_text(encoding="utf-8")
    meta: dict = {}
    body = text
    if text.startswith("---"):
        _, front, body = text.split("---", 2)
        meta = yaml.safe_load(front) or {}
    if meta.get("access") not in ("public", "internal"):
        # Fail closed: a document with missing/unknown access is treated as internal.
        meta["access"] = "internal"
    return ParsedDoc(path=file.relative_to(docs_root).as_posix(), meta=meta, body=body.strip())


def chunk_document(doc: ParsedDoc) -> list[Document]:
    title = doc.meta.get("title", Path(doc.path).stem)
    chunks: list[Document] = []

    for piece in _header_splitter.split_text(doc.body):
        section = piece.metadata.get("section", "Overview")
        for i, part in enumerate(_size_splitter.split_text(piece.page_content)):
            content = f"{title} > {section}\n\n{part}"
            chunk_key = f"{doc.path}#{section}#{i}"
            findings = scan(f"{section}\n{part}")
            chunks.append(Document(
                page_content=content,
                metadata={
                    "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_key)),
                    "source": doc.path,
                    "title": title,
                    "section": section,
                    "category": doc.meta.get("category", "general"),
                    "access": doc.meta["access"],
                    "last_updated": str(doc.meta.get("last_updated", "")),
                    "quarantined": bool(findings),
                    "quarantine_reason": ",".join(findings),
                },
            ))
    return chunks


def load_and_chunk(docs_root: Path) -> list[Document]:
    chunks: list[Document] = []
    for file in sorted(docs_root.rglob("*.md")):
        chunks.extend(chunk_document(parse_markdown(file, docs_root)))
    return chunks
