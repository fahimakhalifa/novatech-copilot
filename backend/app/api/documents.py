"""Document management: staff can list documents, admins can upload and delete them."""
import re

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.auth import require_admin, require_staff
from app.rag import indexer
from app.tools import CallerContext

router = APIRouter(prefix="/api/documents", tags=["documents"])

MAX_DOC_CHARS = 100_000
FILENAME = r"^[a-z0-9][a-z0-9-]{0,60}\.md$"


class UploadRequest(BaseModel):
    category: str = Field(pattern=f"^({'|'.join(indexer.CATEGORIES)})$")
    filename: str = Field(pattern=FILENAME, description="e.g. 'gift-cards.md'")
    content: str = Field(min_length=1, max_length=MAX_DOC_CHARS,
                         description="Markdown, ideally with YAML frontmatter (title, access).")


def _path(category: str, filename: str):
    path = (indexer.docs_root() / category / filename).resolve()
    # Belt and braces: the patterns already forbid "/" and "..", but check the result too.
    if not path.is_relative_to(indexer.docs_root()):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid document path.")
    return path


@router.get("")
def list_documents(_: CallerContext = Depends(require_staff)) -> list[dict]:
    return indexer.list_documents()


@router.post("", status_code=status.HTTP_201_CREATED)
def upload_document(req: UploadRequest, caller: CallerContext = Depends(require_admin)) -> dict:
    path = _path(req.category, req.filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    replaced = path.exists()
    path.write_text(req.content, encoding="utf-8")
    chunks = indexer.index_file(path)
    return {
        "path": f"{req.category}/{req.filename}",
        "replaced": replaced,
        "access": chunks[0].metadata["access"] if chunks else "internal",
        "chunks": len(chunks),
        "quarantined": [{"section": c.metadata["section"], "reason": c.metadata["quarantine_reason"]}
                        for c in chunks if c.metadata["quarantined"]],
    }


@router.delete("/{category}/{filename}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(category: str, filename: str, _: CallerContext = Depends(require_admin)) -> None:
    if category not in indexer.CATEGORIES or not re.fullmatch(FILENAME, filename):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid document path.")
    path = _path(category, filename)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    indexer.remove_file(path)
