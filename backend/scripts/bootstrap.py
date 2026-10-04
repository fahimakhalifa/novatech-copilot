"""Prepare a fresh deployment, then exit. Used by the Docker entrypoint; safe to re-run."""
from sqlalchemy import inspect, select

from app.config import get_settings
from app.db.models import AppUser
from app.db.session import SessionLocal, engine


def main() -> None:
    has_users = False
    if inspect(engine).has_table("app_users"):
        with SessionLocal() as db:
            has_users = db.scalar(select(AppUser).limit(1)) is not None
    if has_users:
        print("bootstrap: database already seeded")
    else:
        from scripts.seed_db import seed
        seed()

    from qdrant_client import QdrantClient
    s = get_settings()
    client = QdrantClient(url=s.qdrant_url) if s.qdrant_url else QdrantClient(path=s.qdrant_path)
    exists = client.collection_exists(s.qdrant_collection)
    client.close()
    if exists:
        print(f"bootstrap: collection '{s.qdrant_collection}' already indexed")
    else:
        from scripts.ingest import main as ingest
        ingest()


if __name__ == "__main__":
    main()
