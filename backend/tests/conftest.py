"""Shared fixtures: a fresh, seeded in-memory database per test session."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.tools import load_caller
from scripts.seed_db import seed


@pytest.fixture(scope="session")
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False},
                        poolclass=StaticPool)
    seed(eng)
    return eng


@pytest.fixture
def db(engine):
    with Session(engine) as session:
        yield session


@pytest.fixture
def caller(db):
    """caller("mila") → the CallerContext the server would build for that user."""
    return lambda username: load_caller(db, username)
