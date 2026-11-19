"""
Postgres/pgvector connection helper.

Registering pgvector's adapter on the connection lets us pass/receive Python
lists directly as the `vector` column type (psycopg would otherwise have no
idea how to serialize a list[float] into pgvector's wire format).
"""

from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from rag.config import get_settings


@contextmanager
def get_connection():
    settings = get_settings()
    conn = psycopg.connect(settings.postgres_dsn, autocommit=True)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()
