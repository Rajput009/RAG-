"""Shared fixtures: isolated Postgres per session, fresh schema per test."""

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING

import pytest
from atlas_api.main import create_app
from atlas_core.config import Settings
from atlas_core.db.models import Base
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from testcontainers.postgres import PostgresContainer

if TYPE_CHECKING:
    from fastapi import FastAPI

PG_IMAGE = "paradedb/paradedb:latest"
PG_USER = "atlas"
PG_PASSWORD = "atlas"
PG_DB = "atlas_test"


@pytest.fixture(scope="session")
def pg_url() -> Iterator[str]:
    with PostgresContainer(
        PG_IMAGE, username=PG_USER, password=PG_PASSWORD, dbname=PG_DB
    ) as postgres:
        host = postgres.get_container_host_ip()
        port = postgres.get_exposed_port(5432)
        yield f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@{host}:{port}/{PG_DB}"


@pytest.fixture()
async def db_engine(pg_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(pg_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture()
async def client(db_engine: AsyncEngine) -> AsyncIterator[tuple[AsyncClient, "FastAPI"]]:
    app = create_app(Settings(database_url=db_engine.url.render_as_string(hide_password=False)))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http, app
