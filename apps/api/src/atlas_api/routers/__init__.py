"""API routers package."""

from atlas_api.routers.documents import router as documents_router
from atlas_api.routers.query import router as query_router

__all__ = ["documents_router", "query_router"]
