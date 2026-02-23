"""IDP Kit API Routes."""

from .auth import router as auth_router
from .documents import router as documents_router
from .indexing import router as indexing_router
from .jobs import router as jobs_router
from .retrieval import router as retrieval_router
from .agent import router as agent_router
from .tools import router as tools_router
from .generation import router as generation_router
from .processing import router as processing_router
from .plugins import router as plugins_router
from .graph import router as graph_router
from .batch import router as batch_router

__all__ = [
    "auth_router",
    "documents_router",
    "indexing_router",
    "jobs_router",
    "retrieval_router",
    "agent_router",
    "tools_router",
    "generation_router",
    "processing_router",
    "plugins_router",
    "graph_router",
    "batch_router",
]
