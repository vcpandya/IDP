"""
IDP Kit Engine — Core document indexing engine (PageIndex).

Vectorless, reasoning-based RAG system that transforms documents
into hierarchical tree-structured indices for LLM retrieval.
"""

from .page_index import page_index, page_index_main
from .page_index_md import md_to_tree
from .utils import config, ConfigLoader

__all__ = [
    "page_index",
    "page_index_main",
    "md_to_tree",
    "config",
    "ConfigLoader",
]
