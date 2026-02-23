"""IDP Kit shared Pydantic models and data schemas."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class TreeNode(BaseModel):
    """A node in the hierarchical document tree structure."""

    title: str
    node_id: Optional[str] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    summary: Optional[str] = None
    prefix_summary: Optional[str] = None
    text: Optional[str] = None
    line_num: Optional[int] = None
    nodes: list[TreeNode] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class IndexResult(BaseModel):
    """Result of indexing a document into a tree structure."""

    doc_name: str
    doc_description: Optional[str] = None
    structure: list[TreeNode] = Field(default_factory=list)


class LLMResponse(BaseModel):
    """Response from an LLM API call."""

    content: str
    finish_reason: str = "stop"  # "stop", "length", etc.
    model: str = ""
    usage: dict = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    """Metadata about a processed document."""

    filename: str
    format: str  # pdf, docx, md, html, xlsx, csv, pptx, image
    file_size: int = 0
    page_count: Optional[int] = None
    total_tokens: Optional[int] = None


class JobStatus(BaseModel):
    """Status of a background processing job."""

    job_id: str
    job_type: str  # index, extract, convert, tool, batch
    status: str = "pending"  # pending, running, completed, failed
    progress: int = 0  # 0-100
    result: Optional[dict] = None
    error: Optional[str] = None


class ToolOptions(BaseModel):
    """Base options for Smart Tools. Each tool extends this."""

    scope: str = "full"  # "full", "sections", "pages"
    section_ids: list[str] = Field(default_factory=list)
    output_format: str = "json"

    model_config = {"extra": "allow"}


class ToolResult(BaseModel):
    """Result from a Smart Tool execution."""

    tool_name: str
    status: str = "success"  # "success", "error"
    data: Optional[dict] = None
    output_file: Optional[str] = None
    error: Optional[str] = None
