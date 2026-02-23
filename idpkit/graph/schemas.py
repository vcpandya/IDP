"""Pydantic request/response schemas for the Knowledge Graph."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# --- Response models ---

class EntitySchema(BaseModel):
    id: str
    canonical_name: str
    entity_type: str
    description: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)
    first_document_id: Optional[str] = None
    document_count: int = 1

    model_config = {"from_attributes": True}


class MentionSchema(BaseModel):
    id: str
    entity_id: str
    document_id: str
    node_id: str
    node_title: Optional[str] = None
    mention_text: Optional[str] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None

    model_config = {"from_attributes": True}


class EdgeSchema(BaseModel):
    id: str
    source_entity_id: str
    source_document_id: str
    source_node_id: Optional[str] = None
    target_entity_id: str
    target_document_id: Optional[str] = None
    target_node_id: Optional[str] = None
    relation_type: str
    scope: str = "intra"
    weight: int = 1
    confidence: int = 80
    context_snippet: Optional[str] = None

    model_config = {"from_attributes": True}


class EntityDetailSchema(BaseModel):
    entity: EntitySchema
    mentions: list[MentionSchema] = Field(default_factory=list)
    edges: list[EdgeSchema] = Field(default_factory=list)


class NeighborSchema(BaseModel):
    entity: EntitySchema
    edge: EdgeSchema


class DocumentGraphSummary(BaseModel):
    document_id: str
    entity_count: int = 0
    edge_count: int = 0
    entity_types: dict[str, int] = Field(default_factory=dict)
    top_entities: list[EntitySchema] = Field(default_factory=list)


class CrossDocLink(BaseModel):
    linked_document_id: str
    linked_document_filename: Optional[str] = None
    shared_entities: list[EntitySchema] = Field(default_factory=list)
    edge_count: int = 0


class VisualizationNode(BaseModel):
    id: str
    label: str
    type: str
    document_id: Optional[str] = None


class VisualizationEdge(BaseModel):
    source: str
    target: str
    relation_type: str
    weight: int = 1


class VisualizationData(BaseModel):
    nodes: list[VisualizationNode] = Field(default_factory=list)
    edges: list[VisualizationEdge] = Field(default_factory=list)


# --- LLM extraction models (used internally by builder) ---

class ExtractedEntity(BaseModel):
    name: str
    entity_type: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


class ExtractedRelation(BaseModel):
    source: str
    target: str
    relation_type: str
    confidence: int = 80
    context: str = ""

    model_config = {"extra": "ignore"}


class NodeExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)

    model_config = {"extra": "ignore"}
