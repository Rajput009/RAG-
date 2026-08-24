"""Declarative corpus specification. All values here are spec literals."""

from pydantic import BaseModel, Field


class TenantSpec(BaseModel):
    tenant_id: str
    display_name: str
    docs_per_type: dict[str, int]


class CorpusSpec(BaseModel):
    seed: int = 42
    tenants: list[TenantSpec]
    # doc types listed here produce v1..v3 version families; family count comes
    # from TenantSpec.docs_per_type
    versioned_types: set[str] = Field(default_factory=set)
    injection_docs: int = 0
    distractor_sets: int = 0
    unanswerable_topics: list[str] = Field(default_factory=list)
