"""Curated learning-resource library and its retriever (tags plus BM25, no embeddings)."""

from .retriever import (
    Hit,
    Resource,
    detect_tags,
    library_urls,
    load_library,
    resources_for_requirements,
    resources_for_skills,
    retrieve,
    skill_gap_resources_block,
    strip_unknown_links,
    unknown_links,
)

__all__ = [
    "Hit",
    "Resource",
    "detect_tags",
    "library_urls",
    "load_library",
    "resources_for_requirements",
    "resources_for_skills",
    "retrieve",
    "skill_gap_resources_block",
    "strip_unknown_links",
    "unknown_links",
]
