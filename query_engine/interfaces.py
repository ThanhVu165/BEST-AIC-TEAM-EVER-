"""Stable Query Engine interfaces.

The Query Engine depends on the shared data-layer contract and exposes one
inference boundary to FastAPI. Model implementations are intentionally kept
behind these interfaces.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from data_layer.datastore import DataStore
from schemas import QueryRequest, SearchResponse


class QueryEngine(ABC):
    """Inference boundary consumed by FastAPI and therefore by the UI."""

    @abstractmethod
    def search(self, request: QueryRequest) -> SearchResponse:
        raise NotImplementedError


__all__ = ["DataStore", "QueryEngine"]
