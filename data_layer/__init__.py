"""Shared data access layer for the AIC 2026 system."""

from .datastore import DataStore, LocalDataStore

__all__ = ["DataStore", "LocalDataStore"]
