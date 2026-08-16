"""Runtime factory for the real CLIP retrieval baseline."""
from __future__ import annotations

from pathlib import Path

from data_layer.datastore import LocalDataStore
from data_layer.faiss_store import FAISSFrameStore

from .clip_encoder import CLIPTextEncoder
from .engine import BaselineQueryEngine
from .retrieval import ClipCandidateRetriever


def build_clip_baseline_engine(
    *,
    db_path: str | Path,
    index_path: str | Path,
    mapping_path: str | Path,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "auto",
    frame_top_k: int = 200,
    video_top_k: int = 50,
    max_frames_per_video: int = 3,
    object_weight: float = 0.10,
) -> BaselineQueryEngine:
    """Construct a real offline-data-backed CLIP retrieval engine."""
    clip_store = FAISSFrameStore(index_path=index_path, mapping_path=mapping_path)
    clip_store.load()
    datastore = LocalDataStore(db_path=db_path, clip_index=clip_store)
    embedder = CLIPTextEncoder(model_name=model_name, device=device)
    retriever = ClipCandidateRetriever(
        datastore,
        embedder,
        frame_top_k=frame_top_k,
        video_top_k=video_top_k,
        max_frames_per_video=max_frames_per_video,
        object_weight=object_weight,
    )
    return BaselineQueryEngine(retriever)
