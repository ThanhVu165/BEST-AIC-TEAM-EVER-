"""Runtime factories for offline AIC Query Engine deployments."""
from __future__ import annotations

from pathlib import Path

from data_layer.datastore import LocalDataStore
from data_layer.faiss_store import FAISSFrameStore

from .clip_encoder import CLIPTextEncoder
from .engine import BaselineQueryEngine
from .retrieval import ClipCandidateRetriever
from .vlm_answering import TransformersImageAnswerExtractor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "database" / "aic2026.sqlite"
DEFAULT_INDEX = ROOT / "indexes" / "clip_vit_b32.faiss"
DEFAULT_MAPPING = ROOT / "indexes" / "clip_vit_b32.mapping.json"


def build_clip_baseline_engine(
    *,
    db_path: str | Path = DEFAULT_DB,
    index_path: str | Path = DEFAULT_INDEX,
    mapping_path: str | Path = DEFAULT_MAPPING,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "auto",
    frame_top_k: int = 200,
    video_top_k: int = 50,
    object_weight: float = 0.10,
    vlm_model_name: str | None = None,
    vlm_device: str | None = None,
) -> BaselineQueryEngine:
    """Construct an offline-data-backed CLIP engine with optional VLM QA."""
    clip_store = FAISSFrameStore(index_path=index_path, mapping_path=mapping_path)
    clip_store.load()
    datastore = LocalDataStore(db_path=db_path, clip_index=clip_store)
    embedder = CLIPTextEncoder(model_name=model_name, device=device)
    retriever = ClipCandidateRetriever(
        datastore,
        embedder,
        frame_top_k=frame_top_k,
        video_top_k=video_top_k,
        object_weight=object_weight,
    )
    answer_extractor = None
    if vlm_model_name:
        answer_extractor = TransformersImageAnswerExtractor(
            vlm_model_name,
            device=vlm_device or device,
        )
    return BaselineQueryEngine(retriever, answer_extractor=answer_extractor)
