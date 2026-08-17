"""Runtime factories for offline AIC Query Engine deployments."""
from __future__ import annotations

from pathlib import Path

from data_layer.datastore import LocalDataStore
from data_layer.faiss_store import FAISSFrameStore

from .clip_encoder import CLIPTextEncoder
from .engine import BaselineQueryEngine
from .retrieval import ClipCandidateRetriever
from .vlm_answering import TransformersImageAnswerExtractor


def build_clip_baseline_engine(
    *,
    db_path: str | Path,
    index_path: str | Path,
    mapping_path: str | Path,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "auto",
    frame_top_k: int = 5000,
    video_top_k: int = 100,
    object_weight: float = 0.05,
    metadata_weight: float = 0.03,
    ocr_weight: float = 0.02,
    asr_weight: float = 0.02,
    vlm_model_name: str | None = None,
    vlm_device: str | None = None,
    fine_temporal_anchors: int = 20,
    fine_temporal_radius: int = 16,
) -> BaselineQueryEngine:
    """Construct the current canonical baseline runtime.

    BTC CLIP remains the primary retrieval signal. Auxiliary weights are kept
    explicit so they can be ablated and benchmarked independently. The same
    CLIP instance is reused for source-frame temporal refinement.
    """
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
        metadata_weight=metadata_weight,
        ocr_weight=ocr_weight,
        asr_weight=asr_weight,
    )
    answer_extractor = None
    if vlm_model_name:
        answer_extractor = TransformersImageAnswerExtractor(
            vlm_model_name,
            device=vlm_device or device,
        )
    return BaselineQueryEngine(
        retriever,
        answer_extractor=answer_extractor,
        image_encoder=embedder,
        fine_temporal_anchors=fine_temporal_anchors,
        fine_temporal_radius=fine_temporal_radius,
    )
