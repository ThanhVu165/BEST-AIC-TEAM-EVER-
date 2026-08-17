"""Query Engine package."""

from .answering import AnswerEvidence, AnswerResult, UnavailableAnswerExtractor
from .clip_encoder import CLIPTextEncoder
from .engine import BaselineQueryEngine
from .late_verification import LateVerificationConfig, verify_candidate_windows
from .mock_engine import MockQueryEngine
from .query_understanding import QueryEventSpec, QuerySpec, understand_query
from .semantic import SemanticQuery, decompose_query, semantic_score
from .semantic_reranker import (
    SemanticRerankConfig,
    SigLIP2ImageTextScorer,
    build_semantic_reranker,
    semantic_text,
)
from .video_verifier import InternVideo3Verifier, VideoVerifierConfig, build_video_verifier
from .vlm_answering import TransformersImageAnswerExtractor
from .vlm_verifier import Qwen3VLVerifier, VLMVerifierConfig, build_vlm_verifier

__all__ = [
    "AnswerEvidence",
    "AnswerResult",
    "BaselineQueryEngine",
    "CLIPTextEncoder",
    "InternVideo3Verifier",
    "LateVerificationConfig",
    "MockQueryEngine",
    "Qwen3VLVerifier",
    "QueryEventSpec",
    "QuerySpec",
    "SemanticQuery",
    "SemanticRerankConfig",
    "SigLIP2ImageTextScorer",
    "TransformersImageAnswerExtractor",
    "UnavailableAnswerExtractor",
    "VLMVerifierConfig",
    "VideoVerifierConfig",
    "build_semantic_reranker",
    "build_video_verifier",
    "build_vlm_verifier",
    "decompose_query",
    "semantic_score",
    "semantic_text",
    "understand_query",
    "verify_candidate_windows",
]
