"""Late-verification query-engine facade.

Keeps the canonical baseline engine intact while providing an explicit,
model-pluggable KIS cascade:

CLIP/FAISS -> semantic/temporal rerank -> bounded video verification -> final rank.

The expensive verifier is only evaluated on the already-ranked candidate pool.
"""
from __future__ import annotations

from typing import Any

from schemas import QueryRequest

from .engine import BaselineQueryEngine
from .late_verification import LateVerificationConfig, verify_candidate_windows
from .video_verifier import VideoVerifierConfig, build_video_verifier


class VerifiedQueryEngine(BaselineQueryEngine):
    """Baseline engine with optional bounded video-model verification for KIS."""

    def __init__(
        self,
        *args: Any,
        video_verifier: Any | None = None,
        video_verifier_config: VideoVerifierConfig | None = None,
        late_verification_config: LateVerificationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.video_verifier_config = video_verifier_config or VideoVerifierConfig()
        self.late_verification_config = late_verification_config or LateVerificationConfig(
            enabled=False,
            candidate_limit=3,
            weight=0.10,
        )
        self.video_verifier = video_verifier or build_video_verifier(self.video_verifier_config)

    def _solve_kis(self, spec):
        base = super()._solve_kis(spec)
        config = self.late_verification_config
        if not config.enabled or self.video_verifier is None or not base:
            return base

        scores = verify_candidate_windows(
            base,
            datastore=self.retriever.datastore,
            query=spec.text,
            verifier=self.video_verifier,
            config=config,
        )
        if not scores:
            return base

        weight = float(config.weight)
        for candidate in base:
            key = (str(candidate["video_id"]), int(candidate["frame_id"]))
            verification = scores.get(key)
            if verification is None:
                continue
            base_score = max(0.0, min(1.0, float(candidate["score"])))
            final_score = (1.0 - weight) * base_score + weight * verification
            candidate["score"] = final_score
            candidate["rerank_score"] = final_score
            evidence = candidate.setdefault("evidence", {})
            evidence["video_verification_score"] = verification
            evidence["video_verification_model"] = getattr(
                self.video_verifier_config, "model_id", None
            )
            evidence["video_verification_weight"] = weight
            evidence["rerank_score"] = final_score

        base.sort(
            key=lambda item: (
                -float(item["score"]),
                str(item["video_id"]),
                int(item["frame_id"]),
            )
        )
        for rank, candidate in enumerate(base, start=1):
            candidate["rank"] = rank
        return base[: self.final_limit]


def build_verified_query_engine(*args: Any, **kwargs: Any) -> VerifiedQueryEngine:
    """Construct the late-verification engine without forcing model loading."""
    return VerifiedQueryEngine(*args, **kwargs)
