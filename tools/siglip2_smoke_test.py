from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from PIL import Image

from query_engine.semantic_reranker import SigLIP2ImageTextScorer


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal SigLIP2 GPU smoke test")
    parser.add_argument("--model", default="google/siglip2-base-patch16-256")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run this test on a CUDA-enabled PyTorch environment.")

    device = torch.device(args.device)
    image = Image.new("RGB", (256, 256), (128, 128, 128))
    texts = [
        "person riding a motorcycle",
        "person standing still",
        "a parked motorcycle",
    ]

    scorer = SigLIP2ImageTextScorer(model_id=args.model, device=str(device))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    scores = scorer.score_images([image] * len(texts), "")
    # score_images accepts one text for a batch of images. For the semantic
    # smoke test we therefore score each phrase independently while reusing
    # the lazily loaded model.
    scores = np.asarray([
        scorer.score_images([image], text)[0] for text in texts
    ], dtype=np.float32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    print(f"model={args.model}")
    print(f"device={device}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        print(f"gpu={props.name}")
        print(f"vram_gb={props.total_memory / 1024**3:.2f}")
        print(f"peak_allocated_gb={allocated:.2f}")
        print(f"peak_reserved_gb={reserved:.2f}")
    print(f"latency_sec={elapsed:.3f}")
    for text, score in zip(texts, scores):
        print(f"score={float(score):.6f}\t{text}")

    if len(scores) != len(texts) or not np.isfinite(scores).all():
        raise SystemExit("SigLIP2 returned invalid scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
