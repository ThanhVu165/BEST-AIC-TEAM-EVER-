from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from query_engine.semantic_reranker import SigLIP2ImageTextScorer


DEFAULT_TEXTS = (
    "person riding a motorcycle",
    "person standing still",
    "a parked motorcycle",
)


def _resolve_images(patterns: list[str], limit: int) -> list[Path]:
    if limit <= 0:
        raise ValueError("--limit must be > 0")
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern, recursive=True)]
        if matches:
            paths.extend(p for p in matches if p.is_file())
        else:
            p = Path(pattern)
            if p.is_file():
                paths.append(p)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path not in seen and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            seen.add(path)
            unique.append(path)
    unique.sort()
    if len(unique) <= limit:
        return unique
    # Deterministic spread across the matched keyframe set rather than taking
    # only the first video/directory in lexical order.
    indices = np.linspace(0, len(unique) - 1, num=limit, dtype=int)
    return [unique[int(i)] for i in indices]


def main() -> int:
    parser = argparse.ArgumentParser(description="SigLIP2 GPU smoke/real-keyframe benchmark")
    parser.add_argument("--model", default="google/siglip2-base-patch16-256")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--images",
        nargs="*",
        default=[],
        help="Image paths or glob patterns. If omitted, use one synthetic image.",
    )
    parser.add_argument("--limit", type=int, default=12, help="Maximum real keyframes to load")
    parser.add_argument("--text", action="append", dest="texts", help="Text query; repeat for multiple queries")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run this test on a CUDA-enabled PyTorch environment.")
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("--warmup must be >= 0 and --repeat must be > 0")

    device = torch.device(args.device)
    image_paths = _resolve_images(args.images, args.limit)
    if image_paths:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        image_labels = [str(path) for path in image_paths]
        source = "real"
    else:
        images = [Image.new("RGB", (256, 256), (128, 128, 128))]
        image_labels = ["synthetic-gray"]
        source = "synthetic"

    texts = args.texts or list(DEFAULT_TEXTS)
    scorer = SigLIP2ImageTextScorer(model_id=args.model, device=str(device))
    scorer._load()

    # Warm-up removes one-time CUDA/context/model execution from latency.
    for _ in range(args.warmup):
        scorer.score_images(images, texts[0])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    timings: list[float] = []
    scores_by_text: dict[str, np.ndarray] = {}
    for _ in range(args.repeat):
        start = time.perf_counter()
        current: dict[str, np.ndarray] = {}
        for text in texts:
            current[text] = np.asarray(scorer.score_images(images, text), dtype=np.float32)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
        scores_by_text = current

    timings_sorted = sorted(timings)
    p50 = timings_sorted[len(timings_sorted) // 2]
    p95_index = min(len(timings_sorted) - 1, max(0, int(np.ceil(0.95 * len(timings_sorted))) - 1))
    p95 = timings_sorted[p95_index]

    print(f"model={args.model}")
    print(f"device={device}")
    print(f"source={source}")
    print(f"num_images={len(images)}")
    print(f"num_texts={len(texts)}")
    print(f"warmup={args.warmup}")
    print(f"repeat={args.repeat}")
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
    print(f"latency_mean_sec={np.mean(timings):.3f}")
    print(f"latency_p50_sec={p50:.3f}")
    print(f"latency_p95_sec={p95:.3f}")

    for text in texts:
        scores = scores_by_text[text]
        print(f"text={text}")
        for label, score in zip(image_labels, scores):
            print(f"  score={float(score):.6f}\t{label}")
        if not np.isfinite(scores).all():
            raise SystemExit("SigLIP2 returned invalid scores")

    for image in images:
        image.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
