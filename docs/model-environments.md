# Model runtime environments

The query engine keeps expensive multimodal backends optional and lazy. Install only the backend required by the experiment or deployment target.

## Base CI / CPU

```bash
pip install -e '.[dev]'
```

## SigLIP2 semantic reranking

```bash
pip install -e '.[ml-siglip]'
```

Backend: `SigLIP2ImageTextScorer`.

## Qwen VLM

```bash
pip install -e '.[ml-vlm]'
```

Use this environment only when the VLM answer/verification stage is enabled.

## Video-language verification

```bash
pip install -e '.[ml-video]'
```

This environment contains the video decoding/runtime dependencies used by bounded video-window verification.

## Important

The default `ml` extra remains intentionally broad enough for the baseline retrieval stack. CI should not download large model checkpoints. Model weights are fetched lazily by the selected backend and must be benchmarked on the actual AIC dataset before production defaults are changed.
