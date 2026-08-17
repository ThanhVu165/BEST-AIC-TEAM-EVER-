# Manual semantic benchmark protocol

The local AIC material does not provide official public ground truth for this experiment. Therefore semantic-model selection must use a human-curated benchmark rather than pretending that a guessed label is ground truth.

## Protocol

1. Select real AIC keyframes from different videos.
2. For each query, create an unambiguous positive frame and a hard-negative frame.
3. Prefer minimal semantic contrasts:
   - riding vs standing beside
   - repairing vs standing near
   - entering vs standing beside
   - carrying vs standing with object nearby
   - opening vs holding closed object
4. Do not include pairs where the distinction cannot be judged from one frame.
5. Keep positive and negative frames from comparable visual contexts when possible.
6. Record the reason for the label in `notes`.
7. Run SigLIP2 scoring only after the human labels are fixed.

## Metrics

For every pair:

`margin = score(query, positive) - score(query, negative)`

Report:

- pairwise accuracy: fraction of pairs where positive > negative
- mean margin
- qualitative failure cases

Pairwise accuracy is a diagnostic, not an official AIC metric. It must not be presented as AIC recall or Final Score.

## Model-selection rule

A semantic model should be kept only when it provides a repeatable qualitative or quantitative benefit on the manually reviewed hard-negative set without an unacceptable inference cost. The official AIC submission metrics still require evaluation against the competition's actual evaluator when available.
