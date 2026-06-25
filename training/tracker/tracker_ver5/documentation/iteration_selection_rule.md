# Iteration Selection Rule

This document summarizes the quality-based selection rule used by the tracker visual test when running with `--iterations_num`.

## Inputs

For each refinement iteration, the visual test stores:

- The predicted position mapped back to the original 256x256 display space.
- The iteration-local prediction.
- The model quality score for that iteration.

The final displayed prediction is selected from the iteration results according to the quality scores.

## Rule

1. If there are no iteration results, reject the detection.
2. If the first iteration quality is below `0.30`, reject the detection.
3. Otherwise, create the list of valid iterations:
   - Iteration `0` is valid.
   - Starting from iteration `1`, keep accepting iterations while `quality >= 0.75`.
   - Stop at the first iteration whose quality is below `0.75`; that iteration and all later iterations are ignored.
4. From the valid iterations:
   - If one or more valid iterations have `quality > 0.90`, select the last such iteration.
   - Otherwise, select the valid iteration with the highest quality score.

## Pseudocode

```text
if qualities is empty:
    reject

if qualities[0] < 0.30:
    reject

valid_iters = [0]
for i in 1..N-1:
    if qualities[i] < 0.75:
        break
    valid_iters.append(i)

high_quality_iters = [i for i in valid_iters if qualities[i] > 0.90]
if high_quality_iters is not empty:
    selected_iter = high_quality_iters[-1]
else:
    selected_iter = argmax(qualities[i] for i in valid_iters)
```

## Display Behavior

- The selected iteration is highlighted as `SELECTED`.
- Iterations excluded after the `0.75` cutoff are marked `IGNORED`.
- If the detection is rejected because the first iteration quality is below `0.30`, iteration `0` is marked `REJECTED`.
- The search-frame prediction marker is drawn at the mapped position of the selected iteration.

## Rationale

The rule keeps the refinement behavior conservative:

- A very weak first match rejects the detection immediately.
- Later refinements are allowed only while quality remains stable enough.
- A high-confidence refinement above `0.90` is preferred, with the latest such refinement used as the final zoomed-in result.
- If no refinement reaches high confidence, the best quality among the valid iterations is used.
