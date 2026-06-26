# Iteration Selection Rule

This document describes the Android tracker rule used when `iterations_num > 1`.

## Current Rule: Independent Reference-Stack Candidates

The current Android implementation does not use iterations as a zoom/refinement chain.

For every camera frame:

1. Build one centered square `search_frame` from the current full frame.
2. Run the model multiple times on the same `search_frame`.
3. For each iteration, use a different reference-stack candidate:
   - Iteration `0` uses the selected base reference stack.
   - Later iterations use progressively cropped/resized versions of the reference stack.
4. Keep every iteration as a candidate.
5. Select the iteration with the highest `quality` score.

The selected heatmap prediction is mapped directly inside the original centered `search_frame`.
There is no per-iteration crop of the search frame and no mapped zoom coordinate chain.

## Pseudocode

```text
search_frame = center_square_crop(current_frame)
reference_stack = selected_reference_stack

for i in 0..iterations_num-1:
    result[i] = model(reference_stack, search_frame)
    if i < iterations_num-1:
        reference_stack = crop_and_resize_reference_stack(reference_stack)

selected_iter = argmax(result[i].quality)
target = result[selected_iter].heatmap_argmax
```

## Quality Display Behavior

Quality controls display state, not iteration selection:

- If quality display is enabled and the selected quality is below the configured target-lost threshold, the marker is drawn red.
- If quality display is disabled, the selected prediction remains green regardless of quality.
- Experimental previous-frame stack updates still use their own fixed quality threshold.

## Previous Rule: Subpixel Convergence Idea

The previous rule is kept here only as a useful idea for a future subpixel convergence experiment.
It is not used by the current Android runtime.

That rule treated iterations as a refinement chain:

1. Reject the detection if the first iteration quality was below `0.30`.
2. Continue accepting later refinements only while quality stayed at or above `0.75`.
3. Prefer the last valid iteration above `0.90`.
4. Otherwise select the highest-quality valid iteration.

This can still be a good basis for a different mode whose goal is subpixel convergence after zooming into the target.
It should not be mixed with the current max-quality reference-stack candidate rule.
