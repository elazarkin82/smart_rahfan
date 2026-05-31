# Tracker Ver 4 (Conceptual Roadmap)

This version will introduce a major shift from the `hist` + `curr` Siamese tracking approach towards a **Multi-Scale Reference Stack** with a **Detection Quality** output. 

## Architectural Changes

1. **Target Reference Stack (Instead of single `hist`):**
   * Rather than a single historical frame, the network will receive a stack of 8-16 cropped layers of the target at different scales. 
   * Example: A small crop resized to `16x16` up to a highly zoomed-in `4x4` crop resized to `16x16`. 
   * This provides the network with a rich, multi-scale reference dictionary of what the target looks like under different magnifications (simulating the UAV getting closer to the landing pad or package).

2. **Dual Outputs:**
   * **Output 1 (Localization):** An exponential heatmap (cone) representing the continuous spatial location of the target within the search frame, similar to `tracker_ver3_lite`.
   * **Output 2 (Quality Score):** A scalar value (0.0 to 1.0) indicating the confidence/quality of the identification. This acts as a tracking failure/drift detector.

## Dataset Generation Strategy (Synthetic)

* **Multi-Scale Crops:** From a single frame, the system will generate the multi-scale crop stack directly around the target point.
* **Randomized Current Frame:** The `curr_frame` will be randomly cropped around the target. The target may appear anywhere within the frame.
* **Negative Samples (Drift Simulation):** Crucially, the dataset will include frames where the target *is not present* (moves outside the frame). For these samples, the heatmap should be empty, and the **Quality Score** output must be 0.0.
* This synthetic generation will yield many samples per original frame, varying the surrounding background significantly to prevent overfitting.
