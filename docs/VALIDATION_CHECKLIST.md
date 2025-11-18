# Semantic Reconstruction Validation Checklist

Use this checklist before running large experiments or drafting new sections of the paper.

1. **Entity coverage** – Have all significant artifacts (prompts, intermediate states, operations) been modeled as entities with clear lifecycle assumptions?
2. **Information sufficiency** – Does the prompt meet the semantic Nyquist threshold for the concept being queried? Document how the threshold was estimated (token count, entropy, mutual information proxy).
3. **Attention diagnostics** – Are attention maps sharp enough to guarantee high signal-to-noise ratio? Flag heads with diffuse or sink-like patterns.
4. **Semantic redundancy ratio** – Estimate the proportion of informative vs. noisy tokens ($\rho$). Lengthen prompts only if $\rho$ remains high; otherwise consider pruning or gating.
5. **Trajectory stability** – Measure latent drift or Lyapunov-style divergence across perturbed prompts before decoding.
6. **Redundancy strategy** – If the task is high risk, specify the error-control mechanism (CoT, self-consistency, external evidence) and expected overhead.
7. **Energy audit** – Record FLOPs or latency deltas for each prompt setting to track the energy–hallucination trade-off.
8. **Registry traceability** – Ensure all artifacts used in the experiment (datasets, configs, notebook versions) are registered and reproducible.

