# Next Work Session TODOs

1. **Scale redundancy/geometry experiments** on open-weight LLMs (e.g., run the semantic redundancy ratio and geometric alignment notebooks against a real model, collect metrics).
2. **Run controllability/observability diagnostics** on real prompts (measure control projections and entropy using open-weight checkpoints or API results).
3. **Document the new findings** in `README.md` and `paper/draft.md`, including any figures/metrics generated above.

A couple of additions could strengthen the draft:

1. **Semantic Redundancy Measurement Results** – Highlight the new TF-IDF/transformer redundancy experiments. A short paragraph in Section 2 or 3 describing the real-prompt heatmap (with the figure) would show how the theory applies beyond toy signals.

2. **Geometric Alignment & Control Diagnostics** – Section 2 could add a sentence or small subsection referencing the alignment metrics and control-projection vs. entropy plot. This ties the control-theoretic framing (1.2.3) to concrete evidence.

3. **Terminology Clarity** – In the draft’s overview or intro, add a line similar to the README note so reviewers know “semantic redundancy threshold” is the measured quantity and “Nyquist” is an analogy/control experiment.

Beyond that, the structure already mirrors the README: framework → experiments → hypotheses → research directions. If we get real LLM data later, we can slot it into Section 2 and the research directions, but for now the new notebooks/figures would be the main additions.