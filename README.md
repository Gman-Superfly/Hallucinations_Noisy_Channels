# LLM Hallucinations, Semantic Drift, and Signal Processing

This repository presents preliminary notes on potential links between large language model (LLM) hallucinations and ideas from signal processing, 
uncertainty principles, and information theory. 
We tentatively explore analogies to the Shannon-Nyquist theorem, the Uncertainty principle, the Platonic Representation Hypothesis, 
in-context learning, error correction via Shannon's noisy channel coding theorem and Hamming bounds, 
and thermodynamic constraints—while fully recognizing these cross-domain connections are unusual but testable and currently in being validated.

At its core, the work modestly suggests that hallucinations could stem from "semantic drift" in LLM latent spaces, 
where sparse or ambiguous prompts fail to guide trajectories toward accurate outputs, 
mirroring reconstruction errors in undersampled signals or noisy channels. 
This view aims to inspire testable hypotheses for enhancing LLM reliability, 
such as better prompt engineering or context accumulation, but we emphasize it's far from a proven theory.

For a deeper dive into the mathematical details, hypotheses, limitations, and interdisciplinary analogies, 
please read the full paper in this repo: (HALLUCINATIONS_AND_NOISY_CHANNELS.md). 
This is an early exploration from notes collected since Nov 2024.
