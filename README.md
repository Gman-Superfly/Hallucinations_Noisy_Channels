# LLM Hallucinations, Semantic Drift, and Signal Processing

This repository presents preliminary notes on potential links between large language model (LLM) hallucinations and ideas from signal processing, 
uncertainty principles, and information theory. 
We tentatively explore analogies to the Shannon-Nyquist theorem, the Uncertainty principle, the Platonic Representation Hypothesis, 
in-context learning, error correction via Shannon's noisy channel coding theorem and Hamming bounds, 
while fully recognizing these cross-domain connections work through analogous mechanisms and may not be direct 
it gives us a glimpse of the interconnected nature of probabilities in deterministic vs stochastic systems  
all work will be testable and is currently being validated, we are not interested in tentative speculation only work as that
is best left to fake ass hipsters and mongoloid grifters.

At its core, the work modestly suggests that hallucinations could stem from "semantic drift" in LLM latent spaces, 
where sparse or ambiguous prompts fail to guide trajectories toward accurate outputs and construct structures which
semantically meaningful to themselves are not semantically relevant to the users intentions,
the hallucinations construct their own context and create their own space where the meaning is self contained in
a context bubble separate from reality allowing the hallucination to live in a self referential space 
mirroring the spiralling reconstruction errors in undersampled signals or noisy channels when they are allowed to feedback and loop on their outputs. 
This view aims to inspire testable hypotheses for enhancing LLM reliability, 
such as better prompt engineering or context accumulation, but we emphasize it's far from a proven theory.

For a deeper dive into the mathematical details, hypotheses, limitations, and interdisciplinary analogies, 
please read the full paper in this repo: (HALLUCINATIONS_AND_NOISY_CHANNELS.md). 
This is an early exploration from notes collected since Nov 2024.
