# Core Framework: LLM Forward Pass as a Semantic Communication Channel

## Abstract

This technical note refines the signal-processing interpretation of LLM hallucinations introduced in the main README. We formalize the forward pass as a noisy communication channel obeying Nyquist-Shannon constraints, quantify how transformer depth and attention patterns determine semantic bandwidth, and derive analytic risk metrics for reconstruction failure. The document serves as the canonical reference for the mathematical apparatus used throughout the repository.

## Framework

### Overview

This document outlines the core framework for understanding LLM hallucinations through signal processing principles, 
specifically treating the transformer forward pass as a communication channel governed by Nyquist-Shannon sampling theory.

### The Core Analogy

### Forward Pass = Communication Channel
- Each layer represents a stage in signal transmission
- The sequence of hidden states forms a trajectory through semantic space
- Layer depth determines the "sampling rate" for semantic reconstruction

### Low-Pass Component
- **Bandwidth**: The ratio of layers to sequence complexity determines maximum semantic resolution
- **Sampling Rate**: More layers = higher sampling rate = better signal reconstruction
- **Undersampling**: Insufficient layers lead to semantic aliasing (hallucinations)
- **Filter Response**: Transformers act as low-pass filters in semantic space, attenuating high-frequency semantic patterns

### Ambiguity Score
- Derived from attention head relationships to the sequence context
- Measures semantic clarity and focus
- High ambiguity = noisy channel = more layers needed for reliable reconstruction

### Mathematical Framework

### Channel Capacity
Following Shannon's formula for communication channels:
```
C = B * log2(1 + SNR)
```
Where:
- **C** = Channel capacity (semantic reconstruction capability)
- **B** = Effective bandwidth (function of model architecture)
- **SNR** = Signal-to-noise ratio (attention clarity vs. semantic complexity)

### Effective Bandwidth
The bandwidth is determined by the model's ability to "sample" semantic space, modified by low-pass filter response:
```
B = base_bandwidth * low_pass_response
```
Where:
- **base_bandwidth** = f(num_layers, attention_focus)
- **low_pass_response** = filter response for the given semantic frequency
- More layers = higher base bandwidth and lower cutoff frequency
- Attention focus determines how efficiently the bandwidth is utilized

### Signal-to-Noise Ratio
```
SNR = concept_clarity / attention_ambiguity
```
- **Concept clarity**: How well-defined the target concept is
- **Attention ambiguity**: How scattered the attention patterns are across the sequence

### Low-Pass Filtering in Semantic Space
Transformers exhibit low-pass filter behavior, inspired by Heaviside's telegraph line analysis and Nyquist's sampling theory:

```
semantic_frequency = concept_complexity / attention_resolution
low_pass_response = 1 / (1 + (semantic_frequency / cutoff_frequency)^2)
```

Where:
- **semantic_frequency**: How "fast" semantic patterns change across layers
- **cutoff_frequency**: Determined by layer depth and attention mechanisms
- **low_pass_response**: How well semantic patterns are preserved (1 = perfect, 0 = completely filtered)

### Energy Landscape and Trajectory Dynamics
The transformer creates an energy landscape where tokens follow different paths:

- **Easy Paths (Low Energy)**: Simple, common concepts flow smoothly through layers with minimal distortion
- **Hard Paths (High Energy)**: Complex, rare concepts get attenuated or distorted, causing tokens to "spin around attractors"
- **Filter Cutoff**: Semantic complexity threshold where the transformer starts acting as a low-pass filter

### Hallucination Risk
```
Hallucination_Risk = max(0, 1 - (C / concept_complexity))
```
- When channel capacity < concept complexity, risk increases
- Risk approaches 1 as capacity becomes insufficient for reconstruction

### Nyquist-Shannon Connection
- **Sampling Rate**: Number of layers determines semantic sampling density
- **Bandwidth**: Layer depth sets maximum concept complexity for faithful reconstruction
- **Aliasing**: Insufficient layers cause semantic distortion (hallucinations)
- **Phase Alignment**: Attention patterns provide temporal coherence for reconstruction

### Key Insights

1. **Predictable Failures**: Hallucinations are not random but predictable reconstruction failures due to insufficient semantic sampling depth
2. **Architecture Constraints**: Model depth creates fundamental limits on semantic complexity - more layers = higher capacity
3. **Input Sensitivity**: Same model may hallucinate on complex concepts but be reliable on simple ones, following the capacity vs. complexity relationship
4. **Attention Quality**: Scattered attention patterns increase semantic noise, reducing effective SNR and channel capacity
5. **Shannon Foundation**: The framework follows established information theory principles, making it mathematically principled rather than ad-hoc
6. **Phase Alignment**: Attention patterns provide the "temporal" coherence needed for semantic reconstruction, analogous to phase information in signal processing
7. **Low-Pass Filtering**: Transformers act as semantic low-pass filters, explaining why simple concepts are recalled easily while complex reasoning gets distorted
8. **Energy Landscape**: The transformer creates predictable energy barriers that determine which semantic paths are "easy" vs. "hard" for tokens to traverse

### Applications

### Model Selection
- Choose appropriate model depth for task complexity
- Balance computational cost vs. semantic bandwidth

### Prompt Engineering
- Understand when additional context is needed
- Predict minimum information requirements

### Architecture Design
- Optimize layer depth for target applications
- Design efficient semantic sampling strategies

### Hallucination Prevention
- Proactively identify risky inputs
- Quantify reliability thresholds

### Research Directions

1. **Ambiguity Metrics**: Develop quantitative measures of attention scatter
2. **Bandwidth Calculation**: Formalize the relationship between layers and semantic capacity
3. **Threshold Prediction**: Determine minimum layers needed for given concept complexity
4. **Empirical Validation**: Test predictions on real transformer architectures

## Experiments

- `rope_accumulation.py`: validates the RoPE-driven phase-accumulation story by comparing latent drift under short vs. long prompts.
- `samplig_reconstruction.py`: reproduces Nyquist sampling and demonstrates how undersampling inflates reconstruction error in the reference domain.
- `simpleLM_test.py`: measures trajectory drift in a trained micro-transformer, showing statistically significant improvements when prompts cross the semantic information threshold.

## Hypotheses

1. **Semantic Bandwidth Limit:** Transformer depth and attention focus set a maximum concept complexity that can be reconstructed without aliasing.
2. **Attention SNR Predictor:** The ratio of concept clarity to attention ambiguity predicts hallucination risk before decoding.
3. **Phase Alignment Requirement:** Sufficient positional rotation (via RoPE or comparable encodings) is necessary to accumulate semantic phase and avoid drift.
4. **Context-Length Optimum:** Increasing context beyond the semantic Nyquist rate introduces attention sinks and raises hallucination probability.

## Conclusion

This framework transforms the "messy practice" of constructing prompts "science" of semantic signal processing, 
providing principled predictions for when hallucinations will occur based on architecture constraints, input complexity, and attention quality.

---

*This document captures the core discussion about treating LLM hallucinations as a signal-processing problem, where insufficient semantic sampling depth leads to predictable reconstruction failure.*
