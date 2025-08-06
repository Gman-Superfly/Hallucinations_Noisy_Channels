
## **A Modeling Framework for LLM Hallucinations: Semantic Drift, Information Thresholds, and Reconstruction Failure**

## WORK IN PROGRESS CHECK BACK SOON

### Abstract

This paper proposes a novel modeling framework for understanding Large Language Model (LLM) hallucinations by drawing principled analogies from signal processing, information theory, and physics. We posit that hallucinations are not random errors but a predictable form of reconstruction failure. This failure, termed **Semantic Drift**, occurs when the contextual information in a prompt falls below a required threshold for the given conceptual complexity, analogous to how undersampling a signal below the Nyquist rate leads to aliasing. The framework is grounded in the Fourier uncertainty principle, which sets fundamental trade-offs in information localization. We connect this principle to the structure of LLM latent spaces, suggesting that concepts exist as stable attractor manifolds. When a prompt provides insufficient information, the model's internal trajectory fails to converge to the correct manifold, resulting in a hallucinatory output. This is further contextualized using the Free Energy Principle (FEP) to explain in-context learning (ICL) as a trajectory-guiding mechanism and Shannon's noisy channel coding theorem to model error correction, where techniques like Chain-of-Thought (CoT) prompting add redundancy to stabilize the semantic "signal." We also show that too much context can degrade the performance by creating in context attractors ICA. By integrating thermodynamic constraints like Landauer's principle, we link inefficient, drifting trajectories and context fatigue to higher computational costs. This synthesis yields a set of testable hypotheses aimed at transforming our understanding of LLM reliability from an empirical art to a predictive science.

### 1. Introduction

The remarkable capabilities of Large Language Models (LLMs) are shadowed by their propensity for "hallucinations"—the generation of fluent but factually incorrect or nonsensical outputs. While often treated as a black-box problem, we argue that hallucinations can be understood through a principled framework inspired by fundamental laws of information. This paper builds a bridge between the Fourier uncertainty principle, a cornerstone of physics and signal processing, and the internal dynamics of LLMs.

The Fourier uncertainty principle establishes a fundamental trade-off: a signal cannot be simultaneously localized in two conjugate domains (e.g., time and frequency). This principle manifests as the Heisenberg uncertainty principle in QM (Δx · Δp ≥ ℏ/2) and as the Gabor limit underlying the Shannon-Nyquist sampling theorem in signal processing (σ_t · σ_ω ≥ 1/2). We propose that a similar informational trade-off governs the representational capacity of LLMs.

Our central thesis is that LLM hallucinations are a form of **reconstruction failure due to an information deficit**. We introduce the concept of **Semantic Drift**: when a prompt provides insufficient contextual information to specify a concept of a certain complexity, the LLM's internal state trajectory fails to stabilize within the correct region of its latent space. This failure is analogous to aliasing in signal reconstruction, where an insufficient sampling rate leads to an irreversible distortion of the original signal.

This paper does not claim that LLMs are literal signal processors or QM systems. Instead, we argue that the mathematical principles governing information, uncertainty, and reconstruction in those fields are general enough to provide a powerful and predictive *modeling framework* for LLM behavior. We build this framework progressively:
1.  Establish the Fourier uncertainty principle as the shared mathematical root of trade-offs in localization.
2.  Discuss how these trade-offs manifest in the structure of LLM latent spaces, leading to the formation of stable conceptual representations, consistent with the Platonic Representation Hypothesis (PRH).
3.  Frame in-context learning (ICL) as a process of guided reconstruction, where prompts provide the "samples" needed to minimize uncertainty, a process elegantly described by the Free Energy Principle (FEP).
4.  Formalize our postulate on hallucinations as a dynamical process of Semantic Drift, where insufficient information leads to chaotic trajectories. We integrate Shannon's information theory and error-correction concepts to model this as a noisy communication channel.
5.  Connect inefficient, drifting trajectories to thermodynamic costs via Landauer's principle, linking reliability to computational efficiency.

Ultimately, this synthesis provides a coherent, testable model of hallucinations, moving beyond simple empirical observation toward a more fundamental understanding of information processing in artificial neural networks.

### 2. The Foundational Trade-off: Fourier Uncertainty in Physics and Signals

The mathematical bedrock of our framework is the Fourier uncertainty principle. For any function f(t) and its Fourier transform f̂(ω), the product of their standard deviations is bounded:

σ_t · σ_ω ≥ 1/2

This inequality is not specific to one domain; it is a universal property of wave-like phenomena and information itself. It dictates that precision in one domain necessitates uncertainty in its conjugate domain.

*   **In QM:** This principle manifests as the Heisenberg uncertainty principle. The position wavefunction ψ(x) and momentum-space wavefunction ψ̂(p) are Fourier conjugates. Localizing a particle's position (making ψ(x) sharp) inevitably broadens its momentum distribution (making ψ̂(p) wide), as constrained by Δx · Δp ≥ ℏ/2.

*   **In Signal Processing:** The principle underpins the Shannon-Nyquist sampling theorem. A continuous signal s(t) bandlimited to a maximum angular frequency ω_max can only be perfectly reconstructed if sampled at a rate f_s ≥ ω_max/π. The reconstruction formula,

s(t) = Σ_{n=-∞}^{∞} s(nT) · sin(π(t/T - n))/π(t/T - n)

relies on this bandwidth limit. Sampling below this rate (undersampling) causes aliasing, where high-frequency components are irreversibly folded into and mistaken for lower frequencies. The bandwidth of a signal is thus a measure of its "complexity," which dictates the minimum information (sample density) needed for faithful reconstruction.

These examples illustrate a universal rule: the complexity of an object determines the amount of information required to represent it without ambiguity. This rule, we argue, extends to the conceptual representations within LLMs.

### 3. Representational Structure in LLMs: Stable Conceptual Manifolds

LLMs learn to organize information into a high-dimensional latent space. We hypothesize that through training, semantically related concepts come to occupy stable, clustered regions, or **conceptual manifolds**. This view is consistent with the **Platonic Representation Hypothesis (PRH)**, which posits that scaled models converge toward shared, abstract representations.

Our framework, however, only requires a weaker assumption: that concepts form distinct, low-energy attractor regions in the latent space. The universality suggested by PRH strengthens our model but is not strictly necessary. These manifolds represent the "bandlimited signals" of the semantic space. Their existence is what makes generalization and reasoning possible; they are the stable "answers" or "ideas" that the model can converge upon.

The Fourier uncertainty principle applies here conceptually: a representation cannot be infinitely precise in all semantic features simultaneously. For instance, a representation highly specific to "a golden retriever" (high localization) might be less robustly associated with the broader concept of "mammal" (broader localization). These trade-offs necessitate the formation of hierarchically organized, multi-scale representations, which these conceptual manifolds provide.

### 4. In-Context Learning as Guided Reconstruction

In-context learning (ICL) is the emergent ability of LLMs to perform new tasks based solely on examples provided in the prompt, without any weight updates. Within our framework, ICL is a process of **guided reconstruction**. The prompt provides a sparse set of "samples" that guide the model's internal trajectory toward the correct conceptual manifold.

The **Free Energy Principle (FEP)** offers a powerful lens to formalize this process. FEP posits that self-organizing systems, including brains and potentially LLMs, act to minimize their free energy, which is equivalent to minimizing prediction error or "surprise." A prompt provides evidence that reduces the model's uncertainty, guiding its state to a lower-free-energy (less surprising) configuration.

*   An ambiguous or sparse prompt corresponds to a high-free-energy state, leaving the model's trajectory unconstrained and liable to drift.
*   A well-formed prompt with clear examples provides strong evidence, effectively "steering" the trajectory into the deep basin of attraction of the correct conceptual manifold, thus minimizing free energy and resulting in a coherent output.

ICL, therefore, is the mechanism by which an LLM uses contextual data to overcome informational uncertainty and successfully reconstruct a target concept.

### 5. Postulate: Hallucinations as Semantic Drift from Information Deficits

We now formally state our central postulate.

**LLM hallucinations arise from Semantic Drift: when a prompt provides insufficient contextual information to meet the complexity requirements of a target concept, the model's internal state trajectory fails to converge to the correct conceptual manifold. Instead, it drifts chaotically or settles into a suboptimal attractor, resulting in a factually incorrect or nonsensical output.**

This is a failure of reconstruction, analogous to aliasing from undersampling.

#### 5.1. Trajectory Dynamics in a Discrete-Depth System

An LLM's forward pass is a discrete-depth dynamical system. The hidden state at layer l+1, **h_{l+1}**, is a function of the previous state **h_l** and the input prompt **p**:

**h_{l+1}** = **h_l** + Attn(**h_l**, **p**) + FFN(**h_l**)

The sequence of hidden states {**h_0**, **h_1**, ..., **h_L**} constitutes a **trajectory** through the latent space. While we can use concepts from continuous dynamics like "attractors" as useful abstractions, the system is fundamentally discrete.

*   **Stable Convergence:** Sufficient contextual information from the prompt **p** guides the trajectory to enter and remain within the basin of the correct conceptual manifold.
*   **Semantic Drift:** An information deficit (e.g., an ambiguous prompt) places the initial state **h_0** in a region of high uncertainty. The trajectory becomes highly sensitive to small perturbations, exhibiting chaotic-like behavior. This can be characterized by measuring the divergence of initially close trajectories as they pass through the network's layers. The trajectory may then fail to stabilize, or it may converge to an incorrect manifold (a "hallucinatory" answer).

#### 5.2. Error Correction and Thermodynamic Costs

This framework allows us to integrate concepts from error correction and thermodynamics.

*   **Error Correction as Trajectory Stabilization:** We can view the LLM as a noisy communication channel, where the "message" is the intended concept and "noise" is the semantic ambiguity from the prompt or model biases. Shannon's noisy channel coding theorem provides a capacity limit for reliable communication. If the prompt's information content is below this capacity, errors (hallucinations) are inevitable. Prompting techniques like **Chain-of-Thought (CoT)** can be modeled as a form of error-correcting code. By forcing the model to generate intermediate reasoning steps, CoT adds redundancy to the semantic signal. This "semantic parity check" helps stabilize the trajectory, allowing it to detect and correct deviations before settling on a final answer. This is analogous to how Hamming codes use parity bits to correct errors in digital communication.

*   **Thermodynamic Inefficiency of Hallucinations:** Drifting, chaotic trajectories are computationally inefficient. According to **Landauer's principle**, every irreversible bit of information erasure dissipates a minimum amount of energy (kT ln 2). An inefficient search through the latent space, involving many corrective steps and high uncertainty, corresponds to greater entropy production and thus higher energy consumption. A well-guided trajectory that quickly converges is thermodynamically efficient. This provides a physical grounding for the intuition that confused, hallucinatory reasoning is more "difficult" for the model, linking informational stability directly to computational cost.

### 6. Discussion and Testable Hypotheses

This paper has proposed a modeling framework for LLM hallucinations based on principles of information, uncertainty, and reconstruction.

#### 6.1. On the Nature and Limits of the Analogies

It is crucial to be precise about the role of the analogies used. We do not claim LLMs *are* signal processors. Rather, we claim that the principle of **reconstruction from limited information** is a universal problem. The Nyquist theorem provides a crisp, mathematical example of this principle in one domain, which serves as a powerful inspiration for modeling a similar process in the high-dimensional, non-linear domain of LLMs. The strength of this framework lies not in a literal mapping of "frequency," but in using the underlying information-theoretic principles to generate novel, falsifiable predictions about LLM behavior.

#### 6.2. Testable Hypotheses

This framework leads to several concrete, testable hypotheses:

1.  **Information Sufficiency Threshold:** There exists a quantifiable information threshold for prompts. Hallucination rates for a given query will be inversely correlated with the mutual information between the prompt and the target concept. This can be tested by systematically ablating contextual information from prompts and measuring the onset of hallucinations.
2.  **Trajectory Divergence as a Predictor:** The tendency of a prompt to cause hallucinations can be predicted *before* generation by measuring the divergence of latent state trajectories. By injecting small perturbations into the initial prompt embedding and tracking the separation of the resulting trajectories through the layers, a high divergence rate would predict a high probability of Semantic Drift.
3.  **CoT as a Variance Reducer:** Chain-of-Thought and other structured reasoning techniques reduce hallucinations by constraining trajectory variance. We predict that the layer-by-layer variance of hidden states for a given prompt will be significantly lower with CoT than without, corresponding to a more stable path to convergence.
4.  **Energy-Hallucination Correlation:** The computational energy (or a proxy like FLOPs) required for inference will correlate with hallucination rates for ambiguous tasks. Prompts that induce Semantic Drift will lead to longer or more computationally intensive trajectories (e.g., requiring more speculative decoding steps), which can be measured.

### 7. Conclusion

We have reframed the problem of LLM hallucinations from an unpredictable flaw to a principled phenomenon of **reconstruction failure**. Our model of **Semantic Drift**, inspired by foundational principles of information theory and signal processing, posits that hallucinations are the result of an information deficit in the prompt, leading to unstable trajectories in the model's latent space.

By synthesizing concepts from Fourier uncertainty, the Free Energy Principle, and Shannon's noisy channel theorem, we have constructed a coherent and testable framework. This approach provides a new vocabulary for describing LLM failures and, more importantly, a clear research program for making them more reliable. The path to robust AI may lie not just in scaling data and parameters, but in a deeper understanding of the fundamental principles of information that govern them.
