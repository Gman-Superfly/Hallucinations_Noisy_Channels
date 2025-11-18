"""
Compute semantic redundancy ratios for realistic prompts and export a heatmap.

Use --backend tfidf (default, no external deps) or --backend transformer to use
contextual embeddings (requires transformers + pretrained model).
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

plt.style.use("seaborn-v0_8")

QA_DATA = [
    {
        "question": "What is the capital of Spain?",
        "answer": "Madrid",
        "prompts": {
            "direct": "Question: What is the capital of Spain?",
            "relevant": "Spain is bordered by France and Portugal. The capital city of Spain is Madrid.",
            "noisy": "Spain shares borders with Portugal. Soccer is popular there. Some people mistakenly cite Barcelona as the capital.",
            "contradictory": "Spain's capital used to be Toledo. Many tourists believe Barcelona is the capital today.",
        },
    },
    {
        "question": "Who developed the theory of relativity?",
        "answer": "Albert Einstein",
        "prompts": {
            "direct": "Who developed the theory of relativity?",
            "relevant": "Physics history highlights Albert Einstein, who formulated the theory of relativity.",
            "noisy": "Relativity is discussed alongside quantum mechanics. Isaac Newton studied gravity.",
            "contradictory": "Some blogs claim Nikola Tesla created relativity. Others credit Einstein.",
        },
    },
    {
        "question": "Which element has the chemical symbol O?",
        "answer": "Oxygen",
        "prompts": {
            "direct": "Which element has the chemical symbol O?",
            "relevant": "Oxygen, symbol O, is essential for respiration.",
            "noisy": "The periodic table lists elements like hydrogen and carbon.",
            "contradictory": "Some sources confuse oxygen with osmium because both start with 'O'.",
        },
    },
]


def build_corpus() -> list[str]:
    corpus = []
    for sample in QA_DATA:
        corpus.append(sample["answer"])
        for prompt_text in sample["prompts"].values():
            corpus.extend([s.strip() for s in prompt_text.split(".") if s.strip()])
    return corpus


def build_tfidf_encoder(corpus: list[str]):
    vectorizer = TfidfVectorizer().fit(corpus)

    def encode(text: str) -> np.ndarray:
        vec = vectorizer.transform([text])
        return normalize(vec).toarray().squeeze(0)

    return encode


def build_transformer_encoder(model_name: str):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    def encode(text: str) -> np.ndarray:
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
        hidden = outputs.last_hidden_state.mean(dim=1)
        hidden = torch.nn.functional.normalize(hidden, dim=-1)
        return hidden.cpu().numpy().squeeze(0)

    return encode


def redundancy_ratio(prompt: str, answer_emb: np.ndarray, encoder, threshold: float = 0.35):
    sentences = [s.strip() for s in prompt.split(".") if s.strip()]
    if not sentences:
        return 0.0, np.array([])
    sims = []
    for sentence in sentences:
        emb = encoder(sentence)
        sims.append(float(np.dot(emb, answer_emb)))
    sims = np.array(sims)
    rho = np.mean(sims >= threshold)
    return float(rho), sims


def main():
    parser = argparse.ArgumentParser(description="Semantic redundancy heatmap generator")
    parser.add_argument(
        "--backend",
        choices=["tfidf", "transformer"],
        default="tfidf",
        help="Embedding backend to use.",
    )
    parser.add_argument(
        "--model_name",
        default="distilbert-base-uncased",
        help="HF model name (used if backend=transformer).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Similarity threshold for counting redundant sentences.",
    )
    args = parser.parse_args()

    if args.backend == "tfidf":
        print("Using TF-IDF embeddings.")
        encoder = build_tfidf_encoder(build_corpus())
    else:
        print(f"Using transformer embeddings ({args.model_name}).")
        encoder = build_transformer_encoder(args.model_name)

    rows = []
    for sample in QA_DATA:
        answer_emb = encoder(sample["answer"])
        for label, prompt in sample["prompts"].items():
            rho, sims = redundancy_ratio(prompt, answer_emb, encoder, args.threshold)
            rows.append(
                {
                    "question": sample["question"],
                    "variant": label,
                    "rho": rho,
                    "avg_similarity": float(np.mean(sims)) if len(sims) else 0.0,
                }
            )

    df = pd.DataFrame(rows)
    df_pivot = df.pivot(index="question", columns="variant", values="rho")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        df_pivot.loc[:, ["direct", "relevant", "noisy", "contradictory"]],
        annot=True,
        cmap="mako",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Semantic redundancy ratio (rho)")
    ax.set_xlabel("Prompt variant")
    ax.set_ylabel("Question")
    fig.tight_layout()
    output_dir = pathlib.Path("figures")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "semantic_redundancy_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    df.to_csv(output_dir / "semantic_redundancy_metrics.csv", index=False)
    print("Saved heatmap and metrics to figures/.")


if __name__ == "__main__":
    main()

