"""Language-model "surprise" features from GPT-2 small.

Idea: a language model assigns every next word a probability. Text an LLM writes tends to be made of
words the model finds *likely* (low surprise); people pick less predictable words more often. We
measure that with four numbers per text:
  mean_surprisal   average -log p(token) in nats (log of perplexity; lower = more predictable)
  surprisal_std    how much surprise varies from token to token
  top1_frac        share of tokens that were the model's single favorite guess
  top10_frac       share of tokens inside the model's 10 favorite guesses (as in GLTR)
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "openai-community/gpt2"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class GPT2Scorer:
    def __init__(self, model_name: str = MODEL_NAME, max_tokens: int = 512, batch_size: int = 8,
                 device: str | None = None):  # fmt: skip
        self.device = device or pick_device()
        self.max_tokens, self.batch_size = max_tokens, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.bos = self.tokenizer.bos_token_id  # <|endoftext|>, lets us score the first real token

    @torch.no_grad()
    def token_stats(self, texts: list[str]) -> list[dict[str, np.ndarray]]:
        """Per-token surprisal (nats), rank of the true token, and its character span in the text.

        Input order is kept. Only the first `max_tokens` tokens of a text are scored; `offsets` lets
        callers map tokens back to sentences (later tokens simply have no entry).
        """
        enc = [self.tokenizer(t, return_offsets_mapping=True) for t in texts]
        encoded = [[self.bos] + e["input_ids"][: self.max_tokens] for e in enc]
        offsets = [np.asarray(e["offset_mapping"][: self.max_tokens], dtype=int).reshape(-1, 2) for e in enc]
        order = sorted(range(len(texts)), key=lambda i: len(encoded[i]))  # similar lengths per batch
        out: list[dict[str, np.ndarray] | None] = [None] * len(texts)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            width = max(len(encoded[i]) for i in idx)
            ids = torch.full((len(idx), width), self.bos, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for row, i in enumerate(idx):
                ids[row, : len(encoded[i])] = torch.tensor(encoded[i])
                mask[row, : len(encoded[i])] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            logits = self.model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            targets = ids[:, 1:]
            logprobs = torch.log_softmax(logits, dim=-1)
            target_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            rank = (logprobs > target_lp.unsqueeze(-1)).sum(-1) + 1  # 1 = the model's top guess
            target_lp, rank = target_lp.cpu().numpy(), rank.cpu().numpy()
            for row, i in enumerate(idx):
                n = len(encoded[i]) - 1
                out[i] = {"surprisal": -target_lp[row, :n], "rank": rank[row, :n], "offsets": offsets[i]}
        return out  # type: ignore[return-value]

    def features(self, texts: list[str]) -> list[dict[str, float]]:
        return [perplexity_features(stats) for stats in self.token_stats(texts)]


def perplexity_features(stats: dict[str, np.ndarray]) -> dict[str, float]:
    """The four document-level GPT-2 features from one text's per-token stats."""
    s, r = stats["surprisal"], stats["rank"]
    return {
        "mean_surprisal": float(s.mean()),
        "surprisal_std": float(s.std()),
        "top1_frac": float((r == 1).mean()),
        "top10_frac": float((r <= 10).mean()),
    }
