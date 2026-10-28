"""Independent open-source detector, used ONLY to evaluate, never to train or tune anything.

`desklib/ai-text-detector-v1.01` is DeBERTa-v3-large fine-tuned on the RAID benchmark (modern LLM
generations across several domains). It was trained on a different dataset than ours, so it is the
honest check that a humanizer did not merely overfit to our own detector. We cannot rule out
incidental overlap of source material (e.g. Wikipedia or Reddit text) with RAID.

The model class below follows the usage code on the model card (MIT license), with dynamic padding
(pad to the longest text in a batch) instead of padding every text to 768 tokens.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

from hd.features.perplexity import pick_device

DESKLIB = "desklib/ai-text-detector-v1.01"


class ExternalDetector(Protocol):
    name: str

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Probability that each text is AI-written."""
        ...


class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, attention_mask=None):
        hidden = self.model(input_ids, attention_mask=attention_mask)[0]
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        pooled = (hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)  # mean pooling
        return {"logits": self.classifier(pooled)}


class DesklibDetector:
    name = "desklib"

    def __init__(self, model_name: str = DESKLIB, max_len: int = 768, batch_size: int = 8, device: str | None = None):
        self.device = device or pick_device()
        self.max_len, self.batch_size = max_len, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = DesklibAIDetectionModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        lengths = [len(t) for t in texts]
        order = np.argsort(lengths)  # similar lengths per batch -> little padding
        out = np.zeros(len(texts))
        for start in range(0, len(texts), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                 max_length=self.max_len, return_tensors="pt").to(self.device)  # fmt: skip
            logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])["logits"]
            out[idx] = torch.sigmoid(logits.float()).squeeze(-1).cpu().numpy()
        return out
