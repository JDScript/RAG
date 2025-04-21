from threading import Lock

import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


class Reranker:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_id: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()

    def re_rank(self, query: str, contexts: list[str]):
        """
        Re-rank the contexts based on the query.
        :param query: The query to re-rank the contexts for.
        :param contexts: The contexts to be re-ranked.
        """
        pairs = [(query, context) for context in contexts]
        inputs = self.tokenizer(
            pairs,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = (
                self.model(**inputs)
                .logits.view(
                    -1,
                )
                .float()
            )
        return torch.argsort(logits, descending=True).tolist()
