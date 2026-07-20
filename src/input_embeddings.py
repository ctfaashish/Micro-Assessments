import torch
from transformers import AutoModelForSeq2SeqLM


class FlanT5EmbeddingBlock:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.shared_embedding = self.model.get_input_embeddings()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        embeddings = self.shared_embedding(input_ids)
        return embeddings

    def prepare_batch(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        x_emb = self.embed_input_ids(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "x_emb": x_emb
        }