from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class RawTextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class FlanT5TokenizerPipeline:
    def __init__(self, model_name: str, max_length: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoded

    def build_dataloader(self, texts: List[str], batch_size: int) -> DataLoader:
        dataset = RawTextDataset(texts)

        def collate_fn(batch_texts):
            return self.tokenize_batch(batch_texts)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )