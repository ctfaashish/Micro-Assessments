from dataclasses import dataclass
import torch


@dataclass
class InputPipelineConfig:
    model_name: str = "google/flan-t5-base"
    max_length: int = 256
    batch_size: int = 8
    text_field: str = "raw_text"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"