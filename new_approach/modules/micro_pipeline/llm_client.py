import json
import re
from typing import Any, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMJsonClient:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 640,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "Return only valid JSON."
    ) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("generate_text received an empty prompt.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                model_inputs = self.tokenizer(
                    formatted,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    return_attention_mask=True,
                )
            else:
                full_prompt = f"{system_prompt}\n\n{prompt}"
                model_inputs = self.tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    return_attention_mask=True,
                )

            model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
            input_len = model_inputs["input_ids"].shape[-1]

            generate_kwargs = {
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0.0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            if self.temperature > 0.0:
                generate_kwargs["temperature"] = self.temperature

            with torch.no_grad():
                outputs = self.model.generate(**generate_kwargs)

            generated_ids = outputs[0][input_len:]
            decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            if not decoded:
                raise RuntimeError("Model returned empty output.")

            return decoded

        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}") from e

    def extract_json(self, text: Optional[str]) -> Optional[Union[dict, list]]:
        if text is None:
            return None

        if not isinstance(text, str):
            text = str(text)

        text = text.strip()
        if not text:
            return None

        try:
            obj = json.loads(text)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass

        fence_match = re.search(
            r"```json\s*(\{.*?\}|\[.*?\])\s*```",
            text,
            flags=re.DOTALL,
        )
        if fence_match:
            try:
                obj = json.loads(fence_match.group(1))
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                pass

        blob_match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if blob_match:
            try:
                obj = json.loads(blob_match.group(1))
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                pass

        return None