from typing import List, Optional

from .config import DEFAULT_MICRO_COUNT
from .llm_client import LLMJsonClient
from .text_utils import clean_text


class MicroAssessmentGenerator:
    def __init__(self, llm_client: LLMJsonClient, micro_count: int = DEFAULT_MICRO_COUNT):
        self.llm_client = llm_client
        self.micro_count = micro_count

    def build_prompt(self, source_text: str, micro_count: int) -> str:
        return f"""
You are decomposing one source assessment item into smaller micro-assessments.

Source assessment:
{source_text}

Generate exactly {micro_count} focused micro-assessments that:
- preserve the original intent and topic
- break the task into smaller assessable units
- avoid adding unrelated content
- avoid repeating the source verbatim
- ignore numbering, OCR fragments, page artifacts, and formatting noise
- keep each micro-assessment concise and meaningful
- write each micro-assessment as a complete assessment prompt

Return ONLY valid JSON in this schema:
{{
  "micro_assessments": [
    "micro assessment 1",
    "micro assessment 2"
  ]
}}
""".strip()

    def normalize_outputs(self, items: List[str], max_items: int) -> List[str]:
        cleaned: List[str] = []
        seen = set()

        for item in items:
            if item is None:
                continue

            text = clean_text(str(item))
            text = text.lstrip("-•*0123456789. )(").strip()
            text = clean_text(text)

            if len(text.split()) < 3:
                continue

            key = text.lower()
            if key in seen:
                continue

            seen.add(key)
            cleaned.append(text)

            if len(cleaned) >= max_items:
                break

        return cleaned

    def _extract_micro_list(self, obj) -> Optional[List[str]]:
        if isinstance(obj, dict):
            micros = obj.get("micro_assessments")
            if isinstance(micros, list):
                return micros

        # Optional fallback if the model directly returns a list
        if isinstance(obj, list):
            if all(isinstance(x, str) for x in obj):
                return obj

        return None

    def generate(self, source_text: str, micro_count: int | None = None) -> List[str]:
        micro_count = micro_count or self.micro_count
        prompt = self.build_prompt(source_text, micro_count)

        raw = self.llm_client.generate_text(prompt)
        if raw is None:
            return [source_text]

        obj = self.llm_client.extract_json(raw)
        micros = self._extract_micro_list(obj)

        if micros is None:
            return [source_text]

        cleaned = self.normalize_outputs(micros, max_items=micro_count)

        return cleaned if cleaned else [source_text]