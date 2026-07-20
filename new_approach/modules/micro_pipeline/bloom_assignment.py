import re
from typing import Any, Dict, Tuple

from .config import BLOOM_HINTS, BLOOM_TO_PROFICIENCY, VALID_BLOOM
from .llm_client import LLMJsonClient
from .text_utils import clean_text


class BloomAssigner:
    def __init__(self, llm_client: LLMJsonClient):
        self.llm_client = llm_client

    def heuristic_fallback(self, text: str) -> Tuple[str, float, str]:
        text_l = clean_text(text).lower()
        scores = {level: 0 for level in VALID_BLOOM}
        for level, hints in BLOOM_HINTS.items():
            for hint in hints:
                if re.search(rf"\b{re.escape(hint)}\b", text_l):
                    scores[level] += 1

        best_level = max(scores.items(), key=lambda kv: kv[1])[0]
        confidence = 0.45 if max(scores.values()) == 0 else min(0.9, 0.55 + 0.1 * max(scores.values()))
        if max(scores.values()) == 0:
            best_level = "Understand"
        rationale = "heuristic fallback based on action verbs"
        return best_level, confidence, rationale

    def build_prompt(self, micro_assessment: str) -> str:
        levels = ", ".join(VALID_BLOOM)
        return f"""
Assign the most appropriate Bloom's Taxonomy level to this micro-assessment.

Micro-assessment:
{micro_assessment}

Choose exactly one of these levels:
{levels}

Return ONLY valid JSON in this schema:
{{
  "bloom_level": "one valid level",
  "confidence": 0.0,
  "rationale": "one short reason"
}}

Rules:
- confidence must be between 0 and 1
- use the task demand of the micro-assessment, not the source file metadata
- do not return markdown
""".strip()

    def assign(self, micro_assessment: str) -> Dict[str, Any]:
        prompt = self.build_prompt(micro_assessment)
        raw = self.llm_client.generate_text(prompt)
        obj = self.llm_client.extract_json(raw)

        if obj is None:
            bloom_level, confidence, rationale = self.heuristic_fallback(micro_assessment)
        else:
            bloom_level = str(obj.get("bloom_level", "")).strip().capitalize()
            confidence = obj.get("confidence", 0.0)
            rationale = str(obj.get("rationale", "")).strip()
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            if bloom_level not in VALID_BLOOM:
                bloom_level, confidence, rationale = self.heuristic_fallback(micro_assessment)

        return {
            "bloom_level": bloom_level,
            "required_proficiency": BLOOM_TO_PROFICIENCY[bloom_level],
            "bloom_confidence": round(float(confidence), 4),
            "bloom_rationale": rationale or "No rationale returned.",
        }
