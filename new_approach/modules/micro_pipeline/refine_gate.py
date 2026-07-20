from .llm_client import LLMJsonClient
from .schemas import GateResult
from .text_utils import clean_text


class RefineGate:
    def __init__(self, llm_client: LLMJsonClient):
        self.llm_client = llm_client

    def build_prompt(self, source_text: str, bad_micro: str, gate_result: GateResult) -> str:
        return f"""
Rewrite this weak micro-assessment so it better matches the original source assessment.

Source assessment:
{source_text}

Current micro-assessment:
{bad_micro}

Diagnostics:
- semantic_alignment: {gate_result.semantic_alignment}
- action_overlap: {gate_result.action_overlap}
- concept_overlap: {gate_result.concept_overlap}
- length_score: {gate_result.length_score}
- overall_score: {gate_result.overall_score}

Rewrite rules:
- preserve the same core topic and intent as the source assessment
- make it narrower and easier to assess than the source item
- do not introduce unrelated concepts
- do not copy the source verbatim unless necessary
- keep it as one concise assessment item
- ignore OCR fragments, numbering, and page artifacts

Return ONLY valid JSON in this schema:
{{
  "micro_assessment": "rewritten micro assessment"
}}
""".strip()

    def refine(self, source_text: str, bad_micro: str, gate_result: GateResult) -> str | None:
        prompt = self.build_prompt(source_text, bad_micro, gate_result)
        raw = self.llm_client.generate_text(prompt)
        obj = self.llm_client.extract_json(raw)
        if obj is None:
            return None
        rewritten = clean_text(str(obj.get("micro_assessment", "")))
        return rewritten or None
