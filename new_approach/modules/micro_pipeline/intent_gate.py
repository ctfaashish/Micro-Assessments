from typing import Any, Dict, Tuple

from sentence_transformers import SentenceTransformer, util

from .config import DEFAULT_GATE_THRESHOLD
from .schemas import GateResult
from .text_utils import has_assessment_signal, jaccard_overlap, tokenize_keywords


class IntentGate:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = DEFAULT_GATE_THRESHOLD,
    ):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.threshold = threshold

    def semantic_alignment_score(self, source_text: str, micro_assessment: str) -> float:
        emb = self.embedder.encode([source_text, micro_assessment], convert_to_tensor=True)
        cosine = util.cos_sim(emb[0], emb[1]).item()
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    def action_overlap_score(self, source_text: str, micro_assessment: str) -> float:
        source_has_action = 1.0 if has_assessment_signal(source_text) else 0.5
        micro_has_action = 1.0 if has_assessment_signal(micro_assessment) else 0.4
        return round((source_has_action + micro_has_action) / 2.0, 4)

    def concept_overlap_score(self, source_text: str, micro_assessment: str) -> float:
        source_tokens = tokenize_keywords(source_text)
        micro_tokens = tokenize_keywords(micro_assessment)
        return jaccard_overlap(source_tokens, micro_tokens)

    def length_score(self, micro_assessment: str) -> float:
        n = len(micro_assessment.split())
        if 5 <= n <= 28:
            return 1.0
        if 4 <= n <= 35:
            return 0.8
        if 3 <= n <= 45:
            return 0.6
        return 0.35

    def score(self, source_text: str, micro_assessment: str) -> GateResult:
        semantic = self.semantic_alignment_score(source_text, micro_assessment)
        action_overlap = self.action_overlap_score(source_text, micro_assessment)
        concept_overlap = self.concept_overlap_score(source_text, micro_assessment)
        length_score = self.length_score(micro_assessment)

        overall = 0.50 * semantic + 0.20 * action_overlap + 0.20 * concept_overlap + 0.10 * length_score
        return GateResult(
            semantic_alignment=round(semantic, 4),
            action_overlap=round(action_overlap, 4),
            concept_overlap=round(concept_overlap, 4),
            length_score=round(length_score, 4),
            overall_score=round(overall, 4),
            passes_intent_gate=overall >= self.threshold,
        )
