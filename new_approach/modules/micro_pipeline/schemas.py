from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SourceQuestion:
    source_file: str
    source_index: int
    source_text: str
    source_header: str

    def to_dict(self) -> Dict[str, Any]:

        return {
            "source_header": self.source_header,
        }


@dataclass
class MicroAssessmentRecord:
    #record_id: str
    micro_assessment: str
    bloom_level: str
    required_proficiency: str
    bloom_confidence: float
    bloom_rationale: str
    gate_scores: Dict[str, Any]
    refined: bool
    refinement_applied: bool
    refinement_failed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            #"record_id": self.record_id,
            "micro_assessment": self.micro_assessment,
            "bloom_level": self.bloom_level,
            "required_proficiency": self.required_proficiency,
            "bloom_confidence": self.bloom_confidence,
            "bloom_rationale": self.bloom_rationale,
            "gate_scores": self.gate_scores,
            "refined": self.refined,
            "refinement_applied": self.refinement_applied,
            "refinement_failed": self.refinement_failed,
        }


@dataclass
class GateResult:
    semantic_alignment: float
    action_overlap: float
    concept_overlap: float
    length_score: float
    overall_score: float
    passes_intent_gate: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_alignment": self.semantic_alignment,
            "action_overlap": self.action_overlap,
            "concept_overlap": self.concept_overlap,
            "length_score": self.length_score,
            "overall_score": self.overall_score,
            "passes_intent_gate": self.passes_intent_gate,
        }
