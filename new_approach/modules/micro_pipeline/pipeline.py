import json
from pathlib import Path
from typing import List, Sequence

from .bloom_assignment import BloomAssigner
from .file_loader import load_questions_from_files
from .intent_gate import IntentGate
from .llm_client import LLMJsonClient
from .micro_generator import MicroAssessmentGenerator
from .refine_gate import RefineGate
from .schemas import MicroAssessmentRecord
from .text_utils import clean_text, make_short_header


class MicroAssessmentPipeline:
    def __init__(
        self,
        model_name: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        micro_count: int = 3,
        gate_threshold: float = 0.68,
        max_new_tokens: int = 384,
    ):
        self.llm_client = LLMJsonClient(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
        self.generator = MicroAssessmentGenerator(
            llm_client=self.llm_client,
            micro_count=micro_count,
        )
        self.intent_gate = IntentGate(
            embedding_model_name=embedding_model_name,
            threshold=gate_threshold,
        )
        self.refine_gate = RefineGate(llm_client=self.llm_client)
        self.bloom_assigner = BloomAssigner(llm_client=self.llm_client)
        self.micro_count = micro_count

    def process_source_question(self, source_question) -> List[MicroAssessmentRecord]:
        generated = self.generator.generate(
            source_text=source_question.source_text,
            micro_count=self.micro_count,
        )

        results: List[MicroAssessmentRecord] = []
        local_seen = set()

        for micro in generated:
            initial_gate = self.intent_gate.score(source_question.source_text, micro)
            refined = False
            refinement_applied = False
            refinement_failed = False
            accepted_micro = micro
            accepted_gate = initial_gate

            if not initial_gate.passes_intent_gate:
                refinement_applied = True
                rewritten = self.refine_gate.refine(source_question.source_text, micro, initial_gate)
                if rewritten:
                    refined = True
                    accepted_micro = rewritten
                    accepted_gate = self.intent_gate.score(source_question.source_text, rewritten)
                else:
                    refinement_failed = True

            normalized_micro = clean_text(accepted_micro)
            key = normalized_micro.lower()
            if key in local_seen:
                continue
            local_seen.add(key)

            bloom = self.bloom_assigner.assign(normalized_micro)
            results.append(
                MicroAssessmentRecord(
                    micro_assessment=normalized_micro,
                    #micro_header=make_short_header(normalized_micro),
                    bloom_level=bloom["bloom_level"],
                    required_proficiency=bloom["required_proficiency"],
                    bloom_confidence=bloom["bloom_confidence"],
                    bloom_rationale=bloom["bloom_rationale"],
                    gate_scores=accepted_gate.to_dict(),
                    refined=refined,
                    refinement_applied=refinement_applied,
                    refinement_failed=refinement_failed,
                )
            )

        return results

    def run(self, input_paths: Sequence[str], output_path: str) -> None:
        source_questions = load_questions_from_files(input_paths)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        total_sources = len(source_questions)
        total_micros = 0
        refined_count = 0
        global_seen = set()

        print(f"Loaded {total_sources} source questions from {len(input_paths)} file(s).")

        with open(output_file, "w", encoding="utf-8") as fout:
            for idx, source_question in enumerate(source_questions, start=1):
                rows = self.process_source_question(source_question)
                for row in rows:
                    key = clean_text(row.micro_assessment).lower()
                    if key in global_seen:
                        continue
                    global_seen.add(key)

                    if row.refined:
                        refined_count += 1
                    fout.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
                    total_micros += 1

                if idx % 10 == 0:
                    print(f"Processed {idx}/{total_sources} source questions | Generated {total_micros} unique micro-assessments")

        print(f"Finished. Source questions: {total_sources}")
        print(f"Generated unique micro-assessments: {total_micros}")
        print(f"Refined micro-assessments: {refined_count}")
        print(f"Saved to: {output_file}")
