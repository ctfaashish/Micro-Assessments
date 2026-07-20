import re
import torch
import torch.nn.functional as F


class RefineGate:
    def __init__(self, gated_model, refine_threshold: float = 0.78):
        self.gated_model = gated_model
        self.refine_threshold = refine_threshold

    def simple_bloom_check(self, text: str, bloom_level: str) -> float:
        """
        Cheap lexical check. Not perfect, but useful as one signal.
        """
        text_lower = text.lower()

        bloom_keywords = {
            "Remember": ["define", "list", "name", "identify"],
            "Understand": ["explain", "describe", "summarize"],
            "Apply": ["solve", "implement", "use", "compute"],
            "Analyze": ["compare", "contrast", "differentiate", "analyze"],
            "Debug": ["debug", "find the bug", "fix", "correct"],
            "Evaluate": ["justify", "evaluate", "defend", "assess"],
            "Create": ["design", "create", "develop", "construct"],
        }

        kws = bloom_keywords.get(bloom_level, [])
        hits = sum(1 for kw in kws if kw in text_lower)
        return min(1.0, hits / max(1, len(kws)))

    def semantic_alignment_score(self, generated_text: str, intent_text: str) -> float:
        # Encode both with the same encoder used in gated model.
        q_batch, q_hidden = self.gated_model.encode_texts([generated_text], max_length=256)
        i_batch, i_hidden = self.gated_model.encode_texts([intent_text], max_length=128)

        q_vec = q_hidden.mean(dim=1)
        i_vec = i_hidden.mean(dim=1)

        cosine = F.cosine_similarity(q_vec, i_vec, dim=-1).item()
        return (cosine + 1.0) / 2.0

    def needs_refinement(self, generated_text: str, intent_text: str, bloom_level: str) -> bool:
        semantic = self.semantic_alignment_score(generated_text, intent_text)
        lexical = self.simple_bloom_check(generated_text, bloom_level)
        final_score = 0.7 * semantic + 0.3 * lexical
        return final_score < self.refine_threshold

    def build_refine_prompt(
        self,
        original_question: str,
        draft_output: str,
        bloom_level: str,
        course: str
    ) -> str:
        return f"""
You are refining draft micro-assessments so they align more closely with the requested Bloom level.

Course:
{course}

Original assessment question:
{original_question}

Requested Bloom level:
{bloom_level}

Draft micro-assessments:
{draft_output}

Refinement instructions:
- Keep only items relevant to the original question.
- Rewrite any item that does not match [{bloom_level}].
- Each item must test exactly one concept.
- Remove bundled skills.
- Preserve concise wording.
- Every line must begin with [{bloom_level}].
- Do not include answers.
- Return only the improved numbered list.
""".strip()

    def refine_if_needed(
        self,
        original_question: str,
        draft_output: str,
        bloom_level: str,
        course: str,
        max_new_tokens: int = 256
    ):
        intent_text = f"{bloom_level}: generate micro-assessments for one concept only"
        if not self.needs_refinement(draft_output, intent_text, bloom_level):
            return {
                "refined": False,
                "final_text": draft_output
            }

        refine_prompt = self.build_refine_prompt(
            original_question=original_question,
            draft_output=draft_output,
            bloom_level=bloom_level,
            course=course
        )

        outputs = self.gated_model.model.generate(
            **self.gated_model.tokenizer(
                refine_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=768
            ).to(self.gated_model.device),
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        refined_text = self.gated_model.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "refined": True,
            "final_text": refined_text
        }