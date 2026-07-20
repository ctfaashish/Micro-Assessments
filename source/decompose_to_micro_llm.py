import json
import re
from pathlib import Path
from typing import List, Dict, Optional

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from bloom_rules import predict_dominant_bloom, predict_bloom_from_micro_text
from decompose_to_micro import decompose_raw_assessment as rule_based_decompose
from concept_extractor import extract_candidate_concepts, is_valid_concept


VALID_BLOOM_LEVELS = {
    "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"
}


def sentence_case(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if not text.endswith("."):
        text += "."
    return text


def extract_json_block(text: str) -> Optional[dict]:
    """
    Try to extract a JSON object from model output.
    """
    text = text.strip()

    # direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # try fenced json
    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except Exception:
            pass

    # try first {...} block
    brace_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except Exception:
            pass

    return None


def normalize_bloom(label: str) -> Optional[str]:
    if not isinstance(label, str):
        return None
    label = label.strip().capitalize()
    if label in VALID_BLOOM_LEVELS:
        return label
    return None


BAD_SINGLE_TARGETS = {
    "the", "a", "an", "compare", "contrast", "working", "function", "process",
    "method", "mechanism", "system", "concept", "idea", "principle"
}


def is_bad_micro_text(text: str) -> bool:
    t = text.strip().lower()

    # obvious vague junk
    forbidden = [
        "main idea",
        "central concept",
        "mechanism involved",
        "concept involved in",
        "required in"
    ]
    if any(f in t for f in forbidden):
        return True

    # bad "Define X" targets
    m = re.match(r"^(define|explain|describe|list)\s+([a-zA-Z\-]+)\.?$", t)
    if m:
        target = m.group(2).strip().lower()
        if target in BAD_SINGLE_TARGETS:
            return True

    # too short
    if len(t.split()) < 2:
        return True

    return False


def validate_micro_items(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned = []
    seen = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        text = item.get("text", "")
        bloom = item.get("bloom", "")

        if not isinstance(text, str):
            continue

        text = sentence_case(text)
        lower = text.lower().strip()

        if is_bad_micro_text(text):
            continue

        if lower in seen:
            continue
        seen.add(lower)

        bloom_norm = normalize_bloom(bloom)
        if bloom_norm is None:
            bloom_norm = predict_bloom_from_micro_text(text)

        # re-predict from wording
        bloom_norm = predict_bloom_from_micro_text(text)

        cleaned.append({
            "text": text,
            "bloom": bloom_norm
        })

    return cleaned

class MicroAssessmentDecomposerLLM:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None,
        max_input_length: int = 512,
        max_output_length: int = 256
    ):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def build_prompt(self, raw_text: str) -> str:
        return f"""
You are an expert instructional designer for computer science and cybersecurity education.

Task:
Decompose the following assessment into 4 to 5 atomic micro-assessments.

Assessment:
{raw_text}

Rules:
- Each micro-assessment must test exactly one concept or skill.
- Do not paraphrase the original assessment as a whole.
- Break compound prompts into smaller concept-level items.
- Use a logical Bloom progression when appropriate, starting lower and moving upward.
- Assign exactly one Bloom level to each item from:
  Remember, Understand, Apply, Analyze, Evaluate, Create.
- Use concise wording.
- Do not include answers.
- Avoid vague phrases like "main idea", "central concept", or "mechanism".

Return ONLY valid JSON in this exact format:
{{
  "micro_assessments": [
    {{"text": "Define paging.", "bloom": "Remember"}},
    {{"text": "Define segmentation.", "bloom": "Remember"}}
  ]
}}
""".strip()

    def generate_raw(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def decompose(self, raw_text: str) -> List[Dict[str, str]]:
        prompt = self.build_prompt(raw_text)
        raw_output = self.generate_raw(prompt)

        parsed = extract_json_block(raw_output)

        if parsed and isinstance(parsed, dict) and "micro_assessments" in parsed:
            items = parsed["micro_assessments"]
            if isinstance(items, list):
                cleaned = validate_micro_items(items)
                if len(cleaned) >= 2:
                    return cleaned

        # fallback to rule-based decomposition
        fallback = rule_based_decompose(raw_text)
        fallback = validate_micro_items(fallback)
        if len(fallback) < 2:
            concepts = extract_candidate_concepts(raw_text)
            concepts = [c for c in concepts if is_valid_concept(c)]
            if len(concepts) >= 1:
                c = concepts[0]
                fallback = [{"text": sentence_case(f"Define {c}."), "bloom": "Remember"},
                           {"text": sentence_case(f"Explain one key feature of {c}."), "bloom": "Understand"},
                ]
        return fallback


def run_llm_decomposition(
    input_path: str,
    output_path: str,
    text_field: str = "raw_text",
    model_name: str = "google/flan-t5-base"
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    decomposer = MicroAssessmentDecomposerLLM(model_name=model_name)

    total = 0
    llm_success = 0
    fallback_used = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            raw_text = row.get(text_field, "")
            if not isinstance(raw_text, str) or not raw_text.strip():
                continue

            raw_text = raw_text.strip()
            dominant_bloom, _ = predict_dominant_bloom(raw_text)

            # run LLM
            prompt = decomposer.build_prompt(raw_text)
            model_output = decomposer.generate_raw(prompt)
            parsed = extract_json_block(model_output)

            if parsed and isinstance(parsed, dict) and "micro_assessments" in parsed:
                items = validate_micro_items(parsed["micro_assessments"])
                if len(items) >= 2:
                    llm_success += 1
                else:
                    items = validate_micro_items(rule_based_decompose(raw_text))
                    fallback_used += 1
            else:
                items = validate_micro_items(rule_based_decompose(raw_text))
                fallback_used += 1

            row["raw_assessment"] = raw_text
            row["dominant_bloom"] = dominant_bloom
            row["micro_assessments"] = items
            row["llm_output_preview"] = model_output[:500]

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1

            if total % 25 == 0:
                print(f"Processed {total} rows | LLM success: {llm_success} | Fallback: {fallback_used}")

    print(f"\nDone.")
    print(f"Processed total rows: {total}")
    print(f"LLM successful decompositions: {llm_success}")
    print(f"Fallback used: {fallback_used}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    run_llm_decomposition(
        input_path="/home/ap2194.UNT/micro/data/original_bloom_labeled.jsonl",
        output_path="/home/ap2194.UNT/micro/data/decomposed_micro_llm.jsonl",
        text_field="raw_text",
        model_name="google/flan-t5-base"
    )