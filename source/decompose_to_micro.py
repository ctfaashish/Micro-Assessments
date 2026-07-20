import json
import re
from pathlib import Path
from typing import List, Dict

from concept_extractor import extract_candidate_concepts, extract_compare_targets, is_valid_concept
from bloom_rules import predict_dominant_bloom


def sentence_case(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def make_item(text: str, bloom: str) -> Dict[str, str]:
    text = re.sub(r"\s+", " ", text.strip())
    text = text[0].upper() + text[1:] if text else text
    if text and not text.endswith("."):
        text += "."
    return {"text": text, "bloom": bloom}


def decompose_compare_item(raw_text: str) -> List[Dict[str, str]]:
    targets = extract_compare_targets(raw_text)
    if len(targets) == 2:
        a, b = targets[0], targets[1]
        return [
            make_item(f"Define {a}", "Remember"),
            make_item(f"Define {b}", "Remember"),
            make_item(f"List one key feature of {a}", "Remember"),
            make_item(f"List one key feature of {b}", "Remember"),
            make_item(f"Compare {a} and {b} in one important aspect", "Analyze"),
        ]
    return []


def decompose_explain_item(raw_text: str) -> List[Dict[str, str]]:
    concepts = extract_candidate_concepts(raw_text)
    if not concepts:
        return []

    primary = concepts[0]
    return [
        make_item(f"Define {primary}", "Remember"),
        make_item(f"Explain the role of {primary}", "Understand"),
        make_item(f"Describe one important property of {primary}", "Understand"),
    ]


def decompose_apply_item(raw_text: str) -> List[Dict[str, str]]:
    concepts = extract_candidate_concepts(raw_text)
    if not concepts:
        return []

    primary = concepts[0]
    return [
        make_item(f"Define {primary}", "Remember"),
        make_item(f"Explain the role of {primary}", "Understand"),
        make_item(f"Apply {primary} in a simple example", "Apply"),
    ]


def decompose_generic(raw_text: str) -> List[Dict[str, str]]:
    concepts = extract_candidate_concepts(raw_text)

    # only keep actually valid concepts
    concepts = [c for c in concepts if is_valid_concept(c)]

    if not concepts:
        return []

    items = []
    for c in concepts[:4]:
        items.append(make_item(f"Define {c}", "Remember"))

    # if we only found one concept, add a more meaningful understand-level follow-up
    if len(concepts) == 1:
        c = concepts[0]
        items.append(make_item(f"Explain one key feature of {c}", "Understand"))

    return items[:5]


def decompose_raw_assessment(raw_text: str) -> List[Dict[str, str]]:
    text_l = raw_text.lower().strip()

    # more specific rules first
    if re.search(r"\bcompare\b|\bcontrast\b", text_l):
        items = decompose_compare_item(raw_text)
        if items:
            return items

    if re.search(r"\bdefine\b|\blist\b|\bidentify\b", text_l):
        items = decompose_define_item(raw_text)
        if items:
            return items

    if re.search(r"\bexplain\b|\bdescribe\b|\bsummarize\b", text_l):
        items = decompose_explain_item(raw_text)
        if items:
            return items

    if re.search(r"\bapply\b|\bsolve\b|\bcompute\b|\bcalculate\b|\bimplement\b", text_l):
        items = decompose_apply_item(raw_text)
        if items:
            return items

    return decompose_generic(raw_text)


def run_decomposition(input_path: str, output_path: str, text_field: str = "raw_text"):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            raw_text = row.get(text_field, "").strip()
            if not raw_text:
                continue

            dominant_bloom, _ = predict_dominant_bloom(raw_text)
            micro_items = decompose_raw_assessment(raw_text)

            row["raw_assessment"] = raw_text
            row["dominant_bloom"] = dominant_bloom
            row["micro_assessments"] = micro_items

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1

    print(f"Decomposed {total} assessments")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    run_decomposition(
        "/home/ap2194.UNT/micro/data/original_bloom_labeled.jsonl",
        "/home/ap2194.UNT/micro/data/decomposed_micro.jsonl"
    )