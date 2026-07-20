import json
from pathlib import Path
from typing import List, Dict

from bloom_rules import predict_bloom_from_micro_text


FORBIDDEN_PHRASES = [
    "main idea",
    "central concept",
    "mechanism involved",
    "required in",
    "the concept involved in",
]


def is_vague(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in FORBIDDEN_PHRASES)


def is_duplicate(text: str, seen: set) -> bool:
    key = text.strip().lower()
    if key in seen:
        return True
    seen.add(key)
    return False


def validate_micro_items(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned = []
    seen = set()

    for item in items:
        text = item.get("text", "").strip()
        bloom = item.get("bloom", "").strip()

        if not text:
            continue
        if is_vague(text):
            continue
        if is_duplicate(text, seen):
            continue

        predicted = predict_bloom_from_micro_text(text)

        # relabel if mismatch
        item["bloom"] = predicted
        cleaned.append(item)

    return cleaned


def run_validation(input_path: str, output_path: str):
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
            items = row.get("micro_assessments", [])
            row["micro_assessments"] = validate_micro_items(items)

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1

    print(f"Validated {total} rows")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    run_validation(
        "/home/ap2194.UNT/micro/data/decomposed_micro.jsonl",
        "/home/ap2194.UNT/micro/data/validated_micro.jsonl"
    )