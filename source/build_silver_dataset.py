import json
from pathlib import Path


PROMPT_PREFIX = (
    "Generate 4-5 micro-assessments.\n"
    "Bloom progression required: start from lower levels and move upward logically.\n"
    "Requirements:\n"
    "- Break the assessment into smaller concept-level items.\n"
    "- Each item must test one concept only.\n"
    "- Do not paraphrase the original assessment as a whole.\n"
    "- Do not use generic phrases such as \"main idea\", \"central concept\", or \"mechanism\".\n"
    "- Output only a numbered list.\n\n"
)


def format_target(micro_items):
    lines = []
    for idx, item in enumerate(micro_items, start=1):
        lines.append(f"{idx}. [{item['bloom']}] {item['text']}")
    return "\n".join(lines)


def build_input_text(raw_assessment: str) -> str:
    return PROMPT_PREFIX + f"Assessment: {raw_assessment}"


def build_silver_dataset(input_path: str, output_path: str, ft_output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)
    ft_output_path = Path(ft_output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ft_output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout_silver, \
         open(ft_output_path, "w", encoding="utf-8") as fout_ft:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            row = json.loads(line)

            raw_assessment = row.get("raw_assessment", "").strip()
            micro_items = row.get("micro_assessments", [])

            if not raw_assessment or len(micro_items) < 2:
                continue

            silver_row = {
                "item_id": row.get("item_id"),
                "course": row.get("course"),
                "document_type": row.get("document_type"),
                "raw_assessment": raw_assessment,
                "dominant_bloom": row.get("dominant_bloom"),
                "micro_assessments": micro_items
            }

            fout_silver.write(json.dumps(silver_row, ensure_ascii=False) + "\n")

            ft_row = {
                "input_text": build_input_text(raw_assessment),
                "target_text": format_target(micro_items)
            }
            fout_ft.write(json.dumps(ft_row, ensure_ascii=False) + "\n")

            kept += 1

    print(f"Processed {total} rows")
    print(f"Kept {kept} rows")
    print(f"Silver dataset: {output_path}")
    print(f"Fine-tuning pairs: {ft_output_path}")


if __name__ == "__main__":
    build_silver_dataset(
        "/home/ap2194.UNT/micro/data/validated_micro.jsonl",
        "/home/ap2194.UNT/micro/data/silver_dataset.jsonl",
        "/home/ap2194.UNT/micro/data/training_pairs_silver.jsonl"
    )