import json
from pathlib import Path
from bloom_rules import predict_dominant_bloom


def label_original_bloom(input_path: str, output_path: str, text_field: str = "raw_text"):
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
            text = row.get(text_field, "").strip()
            if not text:
                continue

            bloom, scores = predict_dominant_bloom(text)
            row["dominant_bloom"] = bloom
            row["bloom_scores"] = scores

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1

    print(f"Labeled {total} rows with dominant Bloom")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    label_original_bloom(
        "/home/ap2194.UNT/micro/data/parsed_items.jsonl",
        "/home/ap2194.UNT/micro/data/original_bloom_labeled.jsonl"
    )