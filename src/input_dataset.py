import json
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_jsonl_folder(folder_path: str) -> List[Dict]:
    folder = Path(folder_path)
    all_rows = []

    for file_path in sorted(folder.glob("*.jsonl")):
        rows = load_jsonl(str(file_path))
        print(f"Loaded {len(rows)} rows from {file_path.name}")
        all_rows.extend(rows)

    print(f"Total loaded rows: {len(all_rows)}")
    return all_rows


def extract_texts(rows: List[Dict], text_field: str = "raw_text") -> List[str]:
    texts = []
    for row in rows:
        text = row.get(text_field, "")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts