import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .config import TEXT_KEYS_PRIORITY
from .schemas import SourceQuestion
from .text_utils import clean_text, looks_like_question, make_short_header, normalize_candidate_question, split_plain_text_into_questions



def extract_text_from_record(record: Dict[str, Any]) -> List[str]:
    results: List[str] = []

    for key in TEXT_KEYS_PRIORITY:
        value = record.get(key)
        if isinstance(value, str):
            value = normalize_candidate_question(value)
            if looks_like_question(value):
                results.append(value)

    if not results:
        for value in record.values():
            if isinstance(value, str):
                value = normalize_candidate_question(value)
                if looks_like_question(value):
                    results.append(value)

    deduped: List[str] = []
    seen = set()
    for item in results:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped



def load_jsonl_file(path: Path) -> List[str]:
    results: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                candidate = normalize_candidate_question(line)
                if looks_like_question(candidate):
                    results.append(candidate)
                continue

            if isinstance(obj, dict):
                results.extend(extract_text_from_record(obj))
            elif isinstance(obj, str):
                candidate = normalize_candidate_question(obj)
                if looks_like_question(candidate):
                    results.append(candidate)
    return results



def load_json_file(path: Path) -> List[str]:
    results: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                results.extend(extract_text_from_record(item))
            elif isinstance(item, str):
                candidate = normalize_candidate_question(item)
                if looks_like_question(candidate):
                    results.append(candidate)
    elif isinstance(obj, dict):
        results.extend(extract_text_from_record(obj))
        for value in obj.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        results.extend(extract_text_from_record(item))
                    elif isinstance(item, str):
                        candidate = normalize_candidate_question(item)
                        if looks_like_question(candidate):
                            results.append(candidate)
    return results



def load_csv_like_file(path: Path, delimiter: str) -> List[str]:
    results: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            results.extend(extract_text_from_record(row))
    return results



def load_plain_text_file(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return split_plain_text_into_questions(text)



def load_texts_from_file(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl_file(path)
    if suffix == ".json":
        return load_json_file(path)
    if suffix == ".csv":
        return load_csv_like_file(path, ",")
    if suffix == ".tsv":
        return load_csv_like_file(path, "\t")
    return load_plain_text_file(path)



def load_questions_from_files(input_paths: Sequence[str]) -> List[SourceQuestion]:
    records: List[SourceQuestion] = []
    global_seen = set()

    for file_path in input_paths:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        texts = load_texts_from_file(path)
        local_index = 0
        for text in texts:
            source_text = normalize_candidate_question(text)
            if not looks_like_question(source_text):
                continue
            key = clean_text(source_text).lower()
            if key in global_seen:
                continue
            global_seen.add(key)
            local_index += 1
            records.append(
                SourceQuestion(
                    source_file=str(path),
                    source_index=local_index,
                    source_text=source_text,
                    source_header=make_short_header(source_text),
                )
            )

    return records
