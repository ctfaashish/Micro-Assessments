import json
import re
import os

input_file = "open_ended_homework.jsonl"
output_file = "cleaned.jsonl"

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.strip()

    # Remove leading markers repeatedly: A. / a. / 1. / Q:
    while True:
        new_text = re.sub(r'^\s*(?:[A-Za-z]\.|\\d+\.|Q:)\s*', '', text, flags=re.IGNORECASE)
        if new_text == text:
            break
        text = new_text.strip()

    # Remove patterns like 2pts, 3 pts, 10pts
    text = re.sub(r'\b\d+\s*pts\b', ' ', text, flags=re.IGNORECASE)

    # Remove standalone junk words
    text = re.sub(r'\b(Question|Points|pts|Page)\b', ' ', text, flags=re.IGNORECASE)

    # Remove repeated dashes
    text = re.sub(r'-{2,}', ' ', text)

    # Clean extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)
        raw_text = obj.get("raw_text", "")
        cleaned_text = clean_text(raw_text)

        outfile.write(json.dumps({"raw_text": cleaned_text}, ensure_ascii=False) + "\n")

print(f"Done. Saved cleaned file to {output_file}")