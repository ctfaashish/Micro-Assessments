import json
import os

input_file = "open_ended_homework.jsonl"
temp_file = "temp.jsonl"

with open(input_file, "r", encoding="utf-8") as infile, open(temp_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)
        new_obj = {"raw_text": obj["raw_text"]} if "raw_text" in obj else {}
        outfile.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

os.replace(temp_file, input_file)

print(f"Done. Updated {input_file} with only raw_text field.")