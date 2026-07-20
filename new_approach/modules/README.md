# Modular Micro-Assessment Pipeline

This pipeline:
1. loads one or more input files,
2. extracts source assessment questions,
3. generates micro-assessments first,
4. runs an intent gate,
5. optionally refines weak micro-assessments,
6. assigns Bloom levels **after** generation,
7. writes JSONL output.

## Run

```bash
python main.py   --input_paths     "/path/to/file1.txt"     "/path/to/file2.jsonl"     "/path/to/file3.csv"   --output_path "/path/to/output.jsonl"   --model_name "meta-llama/Llama-3.1-70B-Instruct"
```

## Output fields
- `micro_assessment`
- `micro_header`
- `bloom_level`
- `required_proficiency`
- `bloom_confidence`
- `bloom_rationale`
- `gate_scores`
- `refined`
- `refinement_applied`
- `refinement_failed`

## Notes
- `micro_header` is capped at **80 characters**.
- Source metadata such as `source_file`, `source_index`, and `source_text` is kept internal and is **not** written to the final JSONL.
