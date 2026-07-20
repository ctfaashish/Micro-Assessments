import argparse
from pathlib import Path
from typing import List

from micro_pipeline.pipeline import MicroAssessmentPipeline

SUPPORTED_EXTENSIONS = {".txt", ".md", ".jsonl", ".json", ".csv", ".tsv"}


def collect_input_files(input_dir: str) -> List[str]:
    directory = Path(input_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Provided input_dir is not a directory: {input_dir}")

    files = [
        str(file)
        for file in sorted(directory.iterdir())
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise ValueError(
            f"No supported files found in directory: {input_dir}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    return files


def validate_input_paths(input_paths: List[str]) -> List[str]:
    valid_files = []

    for path_str in input_paths:
        path = Path(path_str)

        if path.is_dir():
            valid_files.extend(collect_input_files(str(path)))
            continue

        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path_str}")
        if not path.is_file():
            raise ValueError(f"Input path is not a file: {path_str}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path_str}. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        valid_files.append(str(path))

    # remove duplicates while preserving order
    seen = set()
    deduped = []
    for f in valid_files:
        if f not in seen:
            seen.add(f)
            deduped.append(f)

    if not deduped:
        raise ValueError("No valid input files found.")

    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate micro-assessments from one or more input files, then assign Bloom levels."
    )
    parser.add_argument(
        "--input_paths",
        nargs="+",
        required=True,
        help="One or more input files or directories containing supported files.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="HF model name for micro-assessment generation, refinement, and Bloom assignment.",
    )
    parser.add_argument(
        "--embedding_model_name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for the intent gate.",
    )
    parser.add_argument(
        "--micro_count",
        type=int,
        default=3,
        help="Number of micro-assessments to generate per source question.",
    )
    parser.add_argument(
        "--gate_threshold",
        type=float,
        default=0.68,
        help="Intent gate threshold.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=384,
        help="Maximum LLM generation length.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_files = validate_input_paths(args.input_paths)

    print(f"Found {len(input_files)} input file(s):")
    for file_path in input_files:
        print(f"  - {file_path}")

    pipeline = MicroAssessmentPipeline(
        model_name=args.model_name,
        embedding_model_name=args.embedding_model_name,
        micro_count=args.micro_count,
        gate_threshold=args.gate_threshold,
        max_new_tokens=args.max_new_tokens,
    )

    # THIS is the important fix
    pipeline.run(input_paths=input_files, output_path=args.output_path)