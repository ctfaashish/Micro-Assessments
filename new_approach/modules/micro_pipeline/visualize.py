import argparse
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


BLOOM_ORDER = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]

POSSIBLE_ID_KEYS = [
    "id",
    "question_id",
    "micro_id",
    "item_id",
    "source_id",
    "uid",
]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON at line {line_no} in {path}")
    return rows


def ensure_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    return " ".join(str(x).replace("\r", " ").replace("\n", " ").split()).strip()


def find_best_id_key(rows1: List[Dict[str, Any]], rows2: List[Dict[str, Any]]) -> Optional[str]:
    for key in POSSIBLE_ID_KEYS:
        vals1 = {row.get(key) for row in rows1 if row.get(key) is not None}
        vals2 = {row.get(key) for row in rows2 if row.get(key) is not None}
        if vals1 and vals2 and len(vals1.intersection(vals2)) > 0:
            return key
    return None


def row_identifier(row: Dict[str, Any], id_key: Optional[str], fallback_index: int) -> Any:
    if id_key is not None and row.get(id_key) is not None:
        return row.get(id_key)
    return fallback_index


def extract_text(row: Dict[str, Any]) -> str:
    candidates = [
        row.get("micro_assessment"),
        row.get("final_micro_assessment"),
        row.get("initial_micro_assessment"),
        row.get("text"),
        row.get("question"),
        row.get("micro_header"),
    ]
    for c in candidates:
        c = clean_text(c)
        if c:
            return c
    return ""


def extract_bloom(row: Dict[str, Any]) -> Optional[str]:
    return row.get("bloom_level") or row.get("final_bloom_level") or row.get("initial_bloom_level")


def extract_score(row: Dict[str, Any]) -> Optional[float]:
    candidates = [
        row.get("overall_score"),
        row.get("final_overall_score"),
        row.get("initial_overall_score"),
        (row.get("gate_scores") or {}).get("overall_score") if isinstance(row.get("gate_scores"), dict) else None,
    ]
    for c in candidates:
        if isinstance(c, (int, float)):
            return float(c)
    return None


def compute_change_ratio(before: str, after: str) -> Optional[float]:
    before = clean_text(before)
    after = clean_text(after)
    if not before or not after:
        return None
    return 1.0 - SequenceMatcher(None, before, after).ratio()


def align_rows(
    before_rows: List[Dict[str, Any]],
    after_rows: List[Dict[str, Any]],
) -> Tuple[List[Tuple[Any, Dict[str, Any], Dict[str, Any]]], str]:
    id_key = find_best_id_key(before_rows, after_rows)

    if id_key is not None:
        before_map = {}
        after_map = {}

        for i, row in enumerate(before_rows):
            rid = row_identifier(row, id_key, i)
            before_map[rid] = row

        for i, row in enumerate(after_rows):
            rid = row_identifier(row, id_key, i)
            after_map[rid] = row

        common_ids = [rid for rid in before_map if rid in after_map]
        pairs = [(rid, before_map[rid], after_map[rid]) for rid in common_ids]
        return pairs, f"matched by key: {id_key}"

    # fallback: row-order alignment
    min_len = min(len(before_rows), len(after_rows))
    pairs = []
    for i in range(min_len):
        pairs.append((i, before_rows[i], after_rows[i]))
    return pairs, "matched by row order"


def build_dataframe(
    before_rows: List[Dict[str, Any]],
    after_rows: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, str]:
    aligned, match_mode = align_rows(before_rows, after_rows)

    flat = []
    for rid, before_row, after_row in aligned:
        before_text = extract_text(before_row)
        after_text = extract_text(after_row)

        before_bloom = extract_bloom(before_row)
        after_bloom = extract_bloom(after_row)

        before_score = extract_score(before_row)
        after_score = extract_score(after_row)

        change_ratio = compute_change_ratio(before_text, after_text)
        word_delta = None
        if before_text and after_text:
            word_delta = len(after_text.split()) - len(before_text.split())

        flat.append({
            "row_id": rid,
            "before_text": before_text,
            "after_text": after_text,
            "before_bloom": before_bloom,
            "after_bloom": after_bloom,
            "before_score": before_score,
            "after_score": after_score,
            "score_delta": (after_score - before_score) if before_score is not None and after_score is not None else None,
            "change_ratio": change_ratio,
            "word_delta": word_delta,
            "changed_text": before_text != after_text if before_text and after_text else None,
        })

    return pd.DataFrame(flat), match_mode


def save_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(10, 6))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_hist(series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path, bins: int = 20) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        print(f"Skipping {path.name}: no numeric data")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_change_status_plot(df: pd.DataFrame, outdir: Path) -> None:
    changed = int(df["changed_text"].fillna(False).sum())
    unchanged = int((df["changed_text"].fillna(False) == False).sum())

    series = pd.Series({
        "Changed": changed,
        "Unchanged": unchanged,
    })
    save_bar(
        series,
        "Before vs After Text Change Counts",
        "Status",
        "Count",
        outdir / "changed_vs_unchanged.png",
    )


def save_change_ratio_plot(df: pd.DataFrame, outdir: Path) -> None:
    save_hist(
        df["change_ratio"],
        "Text Change Ratio",
        "Change Ratio (0 = same, 1 = very different)",
        "Frequency",
        outdir / "text_change_ratio_histogram.png",
        bins=20,
    )


def save_word_delta_plot(df: pd.DataFrame, outdir: Path) -> None:
    save_hist(
        df["word_delta"],
        "Word Count Difference (After - Before)",
        "Word Delta",
        "Frequency",
        outdir / "word_delta_histogram.png",
        bins=20,
    )


def save_score_delta_plot(df: pd.DataFrame, outdir: Path) -> None:
    subset = df["score_delta"]
    if subset.dropna().empty:
        print("Skipping score delta plot: no score data available")
        return

    save_hist(
        subset,
        "Score Change After Refinement",
        "After Score - Before Score",
        "Frequency",
        outdir / "score_delta_histogram.png",
        bins=20,
    )


def save_bloom_before_after_plot(df: pd.DataFrame, outdir: Path) -> None:
    subset = df[df["before_bloom"].notna() & df["after_bloom"].notna()]
    if subset.empty:
        print("Skipping Bloom before/after plot: no Bloom data")
        return

    before_counts = Counter(subset["before_bloom"])
    after_counts = Counter(subset["after_bloom"])

    x = list(range(len(BLOOM_ORDER)))
    before_vals = [before_counts.get(b, 0) for b in BLOOM_ORDER]
    after_vals = [after_counts.get(b, 0) for b in BLOOM_ORDER]

    width = 0.38
    plt.figure(figsize=(11, 6))
    plt.bar([i - width / 2 for i in x], before_vals, width=width, label="Before")
    plt.bar([i + width / 2 for i in x], after_vals, width=width, label="After")
    plt.xticks(x, BLOOM_ORDER, rotation=25, ha="right")
    plt.xlabel("Bloom Level")
    plt.ylabel("Count")
    plt.title("Bloom Distribution: Before vs After Refinement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bloom_before_after.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_top_changed_examples(df: pd.DataFrame, outdir: Path, top_k: int = 25) -> None:
    subset = df[df["change_ratio"].notna()].copy()
    if subset.empty:
        print("Skipping top changed examples: no before/after text")
        return

    subset = subset.sort_values(by="change_ratio", ascending=False).head(top_k)
    subset.to_csv(outdir / "top_changed_examples.csv", index=False)

    with open(outdir / "top_changed_examples.md", "w", encoding="utf-8") as f:
        f.write("# Top Changed Examples\n\n")
        for _, row in subset.iterrows():
            f.write(f"## Row {row['row_id']}\n\n")
            if pd.notna(row["change_ratio"]):
                f.write(f"**Change ratio:** {row['change_ratio']:.4f}\n\n")
            if pd.notna(row["before_bloom"]) or pd.notna(row["after_bloom"]):
                f.write(f"**Bloom:** {row['before_bloom']} → {row['after_bloom']}\n\n")
            if pd.notna(row["score_delta"]):
                f.write(f"**Score delta:** {row['score_delta']:.4f}\n\n")
            f.write("**Before**\n\n")
            f.write(f"{row['before_text']}\n\n")
            f.write("**After**\n\n")
            f.write(f"{row['after_text']}\n\n")
            f.write("---\n\n")


def save_summary(df: pd.DataFrame, outdir: Path, match_mode: str, before_count: int, after_count: int) -> None:
    summary = {
        "before_rows": before_count,
        "after_rows": after_count,
        "matched_rows": len(df),
        "match_mode": match_mode,
        "changed_count": int(df["changed_text"].fillna(False).sum()),
        "unchanged_count": int((df["changed_text"].fillna(False) == False).sum()),
        "avg_change_ratio": pd.to_numeric(df["change_ratio"], errors="coerce").mean(),
        "avg_word_delta": pd.to_numeric(df["word_delta"], errors="coerce").mean(),
        "avg_score_delta": pd.to_numeric(df["score_delta"], errors="coerce").mean(),
    }
    pd.DataFrame([summary]).to_csv(outdir / "before_after_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Visualize changes between before/after refinement files.")
    parser.add_argument("--before_path", type=str, required=True, help="Path to JSONL file before refinement.")
    parser.add_argument("--after_path", type=str, required=True, help="Path to JSONL file after refinement.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots and reports.")
    args = parser.parse_args()

    before_rows = load_jsonl(args.before_path)
    after_rows = load_jsonl(args.after_path)

    if not before_rows:
        raise ValueError(f"No valid rows found in before file: {args.before_path}")
    if not after_rows:
        raise ValueError(f"No valid rows found in after file: {args.after_path}")

    outdir = ensure_dir(args.output_dir)
    df, match_mode = build_dataframe(before_rows, after_rows)

    if df.empty:
        raise ValueError("No aligned rows found between before and after files.")

    df.to_csv(outdir / "flattened_before_after.csv", index=False)
    save_summary(df, outdir, match_mode, len(before_rows), len(after_rows))

    save_change_status_plot(df, outdir)
    save_change_ratio_plot(df, outdir)
    save_word_delta_plot(df, outdir)
    save_score_delta_plot(df, outdir)
    save_bloom_before_after_plot(df, outdir)
    save_top_changed_examples(df, outdir, top_k=25)

    print(f"Saved before/after refinement visualizations to: {outdir}")
    print(f"Alignment mode: {match_mode}")


if __name__ == "__main__":
    main()