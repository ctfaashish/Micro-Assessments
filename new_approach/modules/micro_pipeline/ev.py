from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline


BLOOM_ORDER = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split()).strip()


def unique_nonempty(items: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        text = clean_text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def ensure_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


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
                print(f"Skipping malformed JSON on line {line_no} in {path}")
    return rows


def split_source_into_aspects(source_text: str) -> List[str]:
    source_text = clean_text(source_text)
    if not source_text:
        return []

    raw_parts = re.split(
        r"\b(?:and|or|while|whereas|using|with|also|then|in terms of|with respect to)\b|[;:]",
        source_text,
        flags=re.IGNORECASE,
    )

    aspects = []
    for part in raw_parts:
        part = clean_text(part)
        if len(part.split()) >= 3:
            aspects.append(part)

    aspects = unique_nonempty(aspects)
    if not aspects:
        aspects = [source_text]
    return aspects


def extract_source_id(row: Dict[str, Any], fallback_idx: int) -> str:
    candidates = [
        row.get("source_id"),
        row.get("source_header"),
        row.get("source_text"),
        row.get("record_id"),
    ]
    for c in candidates:
        c = clean_text(c)
        if c:
            return c
    return f"row_{fallback_idx}"


def extract_source_text(row: Dict[str, Any], fallback_source_id: str) -> str:
    candidates = [
        row.get("source_text"),
        row.get("source_header"),
    ]
    for c in candidates:
        c = clean_text(c)
        if c:
            return c
    return fallback_source_id


def extract_micro_set_from_row(row: Dict[str, Any]) -> List[str]:
    if isinstance(row.get("micro_assessments"), list):
        return unique_nonempty(row["micro_assessments"])

    candidates = [
        row.get("final_micro_assessment"),
        row.get("micro_assessment"),
        row.get("initial_micro_assessment"),
    ]

    micros = []
    for c in candidates:
        c = clean_text(c)
        if c:
            micros.append(c)

    return unique_nonempty(micros[:1])  # only one micro text per row in these schemas


def extract_bloom_list_from_row(row: Dict[str, Any]) -> List[str]:
    blooms = []

    if isinstance(row.get("bloom_levels"), list):
        blooms.extend([clean_text(x) for x in row["bloom_levels"] if clean_text(x)])

    for key in ["final_bloom_level", "bloom_level", "initial_bloom_level"]:
        val = clean_text(row.get(key))
        if val:
            blooms.append(val)

    return unique_nonempty(blooms)


def group_rows_by_source(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for idx, row in enumerate(rows):
        source_id = extract_source_id(row, idx)
        source_text = extract_source_text(row, source_id)
        micros = extract_micro_set_from_row(row)
        blooms = extract_bloom_list_from_row(row)

        if source_id not in grouped:
            grouped[source_id] = {
                "source_id": source_id,
                "source_text": source_text,
                "micro_assessments": [],
                "bloom_levels": [],
            }

        grouped[source_id]["micro_assessments"].extend(micros)
        grouped[source_id]["bloom_levels"].extend(blooms)

    for source_id in grouped:
        grouped[source_id]["micro_assessments"] = unique_nonempty(grouped[source_id]["micro_assessments"])
        grouped[source_id]["bloom_levels"] = unique_nonempty(grouped[source_id]["bloom_levels"])

    return grouped


class SetEvaluator:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        nli_model_name: Optional[str] = None,
        device: int = -1,
    ):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.nli = None
        if nli_model_name:
            self.nli = pipeline(
                task="text-classification",
                model=nli_model_name,
                tokenizer=nli_model_name,
                return_all_scores=True,
                device=device,
            )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [clean_text(t) for t in texts]
        return np.asarray(
            self.embedder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )

    def coverage_score(
        self,
        source_text: str,
        micro_set: Sequence[str],
        aspects: Optional[Sequence[str]] = None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        micro_set = unique_nonempty(micro_set)
        if not micro_set:
            return 0.0, []

        if aspects is None:
            aspects = split_source_into_aspects(source_text)
        aspects = unique_nonempty(aspects)
        if not aspects:
            return 0.0, []

        micro_emb = self.encode(micro_set)
        aspect_emb = self.encode(aspects)

        details: List[Dict[str, Any]] = []
        best_scores: List[float] = []

        for i, aspect in enumerate(aspects):
            sims = np.dot(micro_emb, aspect_emb[i])
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            best_scores.append(best_score)
            details.append({
                "aspect": aspect,
                "best_micro": micro_set[best_idx],
                "best_score": round(best_score, 4),
            })

        return float(np.mean(best_scores)), details

    def redundancy_score(
        self,
        micro_set: Sequence[str],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        micro_set = unique_nonempty(micro_set)
        if len(micro_set) <= 1:
            return 0.0, []

        emb = self.encode(micro_set)
        pair_scores: List[float] = []
        pair_details: List[Dict[str, Any]] = []

        n = len(micro_set)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(emb[i], emb[j]))
                pair_scores.append(sim)
                pair_details.append({
                    "item_i": micro_set[i],
                    "item_j": micro_set[j],
                    "similarity": round(sim, 4),
                })

        return float(np.mean(pair_scores)), pair_details

    def cardinality_deviation(self, micro_set: Sequence[str], target_k: int) -> int:
        return abs(len(unique_nonempty(micro_set)) - int(target_k))

    def cardinality_ratio(self, micro_set: Sequence[str], target_k: int) -> float:
        if target_k <= 0:
            return 0.0
        return len(unique_nonempty(micro_set)) / float(target_k)

    def bloom_distribution(self, bloom_levels: Sequence[str]) -> Dict[str, float]:
        cleaned = [clean_text(x) for x in bloom_levels if clean_text(x)]
        total = len(cleaned)
        if total == 0:
            return {b: 0.0 for b in BLOOM_ORDER}

        counts = Counter(cleaned)
        return {b: counts.get(b, 0) / total for b in BLOOM_ORDER}

    def entailment_rate(
        self,
        source_text: str,
        micro_set: Sequence[str],
    ) -> Dict[str, Any]:
        if self.nli is None:
            raise ValueError("No NLI model configured. Pass --nli_model_name and --include_entailment.")

        micro_set = unique_nonempty(micro_set)
        if not micro_set:
            return {
                "entailment_rate": 0.0,
                "neutral_rate": 0.0,
                "contradiction_rate": 0.0,
                "mean_entailment_confidence": 0.0,
                "per_item": [],
            }

        per_item = []
        entailment_confidences = []

        for micro in micro_set:
            result = self.nli({"text": source_text, "text_pair": micro})
            scores = result[0] if isinstance(result, list) and result and isinstance(result[0], list) else result

            label_to_score = {}
            for item in scores:
                raw_label = str(item["label"]).lower()
                score = float(item["score"])
                if "entail" in raw_label:
                    label_to_score["entailment"] = score
                elif "neutral" in raw_label:
                    label_to_score["neutral"] = score
                elif "contrad" in raw_label:
                    label_to_score["contradiction"] = score
                else:
                    label_to_score[raw_label] = score

            best_label = max(label_to_score.items(), key=lambda x: x[1])[0]
            best_conf = float(label_to_score[best_label])

            per_item.append({
                "micro_assessment": micro,
                "label": best_label,
                "confidence": round(best_conf, 4),
            })
            entailment_confidences.append(label_to_score.get("entailment", 0.0))

        total = len(per_item)
        entailment_count = sum(1 for x in per_item if x["label"] == "entailment")
        neutral_count = sum(1 for x in per_item if x["label"] == "neutral")
        contradiction_count = sum(1 for x in per_item if x["label"] == "contradiction")

        return {
            "entailment_rate": entailment_count / total,
            "neutral_rate": neutral_count / total,
            "contradiction_rate": contradiction_count / total,
            "mean_entailment_confidence": float(np.mean(entailment_confidences)) if entailment_confidences else 0.0,
            "per_item": per_item,
        }

    def evaluate_bundle(
        self,
        source_text: str,
        micro_set: Sequence[str],
        bloom_levels: Sequence[str],
        target_k: int,
        include_entailment: bool = False,
    ) -> Dict[str, Any]:
        coverage, coverage_details = self.coverage_score(source_text, micro_set)
        redundancy, redundancy_details = self.redundancy_score(micro_set)

        result = {
            "coverage_score": round(coverage, 4),
            "coverage_details": coverage_details,
            "redundancy_score": round(redundancy, 4),
            "redundancy_details": redundancy_details,
            "cardinality_deviation": self.cardinality_deviation(micro_set, target_k),
            "cardinality_ratio": round(self.cardinality_ratio(micro_set, target_k), 4),
            "set_size": len(unique_nonempty(micro_set)),
            "target_k": int(target_k),
            "bloom_distribution": self.bloom_distribution(bloom_levels),
        }

        if include_entailment:
            result["entailment"] = self.entailment_rate(source_text, micro_set)

        return result


def evaluate_one_file(
    rows: List[Dict[str, Any]],
    evaluator: SetEvaluator,
    target_k: int,
    include_entailment: bool,
) -> pd.DataFrame:
    grouped = group_rows_by_source(rows)
    flat = []

    for source_id, bundle in grouped.items():
        result = evaluator.evaluate_bundle(
            source_text=bundle["source_text"],
            micro_set=bundle["micro_assessments"],
            bloom_levels=bundle["bloom_levels"],
            target_k=target_k,
            include_entailment=include_entailment,
        )

        row = {
            "source_id": source_id,
            "source_text": bundle["source_text"],
            "set_size": result["set_size"],
            "target_k": result["target_k"],
            "coverage_score": result["coverage_score"],
            "redundancy_score": result["redundancy_score"],
            "cardinality_deviation": result["cardinality_deviation"],
            "cardinality_ratio": result["cardinality_ratio"],
            "micro_assessments": bundle["micro_assessments"],
            "bloom_distribution": result["bloom_distribution"],
        }

        if include_entailment and "entailment" in result:
            row["entailment_rate"] = result["entailment"]["entailment_rate"]
            row["neutral_rate"] = result["entailment"]["neutral_rate"]
            row["contradiction_rate"] = result["entailment"]["contradiction_rate"]
            row["mean_entailment_confidence"] = result["entailment"]["mean_entailment_confidence"]

        flat.append(row)

    return pd.DataFrame(flat)


def evaluate_before_after(
    before_rows: List[Dict[str, Any]],
    after_rows: List[Dict[str, Any]],
    evaluator: SetEvaluator,
    target_k: int,
    include_entailment: bool,
) -> pd.DataFrame:
    before_grouped = group_rows_by_source(before_rows)
    after_grouped = group_rows_by_source(after_rows)

    common_sources = sorted(set(before_grouped.keys()) & set(after_grouped.keys()))
    if not common_sources:
        raise ValueError("No overlapping source-level IDs found between before and after files.")

    flat = []
    for source_id in common_sources:
        before_bundle = before_grouped[source_id]
        after_bundle = after_grouped[source_id]

        # use after source text if available, else before
        source_text = after_bundle["source_text"] or before_bundle["source_text"]

        before_eval = evaluator.evaluate_bundle(
            source_text=source_text,
            micro_set=before_bundle["micro_assessments"],
            bloom_levels=before_bundle["bloom_levels"],
            target_k=target_k,
            include_entailment=include_entailment,
        )
        after_eval = evaluator.evaluate_bundle(
            source_text=source_text,
            micro_set=after_bundle["micro_assessments"],
            bloom_levels=after_bundle["bloom_levels"],
            target_k=target_k,
            include_entailment=include_entailment,
        )

        row = {
            "source_id": source_id,
            "source_text": source_text,
            "before_set_size": before_eval["set_size"],
            "after_set_size": after_eval["set_size"],
            "before_coverage_score": before_eval["coverage_score"],
            "after_coverage_score": after_eval["coverage_score"],
            "coverage_delta": after_eval["coverage_score"] - before_eval["coverage_score"],
            "before_redundancy_score": before_eval["redundancy_score"],
            "after_redundancy_score": after_eval["redundancy_score"],
            "redundancy_delta": after_eval["redundancy_score"] - before_eval["redundancy_score"],
            "before_cardinality_deviation": before_eval["cardinality_deviation"],
            "after_cardinality_deviation": after_eval["cardinality_deviation"],
            "cardinality_deviation_delta": after_eval["cardinality_deviation"] - before_eval["cardinality_deviation"],
            "before_micro_assessments": before_bundle["micro_assessments"],
            "after_micro_assessments": after_bundle["micro_assessments"],
            "before_bloom_distribution": before_eval["bloom_distribution"],
            "after_bloom_distribution": after_eval["bloom_distribution"],
        }

        if include_entailment:
            row["before_entailment_rate"] = before_eval["entailment"]["entailment_rate"]
            row["after_entailment_rate"] = after_eval["entailment"]["entailment_rate"]
            row["entailment_delta"] = row["after_entailment_rate"] - row["before_entailment_rate"]

            row["before_neutral_rate"] = before_eval["entailment"]["neutral_rate"]
            row["after_neutral_rate"] = after_eval["entailment"]["neutral_rate"]
            row["neutral_delta"] = row["after_neutral_rate"] - row["before_neutral_rate"]

            row["before_contradiction_rate"] = before_eval["entailment"]["contradiction_rate"]
            row["after_contradiction_rate"] = after_eval["entailment"]["contradiction_rate"]
            row["contradiction_delta"] = row["after_contradiction_rate"] - row["before_contradiction_rate"]

        flat.append(row)

    return pd.DataFrame(flat)


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


def save_single_file_visuals(df: pd.DataFrame, outdir: Path, include_entailment: bool) -> None:
    save_hist(df["coverage_score"], "Coverage Score Distribution", "Coverage Score", "Frequency", outdir / "coverage_score_histogram.png")
    save_hist(df["redundancy_score"], "Redundancy Score Distribution", "Redundancy Score", "Frequency", outdir / "redundancy_score_histogram.png")
    save_hist(df["cardinality_deviation"], "Cardinality Deviation Distribution", "Cardinality Deviation", "Frequency", outdir / "cardinality_deviation_histogram.png")
    save_hist(df["set_size"], "Set Size Distribution", "Set Size", "Frequency", outdir / "set_size_histogram.png")

    if include_entailment and "entailment_rate" in df.columns:
        save_hist(df["entailment_rate"], "Entailment Rate Distribution", "Entailment Rate", "Frequency", outdir / "entailment_rate_histogram.png")
        save_hist(df["neutral_rate"], "Neutral Rate Distribution", "Neutral Rate", "Frequency", outdir / "neutral_rate_histogram.png")
        save_hist(df["contradiction_rate"], "Contradiction Rate Distribution", "Contradiction Rate", "Frequency", outdir / "contradiction_rate_histogram.png")

    # Bloom aggregate
    bloom_totals = {b: 0.0 for b in BLOOM_ORDER}
    count = 0
    for _, row in df.iterrows():
        dist = row.get("bloom_distribution", {})
        if isinstance(dist, dict):
            for b in BLOOM_ORDER:
                bloom_totals[b] += float(dist.get(b, 0.0))
            count += 1

    if count > 0:
        bloom_avg = pd.Series({b: bloom_totals[b] / count for b in BLOOM_ORDER})
        save_bar(
            bloom_avg,
            "Average Bloom Distribution Across Sources",
            "Bloom Level",
            "Average Proportion",
            outdir / "average_bloom_distribution.png",
        )


def save_before_after_visuals(df: pd.DataFrame, outdir: Path, include_entailment: bool) -> None:
    save_hist(df["coverage_delta"], "Coverage Change After Refinement", "After - Before Coverage", "Frequency", outdir / "coverage_delta_histogram.png")
    save_hist(df["redundancy_delta"], "Redundancy Change After Refinement", "After - Before Redundancy", "Frequency", outdir / "redundancy_delta_histogram.png")
    save_hist(df["cardinality_deviation_delta"], "Cardinality Deviation Change", "After - Before Cardinality Deviation", "Frequency", outdir / "cardinality_deviation_delta_histogram.png")
    save_hist(df["before_set_size"], "Before Set Size Distribution", "Before Set Size", "Frequency", outdir / "before_set_size_histogram.png")
    save_hist(df["after_set_size"], "After Set Size Distribution", "After Set Size", "Frequency", outdir / "after_set_size_histogram.png")

    if include_entailment and "entailment_delta" in df.columns:
        save_hist(df["entailment_delta"], "Entailment Change After Refinement", "After - Before Entailment", "Frequency", outdir / "entailment_delta_histogram.png")
        save_hist(df["neutral_delta"], "Neutral Rate Change", "After - Before Neutral Rate", "Frequency", outdir / "neutral_delta_histogram.png")
        save_hist(df["contradiction_delta"], "Contradiction Rate Change", "After - Before Contradiction Rate", "Frequency", outdir / "contradiction_delta_histogram.png")

    # Improved vs worsened bars
    summary_series = pd.Series({
        "Coverage Improved": int((pd.to_numeric(df["coverage_delta"], errors="coerce") > 0).sum()),
        "Coverage Worsened": int((pd.to_numeric(df["coverage_delta"], errors="coerce") < 0).sum()),
        "Redundancy Reduced": int((pd.to_numeric(df["redundancy_delta"], errors="coerce") < 0).sum()),
        "Redundancy Increased": int((pd.to_numeric(df["redundancy_delta"], errors="coerce") > 0).sum()),
    })
    save_bar(
        summary_series,
        "Before vs After Outcome Counts",
        "Outcome",
        "Count",
        outdir / "before_after_outcome_counts.png",
    )

    # Bloom before/after aggregate
    before_totals = {b: 0.0 for b in BLOOM_ORDER}
    after_totals = {b: 0.0 for b in BLOOM_ORDER}
    count = 0
    for _, row in df.iterrows():
        before_dist = row.get("before_bloom_distribution", {})
        after_dist = row.get("after_bloom_distribution", {})
        if isinstance(before_dist, dict) and isinstance(after_dist, dict):
            for b in BLOOM_ORDER:
                before_totals[b] += float(before_dist.get(b, 0.0))
                after_totals[b] += float(after_dist.get(b, 0.0))
            count += 1

    if count > 0:
        x = list(range(len(BLOOM_ORDER)))
        before_vals = [before_totals[b] / count for b in BLOOM_ORDER]
        after_vals = [after_totals[b] / count for b in BLOOM_ORDER]

        width = 0.38
        plt.figure(figsize=(11, 6))
        plt.bar([i - width / 2 for i in x], before_vals, width=width, label="Before")
        plt.bar([i + width / 2 for i in x], after_vals, width=width, label="After")
        plt.xticks(x, BLOOM_ORDER, rotation=25, ha="right")
        plt.xlabel("Bloom Level")
        plt.ylabel("Average Proportion")
        plt.title("Average Bloom Distribution: Before vs After")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "bloom_before_after.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_summary_csv(df: pd.DataFrame, outdir: Path, mode: str) -> None:
    if mode == "single":
        summary = {
            "num_sources": len(df),
            "avg_coverage_score": pd.to_numeric(df["coverage_score"], errors="coerce").mean(),
            "avg_redundancy_score": pd.to_numeric(df["redundancy_score"], errors="coerce").mean(),
            "avg_cardinality_deviation": pd.to_numeric(df["cardinality_deviation"], errors="coerce").mean(),
            "avg_set_size": pd.to_numeric(df["set_size"], errors="coerce").mean(),
        }
        if "entailment_rate" in df.columns:
            summary["avg_entailment_rate"] = pd.to_numeric(df["entailment_rate"], errors="coerce").mean()
            summary["avg_neutral_rate"] = pd.to_numeric(df["neutral_rate"], errors="coerce").mean()
            summary["avg_contradiction_rate"] = pd.to_numeric(df["contradiction_rate"], errors="coerce").mean()
    else:
        summary = {
            "num_sources_compared": len(df),
            "avg_coverage_delta": pd.to_numeric(df["coverage_delta"], errors="coerce").mean(),
            "avg_redundancy_delta": pd.to_numeric(df["redundancy_delta"], errors="coerce").mean(),
            "avg_cardinality_deviation_delta": pd.to_numeric(df["cardinality_deviation_delta"], errors="coerce").mean(),
            "avg_before_set_size": pd.to_numeric(df["before_set_size"], errors="coerce").mean(),
            "avg_after_set_size": pd.to_numeric(df["after_set_size"], errors="coerce").mean(),
        }
        if "entailment_delta" in df.columns:
            summary["avg_entailment_delta"] = pd.to_numeric(df["entailment_delta"], errors="coerce").mean()
            summary["avg_neutral_delta"] = pd.to_numeric(df["neutral_delta"], errors="coerce").mean()
            summary["avg_contradiction_delta"] = pd.to_numeric(df["contradiction_delta"], errors="coerce").mean()

    pd.DataFrame([summary]).to_csv(outdir / "summary.csv", index=False)


def save_top_examples(df: pd.DataFrame, outdir: Path, mode: str, top_k: int = 25) -> None:
    if mode == "single":
        subset = df.sort_values(by="coverage_score", ascending=False).head(top_k)
        subset.to_csv(outdir / "top_sources_by_coverage.csv", index=False)
    else:
        improved = df.sort_values(by="coverage_delta", ascending=False).head(top_k)
        worsened = df.sort_values(by="coverage_delta", ascending=True).head(top_k)
        improved.to_csv(outdir / "top_improved_sources.csv", index=False)
        worsened.to_csv(outdir / "top_worsened_sources.csv", index=False)

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute set-level evaluation metrics and create visualizations.")
    parser.add_argument("--input_path", type=str, default=None, help="Single JSONL file to evaluate.")
    parser.add_argument("--before_path", type=str, default=None, help="Before-refinement JSONL.")
    parser.add_argument("--after_path", type=str, default=None, help="After-refinement JSONL.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation tables and plots.")
    parser.add_argument("--target_k", type=int, default=3, help="Target number of micro-assessments per source.")
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for coverage and redundancy.",
    )
    parser.add_argument(
        "--nli_model_name",
        type=str,
        default=None,
        help="Optional NLI model for entailment metrics.",
    )
    parser.add_argument(
        "--include_entailment",
        action="store_true",
        help="Enable entailment, neutral, contradiction metrics.",
    )
    args = parser.parse_args()

    outdir = ensure_dir(args.output_dir)

    if args.input_path and (args.before_path or args.after_path):
        raise ValueError("Use either --input_path OR (--before_path and --after_path), not both.")

    if not args.input_path and not (args.before_path and args.after_path):
        raise ValueError("Provide either --input_path or both --before_path and --after_path.")

    evaluator = SetEvaluator(
        embedding_model_name=args.embedding_model_name,
        nli_model_name=args.nli_model_name,
    )

    if args.input_path:
        rows = load_jsonl(args.input_path)
        if not rows:
            raise ValueError(f"No valid rows found in: {args.input_path}")

        df = evaluate_one_file(
            rows=rows,
            evaluator=evaluator,
            target_k=args.target_k,
            include_entailment=args.include_entailment,
        )
        df.to_json(outdir / "source_level_eval.jsonl", orient="records", lines=True, force_ascii=False)
        df.to_csv(outdir / "source_level_eval.csv", index=False)

        save_summary_csv(df, outdir, mode="single")
        save_single_file_visuals(df, outdir, include_entailment=args.include_entailment)
        save_top_examples(df, outdir, mode="single", top_k=25)

        print(f"Saved single-file evaluation and visualizations to: {outdir}")

    else:
        before_rows = load_jsonl(args.before_path)
        after_rows = load_jsonl(args.after_path)

        if not before_rows:
            raise ValueError(f"No valid rows found in before file: {args.before_path}")
        if not after_rows:
            raise ValueError(f"No valid rows found in after file: {args.after_path}")

        df = evaluate_before_after(
            before_rows=before_rows,
            after_rows=after_rows,
            evaluator=evaluator,
            target_k=args.target_k,
            include_entailment=args.include_entailment,
        )
        df.to_json(outdir / "before_after_source_eval.jsonl", orient="records", lines=True, force_ascii=False)
        df.to_csv(outdir / "before_after_source_eval.csv", index=False)

        save_summary_csv(df, outdir, mode="before_after")
        save_before_after_visuals(df, outdir, include_entailment=args.include_entailment)
        save_top_examples(df, outdir, mode="before_after", top_k=25)

        print(f"Saved before/after source-level evaluation and visualizations to: {outdir}")


if __name__ == "__main__":
    main()