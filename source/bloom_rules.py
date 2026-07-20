import re
from typing import Dict, List, Tuple

BLOOM_LEVELS = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]

BLOOM_VERB_RULES: Dict[str, List[str]] = {
    "Remember": [
        "define", "list", "identify", "name", "state", "recall", "label"
    ],
    "Understand": [
        "explain", "describe", "summarize", "discuss", "interpret", "outline"
    ],
    "Apply": [
        "apply", "use", "solve", "compute", "calculate", "implement", "demonstrate"
    ],
    "Analyze": [
        "analyze", "compare", "contrast", "differentiate", "examine", "distinguish"
    ],
    "Evaluate": [
        "evaluate", "justify", "assess", "critique", "argue", "defend"
    ],
    "Create": [
        "design", "construct", "develop", "propose", "create", "build"
    ],
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def detect_bloom_scores(text: str) -> Dict[str, int]:
    text_norm = normalize_text(text)
    scores = {level: 0 for level in BLOOM_LEVELS}

    for level, verbs in BLOOM_VERB_RULES.items():
        for verb in verbs:
            # whole word match
            if re.search(rf"\b{re.escape(verb)}\b", text_norm):
                scores[level] += 1

    # task-pattern boosts
    if re.search(r"\bcompare\b|\bcontrast\b", text_norm):
        scores["Analyze"] += 2

    if re.search(r"\bdefine\b|\blist\b|\bidentify\b", text_norm):
        scores["Remember"] += 2

    if re.search(r"\bexplain\b|\bdescribe\b", text_norm):
        scores["Understand"] += 2

    if re.search(r"\bcompute\b|\bsolve\b|\bcalculate\b", text_norm):
        scores["Apply"] += 2

    if re.search(r"\bjustify\b|\bevaluate\b|\bcritique\b", text_norm):
        scores["Evaluate"] += 2

    if re.search(r"\bdesign\b|\bdevelop\b|\bcreate\b|\bpropose\b", text_norm):
        scores["Create"] += 2

    return scores


def predict_dominant_bloom(text: str) -> Tuple[str, Dict[str, int]]:
    scores = detect_bloom_scores(text)
    best = max(scores.items(), key=lambda x: x[1])[0]

    # fallback when everything is zero
    if all(v == 0 for v in scores.values()):
        best = "Understand"

    return best, scores


def predict_bloom_from_micro_text(text: str) -> str:
    best, _ = predict_dominant_bloom(text)
    return best