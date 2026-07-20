import re
from concept_extractor import (
    extract_candidate_concepts,
    extract_compare_targets,
    is_valid_concept,
)


def make_item(text: str, bloom: str) -> dict[str, str]:
    text = re.sub(r"\s+", " ", text.strip())
    text = text[0].upper() + text[1:] if text else text
    if text and not text.endswith("."):
        text += "."
    return {"text": text, "bloom": bloom}


def decompose_compare_item(raw_text: str) -> list[dict[str, str]]:
    targets = extract_compare_targets(raw_text)
    if len(targets) == 2:
        a, b = targets[0], targets[1]
        return [
            make_item(f"Define {a}", "Remember"),
            make_item(f"Define {b}", "Remember"),
            make_item(f"List one key feature of {a}", "Remember"),
            make_item(f"List one key feature of {b}", "Remember"),
            make_item(f"Compare {a} and {b} in one important aspect", "Analyze"),
        ]
    return []


def decompose_explain_item(raw_text: str) -> list[dict[str, str]]:
    concepts = extract_candidate_concepts(raw_text)
    if not concepts:
        return []

    primary = concepts[0]
    return [
        make_item(f"Define {primary}", "Remember"),
        make_item(f"Explain the role of {primary}", "Understand"),
        make_item(f"Describe one important property of {primary}", "Understand"),
    ]


def decompose_apply_item(raw_text: str) -> list[dict[str, str]]:
    concepts = extract_candidate_concepts(raw_text)
    if not concepts:
        return []

    primary = concepts[0]
    return [
        make_item(f"Define {primary}", "Remember"),
        make_item(f"Explain the role of {primary}", "Understand"),
        make_item(f"Apply {primary} in a simple example", "Apply"),
    ]


def decompose_generic(raw_text: str) -> list[dict[str, str]]:
    concepts = extract_candidate_concepts(raw_text)
    concepts = [c for c in concepts if is_valid_concept(c)]

    if not concepts:
        return []

    items = []
    for c in concepts[:4]:
        items.append(make_item(f"Define {c}", "Remember"))

    if len(concepts) == 1:
        c = concepts[0]
        items.append(make_item(f"Explain one key feature of {c}", "Understand"))

    return items[:5]