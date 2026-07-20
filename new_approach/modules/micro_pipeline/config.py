from typing import Dict, List

VALID_BLOOM: List[str] = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]

BLOOM_TO_PROFICIENCY: Dict[str, str] = {
    "Remember": "Beginner",
    "Understand": "Beginner",
    "Apply": "Intermediate",
    "Analyze": "Intermediate",
    "Evaluate": "Advanced",
    "Create": "Advanced",
}

HEADER_MAX_CHARS = 150
MIN_QUESTION_WORDS = 4
MAX_QUESTION_WORDS = 160
DEFAULT_MICRO_COUNT = 3
DEFAULT_GATE_THRESHOLD = 0.68
DEFAULT_MAX_NEW_TOKENS = 384

TEXT_KEYS_PRIORITY = [
    "micro_assessment",
    "question",
    "text",
    "raw_text",
    "assessment",
    "question_text",
    "prompt",
    "item",
]

QUESTION_STARTERS = {
    "what", "why", "how", "which", "when", "where", "who",
    "define", "explain", "describe", "compare", "contrast",
    "analyze", "evaluate", "justify", "design", "create",
    "implement", "compute", "solve", "list", "identify",
    "apply", "show", "find", "trace", "write", "discuss",
    "derive", "classify", "differentiate", "examine", "argue",
    "assess", "critique", "demonstrate", "outline", "interpret",
    "state", "recall", "name",
}

BLOOM_HINTS = {
    "Remember": ["define", "list", "identify", "name", "state", "recall"],
    "Understand": ["explain", "describe", "summarize", "interpret", "outline"],
    "Apply": ["apply", "use", "solve", "compute", "calculate", "implement", "demonstrate"],
    "Analyze": ["analyze", "compare", "contrast", "differentiate", "examine", "distinguish"],
    "Evaluate": ["evaluate", "justify", "assess", "critique", "defend", "argue"],
    "Create": ["design", "construct", "develop", "propose", "create", "build"],
}

ALL_ACTION_HINTS = sorted({token for values in BLOOM_HINTS.values() for token in values} | {
    "show", "find", "trace", "write", "discuss", "derive", "classify"
})
