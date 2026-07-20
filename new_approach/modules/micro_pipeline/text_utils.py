import re
from typing import Iterable, List, Set

from .config import (
    ALL_ACTION_HINTS,
    HEADER_MAX_CHARS,
    MAX_QUESTION_WORDS,
    MIN_QUESTION_WORDS,
    QUESTION_STARTERS,
)


STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "of", "to", "for", "in", "on", "with", "by",
    "at", "from", "into", "that", "this", "these", "those", "is", "are", "be",
    "as", "it", "its", "their", "your", "you", "using", "use", "given", "based",
    "can", "should", "would", "could", "must", "do", "does", "did", "about",
}


def clean_text(text: str) -> str:
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text



def make_short_header(text: str, max_chars: int = HEADER_MAX_CHARS) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    if not truncated:
        truncated = text[:max_chars].strip()
    return truncated[:max_chars].rstrip(" ,;:.")



def strip_noise(text: str) -> str:
    text = str(text).replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"(?<=\w)\n(?=[a-z(])", " ", text)

    kept_lines: List[str] = []
    for raw_line in text.split("\n"):
        line = clean_text(raw_line)
        lower = line.lower()
        if not line:
            continue
        if re.fullmatch(r"[\d\W_]+", line):
            continue
        junk_patterns = [
            r"^page\s+\d+(\s+of\s+\d+)?$",
            r"^chapter\s+\d+$",
            r"^table of contents$",
            r"^(name|student name|student id|score|submitted|instructions)[: ]*$",
            r"^(downloaded from|copyright|all rights reserved).*",
            r"^(http|https)://",
            r"^www\.",
            r"^\(?\d+\s*(pt|pts|point|points)\)?[: ]*$",
            r"^(course|semester|term|department)[: ].*",
        ]
        if any(re.match(p, lower) for p in junk_patterns):
            continue
        if re.fullmatch(r"[A-Z]{2,}\s*\d{3,4}[A-Z]?(?:\s+[-_:A-Za-z0-9]+){0,5}", line) and len(line.split()) <= 8:
            continue
        digits = sum(ch.isdigit() for ch in line)
        letters = sum(ch.isalpha() for ch in line)
        if digits > letters and letters < 6:
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()



def has_assessment_signal(text: str) -> bool:
    lower = clean_text(text).lower()
    if "?" in lower:
        return True
    if re.match(r"^(q\d+|question\s*\d+|\d+[\.)])\s*", lower):
        return True
    starter_pattern = r"^(" + "|".join(sorted(re.escape(x) for x in QUESTION_STARTERS)) + r")\b"
    if re.match(starter_pattern, lower):
        return True
    return any(re.search(rf"\b{re.escape(token)}\b", lower) for token in ALL_ACTION_HINTS)



def is_probably_garbage(text: str) -> bool:
    text = clean_text(text)
    if not text or len(text) < 12:
        return True
    if re.fullmatch(r"[\W\d_]+", text):
        return True

    words = re.findall(r"\b[a-zA-Z]{2,}\b", text)
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    symbols = sum((not c.isalnum()) and (not c.isspace()) for c in text)

    if len(words) < 3 or letters < 10:
        return True

    has_task_signal = has_assessment_signal(text)
    if digits / max(len(text), 1) > 0.35 and not has_task_signal:
        return True
    if symbols / max(len(text), 1) > 0.25 and not has_task_signal:
        return True

    lower = text.lower()
    junk_patterns = [
        r"\bpage\s+\d+\b",
        r"\bstudent\s+name\b",
        r"\bstudent\s+id\b",
        r"\bscore\b",
        r"\bsubmitted\b",
        r"\binstructions\b",
        r"\banswer\s+key\b",
        r"\bcopyright\b",
        r"\bdownloaded\s+from\b",
        r"\bwww\.",
        r"\bhttp",
    ]
    return any(re.search(p, lower) for p in junk_patterns) and not has_task_signal



def normalize_candidate_question(text: str) -> str:
    text = strip_noise(text)
    text = clean_text(text)
    text = re.sub(r"^(q\d+|question\s*\d+)\s*[:.)-]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\d+[\.)]\s*", "", text)
    text = re.sub(r"^\(?\d+\s*(pt|pts|point|points)\)?\s*[:.)-]?\s*", "", text, flags=re.IGNORECASE)
    return clean_text(text)



def looks_like_question(text: str) -> bool:
    if not text:
        return False
    text = normalize_candidate_question(text)
    if is_probably_garbage(text):
        return False
    word_count = len(text.split())
    if word_count < MIN_QUESTION_WORDS or word_count > MAX_QUESTION_WORDS:
        return False
    if has_assessment_signal(text):
        return True
    alpha_words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
    return len(alpha_words) >= 5



def is_question_start(line: str) -> bool:
    lower = clean_text(line).lower()
    if not lower:
        return False
    if re.match(r"^(q\d+|question\s*\d+|\d+[\.)])\s*", lower):
        return True
    starter_pattern = r"^(" + "|".join(sorted(re.escape(x) for x in QUESTION_STARTERS)) + r")\b"
    return re.match(starter_pattern, lower) is not None



def split_plain_text_into_questions(text: str) -> List[str]:
    text = strip_noise(text)
    lines = [clean_text(x) for x in text.split("\n") if clean_text(x)]
    if not lines:
        return []

    chunks: List[str] = []
    current: List[str] = []
    for line in lines:
        if is_question_start(line):
            if current:
                chunk = normalize_candidate_question(" ".join(current))
                if looks_like_question(chunk):
                    chunks.append(chunk)
            current = [line]
        else:
            if current:
                current.append(line)
            else:
                current = [line]
    if current:
        chunk = normalize_candidate_question(" ".join(current))
        if looks_like_question(chunk):
            chunks.append(chunk)

    if not chunks:
        for para in re.split(r"\n\s*\n+", text):
            para = normalize_candidate_question(para)
            if looks_like_question(para):
                chunks.append(para)

    if not chunks:
        for line in lines:
            line = normalize_candidate_question(line)
            if looks_like_question(line):
                chunks.append(line)

    deduped: List[str] = []
    seen: Set[str] = set()
    for chunk in chunks:
        key = chunk.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(chunk)
    return deduped



def tokenize_keywords(text: str) -> Set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", text)
        if token.lower() not in STOPWORDS
    }
    return tokens



def jaccard_overlap(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
    a = set(tokens_a)
    b = set(tokens_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
