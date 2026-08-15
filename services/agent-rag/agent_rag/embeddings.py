import hashlib
import math
import re


def stable_embedding(text: str, dimensions: int = 64) -> list[float]:
    """Deterministic demo embedding; production can swap in a model endpoint."""
    values = [0.0] * dimensions
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower()) or [text.lower()]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        values[index] += 1.0 if digest[4] % 2 else -1.0
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))
