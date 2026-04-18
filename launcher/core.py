from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FILENAME_TOKEN_PATTERN = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)

BEFORE_HINT_TOKENS = {
    "before",
    "до",
    "pre",
    "raw",
    "orig",
    "original",
    "source",
    "dirty",
    "trash",
    "litter",
    "unclean",
}

AFTER_HINT_TOKENS = {
    "after",
    "после",
    "post",
    "result",
    "output",
    "clean",
    "cleaned",
    "final",
}

BEFORE_HINT_SUBSTRINGS = ("before", "dirty", "trash", "litter", "unclean", "orig")
AFTER_HINT_SUBSTRINGS = ("after", "после", "clean", "result", "output", "final")


@dataclass
class AnalysisResult:
    method_id: str
    method_name: str
    summary: str
    metrics: dict[str, Any]
    before_path: Path
    after_path: Path
    preview_text: str
    preview_image_path: Path | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)


def list_image_files(folder: str | Path) -> list[Path]:
    folder_path = Path(folder)
    files = [path for path in sorted(folder_path.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS]
    return files


def select_before_after(files: list[Path]) -> tuple[Path, Path]:
    if len(files) < 2:
        raise ValueError("At least two image files are required")

    ordered = sorted(files, key=lambda path: path.name.casefold())
    fallback = (ordered[0], ordered[1])

    scored = {path: _filename_role_scores(path) for path in ordered}
    best_pair = fallback
    best_score = 0

    for before_path in ordered:
        before_score, before_after_score = scored[before_path]
        before_margin = before_score - before_after_score

        for after_path in ordered:
            if after_path == before_path:
                continue

            after_before_score, after_score = scored[after_path]
            after_margin = after_score - after_before_score
            pair_score = before_margin + after_margin

            if pair_score > best_score:
                best_score = pair_score
                best_pair = (before_path, after_path)

    if best_score > 0:
        return best_pair
    return fallback


def _filename_role_scores(path: Path) -> tuple[int, int]:
    stem = path.stem.casefold()
    tokens = FILENAME_TOKEN_PATTERN.findall(stem)

    before_score = 0
    after_score = 0

    for token in tokens:
        if token in BEFORE_HINT_TOKENS:
            before_score += 3
        if token in AFTER_HINT_TOKENS:
            after_score += 3

    for hint in BEFORE_HINT_SUBSTRINGS:
        if hint in stem:
            before_score += 1

    for hint in AFTER_HINT_SUBSTRINGS:
        if hint in stem:
            after_score += 1

    return before_score, after_score
