#!/usr/bin/env python3
"""Prepare the ImageNet-Sketch list used by closed-set evaluation.

This script combines the old helper steps:
1. read sketchPath.txt entries in "relative/path label" or "relative/path" form
2. strip labels from the source list
3. reuse an existing sketchPath_no_labels.txt if it is present
4. keep only files that exist under the ImageNet-Sketch root
5. resolve labels from imagenet_class_to_id_map.json for reporting

Usage:
    python prepare_imagenet_sketch_list.py imagenet/robustbench/data/sketchPath.txt

By default, the script expects ImageNet-Sketch images under
/mnt/d/stamp_lib/datasets/ImageNet-Sketch/sketch. Set IMAGENET_SKETCH_ROOT
if your dataset lives elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, NamedTuple


DEFAULT_DATASET_ROOT = Path(
    os.environ.get("IMAGENET_SKETCH_ROOT", "/mnt/d/stamp_lib/datasets/ImageNet-Sketch/sketch")
)


class Entry(NamedTuple):
    rel_path: str
    source_label: int | None


class PreparedEntry(NamedTuple):
    rel_path: str
    mapped_label: int


def parse_source_line(line: str, line_number: int) -> Entry | None:
    """Parse one source line and return a relative path plus an optional label."""
    stripped = line.strip()
    if not stripped:
        return None

    parts = stripped.split()
    if len(parts) == 1:
        return Entry(parts[0], None)
    if len(parts) == 2:
        try:
            return Entry(parts[0], int(parts[1]))
        except ValueError as exc:
            raise ValueError(f"Line {line_number} has a non-integer label: {stripped}") from exc

    raise ValueError(f"Line {line_number} should contain a path and optional label: {stripped}")


def read_source_entries(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            entry = parse_source_line(line, line_number)
            if entry is not None:
                entries.append(entry)
    return entries


def read_path_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def load_class_map(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        raw_map = json.load(handle)
    return {str(wnid): int(label) for wnid, label in raw_map.items()}


def align_to_reference(entries: list[Entry], reference_paths: list[str]) -> list[Entry]:
    """Keep entries in reference-list order while preserving labels from the source list."""
    by_path: dict[str, Entry] = {}
    duplicates = 0
    for entry in entries:
        if entry.rel_path in by_path:
            duplicates += 1
            continue
        by_path[entry.rel_path] = entry

    aligned: list[Entry] = []
    missing_from_source: list[str] = []
    for rel_path in reference_paths:
        entry = by_path.get(rel_path)
        if entry is None:
            missing_from_source.append(rel_path)
            aligned.append(Entry(rel_path, None))
        else:
            aligned.append(entry)

    if duplicates:
        print(f"Source duplicate paths ignored while aligning: {duplicates}")
    if missing_from_source:
        print(f"Reference paths not found in source: {len(missing_from_source)}")
        for rel_path in missing_from_source[:10]:
            print(f"  missing in source: {rel_path}")

    return aligned


def prepare_entries(
    entries: list[Entry],
    dataset_root: Path,
    class_to_idx: dict[str, int],
) -> tuple[list[PreparedEntry], list[str], list[tuple[str, int | None, int]]]:
    prepared: list[PreparedEntry] = []
    missing_files: list[str] = []
    label_mismatches: list[tuple[str, int | None, int]] = []
    skipped_unknown_wnids: list[str] = []

    for entry in entries:
        wnid = entry.rel_path.split("/", 1)[0]
        mapped_label = class_to_idx.get(wnid)
        if mapped_label is None:
            skipped_unknown_wnids.append(entry.rel_path)
            continue

        if entry.source_label is not None and entry.source_label != mapped_label:
            label_mismatches.append((entry.rel_path, entry.source_label, mapped_label))

        if (dataset_root / entry.rel_path).is_file():
            prepared.append(PreparedEntry(entry.rel_path, mapped_label))
        else:
            missing_files.append(entry.rel_path)

    if skipped_unknown_wnids:
        print(f"Skipped paths with unknown wnids: {len(skipped_unknown_wnids)}")
        for rel_path in skipped_unknown_wnids[:10]:
            print(f"  unknown wnid: {rel_path}")

    return prepared, missing_files, label_mismatches


def print_balance_report(labels: list[int], title: str, expected_classes: int) -> None:
    if not labels:
        print(f"{title}: no samples")
        return

    counts = Counter(labels)
    count_hist = Counter(counts.values())
    print(f"{title}:")
    print(f"  samples: {len(labels)}")
    print(f"  classes: {len(counts)} / {expected_classes}")
    print(f"  min per class: {min(counts.values())}")
    print(f"  max per class: {max(counts.values())}")
    print(f"  count histogram: {dict(sorted(count_hist.items()))}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build sketchPath_no_labels_validated.txt from sketchPath.txt."
        )
    )
    parser.add_argument("input", type=Path, help="Source sketchPath.txt file.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    input_path = args.input
    data_dir = input_path.parent
    class_map_path = data_dir / "imagenet_class_to_id_map.json"
    reference_path = data_dir / f"{input_path.stem}_no_labels.txt"
    output_path = data_dir / f"{input_path.stem}_no_labels_validated.txt"
    dataset_root = DEFAULT_DATASET_ROOT

    if not input_path.is_file():
        raise FileNotFoundError(f"Source file not found: {input_path}")
    if not class_map_path.is_file():
        raise FileNotFoundError(f"Class map not found: {class_map_path}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"ImageNet-Sketch root not found: {dataset_root}. "
            "Set IMAGENET_SKETCH_ROOT if the dataset is stored elsewhere."
        )

    entries = read_source_entries(input_path)
    class_to_idx = load_class_map(class_map_path)

    print(f"Source file: {input_path}")
    print(f"Source entries: {len(entries)}")
    print(f"Class map: {class_map_path}")
    print(f"Dataset root: {dataset_root}")

    if reference_path.is_file():
        reference_paths = read_path_list(reference_path)
        entries = align_to_reference(entries, reference_paths)
        print(f"Reference path-only file: {reference_path}")
        print(f"Reference path-only entries: {len(reference_paths)}")
    else:
        print(f"Reference path-only file not found, using stripped source entries: {reference_path}")

    prepared, missing_files, label_mismatches = prepare_entries(
        entries=entries,
        dataset_root=dataset_root,
        class_to_idx=class_to_idx,
    )

    write_lines(output_path, (entry.rel_path for entry in prepared))
    print(f"Wrote validated path list: {output_path}")

    print("Validation summary:")
    print(f"  checked paths: {len(entries)}")
    print(f"  existing files: {len(prepared)}")
    print(f"  missing files: {len(missing_files)}")
    print(f"  source-label mismatches: {len(label_mismatches)}")

    if missing_files:
        for rel_path in missing_files[:20]:
            print(f"  missing file: {rel_path}")

    if label_mismatches:
        for rel_path, source_label, mapped_label in label_mismatches[:10]:
            print(f"  label mismatch: {rel_path} source={source_label} mapped={mapped_label}")

    prepared_labels = [entry.mapped_label for entry in prepared]
    print_balance_report(prepared_labels, "All prepared samples", len(class_to_idx))
    print_balance_report(
        prepared_labels[:5000],
        f"First {min(5000, len(prepared_labels))} prepared samples",
        len(class_to_idx),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
