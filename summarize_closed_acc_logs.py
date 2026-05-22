#!/usr/bin/env python3
"""Summarize closed benchmark ACC logs under the current result directory.

Run from this directory, for example:

    cd ~/bDOCO0422/output_closed_benchmarks
    python summarize_closed_acc_logs.py

Directory examples supported:

    imagenet-a/dpcore/dpcore_260424_005406_closed_IA_L8.txt
    imagenet-c/doco/doco_260423_231108_closed_IC_L8.txt
    imagenet-r/source/source_260423_233633_closed_IR.txt
    imagenet-sketch/tent/tent_260423_233313_closed_IS_lr0.001.txt
    laion-c/sev3/ORDER_1/.../*.txt

The script recursively scans *.txt logs, extracts method, dataset and ACC
/ Mean Accuracy, then writes Markdown and CSV tables next to this script.

If multiple logs have the same method x dataset pair, only the latest log is
selected by LOG_TIME / filename timestamp / file mtime.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


OUTPUT_MD = "closed_benchmark_acc_summary_table.md"
OUTPUT_CSV = "closed_benchmark_acc_summary_table.csv"


METHOD_RE = re.compile(r"^\s*ADAPTATION:\s*(\S+)\s*$", re.MULTILINE)
OOD_RE = re.compile(r"^\s*OOD_DATASET:\s*(\S+)\s*$", re.MULTILINE)
LOG_TIME_RE = re.compile(r"^\s*LOG_TIME:\s*(\d{6}_\d{6})\s*$", re.MULTILINE)
FILENAME_TIME_RE = re.compile(r"_(\d{6}_\d{6})_")

# Prefer Mean Accuracy. The other variants are fallbacks for slightly different logs.
ACC_PATTERNS = (
    re.compile(r"Mean\s+Accuracy:\s*([-+]?\d+(?:\.\d+)?)\s*%?", re.IGNORECASE),
    re.compile(r"Mean\s+Acc:\s*([-+]?\d+(?:\.\d+)?)\s*%?", re.IGNORECASE),
    re.compile(r"Final\s+Accuracy:\s*([-+]?\d+(?:\.\d+)?)\s*%?", re.IGNORECASE),
    re.compile(r"Final\s+Acc:\s*([-+]?\d+(?:\.\d+)?)\s*%?", re.IGNORECASE),
)


KNOWN_METHODS = {
    "source",
    "tent",
    "cotta",
    "eata",
    "eata_unient",
    "eataunient",
    "eatacome",
    "sar",
    "sarcome",
    "ostta",
    "vida",
    "stamp",
    "dpcore",
    "doco",
}


METHOD_ORDER = {
    "source": 0,
    "tent": 1,
    "cotta": 2,
    "eata": 3,
    "sar": 4,
    "ostta": 5,
    "vida": 6,
    "eataunient": 7,
    "stamp": 8,
    "eatacome": 9,
    "sarcome": 10,
    "dpcore": 11,
    "doco": 12,
}


DATASET_ORDER = {
    "imagenet-a": 0,
    "imagenet-c": 1,
    "imagenet-r": 2,
    "imagenet-sketch": 3,
    "laion-c/sev3/ORDER_1": 4,

    # Kept for backward compatibility with previous OCTTA-style outputs.
    "places365": 100,
    "textures": 101,
    "inaturalist": 102,
    "sun": 103,
    "ssb-hard": 104,
    "ninco_ood_classes": 105,
}


@dataclass(frozen=True)
class LogRecord:
    method: str
    dataset: str
    accuracy: float
    log_time: str
    mtime_ns: int
    path: Path


@dataclass(frozen=True)
class SummaryRow:
    method: str
    dataset: str
    n_logs: int
    accuracy: float
    files: tuple[str, ...]


def normalize_method(method: str, text: str = "") -> str:
    normalized = method.strip().lower()

    if normalized == "eata_unient":
        normalized = "eataunient"

    # Some old logs may say eataunient while actually running original EATA.
    if normalized == "eataunient" and "using the original EATA" in text:
        return "eata"

    return normalized


def extract_last_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def extract_accuracy(text: str) -> float | None:
    for pattern in ACC_PATTERNS:
        value = extract_last_float(pattern, text)
        if value is not None:
            return value
    return None


def path_parts_without_file(path: Path, root: Path) -> tuple[str, ...]:
    rel = path.relative_to(root)
    return rel.parts[:-1]


def infer_method_from_path(path: Path, root: Path) -> str | None:
    parts = path_parts_without_file(path, root)

    for part in reversed(parts):
        normalized = normalize_method(part)
        if normalized in KNOWN_METHODS:
            return normalized

    if parts:
        # Fallback: parent directory is usually the method directory.
        return normalize_method(parts[-1])

    return None


def infer_dataset_from_path(path: Path, root: Path, method: str | None) -> str | None:
    parts = path_parts_without_file(path, root)
    if not parts:
        return None

    normalized_parts = [normalize_method(part) for part in parts]

    if method:
        method = normalize_method(method)
        for idx, part in enumerate(normalized_parts):
            if part == method and idx > 0:
                return "/".join(parts[:idx])

    # Special handling for laion-c nested structure:
    # laion-c/sev3/ORDER_1/<method>/log.txt
    # or deeper under laion-c/sev3/ORDER_1/...
    if parts[0] == "laion-c":
        if len(parts) >= 3 and parts[1].startswith("sev") and parts[2].startswith("ORDER"):
            return "/".join(parts[:3])
        if len(parts) >= 2 and parts[1].startswith("sev"):
            return "/".join(parts[:2])
        return parts[0]

    # Standard structure:
    # imagenet-a/<method>/log.txt
    # imagenet-c/<method>/log.txt
    return parts[0]


def parse_log(path: Path, root: Path) -> tuple[LogRecord | None, str | None]:
    text = path.read_text(encoding="utf-8", errors="replace")

    method_match = METHOD_RE.search(text)
    method = normalize_method(method_match.group(1), text) if method_match else infer_method_from_path(path, root)

    dataset = infer_dataset_from_path(path, root, method)
    if dataset is None:
        ood_match = OOD_RE.search(text)
        dataset = ood_match.group(1) if ood_match else None

    log_time_match = LOG_TIME_RE.search(text) or FILENAME_TIME_RE.search(path.name)
    accuracy = extract_accuracy(text)

    missing = []
    if method is None:
        missing.append("ADAPTATION/method")
    if dataset is None:
        missing.append("OOD_DATASET/dataset")
    if accuracy is None:
        missing.append("Mean Accuracy/ACC")

    if missing:
        return None, f"{path.relative_to(root)}: missing {', '.join(missing)}"

    return (
        LogRecord(
            method=method,
            dataset=dataset,
            accuracy=accuracy,
            log_time=log_time_match.group(1) if log_time_match else "",
            mtime_ns=path.stat().st_mtime_ns,
            path=path,
        ),
        None,
    )


def fmt(value: float) -> str:
    return f"{value:.2f}"


def sort_method(value: str) -> tuple[int, str]:
    return METHOD_ORDER.get(value, 100), value


def sort_dataset(value: str) -> tuple[int, str]:
    return DATASET_ORDER.get(value, 1000), value


def aggregate_records(records: list[LogRecord], root: Path) -> list[SummaryRow]:
    groups: dict[tuple[str, str], list[LogRecord]] = defaultdict(list)

    for record in records:
        groups[(record.method, record.dataset)].append(record)

    rows: list[SummaryRow] = []

    for (method, dataset), group in groups.items():
        selected = max(
            group,
            key=lambda record: (
                bool(record.log_time),
                record.log_time,
                record.mtime_ns,
                str(record.path),
            ),
        )

        rows.append(
            SummaryRow(
                method=method,
                dataset=dataset,
                n_logs=len(group),
                accuracy=selected.accuracy,
                files=(str(selected.path.relative_to(root)),),
            )
        )

    return sorted(rows, key=lambda row: (sort_dataset(row.dataset), sort_method(row.method)))


def method_average_rows(rows: list[SummaryRow]) -> list[SummaryRow]:
    by_method: dict[str, list[SummaryRow]] = defaultdict(list)

    for row in rows:
        by_method[row.method].append(row)

    averages: list[SummaryRow] = []

    for method, group in by_method.items():
        averages.append(
            SummaryRow(
                method=method,
                dataset="AVG",
                n_logs=sum(row.n_logs for row in group),
                accuracy=mean(row.accuracy for row in group),
                files=(),
            )
        )

    return sorted(averages, key=lambda row: sort_method(row.method))


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows)) if rows else len(headers[col])
        for col in range(len(headers))
    ]

    header_line = "| " + " | ".join(headers[col].ljust(widths[col]) for col in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[col] for col in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[col].ljust(widths[col]) for col in range(len(headers))) + " |"
        for row in rows
    ]

    return "\n".join([header_line, sep_line, *body])


def markdown_separator_row(headers: list[str]) -> list[str]:
    return ["---" for _ in headers]


def build_markdown(root: Path, rows: list[SummaryRow], averages: list[SummaryRow], skipped: list[str]) -> str:
    detail_headers = ["Dataset", "Method", "ACC"]

    detail_rows: list[list[str]] = []
    previous_dataset: str | None = None

    for row in rows:
        if previous_dataset is not None and previous_dataset != row.dataset:
            detail_rows.append(markdown_separator_row(detail_headers))

        detail_rows.append(
            [
                row.dataset,
                row.method,
                fmt(row.accuracy),
            ]
        )
        previous_dataset = row.dataset

    average_rows = [
        [
            row.method,
            str(len([detail for detail in rows if detail.method == row.method])),
            fmt(row.accuracy),
        ]
        for row in averages
    ]

    lines = [
        "# Closed Benchmark ACC Summary",
        "",
        f"- Root: `{root}`",
        f"- Selected method x dataset pairs: {len(rows)}",
        "- Metric: ACC / Mean Accuracy only.",
        "- Values are rounded to two decimals.",
        "",
        "## Method Averages",
        "",
        markdown_table(
            ["Method", "Datasets", "ACC"],
            average_rows,
        ),
        "",
        "## Method x Dataset",
        "",
        markdown_table(
            detail_headers,
            detail_rows,
        ),
    ]

    duplicate_rows = [
        [
            row.dataset,
            row.method,
            str(row.n_logs),
            row.files[0],
        ]
        for row in rows
        if row.n_logs > 1
    ]

    if duplicate_rows:
        lines.extend(
            [
                "",
                "## Duplicate Selections",
                "",
                markdown_table(
                    ["Dataset", "Method", "Candidates", "Selected Latest File"],
                    duplicate_rows,
                ),
            ]
        )

    if skipped:
        lines.extend(["", "## Skipped Files", ""])
        lines.extend(f"- {item}" for item in skipped)

    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[SummaryRow], averages: list[SummaryRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "row_type",
                "method",
                "dataset",
                "n_logs",
                "acc",
                "log_files",
            ]
        )

        for row in rows:
            writer.writerow(
                [
                    "dataset",
                    row.method,
                    row.dataset,
                    row.n_logs,
                    fmt(row.accuracy),
                    ";".join(row.files),
                ]
            )

        for row in averages:
            writer.writerow(
                [
                    "method_average",
                    row.method,
                    row.dataset,
                    row.n_logs,
                    fmt(row.accuracy),
                    "",
                ]
            )


def main() -> int:
    root = Path(__file__).resolve().parent

    records: list[LogRecord] = []
    skipped: list[str] = []

    ignored_names = {
        Path(__file__).name,
        OUTPUT_MD,
        OUTPUT_CSV,
    }

    for path in sorted(root.rglob("*.txt")):
        if path.name in ignored_names:
            continue

        record, error = parse_log(path, root)

        if record is None:
            if error:
                skipped.append(error)
            continue

        records.append(record)

    if not records:
        print(f"No parseable .txt logs found under {root}")
        return 1

    rows = aggregate_records(records, root)
    averages = method_average_rows(rows)

    md_path = root / OUTPUT_MD
    csv_path = root / OUTPUT_CSV

    md_path.write_text(build_markdown(root, rows, averages, skipped), encoding="utf-8")
    write_csv(csv_path, rows, averages)

    print(f"Parsed {len(records)} log file(s).")
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")

    if skipped:
        print(f"Skipped {len(skipped)} file(s); see {md_path.name} for details.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())