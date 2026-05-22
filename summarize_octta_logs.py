#!/usr/bin/env python3
"""Summarize OCTTA log metrics under the current result directory.

Run from this directory:
    python summarize_octta_logs.py

It recursively scans *.txt logs, extracts method/OOD dataset and four mean
metrics, then writes readable Markdown and CSV tables next to this script.
If multiple logs have the same method/OOD dataset, only the latest log is used.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


METRICS = (
    ("accuracy", "Mean Accuracy", "Mean Accuracy"),
    ("auc_energy", "Mean AUC_Energy", "Mean AUC_Energy"),
    ("auoscr_energy", "Mean AUOSCR_Energy", "Mean AUOSCR_Energy"),
    ("hscore_energy", "Mean H-Score_Energy", "Mean H-Score_Energy"),
)

METHOD_RE = re.compile(r"^\s*ADAPTATION:\s*(\S+)\s*$", re.MULTILINE)
OOD_RE = re.compile(r"^\s*OOD_DATASET:\s*(\S+)\s*$", re.MULTILINE)
LOG_TIME_RE = re.compile(r"^\s*LOG_TIME:\s*(\d{6}_\d{6})\s*$", re.MULTILINE)
FILENAME_TIME_RE = re.compile(r"_(\d{6}_\d{6})_")
EATA_CRITERION_RE = re.compile(
    r"(?:EATA_UNIENT_CRITERION:\s*|criterion\s*=\s*|criterion is\s*)(ent_unf|ent)\b",
    re.IGNORECASE,
)
METRIC_RES = {
    key: re.compile(rf"{re.escape(log_label)}:\s*([-+]?\d+(?:\.\d+)?)\s*%")
    for key, log_label, _ in METRICS
}

OUTPUT_MD = "octta_summary_table.md"
OUTPUT_CSV = "octta_summary_table.csv"


@dataclass(frozen=True)
class LogRecord:
    method: str
    ood_dataset: str
    accuracy: float
    auc_energy: float
    auoscr_energy: float
    hscore_energy: float
    log_time: str
    mtime_ns: int
    path: Path


@dataclass(frozen=True)
class SummaryRow:
    method: str
    ood_dataset: str
    n_logs: int
    accuracy: float
    auc_energy: float
    auoscr_energy: float
    hscore_energy: float
    files: tuple[str, ...]


def extract_last_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def normalize_method(method: str, text: str) -> str:
    normalized = method.strip().lower()
    if normalized == "eata_unient":
        normalized = "eataunient"

    if normalized == "eataunient" and "using the original EATA" in text:
        return "eata"

    return normalized


def parse_log(path: Path, root: Path) -> tuple[LogRecord | None, str | None]:
    text = path.read_text(encoding="utf-8", errors="replace")

    method_match = METHOD_RE.search(text)
    ood_match = OOD_RE.search(text)
    log_time_match = LOG_TIME_RE.search(text) or FILENAME_TIME_RE.search(path.name)
    values = {key: extract_last_float(pattern, text) for key, pattern in METRIC_RES.items()}

    missing = []
    if method_match is None:
        missing.append("ADAPTATION")
    if ood_match is None:
        missing.append("OOD_DATASET")
    missing.extend(label for key, _, label in METRICS if values[key] is None)
    if missing:
        return None, f"{path.relative_to(root)}: missing {', '.join(missing)}"

    return (
        LogRecord(
            method=normalize_method(method_match.group(1), text),
            ood_dataset=ood_match.group(1),
            accuracy=values["accuracy"],
            auc_energy=values["auc_energy"],
            auoscr_energy=values["auoscr_energy"],
            hscore_energy=values["hscore_energy"],
            log_time=log_time_match.group(1) if log_time_match else "",
            mtime_ns=path.stat().st_mtime_ns,
            path=path,
        ),
        None,
    )


def fmt(value: float) -> str:
    return f"{value:.2f}"


def sort_name(value: str) -> tuple[int, str]:
    preferred = {
        "source": 0,
        "tent": 1,
        "cotta": 2,
        "eata": 3,
        "sar":4, 
        "ostta": 5,
        "vida": 6,
        "eataunient": 7,
        "stamp": 8,
        "eatacome": 9,
        "sarcome": 10,
        "dpcore": 11,
        "doco": 12,
    }
    return preferred.get(value, 100), value


def sort_dataset(value: str) -> tuple[int, str]:
    preferred = {
        "places365": 0,
        "textures": 1,
        "inaturalist": 2,
        "sun": 3,
        "ssb-hard": 4,
        "ninco_ood_classes": 5,
    }
    return preferred.get(value, 100), value


def aggregate_records(records: list[LogRecord], root: Path) -> list[SummaryRow]:
    groups: dict[tuple[str, str], list[LogRecord]] = defaultdict(list)
    for record in records:
        groups[(record.method, record.ood_dataset)].append(record)

    rows = []
    for (method, dataset), group in groups.items():
        selected = max(group, key=lambda record: (record.log_time, record.mtime_ns, str(record.path)))
        rows.append(
            SummaryRow(
                method=method,
                ood_dataset=dataset,
                n_logs=len(group),
                accuracy=selected.accuracy,
                auc_energy=selected.auc_energy,
                auoscr_energy=selected.auoscr_energy,
                hscore_energy=selected.hscore_energy,
                files=(str(selected.path.relative_to(root)),),
            )
        )
    return sorted(rows, key=lambda row: (sort_dataset(row.ood_dataset), sort_name(row.method)))


def method_average_rows(rows: list[SummaryRow]) -> list[SummaryRow]:
    by_method: dict[str, list[SummaryRow]] = defaultdict(list)
    for row in rows:
        by_method[row.method].append(row)

    averages = []
    for method, group in by_method.items():
        averages.append(
            SummaryRow(
                method=method,
                ood_dataset="AVG",
                n_logs=sum(row.n_logs for row in group),
                accuracy=mean(row.accuracy for row in group),
                auc_energy=mean(row.auc_energy for row in group),
                auoscr_energy=mean(row.auoscr_energy for row in group),
                hscore_energy=mean(row.hscore_energy for row in group),
                files=(),
            )
        )
    return sorted(averages, key=lambda row: sort_name(row.method))


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


def bold_if_doco(row: SummaryRow, values: list[str]) -> list[str]:
    if row.method != "doco":
        return values
    return [f"**{value}**" for value in values]


def build_markdown(root: Path, rows: list[SummaryRow], averages: list[SummaryRow], skipped: list[str]) -> str:
    detail_headers = [
        "OOD Dataset",
        "Method",
        "Mean Accuracy",
        "Mean AUC_Energy",
        "Mean AUOSCR_Energy",
        "Mean H-Score_Energy",
    ]
    detail_rows = []
    previous_dataset = None
    for row in rows:
        if previous_dataset is not None and previous_dataset != row.ood_dataset:
            detail_rows.append(markdown_separator_row(detail_headers))
        detail_rows.append(
            bold_if_doco(
                row,
                [
                    row.ood_dataset,
                    row.method,
                    fmt(row.accuracy),
                    fmt(row.auc_energy),
                    fmt(row.auoscr_energy),
                    fmt(row.hscore_energy),
                ],
            )
        )
        previous_dataset = row.ood_dataset

    average_rows = [
        bold_if_doco(
            row,
            [
                row.method,
                str(len([detail for detail in rows if detail.method == row.method])),
                fmt(row.accuracy),
                fmt(row.auc_energy),
                fmt(row.auoscr_energy),
                fmt(row.hscore_energy),
            ],
        )
        for row in averages
    ]

    lines = [
        "# OCTTA Summary",
        "",
        f"- Selected method x OOD pairs: {len(rows)}",
        f"- Metrics are percentages and rounded to two decimals.",
        "",
        "## Method Averages",
        "",
        markdown_table(
            [
                "Method",
                "OOD Datasets",
                "Mean Accuracy",
                "Mean AUC_Energy",
                "Mean AUOSCR_Energy",
                "Mean H-Score_Energy",
            ],
            average_rows,
        ),
        "",
        "## Method x OOD Dataset",
        "",
        markdown_table(
            detail_headers,
            detail_rows,
        ),
    ]

    duplicate_rows = [
        [
            row.ood_dataset,
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
                    ["OOD Dataset", "Method", "Candidates", "Selected Latest File"],
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
                "ood_dataset",
                "n_logs",
                "mean_accuracy",
                "mean_auc_energy",
                "mean_auoscr_energy",
                "mean_hscore_energy",
                "log_files",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    "dataset",
                    row.method,
                    row.ood_dataset,
                    row.n_logs,
                    fmt(row.accuracy),
                    fmt(row.auc_energy),
                    fmt(row.auoscr_energy),
                    fmt(row.hscore_energy),
                    ";".join(row.files),
                ]
            )
        for row in averages:
            writer.writerow(
                [
                    "method_average",
                    row.method,
                    row.ood_dataset,
                    row.n_logs,
                    fmt(row.accuracy),
                    fmt(row.auc_energy),
                    fmt(row.auoscr_energy),
                    fmt(row.hscore_energy),
                    "",
                ]
            )


def main() -> int:
    root = Path(__file__).resolve().parent
    records: list[LogRecord] = []
    skipped: list[str] = []

    ignored_names = {Path(__file__).name, OUTPUT_MD, OUTPUT_CSV}
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
