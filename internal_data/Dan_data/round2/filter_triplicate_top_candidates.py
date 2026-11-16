from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter triplicate top-N summaries by support count and mean rank."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("internal_data/Dan_data/round2/triplicate_perSite_summary_requested_backgrounds_top100"),
        help="Directory containing backbone-specific top-N CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("internal_data/Dan_data/round2/triplicate_perSite_summary_requested_backgrounds_top20_mean_rank"),
        help="Directory where filtered CSV files will be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of rows to keep after filtering and sorting.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_rows(rows: list[dict[str, str]], top_n: int) -> list[dict[str, str]]:
    filtered = [row for row in rows if int(row["support_runs"]) != 1]
    filtered.sort(key=lambda row: (float(row["mean_rank"]), -float(row["priority_score"])))
    return filtered[:top_n]


def output_name(input_name: str, top_n: int) -> str:
    if input_name.endswith("_top100.csv"):
        return input_name.replace("_top100.csv", f"_top{top_n}_mean_rank.csv")
    return input_name.replace(".csv", f"_top{top_n}_mean_rank.csv")


def main() -> None:
    args = parse_args()
    input_paths = sorted(
        path
        for path in args.input_dir.glob("*_top100.csv")
        if not path.name.startswith("combined_")
    )

    for input_path in input_paths:
        rows = read_rows(input_path)
        if not rows:
            continue
        filtered_rows = filter_rows(rows, args.top_n)
        output_path = args.output_dir / output_name(input_path.name, args.top_n)
        write_rows(output_path, filtered_rows, list(rows[0].keys()))


if __name__ == "__main__":
    main()