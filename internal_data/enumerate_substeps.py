#!/usr/bin/env python3
import argparse
import os
import re
from itertools import combinations

import pandas as pd


TOKEN_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


def parse_mutation_token(token: str) -> tuple[int, str, str]:
    match = TOKEN_RE.fullmatch(token.strip())
    if match is None:
        raise ValueError(f"Invalid mutation token: {token}")
    wt_aa, position, mut_aa = match.groups()
    return int(position), wt_aa.upper(), mut_aa.upper()


def normalize_mutant(mutant: str) -> tuple[tuple[int, str, str], ...]:
    tokens = [part.strip() for part in str(mutant).split(":") if part.strip()]
    parsed = [parse_mutation_token(token) for token in tokens]
    positions = [position for position, _, _ in parsed]
    if len(positions) != len(set(positions)):
        raise ValueError(f"Duplicate mutated position found in mutant string: {mutant}")
    return tuple(sorted(parsed, key=lambda token: token[0]))


def normalize_mutant_for_wt(mutant: str, wt_sequence: str) -> tuple[tuple[tuple[int, str, str], ...], bool]:
    repaired = False
    normalized_tokens = []
    for position, stated_wt, mut_aa in normalize_mutant(mutant):
        index = position - 1
        if index < 0 or index >= len(wt_sequence):
            raise ValueError(f"Position {position} is out of bounds for sequence length {len(wt_sequence)}")
        actual_wt = wt_sequence[index]
        if actual_wt != stated_wt:
            repaired = True
        if actual_wt == mut_aa:
            repaired = True
            continue
        normalized_tokens.append((position, actual_wt, mut_aa))

    positions = [position for position, _, _ in normalized_tokens]
    if len(positions) != len(set(positions)):
        raise ValueError(f"Duplicate mutated position found after WT repair in mutant string: {mutant}")
    return tuple(sorted(normalized_tokens, key=lambda token: token[0])), repaired


def format_mutant(tokens: tuple[tuple[int, str, str], ...]) -> str:
    return ":".join(f"{wt_aa}{position}{mut_aa}" for position, wt_aa, mut_aa in tokens)


def apply_mutations(wt_sequence: str, tokens: tuple[tuple[int, str, str], ...]) -> str:
    sequence = list(wt_sequence)
    for position, wt_aa, mut_aa in tokens:
        index = position - 1
        if index < 0 or index >= len(sequence):
            raise ValueError(f"Position {position} is out of bounds for sequence length {len(sequence)}")
        if sequence[index] != wt_aa:
            raise ValueError(
                f"WT mismatch for {wt_aa}{position}{mut_aa}: expected {wt_aa}, found {sequence[index]}"
            )
        sequence[index] = mut_aa
    return "".join(sequence)


def build_base_rows(data: pd.DataFrame) -> tuple[list[dict[str, object]], int, int]:
    rows = []
    repaired_rows = 0
    skipped_rows = 0
    for record in data.to_dict("records"):
        wt_sequence = str(record["wt_sequence"])
        normalized_tokens, repaired = normalize_mutant_for_wt(record["mutant"], wt_sequence)
        if repaired:
            repaired_rows += 1
        if not normalized_tokens:
            skipped_rows += 1
            continue
        normalized_mutant = format_mutant(normalized_tokens)
        rows.append(
            {
                "mutant": normalized_mutant,
                "target": float(record["target"]),
                "wt_sequence": wt_sequence,
                "is_enumerated": False,
                "derivation_depth": 0,
                "source_mutant": pd.NA,
                "parent_mutant": pd.NA,
                "source_target": pd.NA,
                "parent_target": pd.NA,
                "source_wt_sequence": wt_sequence,
                "derivation_type": "original",
                "input_mutant": str(record["mutant"]),
                "mutation_label_repaired": repaired,
                "_tokens": normalized_tokens,
            }
        )
    return rows, repaired_rows, skipped_rows


def row_identity(row: dict[str, object]) -> tuple[str, str, float]:
    return row["wt_sequence"], row["mutant"], round(float(row["target"]), 10)


def derive_preferred_jump_rows(base_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], set[int]]:
    derived_rows = []
    replaced_base_indices: set[int] = set()
    seen = {
        row_identity(row)
        for row in base_rows
        if len(row["_tokens"]) == 1
    }
    grouped: dict[str, list[dict[str, object]]] = {}

    for index, row in enumerate(base_rows):
        grouped.setdefault(str(row["wt_sequence"]), []).append((index, row))

    for wt_sequence, indexed_rows in grouped.items():
        parent_lookup: dict[tuple[tuple[int, str, str], ...], list[dict[str, object]]] = {}
        for _, row in indexed_rows:
            parent_lookup.setdefault(row["_tokens"], []).append(row)

        for row_index, child_row in indexed_rows:
            child_tokens = child_row["_tokens"]
            if len(child_tokens) <= 1:
                continue

            chosen_parent_tokens = None
            for subset_size in range(len(child_tokens) - 1, 0, -1):
                candidates = []
                for subset in combinations(child_tokens, subset_size):
                    parent_tokens = tuple(sorted(subset, key=lambda token: token[0]))
                    if parent_tokens in parent_lookup:
                        candidates.append(parent_tokens)
                if candidates:
                    chosen_parent_tokens = candidates
                    break

            if not chosen_parent_tokens:
                continue

            replaced_base_indices.add(row_index)

            for parent_tokens in chosen_parent_tokens:
                parent_token_set = set(parent_tokens)
                remaining_tokens = tuple(token for token in child_tokens if token not in parent_token_set)
                if not remaining_tokens:
                    continue

                derived_wt = apply_mutations(wt_sequence, parent_tokens)
                derived_mutant = format_mutant(remaining_tokens)
                parent_mutant = format_mutant(parent_tokens)
                derivation_type = "enumerated_single_step" if len(remaining_tokens) == 1 else "enumerated_jump"

                for parent_row in parent_lookup[parent_tokens]:
                    derived_row = {
                        "mutant": derived_mutant,
                        "target": float(child_row["target"] - parent_row["target"]),
                        "wt_sequence": derived_wt,
                        "is_enumerated": True,
                        "derivation_depth": max(
                            int(child_row["derivation_depth"]),
                            int(parent_row["derivation_depth"]),
                        )
                        + 1,
                        "source_mutant": child_row["mutant"],
                        "parent_mutant": parent_mutant,
                        "source_target": float(child_row["target"]),
                        "parent_target": float(parent_row["target"]),
                        "source_wt_sequence": wt_sequence,
                        "derivation_type": derivation_type,
                        "input_mutant": pd.NA,
                        "mutation_label_repaired": False,
                        "_tokens": remaining_tokens,
                    }
                    identity = row_identity(derived_row)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    derived_rows.append(derived_row)

    return derived_rows, replaced_base_indices


def augment_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    required_columns = {"mutant", "target", "wt_sequence"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing_columns)}")

    base_rows, repaired_rows, skipped_rows = build_base_rows(data)
    derived_rows, replaced_base_indices = derive_preferred_jump_rows(base_rows)
    kept_base_rows = []
    for index, row in enumerate(base_rows):
        if len(row["_tokens"]) == 1 or index not in replaced_base_indices:
            kept_base_rows.append(row)

    output = pd.DataFrame(kept_base_rows + derived_rows).drop(columns=["_tokens"])
    output = output[
        [
            "mutant",
            "target",
            "wt_sequence",
            "is_enumerated",
            "derivation_depth",
            "source_mutant",
            "parent_mutant",
            "source_target",
            "parent_target",
            "source_wt_sequence",
            "derivation_type",
            "input_mutant",
            "mutation_label_repaired",
        ]
    ]
    output = output.sort_values(
        ["is_enumerated", "derivation_depth", "wt_sequence", "mutant", "target"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)
    derived_only = output.loc[output["is_enumerated"]].reset_index(drop=True)
    return output, derived_only, repaired_rows, skipped_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enumerate single-mutant residual steps by subtracting immediate parents and updating the WT background."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--derived_only_csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_df = pd.read_csv(args.input_csv)
    output_df, derived_df, repaired_rows, skipped_rows = augment_dataset(input_df)

    if args.output_csv is None:
        stem, ext = os.path.splitext(args.input_csv)
        output_csv = f"{stem}_enumerated{ext or '.csv'}"
    else:
        output_csv = args.output_csv

    if args.derived_only_csv is None:
        stem, ext = os.path.splitext(output_csv)
        derived_only_csv = f"{stem}_derived_only{ext or '.csv'}"
    else:
        derived_only_csv = args.derived_only_csv

    output_df.to_csv(output_csv, index=False)
    derived_df.to_csv(derived_only_csv, index=False)

    print(f"Original rows: {len(input_df)}")
    print(f"WT-label repaired rows: {repaired_rows}")
    print(f"Skipped no-op rows after repair: {skipped_rows}")
    print(f"Derived rows: {len(derived_df)}")
    print(f"Augmented dataset written to {output_csv}")
    print(f"Derived-only dataset written to {derived_only_csv}")


if __name__ == "__main__":
    main()
