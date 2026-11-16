"""Local ProteinGym alignments, novel-target MMseqs2 retrieval and PSSM scores."""

import io
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .protocol import AA


def parse_a3m(path):
    sequences = []
    current = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append("".join(character for character in line
                                       if not character.islower() and character != "."))
    if current:
        sequences.append("".join(current))
    if not sequences or len({len(sequence) for sequence in sequences}) != 1:
        raise ValueError("MSA is empty or has inconsistent match-column lengths")
    if any(set(sequence) - set(AA + "-XBZUOJ") for sequence in sequences):
        raise ValueError("Invalid alignment characters")
    return list(dict.fromkeys(sequences))


def pssm_predict(path, reference, mutants, start=1, pseudocount=1.0):
    if pseudocount <= 0:
        raise ValueError("Pseudocount must be positive")
    sequences = parse_a3m(path)
    query = sequences[0]
    query_positions = [index for index, residue in enumerate(query) if residue != "-"]
    ungapped_query = query.replace("-", "")
    if start < 1 or reference[start - 1:start - 1 + len(ungapped_query)] != ungapped_query:
        raise ValueError("MSA query does not match the reference at the configured start")
    counts = np.full((len(ungapped_query), len(AA)), pseudocount, dtype=float)
    for sequence in sequences:
        for position, column in enumerate(query_positions):
            residue = sequence[column]
            if residue in AA:
                counts[position, AA.index(residue)] += 1
    log_probabilities = np.log(counts / counts.sum(axis=1, keepdims=True))
    scores = []
    covered = []
    for mutant in mutants:
        score = 0.0
        coverage = 0
        tokens = mutant.split(":")
        for token in tokens:
            position = int(token[1:-1]) - start
            if 0 <= position < len(ungapped_query):
                if ungapped_query[position] != token[0]:
                    raise ValueError(f"MSA reference mismatch for {token}")
                score += log_probabilities[position, AA.index(token[-1])] - \
                         log_probabilities[position, AA.index(token[0])]
                coverage += 1
        scores.append(score)
        covered.append(coverage / len(tokens))
    return np.asarray(scores), np.asarray(covered)


def proteingym_msa(assay, reference_csv, msa_directory):
    reference = pd.read_csv(reference_csv)
    rows = reference.loc[reference.DMS_id == assay]
    if len(rows) != 1:
        raise ValueError(f"Expected one ProteinGym reference entry for {assay}")
    row = rows.iloc[0]
    filename = row.MSA_filename
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Invalid ProteinGym MSA filename")
    path = Path(msa_directory) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Precomputed ProteinGym MSA missing: {path}; no API fallback")
    return path, int(row.MSA_start)


def fetch_mmseqs2(sequence, output, api_url="https://api.colabfold.com", timeout=900,
                  session=None, poll_seconds=5):
    output = Path(output)
    if output.exists():
        if parse_a3m(output)[0].replace("-", "") != sequence:
            raise ValueError("Cached novel-target MSA query mismatch")
        return output
    session = session or requests.Session()
    response = session.post(api_url.rstrip("/") + "/ticket/msa", data={
        "q": f">101\n{sequence}\n", "mode": "env"}, timeout=60)
    response.raise_for_status()
    ticket = response.json()
    deadline = time.monotonic() + timeout
    while ticket.get("status") in {"UNKNOWN", "PENDING", "RUNNING"}:
        if time.monotonic() >= deadline:
            raise TimeoutError("MMseqs2 job exceeded polling deadline")
        time.sleep(poll_seconds)
        response = session.get(api_url.rstrip("/") + f"/ticket/{ticket['id']}", timeout=60)
        response.raise_for_status()
        ticket = response.json()
    if ticket.get("status") != "COMPLETE":
        raise RuntimeError(f"MMseqs2 job failed: {ticket.get('status')}")
    response = session.get(api_url.rstrip("/") + f"/result/download/{ticket['id']}", timeout=120)
    response.raise_for_status()
    alignments = []
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile() and Path(member.name).name in {"uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"}:
                if member.size > 100_000_000:
                    raise ValueError("MMseqs2 alignment exceeds size limit")
                alignments.append(archive.extractfile(member).read().decode().replace("\x00", ""))
    if not alignments:
        raise ValueError("MMseqs2 result contains no supported alignments")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.a3m")
    temporary.write_text("\n".join(alignments))
    if parse_a3m(temporary)[0].replace("-", "") != sequence:
        temporary.unlink()
        raise ValueError("MMseqs2 query mismatch")
    temporary.replace(output)
    return output