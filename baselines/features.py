"""Sequence encodings and sequence-keyed, frozen ESMC embedding caches."""

import hashlib
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .protocol import AA


def one_hot(sequences):
    length = len(sequences[0])
    if any(len(sequence) != length or set(sequence) - set(AA) for sequence in sequences):
        raise ValueError("One-hot encoding requires equal-length canonical protein sequences")
    positions = np.arange(length) * len(AA)
    columns = np.concatenate([positions + np.array([AA.index(residue) for residue in sequence])
                              for sequence in sequences])
    rows = np.repeat(np.arange(len(sequences)), length)
    return csr_matrix((np.ones(len(columns), dtype=np.float32), (rows, columns)),
                      shape=(len(sequences), length * len(AA)))


def synthetic_embeddings(sequences, seed, dimension=32):
    features = []
    for sequence in sequences:
        digest = hashlib.sha256(f"{seed}:{sequence}".encode()).digest()
        generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        features.append(generator.normal(size=dimension))
    return np.asarray(features, dtype=np.float32)


def esmc_embeddings(sequences, cache_path, device, model_name="esmc_300m", batch_size=16):
    cache_path = Path(cache_path)
    lookup = {}
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache["model"]) != model_name or str(cache["pooling"]) != "benchmark_token_mean":
                raise ValueError("Embedding cache model/pooling mismatch")
            if len(set(cache["sequences"])) != len(cache["sequences"]):
                raise ValueError("Duplicate embedding cache sequences")
            lookup = dict(zip(cache["sequences"].tolist(), cache["embeddings"]))
    missing = list(dict.fromkeys(sequence for sequence in sequences if sequence not in lookup))
    if missing:
        if not device.startswith("cuda"):
            raise ValueError("Uncached ESMC inference requires an explicit CUDA device. "
                             "Use --synthetic-encoder for CPU smoke tests.")
        import torch
        from esm.models.esmc import ESMC

        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; refusing expensive CPU fallback")
        if max(map(len, missing)) > 2046:
            raise ValueError("Sequence exceeds the ESMC 2046-residue limit")
        model = ESMC.from_pretrained(model_name).to(device).eval()
        tokenizer = model.tokenizer
        with torch.inference_mode():
            for start in range(0, len(missing), batch_size):
                batch = missing[start:start + batch_size]
                tokens = tokenizer.batch_encode_plus(batch, padding="longest", truncation=False,
                                                     return_tensors="pt")
                with torch.autocast("cuda", dtype=torch.bfloat16
                                    if torch.cuda.is_bf16_supported() else torch.float16):
                    output = model.forward(tokens["input_ids"].to(device),
                                           tokens["attention_mask"].to(device))
                embeddings = output.embeddings.float().mean(dim=1).cpu().numpy()
                lookup.update(zip(batch, embeddings))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(lookup)
        temporary = cache_path.with_name(cache_path.name + ".tmp.npz")
        np.savez_compressed(temporary, sequences=np.asarray(ordered),
                            embeddings=np.stack([lookup[sequence] for sequence in ordered]),
                            model=model_name, pooling="benchmark_token_mean")
        temporary.replace(cache_path)
        del model
    values = np.stack([lookup[sequence] for sequence in sequences]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite ESMC embeddings")
    return values