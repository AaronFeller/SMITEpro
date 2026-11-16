#!/usr/bin/env python3
import os
from typing import Dict, List, Optional, Tuple

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.neighbors import KNeighborsRegressor
# import xgboost
# from sklearn.ensemble import RandomForestRegressor

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time

import argparse
from copy import deepcopy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from esm.models.esmc import ESMC

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # "high" or "highest"

# === Model loader ===
def load_model(model_name="esmc_300m"):
    model = ESMC.from_pretrained(model_name)
    return model, model.tokenizer


AA20 = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

@torch.no_grad()
def predict_all_site_deltas(model, tokenizer, wt_seq: str, device, batch_size: int = 64, position = None):
    """
    For each position i in wt_seq:
      - builds a sequence with that residue replaced by <mask>
      - gets log p(vocab) at the masked token
      - returns Δ(log p) for the 20 AAs vs the WT residue, plus entropy.

    Returns dict with:
      'positions' : (L,) int32  1-based positions
      'wt'        : (L,) str    WT residue at each position
      'entropy'   : (L,) float32  site entropy H = -sum p log p
      'delta20'   : (L, 20) float32, order given by 'aa_order'
      'aa_order'  : list[str] = AA20
    """
    model.eval().to(device)

    mask_tok = tokenizer.mask_token
    mask_id  = tokenizer.mask_token_id
    vocab    = tokenizer.get_vocab()

    # map AA→id (raise if any missing from the tokenizer vocab)
    aa_ids = torch.tensor([vocab[a] for a in AA20], dtype=torch.long, device=device)

    # build masked sequences: replace exactly one residue with <mask>
    if position is not None:
        L = 1  # only one position
        masked = [wt_seq[:position-1] + mask_tok + wt_seq[position:]]
    else:
        L = len(wt_seq)
        masked = [wt_seq[:i] + mask_tok + wt_seq[i+1:] for i in range(L)]

    # tokenize all at once (then batch)
    enc = tokenizer.batch_encode_plus(
        masked, padding="longest", truncation=True, max_length=2046, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    # find the <mask> token index in each row
    mask_positions = (input_ids == mask_id).nonzero(as_tuple=False)
    # mask_positions has rows [row_idx, token_pos]; enforce exactly one per row
    # sort by row_idx to make extraction easy
    mask_positions = mask_positions[mask_positions[:,0].argsort()]
    # sanity: there should be exactly one <mask> per sequence
    # (if your tokenizer adds special tokens, that's fine; we searched by id)
    # Optional strict check:
    # assert (torch.bincount(mask_positions[:,0], minlength=input_ids.size(0)) == 1).all()

    # outputs to fill
    entropy = np.empty(L, dtype=np.float32)
    delta20 = np.empty((L, 20), dtype=np.float32)
    wt_arr  = np.array(list(wt_seq))

    # run in batches
    for start in range(0, L, batch_size):
        end = min(start + batch_size, L)
        x = input_ids[start:end]
        am = attn_mask[start:end] if attn_mask is not None else None

        out = model(x, am) if am is not None else model(x)
        logp = F.log_softmax(out.sequence_logits, dim=-1)  # [B, T, V]
        p    = torch.exp(logp)

        # masked token positions for this batch (convert to row-local index)
        batch_mask_pos = (x == mask_id).nonzero(as_tuple=False)
        # row-local sorting
        batch_mask_pos = batch_mask_pos[batch_mask_pos[:,0].argsort()]
        # (B, ) token indices
        kpos = batch_mask_pos[:,1]

        # gather per-row masked distributions
        B = end - start
        rows = torch.arange(B, device=device)
        g = logp[rows, kpos]    # [B, V]
        pv= p[rows, kpos]       # [B, V]

        # entropy per site
        H = -(pv * g).sum(dim=-1)   # [B]
        entropy[start:end] = H.float().cpu().numpy()

        # Δ(log p) for AA20 vs WT residue
        # log p for candidate amino acids
        lp_aa = g[:, aa_ids]                # [B, 20]
        # log p for WT letter at each masked site
        wt_ids = torch.tensor([vocab[a] for a in wt_arr[start:end]],
                              dtype=torch.long, device=device)  # [B]
        lp_wt = g[rows, wt_ids].unsqueeze(1)                    # [B,1]
        d = (lp_aa - lp_wt).float().cpu().numpy()               # [B,20]
        delta20[start:end, :] = d

    # --- FIX RETURN VALUES IF ONLY ONE POSITION ---
    if position is not None:
        true_positions = np.array([position], dtype=np.int32)
        true_wt        = np.array([wt_seq[position-1]])
    else:
        true_positions = np.arange(1, L+1, dtype=np.int32)
        true_wt        = wt_arr

    return {
        "positions": true_positions,
        "wt": true_wt,
        "entropy": entropy,
        "delta20": delta20,
        "aa_order": AA20,
    }

# === Data preparation ===
def prep_data(
    df: pd.DataFrame,
    tokenizer,
    vocab: Dict[str, int],
    max_length: int = 2046,
) -> Dict[str, torch.Tensor]:
    """
    Prepare batched inputs for N-mutation scoring with MASK-at-site.

    Input dataframe columns expected:
      - "wt_sequence": str, wild-type sequence
      - "mutant":      str, colon-separated mutations like "F42L:R87K" (1-based positions)
      - optional "mutated_sequence": str, for consistency checks (same length as wt_sequence)
      - optional "target" or "DMS_score": float, per-row target response

    Returns a dict of PyTorch tensors:
      input_ids      : [N, L] long
      attention_mask : [N, L] long
      pos_idx        : [N, K_max] long     (token indices of <mask> in token space)
      wt_index       : [N, K_max] long     (token IDs for WT amino acids)
      mut_index      : [N, K_max] long     (token IDs for mutant amino acids)
      mut_mask       : [N, K_max] bool     (True where a site is present)
      target         : [N] float32         (only if supplied/derivable)
    """

    # --------------------
    # Helpers
    # --------------------
    def parse_mutation(token: str) -> Optional[Tuple[str, int, str]]:
        """
        Parse a single mutation token like 'F42L' into (wt_aa, pos_1based, mut_aa).
        Returns None if token is empty/invalid.
        """
        token = (token or "").strip()
        if len(token) < 3:
            return None
        wt_aa = token[0]
        mut_aa = token[-1]
        try:
            pos_1based = int(token[1:-1])
        except ValueError:
            return None
        return wt_aa, pos_1based, mut_aa

    def apply_mutations_to_sequence(
        wt_sequence: str,
        positions_1based: List[int],
        mutant_aas: List[str],
    ) -> str:
        """
        Return mutated sequence after applying (pos -> mutant_aa).
        """
        seq_chars = list(wt_sequence)
        for pos_1based, mut_aa in zip(positions_1based, mutant_aas):
            seq_chars[pos_1based - 1] = mut_aa
        return "".join(seq_chars)

    def make_masked_sequence(wt_sequence: str, positions_1based: List[int]) -> str:
        """
        Return a string with <mask> token placed at mutated positions (replacing WT char),
        truncated to `max_length` if necessary.
        """
        if not positions_1based:
            return wt_sequence[:max_length]

        zero_based_positions = sorted({p - 1 for p in positions_1based})
        parts: List[str] = []
        last_idx = 0
        for pos0 in zero_based_positions:
            parts.append(wt_sequence[last_idx:pos0])
            parts.append(tokenizer.mask_token)  # replace the residue at this position
            last_idx = pos0 + 1
        parts.append(wt_sequence[last_idx:])
        masked = "".join(parts)
        return masked[:max_length]

    def to_token_id(amino_acid: str) -> int:
        try:
            return int(vocab[str(amino_acid)])
        except Exception as exc:
            raise KeyError(f"Vocab has no token id for amino acid '{amino_acid}'") from exc

    # --------------------
    # Copy and target normalization (optional)
    # --------------------
    data = df.copy()

    if "target" not in data.columns and "DMS_score" in data.columns:
        # Use DMS_score as target and optionally apply midpoint scaling if available.
        data["target"] = data["DMS_score"]
        if "DMS_score_bin" in data.columns:
            try:
                midpoint_scale_whole_df(  # noqa: F821  (assumed available in your codebase)
                    data,
                    target_col="target",
                    bin_col="DMS_score_bin",
                    out_col="target_norm",
                )
                data["target"] = data["target_norm"]
                data.drop(columns=["target_norm"], inplace=True, errors="ignore")
            except NameError:
                # If the normalization helper is not present, continue without it.
                pass

    # --------------------
    # Parse mutations row-by-row
    # --------------------
    per_row_positions: List[List[int]] = []
    per_row_wt_aas:   List[List[str]] = []
    per_row_mut_aas:  List[List[str]] = []
    mutated_sequences: List[str] = []

    for row_index, row in data.iterrows():
        wt_sequence: str = str(row.get("wt_sequence", "") or "")
        raw_mutant_spec: str = str(row.get("mutant", "") or "")

        # Build a dict pos_1based -> (wt_aa, mut_aa); last token for a position wins
        pos_to_pair: Dict[int, Tuple[str, str]] = {}
        for token in [t.strip() for t in raw_mutant_spec.split(":") if str(t).strip()]:
            parsed = parse_mutation(token)
            if parsed is None:
                continue
            wt_aa, pos_1based, mut_aa = parsed
            # Validate WT consistency (1-based position)
            if pos_1based < 1 or pos_1based > len(wt_sequence):
                raise ValueError(
                    f"Row {row_index}: position {pos_1based} out of bounds for sequence length {len(wt_sequence)}"
                )
            wt_at_pos = wt_sequence[pos_1based - 1]
            if wt_at_pos != wt_aa:
                raise AssertionError(
                    f"Row {row_index}: WT mismatch at position {pos_1based}: sequence has '{wt_at_pos}' vs token '{wt_aa}'"
                )
            if wt_aa == mut_aa:
                # Skip no-op edits
                continue
            pos_to_pair[pos_1based] = (wt_aa, mut_aa)

        if not pos_to_pair:
            # No valid edits: mutated_sequence is WT
            per_row_positions.append([])
            per_row_wt_aas.append([])
            per_row_mut_aas.append([])
            mutated_sequences.append(wt_sequence)
            continue

        sorted_positions = sorted(pos_to_pair.keys())
        wt_aas = [pos_to_pair[p][0] for p in sorted_positions]
        mut_aas = [pos_to_pair[p][1] for p in sorted_positions]

        mutated_seq = apply_mutations_to_sequence(wt_sequence, sorted_positions, mut_aas)

        # Optional consistency check with provided mutated_sequence
        given_mutated_seq = row.get("mutated_sequence")
        if isinstance(given_mutated_seq, str) and len(given_mutated_seq) == len(mutated_seq):
            if given_mutated_seq != mutated_seq:
                raise AssertionError(f"Row {row_index}: provided mutated_sequence does not match parsed mutations")

        per_row_positions.append(sorted_positions)
        per_row_wt_aas.append(wt_aas)
        per_row_mut_aas.append(mut_aas)
        mutated_sequences.append(mutated_seq)

    data["mutated_sequence"] = mutated_sequences

    # --------------------
    # Build masked sequences for tokenization
    # --------------------
    masked_strings: List[str] = []
    for positions_1based, row in zip(per_row_positions, data.itertuples(index=False)):
        wt_sequence: str = getattr(row, "wt_sequence")
        masked_strings.append(make_masked_sequence(wt_sequence, positions_1based))

    encodings = tokenizer.batch_encode_plus(
        masked_strings,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids: torch.Tensor = encodings["input_ids"]           # [N, L]
    attention_mask: torch.Tensor = encodings["attention_mask"] # [N, L]
    num_items, seq_len = input_ids.shape

    # --------------------
    # Find token positions of <mask> in token space
    # --------------------
    mask_token_id: int = tokenizer.mask_token_id
    token_mask_positions: List[List[int]] = []
    for item_idx in range(num_items):
        ids_row = input_ids[item_idx]
        positions = (ids_row == mask_token_id).nonzero(as_tuple=False).squeeze(1).tolist()
        token_mask_positions.append(positions)

    max_sites = max((len(lst) for lst in token_mask_positions), default=0)
    if max_sites == 0:
        raise ValueError("No <mask> tokens found; parsing produced zero mutations.")

    # --------------------
    # Build padded site tensors
    # --------------------
    pos_idx   = torch.full((num_items, max_sites), -1, dtype=torch.long)
    wt_index  = torch.zeros((num_items, max_sites), dtype=torch.long)
    mut_index = torch.zeros((num_items, max_sites), dtype=torch.long)
    mut_mask  = torch.zeros((num_items, max_sites), dtype=torch.bool)

    for item_idx in range(num_items):
        site_count = len(token_mask_positions[item_idx])
        if site_count == 0:
            continue

        # token-space indices of <mask> (already account for BOS/EOS via tokenizer)
        pos_idx[item_idx, :site_count] = torch.tensor(token_mask_positions[item_idx], dtype=torch.long)

        # amino-acid token ids for WT and MUT at each site
        wt_token_ids = [to_token_id(aa) for aa in per_row_wt_aas[item_idx]]
        mut_token_ids = [to_token_id(aa) for aa in per_row_mut_aas[item_idx]]
        wt_index[item_idx, :site_count] = torch.tensor(wt_token_ids, dtype=torch.long)
        mut_index[item_idx, :site_count] = torch.tensor(mut_token_ids, dtype=torch.long)

        mut_mask[item_idx, :site_count] = True

    # --------------------
    # Assemble output
    # --------------------
    output: Dict[str, torch.Tensor] = {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "pos_idx":        pos_idx,      # token indices of the <mask> tokens
        "wt_index":       wt_index,
        "mut_index":      mut_index,
        "mut_mask":       mut_mask,
    }

    if "target" in data.columns:
        output["target"] = torch.tensor(data["target"].values, dtype=torch.float32)

    return output

# === Target normalization ===
def midpoint_scale_whole_df(df, target_col="target", bin_col="DMS_score_bin", out_col="target_norm"):
    bad  = df.loc[df[bin_col] == 0, target_col]
    good = df.loc[df[bin_col] == 1, target_col]

    # midpoint between best-bad and worst-good
    if bad.empty or good.empty:
        c = df[target_col].median()
    else:
        c = 0.5 * (bad.max() + good.min())

    x = df[target_col].to_numpy()

    # robust spreads to avoid outliers & div-by-zero
    s_neg = (c - bad.quantile(0.05)) if not bad.empty else (c - df[target_col].min())
    s_pos = (good.quantile(0.95) - c) if not good.empty else (df[target_col].max() - c)
    s_neg = float(max(s_neg, 1e-8))
    s_pos = float(max(s_pos, 1e-8))

    y = np.empty_like(x, dtype=float)
    left  = x <= c
    right = ~left
    y[left]  = -(c - x[left])  / s_neg
    y[right] =  (x[right] - c) / s_pos

    df[out_col] = np.clip(y, -1.0, 1.0)
    # return c, s_neg, s_pos  # return params if you want to log them

# === Random masking augmentation ===
def random_mask_batch(input_ids, mask_token_id, p=0.15):
    """
    Randomly mask a batch of input_ids with probability p.
    input_ids: [B, L] tensor of token ids
    mask_token_id: id of the mask token
    p: probability of masking each token
    """
    rand = torch.rand(input_ids.shape, device=input_ids.device)
    mask_arr = (rand < p) & (input_ids != mask_token_id) & (input_ids != 0)  # do not mask padding or existing masks
    input_ids_masked = input_ids.masked_fill(mask_arr, mask_token_id)
    return input_ids_masked

# === Dataset ===
class ProteinDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data["input_ids"].size(0)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

# === Loss & training step ===
import torch
import torch.nn.functional as F


def batch_step(
    model,
    batch,
    device,
    margin: float = 1.0,   # desired min( mut_logp - wt_logp )
    alpha: float  = 0.0,   # KL weight
    beta: float   = 0.01,  # entropy bonus (subtracts from loss)
    gamma: float  = 0.0,   # regression weight
    model_base=None,
    tokenizer=None,        # needed only if using random masking
    rand_mask_p: float = 0.0,  # extra random token masking prob (never touches true mutant sites)
):
    """
    Multi-site version (K >= 1). Computes loss from the average Δ across sites per sequence.

    Batch must contain (from `prep_data`):
      input_ids : [B, L] long
      pos_idx   : [B, K] long  (token indices of <mask> per site)
      wt_index  : [B, K] long  (token IDs of WT residues at sites)
      mut_index : [B, K] long  (token IDs of MUT residues at sites)
      mut_mask  : [B, K] bool  (True where a site exists in this row)
      target    : [B]    float (signed; sign used as ranking label; optional for unlabeled batches)
    """

    # ----- Move to device -----
    token_ids        = batch["input_ids"].to(device, non_blocking=True)            # [B, L]
    mask_positions   = batch["pos_idx"].to(device, non_blocking=True).long()       # [B, K]
    wt_token_ids     = batch["wt_index"].to(device, non_blocking=True).long()      # [B, K]
    mut_token_ids    = batch["mut_index"].to(device, non_blocking=True).long()     # [B, K]
    site_exists_mask = batch["mut_mask"].to(device, non_blocking=True).bool()      # [B, K]
    targets = batch.get("target", None)
    if targets is not None:
        targets = batch["target"].to(device, non_blocking=True).to(torch.float32)  # [B]

    batch_size, seq_len = token_ids.shape
    _, num_sites = mask_positions.shape

    # ----- Optional extra random masking (never touch true <mask> sites) -----
    if tokenizer is not None and rand_mask_p and rand_mask_p > 0.0:
        mask_token_id = int(tokenizer.mask_token_id)
        pad_token_id  = int(getattr(tokenizer, "pad_token_id", 0))

        protected = torch.zeros_like(token_ids, dtype=torch.bool)                   # [B, L]
        row_ix = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, num_sites)
        valid_sites = site_exists_mask & mask_positions.ge(0)
        if valid_sites.any():
            protected[row_ix[valid_sites], mask_positions[valid_sites]] = True

        rnd = torch.rand_like(token_ids, dtype=torch.float32)
        random_mask = (
            (rnd < float(rand_mask_p))
            & (~protected)
            & (token_ids != mask_token_id)
            & (token_ids != pad_token_id)
        )
        token_ids = token_ids.masked_fill(random_mask, mask_token_id)

    # ----- Forward pass -----
    model.train()
    logits   = model(token_ids).sequence_logits                                      # [B, L, V]
    log_probs= F.log_softmax(logits, dim=-1)                                         # [B, L, V]
    vocab_sz = logits.size(-1)

    # ----- Select distributions at true <mask> positions -----
    safe_positions = mask_positions.clamp_min(0)                                     # [B, K]
    row_ix = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, num_sites)
    site_log_probs = log_probs[row_ix, safe_positions]                                # [B, K, V]

    # Guard WT/MUT indices for masked-out sites (gather-safe)
    wt_ids_safe  = torch.where(site_exists_mask, wt_token_ids,  torch.zeros_like(wt_token_ids))
    mut_ids_safe = torch.where(site_exists_mask, mut_token_ids, torch.zeros_like(mut_token_ids))

    wt_logp  = site_log_probs.gather(-1, wt_ids_safe.unsqueeze(-1)).squeeze(-1)      # [B, K]
    mut_logp = site_log_probs.gather(-1, mut_ids_safe.unsqueeze(-1)).squeeze(-1)     # [B, K]

    # ----- Per-sequence delta: average over available sites -----
    sites_per_row = site_exists_mask.sum(dim=1).clamp_min(1).to(logits.dtype)        # [B]
    per_site_delta = (mut_logp - wt_logp) * site_exists_mask.to(logits.dtype)        # [B, K]
    delta_per_row  = per_site_delta.sum(dim=1) / sites_per_row                       # [B]

    # ----- Unlabeled: entropy-only regularization (optional) -----
    if targets is None:
        site_probs = site_log_probs.exp()                                            # [B, K, V]
        per_site_entropy = -(site_probs * site_log_probs).sum(dim=-1)                # [B, K]
        mean_entropy = (per_site_entropy * site_exists_mask).sum(dim=1) / sites_per_row
        return -(beta * mean_entropy.mean())

    # ----- Labeled losses -----
    # Ranking: prefer delta_per_row to exceed 0 by `margin` with label in {-1, +1}
    labels = torch.sign(targets).to(torch.float32)                                    # [B]
    valid_rows = labels.ne(0)
    if valid_rows.any():
        # margin_ranking_loss expects y ∈ {-1,+1}, compares input1 vs input2
        rank_loss = F.margin_ranking_loss(
            input1=delta_per_row[valid_rows],
            input2=torch.zeros_like(delta_per_row[valid_rows]),
            target=labels[valid_rows],
            margin=float(margin),
            reduction="mean",
        )
    else:
        rank_loss = (delta_per_row.sum() * 0.0)

    # Regression: calibrate delta_per_row to target
    reg_loss = F.mse_loss(delta_per_row, targets)

    # KL to base model (optional)
    if model_base is not None and alpha > 0.0:
        with torch.no_grad():
            model_base.eval()
            base_logits = model_base(token_ids).sequence_logits
            base_probs  = F.softmax(base_logits, dim=-1)
        kl_loss = F.kl_div(log_probs, base_probs, reduction="batchmean")
    else:
        kl_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    # Entropy bonus at the masked sites (encourages confident distributions)
    site_probs = site_log_probs.exp()
    per_site_entropy = -(site_probs * site_log_probs).sum(dim=-1)                    # [B, K]
    mean_entropy = (per_site_entropy * site_exists_mask).sum(dim=1) / sites_per_row  # [B]
    entropy_bonus = mean_entropy.mean()                                              # scalar

    # ----- Total loss -----
    total_loss = rank_loss + (gamma * reg_loss) + (alpha * kl_loss) - (beta * entropy_bonus)
    return total_loss

# def batch_step(
#     model,
#     batch,
#     device,
#     margin: float = 0.0,
#     alpha: float = 0.0,          # KL weight
#     beta: float = 0.01,          # entropy bonus weight
#     gamma: float = 0.1,          # regression weight
#     model_base=None,
#     tokenizer=None,
#     rand_mask_p: float = 0.0,    # probability for extra random masking (excludes true mutant sites)
# ):
#     """
#     Compute loss for N-site mutation scoring.

#     Required batch keys from `prep_data`:
#       input_ids:  [B, L] long
#       pos_idx:    [B, K] long (token indices of <mask> tokens)
#       wt_index:   [B, K] long (token IDs for WT residues at each site)
#       mut_index:  [B, K] long (token IDs for MUT residues at each site)
#       mut_mask:   [B, K] bool (True where a site exists in this row)

#     Optional:
#       target:     [B] float (signed regression target; sign used for ranking)

#     Loss:
#       total = rank_loss + gamma * mse(delta_row, target) + alpha * KL(logp || base)
#               - beta * mean_entropy_at_sites
#     """

#     # ----- Move batch to device -----
#     token_ids         = batch["input_ids"].to(device, non_blocking=True)          # [B, L]
#     per_site_mask     = batch["mut_mask"].to(device, non_blocking=True).bool()    # [B, K]
#     mask_positions    = batch["pos_idx"].to(device, non_blocking=True).long()     # [B, K]
#     wt_token_indices  = batch["wt_index"].to(device, non_blocking=True).long()    # [B, K]
#     mut_token_indices = batch["mut_index"].to(device, non_blocking=True).long()   # [B, K]
#     targets           = batch.get("target", None)
#     if targets is not None:
#         targets = batch["target"].to(device, non_blocking=True).to(torch.float32) # [B]

#     batch_size, seq_len = token_ids.shape
#     _, max_sites = mask_positions.shape

#     # ----- Optional extra random masking (never touch the true mutant sites) -----
#     if tokenizer is not None and rand_mask_p and rand_mask_p > 0.0:
#         mask_token_id = int(tokenizer.mask_token_id)
#         pad_token_id  = int(getattr(tokenizer, "pad_token_id", 0))

#         # Build a protection map for real mutant-site masks
#         protected_positions = torch.zeros_like(token_ids, dtype=torch.bool)       # [B, L]
#         row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, max_sites)  # [B, K]
#         valid_sites = per_site_mask & mask_positions.ge(0)                        # [B, K]
#         if valid_sites.any():
#             protected_positions[row_indices[valid_sites], mask_positions[valid_sites]] = True

#         # Sample candidate tokens to mask, excluding protected + PAD + existing MASK
#         random_draw = torch.rand_like(token_ids, dtype=torch.float32)
#         candidate_mask = (
#             (random_draw < float(rand_mask_p))
#             & (~protected_positions)
#             & (token_ids != mask_token_id)
#             & (token_ids != pad_token_id)
#         )
#         token_ids = token_ids.masked_fill(candidate_mask, mask_token_id)

#     # ----- Forward pass -----
#     logits = model(token_ids).sequence_logits                                      # [B, L, V]
#     log_probs = F.log_softmax(logits, dim=-1)                                      # [B, L, V]
#     vocab_size = logits.size(-1)

#     # ----- Gather distributions at intended <mask> positions -----
#     # Clamp indices for gather; invalid sites will be zeroed out by `per_site_mask` later.
#     safe_positions = mask_positions.clamp_min(0)                                   # [B, K]
#     batch_index_matrix = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, max_sites)  # [B, K]
#     site_distributions = log_probs[batch_index_matrix, safe_positions]             # [B, K, V]

#     # Clamp WT/MUT token indices where site does not exist (prevent accidental gather oob in debug)
#     wt_indices_safe  = torch.where(per_site_mask, wt_token_indices,  torch.zeros_like(wt_token_indices))
#     mut_indices_safe = torch.where(per_site_mask, mut_token_indices, torch.zeros_like(mut_token_indices))

#     wt_logp  = site_distributions.gather(-1, wt_indices_safe.unsqueeze(-1)).squeeze(-1)   # [B, K]
#     mut_logp = site_distributions.gather(-1, mut_indices_safe.unsqueeze(-1)).squeeze(-1)  # [B, K]

#     # ----- Per-row delta: masked mean across available sites -----
#     sites_per_row = per_site_mask.sum(dim=1).clamp_min(1).to(logits.dtype)         # [B]
#     per_site_delta = (mut_logp - wt_logp) * per_site_mask.to(logits.dtype)         # [B, K]
#     delta_per_row = per_site_delta.sum(dim=1) / sites_per_row                      # [B]

#     # ----- Unlabeled branch: entropy-only regularization -----
#     if targets is None:
#         site_probs = site_distributions.exp()                                      # [B, K, V]
#         per_site_entropy = -(site_probs * site_distributions).sum(dim=-1)          # [B, K]
#         mean_entropy_per_row = (per_site_entropy * per_site_mask).sum(dim=1) / sites_per_row
#         return -(beta * mean_entropy_per_row.mean())

#     # ----- Labeled branch -----
#     # Ranking targets in {-1, 0, +1}; ignore zeros
#     labels = torch.sign(targets).to(torch.float32)                                  # [B]
#     row_is_valid = labels.ne(0)                                                     # [B]
#     if not row_is_valid.any():
#         # No labeled items; return zero on correct device/dtype
#         return (delta_per_row.sum() * 0.0) + (logits.sum() * 0.0)

#     # Soft hinge ranking loss with optional margin
#     margin_tensor = torch.as_tensor(margin, device=device, dtype=torch.float32)
#     rank_loss = F.softplus(-labels[row_is_valid] * (delta_per_row[row_is_valid] - margin_tensor)).mean()

#     # Regression loss (all rows)
#     regression_loss = F.mse_loss(delta_per_row[row_is_valid], targets[row_is_valid])

#     # KL to base model (optional)
#     if model_base is not None and alpha > 0.0:
#         with torch.no_grad():
#             model_base.eval()
#             base_logits = model_base(token_ids).sequence_logits                    # [B, L, V]
#             base_probs  = F.softmax(base_logits, dim=-1)
#         kl_loss = F.kl_div(log_probs, base_probs, reduction="batchmean")
#     else:
#         kl_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

#     # Entropy bonus at mutant sites (encourage sharper distributions)
#     site_probs = site_distributions.exp()                                          # [B, K, V]
#     per_site_entropy = -(site_probs * site_distributions).sum(dim=-1)              # [B, K]
#     mean_entropy_per_row = (per_site_entropy * per_site_mask).sum(dim=1) / sites_per_row
#     entropy_bonus = mean_entropy_per_row.mean()

#     # ----- Total loss -----
#     total_loss = rank_loss + (gamma * regression_loss) + (alpha * kl_loss) - (beta * entropy_bonus)
#     return total_loss

# === Training loop ===
def train(model, loader, optimizer, device, epochs, alpha, beta, gamma, seed: int | None = None, tokenizer=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            loss = batch_step(model, batch, device, alpha=alpha, beta=beta, gamma=gamma, tokenizer=tokenizer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# === Evaluation ===
@torch.no_grad()
def predict_deltas(model, tokenizer, df, device, batch_size=32):
    """
    Returns four [N] arrays:
      deltas_mean, entropy_mean, mut_logp_mean, wt_logp_mean
    """
    data  = prep_data(df, tokenizer, tokenizer.get_vocab())  # prep needs vocab for wt/mut ids
    ds    = ProteinDataset(data)
    loader= DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    deltas_out, ents_out, mutlp_out, wtlp_out = [], [], [], []

    with torch.inference_mode():
        for batch in loader:
            tokens   = batch["input_ids"].to(device)     # [B,L]
            pos_idx  = batch["pos_idx"].to(device)        # [B,K]
            wt_index = batch["wt_index"].to(device)       # [B,K]
            mut_index= batch["mut_index"].to(device)      # [B,K]
            site_mask= batch["mut_mask"].to(device).bool()# [B,K]

            logits = model(tokens).sequence_logits        # [B,L,V]
            logp   = F.log_softmax(logits, dim=-1)

            B, K   = pos_idx.shape
            L, V   = tokens.size(1), logits.size(-1)
            row_ix = torch.arange(B, device=device).unsqueeze(1).expand(B, K)

            # --- Hard sanity checks (fail fast if anything is misaligned) ---
            mask_token_id = int(tokenizer.mask_token_id)
            # every valid site must be a <mask> in tokens
            assert ((tokens[row_ix[site_mask], pos_idx[site_mask]] == mask_token_id).all()), "pos_idx not pointing at <mask>"
            # pos must be in-range wherever site_mask is True
            assert (((pos_idx[site_mask] >= 0) & (pos_idx[site_mask] < L)).all()), "pos_idx out of bounds"
            # wt/mut ids must be in vocab range
            if site_mask.any():
                wv = wt_index[site_mask]; mv = mut_index[site_mask]
                assert (((wv >= 0) & (wv < V)).all()), "wt_index out of range"
                assert (((mv >= 0) & (mv < V)).all()), "mut_index out of range"

            # --- Safe indexing: never index with invalid positions ---
            safe_pos = torch.where(site_mask, pos_idx, torch.zeros_like(pos_idx))
            per_site_logp = logp[row_ix, safe_pos]               # [B,K,V]

            # Clamp wt/mut ids only where masked-out; gather is safe now
            wt_safe  = torch.where(site_mask, wt_index,  torch.zeros_like(wt_index))
            mut_safe = torch.where(site_mask, mut_index, torch.zeros_like(mut_index))

            wt_logp  = per_site_logp.gather(-1, wt_safe.unsqueeze(-1)).squeeze(-1)   # [B,K]
            mut_logp = per_site_logp.gather(-1, mut_safe.unsqueeze(-1)).squeeze(-1)  # [B,K]

            per_site_delta = (mut_logp - wt_logp) * site_mask.to(logits.dtype)       # [B,K]
            per_site_prob  = per_site_logp.exp()
            per_site_entropy = -(per_site_prob * per_site_logp).sum(dim=-1)          # [B,K]

            sites_per_row = site_mask.sum(dim=1).clamp_min(1).to(logits.dtype)       # [B]
            delta_mean = per_site_delta.sum(dim=1) / sites_per_row                   # [B]
            entropy_mean = (per_site_entropy * site_mask).sum(dim=1) / sites_per_row # [B]
            mut_mean = (mut_logp * site_mask).sum(dim=1) / sites_per_row             # [B]
            wt_mean  = (wt_logp  * site_mask).sum(dim=1) / sites_per_row             # [B]

            deltas_out.append(delta_mean.cpu())
            ents_out.append(entropy_mean.cpu())
            mutlp_out.append(mut_mean.cpu())
            wtlp_out.append(wt_mean.cpu())

    to_np32 = lambda t: torch.cat(t).contiguous().to(torch.float32).numpy()
    return to_np32(deltas_out), to_np32(ents_out), to_np32(mutlp_out), to_np32(wtlp_out)


# === Full evaluation on a candidate dataframe ===
def evaluate_candidate_df(model, tokenizer, df, device, margin=0.1):
    # Requires df['target']
    deltas, _, _, _ = predict_deltas(model, tokenizer, df, device)
    t = df["target"].to_numpy()
    labels = np.sign(t)
    valid = labels != 0

    margin_loss = float(np.mean(np.maximum(0.0, margin - labels[valid]*deltas[valid]))) if valid.any() else 0.0
    corr = spearmanr(deltas, t).correlation if np.unique(t).size > 1 else np.nan

    return deltas, margin_loss, corr

# === Helper functions ===
def seq_diff_to_mutants(wt: str, mut: str) -> str:
    """
    Compare two equal-length sequences and return mutations like 'A114C:G215V'.
    Uses 1-based indexing, no zero padding.
    """
    if wt == mut:
        return "WT_sequence"

    if len(wt) != len(mut):
        raise ValueError(f"Sequences must be same length (got {len(wt)} vs {len(mut)})")
    muts = []
    for i, (a, b) in enumerate(zip(wt, mut), start=1):  # 1-based
        if a != b:
            muts.append(f"{a}{i}{b}")
    return ":".join(muts)

@torch.no_grad()
def _extract_mean_embeddings(df_seq: pd.DataFrame, tokenizer, encoder, device, bs=256):
    # build mut_sequence from mutant and wt_sequence columns if not present
    if "mut_sequence" not in df_seq.columns or df_seq["mut_sequence"].isnull().all():
        df_seq["mut_sequence"] = None
        for i, row in df_seq.iterrows():
            seq = row.get("wt_sequence", "")
            raw = row.get("mutant", "")
            parts = [p.strip() for p in str(raw).split(":") if p and str(p).strip()]
            by_pos = {}
            for p in parts:
                wt, pos, mu = p[0], int(p[1:-1]), p[-1]
                if wt == mu:
                    continue
                by_pos[pos] = (wt, mu)  # keep last per position
            if not by_pos:
                df_seq.at[i, 'mut_sequence'] = seq
                continue
            pos_sorted = sorted(by_pos.keys())
            mu_list = [by_pos[p][1] for p in pos_sorted]
            s = list(seq)
            for p, mu in zip(pos_sorted, mu_list):
                s[p-1] = mu
            ms = "".join(s)
            df_seq.at[i, 'mut_sequence'] = ms
    seqs = df_seq["mut_sequence"].tolist()
    feats, i = [], 0
    encoder.eval().to(device)
    # if device.type == "cuda":
        # encoder = encoder.float()
    while i < len(seqs):
        b = seqs[i:i+bs]
        enc = tokenizer.batch_encode_plus(
            b,
            padding="longest",
            truncation=True,
            max_length=2046,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = encoder.forward(enc["input_ids"], enc["attention_mask"])
        feats.append(out.embeddings.float().mean(dim=1).cpu().numpy())
        i += bs
    return np.concatenate(feats, axis=0)

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",     required=True)
    parser.add_argument("--multi_mutant",  default=None)
    parser.add_argument("--protein",       required=True)
    parser.add_argument("--batch_size",    type=int, default=8)
    parser.add_argument("--epochs",        type=int, default=10)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--model",         type=str, default="esmc_300m")
    parser.add_argument("--gpu",           type=int, default=0)
    parser.add_argument("--output_dir",    type=str, required=True)
    parser.add_argument("--metrics_file",  type=str, default="metrics.txt")
    parser.add_argument("--alpha",         type=float, default=0)
    parser.add_argument("--beta",          type=float, default=0.01)
    parser.add_argument("--gamma",         type=float, default=0)
    parser.add_argument("--delta",         type=float, default=0.1)
    parser.add_argument("--topk",          type=int, default=16)
    parser.add_argument("--seed",          type=int, default=None)
    parser.add_argument("--zeroshot",      action="store_true")
    parser.add_argument("--rf_pred",       action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Set up device, dataset, and model ---
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    all_df = pd.read_csv(args.data_file)

    # if just 'target', or 'DMS_score' and 'DMS_score_bin' are  missing, end run
    if "target" not in all_df.columns and not ("DMS_score" in all_df.columns and "DMS_score_bin" in all_df.columns):
        print(f"No target or DMS_score/DMS_score_bin columns found, skipping {args.protein}.")
        exit(0)

    if 'mutated_sequence' not in all_df.columns or all_df['mutated_sequence'].isnull().all():
        print(f"No mutated_sequence column found or all values are null, generating mutated_sequence.")
        all_df['mutated_sequence'] = None
        
        # generate mutated_sequence from first entry
        all_df['mutated_sequence'] = None
        for i, row in all_df.iterrows():
            seq = row.get("wt_sequence", "")
            raw = row.get("mutant", "")
            parts = [p.strip() for p in str(raw).split(":") if p and str(p).strip()]
            by_pos = {}
            for p in parts:
                wt, pos, mu = p[0], int(p[1:-1]), p[-1]
                if wt == mu:
                    continue
                by_pos[pos] = (wt, mu)  # keep last per position
            if not by_pos:
                all_df.at[i, 'mutated_sequence'] = seq
                continue
            pos_sorted = sorted(by_pos.keys())
            mu_list = [by_pos[p][1] for p in pos_sorted]
            s = list(seq)
            for p, mu in zip(pos_sorted, mu_list):
                s[p-1] = mu
            ms = "".join(s)
            all_df.at[i, 'mutated_sequence'] = ms
    
    # if seq is longer than 2046, end run
    if len(seq) > 2046:
        print(f"Sequence length {len(seq)} exceeds 2046, skipping {args.protein}.")
        exit(0)
    
    # if no 'target' column, but DMS_score and DMS_score_bin exist, use them
    if "target" not in all_df.columns and "DMS_score" in all_df.columns and "DMS_score_bin" in all_df.columns:
        all_df["target"] = all_df["DMS_score"]
        midpoint_scale_whole_df(all_df, target_col="target",
                                bin_col="DMS_score_bin",
                                out_col="target_norm")
        all_df["target"] = all_df["target_norm"]
        all_df.drop(columns=["target_norm"], inplace=True, errors="ignore")

    # === MODEL LOADING ===
    model, tokenizer = load_model(args.model)
    # cast model to float
    if device.type == "cuda":
        model = model.to(device).float()
    base_state = deepcopy(model.state_dict())
    vocab = tokenizer.get_vocab()

    # === Few-shot iterative training ===
    # make position column
    all_df['position'] = all_df['mutant'].str[1:-1].astype(int)
    # for unique positions in 'position' column
    unique_positions = all_df['position'].unique().tolist()
    unique_positions.sort()
    print(f"Unique mutation positions in data: {unique_positions}")

    def training_part(model, base_state, all_df, tokenizer, vocab, device, args, position=None):    
        train_df = all_df.copy()

        model.load_state_dict(base_state)
        model.to(device)

        data_pre = prep_data(train_df, tokenizer, vocab)  # requires 'target' present in train_df
        ds = ProteinDataset(data_pre)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # freeze all
        for p in model.parameters():
            p.requires_grad = False
        # unfreeze last two blocks
        for block in model.transformer.blocks[-2:]:
            for p in block.parameters():
                p.requires_grad = True
        # unfreeze sequence head
        for p in model.sequence_head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # train — pass tokenizer to protect true mask sites
        if args.zeroshot == False:
            train(model, loader, optimizer, device, args.epochs, args.alpha, args.beta, args.gamma,
                seed=(None if args.seed is None else args.seed),
                tokenizer=tokenizer)
        
        model.eval()

        # =======================================
        # === Evaluate on candidate dataframe ===
        # =======================================

        for seq_num, seq in [(0, all_df['wt_sequence'][0])]:
            print(f"Evaluating sequence {seq_num+1}/{all_df['wt_sequence'].nunique()}")
            candidate_deltas = predict_all_site_deltas(model=model, tokenizer=tokenizer, wt_seq=seq, device=device, batch_size=64, position=position)

            # build candidate dataframe with one A1R style mutation per row
            candidate_rows = []
            if position is not None:
                wt = seq[position-1]
                for mu in AA20:
                    if mu == wt:
                        continue
                    mut_str = f"{wt}{position}{mu}"
                    candidate_rows.append({
                        "mutant": mut_str,
                        "delta_logp": candidate_deltas["delta20"][0][AA20.index(mu)],
                        "entropy": candidate_deltas["entropy"],
                        "score": candidate_deltas["delta20"][0][AA20.index(mu)] + args.delta * candidate_deltas["entropy"],
                        "mutated_sequence": seq[:position-1] + mu + seq[position:],
                        "wt_sequence": seq
                    })
            else:
                L = len(seq)
                for i in range(L):
                    wt = seq[i]
                    for mu in AA20:
                        if mu == wt:
                            continue
                        mut_str = f"{wt}{i+1}{mu}"
                        candidate_rows.append({
                            "mutant": mut_str,
                            "delta_logp": candidate_deltas["delta20"][i][AA20.index(mu)],
                            "entropy": candidate_deltas["entropy"][i],
                            "score": candidate_deltas["delta20"][i][AA20.index(mu)] + args.delta * candidate_deltas["entropy"][i],
                            "mutated_sequence": seq[:i] + mu + seq[i+1:],
                            "wt_sequence": seq
                        })
            return candidate_rows

    # set selected to accrue selected positions        
    selected = []

    for position in [-1] + unique_positions:
        # for -1, predict on all positions w/o data
        if position == -1:
            print("Predicting on all positions without training data")
            # train on all data and evaluate on all positions not in unique_positions
            train_df = all_df.copy()
            candidate_rows = training_part(model, base_state, train_df, tokenizer, vocab, device, args)
            # drop candidate rows if mutant is in unique_positions
            candidate_rows = [row for row in candidate_rows if int(row["mutant"][1:-1]) not in unique_positions]
            selected.extend(candidate_rows)
            print(f"Now evaluating {len(unique_positions)} individual positions, with training data removed individually.")

        # for list of positions, remove each position from training data and predict on only that position
        elif isinstance(position, int):
            # for each position, train on all data but that position
            mask = all_df['position'].isin([position])
            # remove rows with these positions
            train_df = all_df[~mask].reset_index(drop=True)
            candidate_rows = training_part(model, base_state, train_df, tokenizer, vocab, device, args, position=position)
            # keep only candidate rows with mutant at this position
            selected.extend([row for row in candidate_rows if int(row["mutant"][1:-1]) == position])


    candidate_df = pd.DataFrame(selected)
    candidate_df = candidate_df.sort_values("score", ascending=False).reset_index(drop=True)

    # This removes predictions that are already in all_df 
    # We actually want to keep these to see if the model predicted the already measured mutation higher than the others
    # candidate_df = candidate_df[~candidate_df['mutated_sequence'].isin(all_df['mutated_sequence'])]

    # min-max scale score to [0,1]
    score_min = candidate_df["score"].min()
    score_max = candidate_df["score"].max()
    candidate_df["score"] = (candidate_df["score"] - score_min) / (score_max - score_min + 1e-8)

    # for unique sequence in wt_sequence
    original_wt = all_df['wt_sequence'].iloc[0]

    sequence_name = seq_diff_to_mutants(original_wt, seq)
    candidate_df.insert(0, "sequence_name", sequence_name)

    # ================================
    # === Random Forest Prediction ===
    # ================================

    # 2) **Reload a clean ESMC**
    clean_model, clean_tokenizer = load_model(args.model)
    clean_model.to(device).eval()
    if device.type == "cuda":
        clean_model = clean_model.float()

    # 3) Embed (no saving)
    X_train = _extract_mean_embeddings(all_df, clean_tokenizer, clean_model, device, bs=args.batch_size)
    y_train = all_df["target"].to_numpy()
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min() + 1e-8)  # min-max scale to [0,1]


    # create all singles mutant_df
    multi_df = candidate_df.copy()
    multi_df = multi_df.rename(columns={"mutated_sequence": "sequence"})
    X_pred  = _extract_mean_embeddings(multi_df, clean_tokenizer, clean_model, device, bs=args.batch_size)

    # 4) Fit RF and predict
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(
            n_estimators=100,
            criterion="friedman_mse",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=1,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
        )
    
    rf.fit(X_train, y_train)

    multi_df_out = multi_df.copy()
    multi_df_out["rf_pred"] = rf.predict(X_pred)

    # now, add rf_pred to candidate_df
    candidate_df = candidate_df.merge(multi_df_out[["mutant", "rf_pred"]], on="mutant", how="left")


    # 0-1 scale rf_pred
    rf_min = candidate_df["rf_pred"].min()
    rf_max = candidate_df["rf_pred"].max()
    candidate_df["rf_pred"] = (candidate_df["rf_pred"] - rf_min) / (rf_max - rf_min + 1e-8)

    # average the two scores
    candidate_df["final_score"] = (candidate_df["score"] + candidate_df["rf_pred"]) / 2
    candidate_df = candidate_df[['mutant', 'score', 'rf_pred', 'final_score', 'mutated_sequence', 'wt_sequence']]
    

    # save with different names
    out_path = os.path.join(args.output_dir, f"{args.protein}_allMutations_results_withRF.csv")
    candidate_df.to_csv(out_path, index=False)

    # create best_per_site selecting the top mutant per position based on "score"
    best_per_site = candidate_df.loc[candidate_df.groupby(candidate_df['mutant'].str[1:-1].astype(int))['final_score'].idxmax()].reset_index(drop=True)
    best_out_path = os.path.join(args.output_dir, f"{args.protein}_perSite_results_withRF.csv")
    best_per_site.to_csv(best_out_path, index=False)

    # else:
    #     # Save
    #     out_path = os.path.join(args.output_dir, f"{args.protein}_allSingles_results.csv")
    #     candidate_df.to_csv(out_path, index=False)
        
    #     # create best_per_site selecting the top mutant per position based on "score"
    #     best_per_site = candidate_df.loc[candidate_df.groupby(candidate_df['mutant'].str[1:-1].astype(int))['score'].idxmax()].reset_index(drop=True)
    #     best_per_site = best_per_site.merge(candidate_df[["mutant", "score"]], on="mutant", how="left")
        
    #     best_out_path = os.path.join(args.output_dir, f"{args.protein}_perSite_results.csv")
    #     best_per_site.to_csv(best_out_path, index=False)