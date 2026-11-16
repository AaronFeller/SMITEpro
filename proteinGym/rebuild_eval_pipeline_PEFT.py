#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
start_time = time.time()

import datetime
import argparse
from copy import deepcopy
from collections import deque

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from esm.models.esmc import ESMC
from peft import LoraConfig, get_peft_model

# =========================
# Data preparation helpers
# =========================
def midpoint_scale_whole_df(df, target_col="target", bin_col="DMS_score_bin", out_col="target_norm"):
    bad  = df.loc[df[bin_col] == 0, target_col]
    good = df.loc[df[bin_col] == 1, target_col]
    if bad.empty or good.empty:
        c = df[target_col].median()
    else:
        c = 0.5 * (bad.max() + good.min())

    x = df[target_col].to_numpy()
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


def _parse_mutant_string(mutant_str: str):
    """Parse 'A124L:G200V:Y305H' -> ([124,200,305], ['A','G','Y'], ['L','V','H'])."""
    pos_list, wt_list, mut_list = [], [], []
    for m in str(mutant_str).split(":"):
        if not m:
            continue
        wt, mut = m[0], m[-1]
        pos = int(m[1:-1])
        pos_list.append(pos)
        wt_list.append(wt)
        mut_list.append(mut)
    return pos_list, wt_list, mut_list


def _mask_seq_k(seq: str, positions_1b, mask_token: str, max_length: int):
    """Mask residues at 1-based positions in a sequence using mask_token."""
    if not positions_1b:
        return seq[:max_length]
    parts, prev = [], 0
    for p in sorted({int(p) for p in positions_1b}):
        p0 = p - 1
        parts.append(seq[prev:p0])
        parts.append(mask_token)
        prev = p0 + 1
    parts.append(seq[prev:])
    masked = "".join(parts)
    return masked if len(masked) <= max_length else masked[:max_length]


def prep_data_from_mutant(
    df: pd.DataFrame,
    tokenizer,
    sequence: str | None,
    vocab: dict,
    max_length: int = 2046,
    mutant_col: str = "mutant",
):
    """
    Prepare DMS data (singles or multis) for scoring.
    Returns:
      input_ids [N,L], attention_mask [N,L],
      pos_idx [N,Kmax], wt_index [N,Kmax], mut_index [N,Kmax],
      k_mask [N,Kmax], and optional target [N]
    """
    mask_token = tokenizer.mask_token

    if "wt_sequence" in df.columns:
        sequences = df["wt_sequence"].astype(str).tolist()
    elif sequence is not None:
        sequences = [sequence] * len(df)
    else:
        raise ValueError("prep_data_from_mutant requires either a shared sequence or a wt_sequence column.")

    triplets = [_parse_mutant_string(m) for m in df[mutant_col].astype(str)]
    N = len(triplets)
    Kmax = max((len(p) for p, _, _ in triplets), default=0)

    masked_strings = [
        _mask_seq_k(seq, pos, mask_token, max_length)
        for seq, (pos, _, _) in zip(sequences, triplets)
    ]

    enc = tokenizer.batch_encode_plus(
        masked_strings,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    pos_idx  = np.full((N, Kmax), -1, dtype=np.int64)
    wt_index = np.full((N, Kmax), -1, dtype=np.int64)
    mu_index = np.full((N, Kmax), -1, dtype=np.int64)
    k_mask   = np.zeros((N, Kmax), dtype=np.float32)

    for i, (pos_1b, wt_list, mu_list) in enumerate(triplets):
        k = len(pos_1b)
        if k == 0:
            continue
        pos0 = [p - 1 for p in pos_1b]
        wt_ids = [vocab[a] for a in wt_list]
        mu_ids = [vocab[a] for a in mu_list]
        pos_idx[i, :k]  = pos0
        wt_index[i, :k] = wt_ids
        mu_index[i, :k] = mu_ids
        k_mask[i, :k]   = 1.0

    out = {
        "input_ids":      enc["input_ids"].long(),
        "attention_mask": enc["attention_mask"].long(),
        "pos_idx":        torch.as_tensor(pos_idx,  dtype=torch.long),
        "wt_index":       torch.as_tensor(wt_index, dtype=torch.long),
        "mut_index":      torch.as_tensor(mu_index, dtype=torch.long),
        "k_mask":         torch.as_tensor(k_mask,   dtype=torch.float32),
    }
    if "target" in df.columns:
        out["target"] = torch.tensor(df["target"].values, dtype=torch.float32)
    return out


# ==========
# Dataset
# ==========
class ProteinDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data["input_ids"].size(0)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# =================
# Model loader
# =================
def load_model(model_name="esmc_300m"):
    model = ESMC.from_pretrained(model_name)
    return model, model.tokenizer


# ===========================
# Loss & training utilities
# ===========================
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

import torch
import torch.nn.functional as F

# def batch_step(model, batch, device, margin=0.1, beta=0.1):
#     """
#     K-aware training step.
#     Uses summed per-site Δ = Σ_k [logP(mut_k) - logP(wt_k)].
#     loss = margin-ranking(labels, Δ) - β·H
#     """
#     global tokenizer

#     batch['input_ids'] = random_mask_batch(
#         batch['input_ids'], tokenizer.mask_token_id, p=0.1
#     )

#     x    = batch["input_ids"].to(device)
#     pos  = batch["pos_idx"].to(device).long()
#     wt   = batch["wt_index"].to(device).long()
#     mu   = batch["mut_index"].to(device).long()
#     mask = batch["k_mask"].to(device)
    
#     if "target" not in batch:
#         raise ValueError("Batch must contain 'target' for ranking loss.")
#     y    = batch["target"].to(device)

#     out   = model(x)
#     logits= out.sequence_logits
#     logp  = F.log_softmax(logits, dim=-1)

#     first_tok = x[:, 0]
#     prefix = torch.zeros(first_tok.size(0), dtype=torch.long, device=x.device)
#     if getattr(tokenizer, "cls_token_id", None) is not None:
#         prefix |= (first_tok == tokenizer.cls_token_id).long()
#     if getattr(tokenizer, "bos_token_id", None) is not None:
#         prefix |= (first_tok == tokenizer.bos_token_id).long()

#     if pos.dim() == 1:
#         pos_tok = pos + prefix
#     else:
#         pos_tok = pos + prefix.unsqueeze(1)

#     B, L, V = logits.shape
#     if pos_tok.max() >= L:
#         raise ValueError(f"pos_idx contains index >= L ({L})")

#     bidx = torch.arange(B, device=device)[:, None].expand_as(pos_tok)
#     logp_at_pos = logp[bidx, pos_tok]
#     mut_lp = logp_at_pos.gather(-1, mu.unsqueeze(-1)).squeeze(-1)
#     wt_lp  = logp_at_pos.gather(-1, wt.unsqueeze(-1)).squeeze(-1)

#     delta_k = (mut_lp - wt_lp) * mask
#     delta   = delta_k.sum(dim=1)

#     probs_pos = F.softmax(logp_at_pos, dim=-1)
#     ent_k     = -(probs_pos * torch.log(probs_pos + 1e-8)).sum(-1)
#     denom     = mask.sum(dim=1).clamp(min=1.0)
#     ent       = (ent_k * mask).sum(dim=1) / denom
#     loss_ent  = ent.mean()

#     labels = torch.zeros_like(y)
#     labels[y >   0] =  1
#     labels[y <   0] = -1

#     valid = labels != 0

#     if valid.any():
#         loss_rank = F.softplus(-labels[valid] * (delta[valid] - margin)).mean()
#     else:
#         loss_rank = torch.tensor(0.0, device=device, requires_grad=True)

#     # Subtracted entropy to match the docstring equation
#     return loss_rank - beta * loss_ent

def batch_step(model, batch, device, margin=0.1, beta=0.1, return_metrics=False):
    """
    K-aware training step for SMITEpro.
    Uses summed per-site Delta = Sigma_k [logP(mut_k) - logP(wt_k)].
    loss = dynamic_margin_ranking(target, Delta) - beta * H
    """
    global tokenizer

    batch['input_ids'] = random_mask_batch(
        batch['input_ids'], tokenizer.mask_token_id, p=0.15
    )

    x    = batch["input_ids"].to(device)
    pos  = batch["pos_idx"].to(device).long()
    wt   = batch["wt_index"].to(device).long()
    mu   = batch["mut_index"].to(device).long()
    mask = batch["k_mask"].to(device)
    
    if "target" not in batch:
        raise ValueError("Batch must contain 'target' for ranking loss.")
    y    = batch["target"].to(device)

    out   = model(x)
    logits= out.sequence_logits
    logp  = F.log_softmax(logits, dim=-1)

    first_tok = x[:, 0]
    prefix = torch.zeros(first_tok.size(0), dtype=torch.long, device=x.device)
    if getattr(tokenizer, "cls_token_id", None) is not None:
        prefix |= (first_tok == tokenizer.cls_token_id).long()
    if getattr(tokenizer, "bos_token_id", None) is not None:
        prefix |= (first_tok == tokenizer.bos_token_id).long()

    if pos.dim() == 1:
        pos_tok = pos + prefix
    else:
        pos_tok = pos + prefix.unsqueeze(1)

    # Multi-mutant batches use -1 padding for inactive mutation slots.
    # Clamp padded indices to 0 before indexing/gather; k_mask removes them afterward.
    pos_tok = pos_tok.clamp(min=0)
    wt = wt.clamp(min=0)
    mu = mu.clamp(min=0)

    B, L, V = logits.shape
    if pos_tok.max() >= L:
        raise ValueError(f"pos_idx contains index >= L ({L})")

    bidx = torch.arange(B, device=device)[:, None].expand_as(pos_tok)
    logp_at_pos = logp[bidx, pos_tok]
    mut_lp = logp_at_pos.gather(-1, mu.unsqueeze(-1)).squeeze(-1)
    wt_lp  = logp_at_pos.gather(-1, wt.unsqueeze(-1)).squeeze(-1)

    delta_k = (mut_lp - wt_lp) * mask
    delta   = delta_k.sum(dim=1)

    probs_pos = F.softmax(logp_at_pos, dim=-1)
    ent_k     = -(probs_pos * torch.log(probs_pos + 1e-8)).sum(-1)
    denom     = mask.sum(dim=1).clamp(min=1.0)
    ent       = (ent_k * mask).sum(dim=1) / denom
    loss_ent  = ent.mean()

    # Dynamic continuous ranking loss
    valid = y != 0
    
    if valid.any():
        y_valid = y[valid]
        delta_valid = delta[valid]
        
        target_sign = torch.sign(y_valid)
        # target_weight = torch.abs(y_valid)
        target_weight = torch.pow(y_valid, 2)  # Exaggerate differences for larger magnitudes
        
        # Scale margin and loss by the magnitude of the DMS score
        dynamic_margin = margin + target_weight 
        base_loss = F.softplus(-target_sign * (delta_valid - dynamic_margin))
        loss_rank = (target_weight * base_loss).mean()
    else:
        loss_rank = torch.tensor(0.0, device=device, requires_grad=True)

    loss_total = loss_rank - beta * loss_ent

    if return_metrics:
        metrics = {
            "loss_total": loss_total.detach(),
            "loss_rank": loss_rank.detach(),
            "loss_entropy": loss_ent.detach(),
            "loss_entropy_weighted": (beta * loss_ent).detach(),
        }
        return loss_total, metrics

    return loss_total

def train(model, loader, optimizer, device, epochs, kwargs, accumulation_steps=1):
    model.train()
    for _ in range(epochs):
        # 1. Zero gradients at the start of the epoch, not inside the batch loop
        optimizer.zero_grad() 
        
        for i, batch in enumerate(loader):
            loss = batch_step(model, batch, device, **kwargs)
            
            # 2. Scale the loss down, because we are summing gradients over multiple steps
            loss = loss / accumulation_steps
            
            # 3. Accumulate gradients
            loss.backward()
            
            # 4. Step the optimizer only when we hit the accumulation target or the end of the data
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()

def train_fixed_steps(model, loader, optimizer, scheduler, device, total_steps=250, kwargs={}):
    """
    Fixed-step training budget to prevent signal dilution at high K.
    """
    model.train()
    step_count = 0
    recent_total = deque(maxlen=10)
    recent_rank = deque(maxlen=10)
    recent_entropy = deque(maxlen=10)
    
    # We loop until we hit the step budget
    while step_count < total_steps:
        for batch in loader:
            if step_count >= total_steps:
                break
                
            optimizer.zero_grad()
            
            # 1. Calculate loss with squared importance weighting
            loss, metrics = batch_step(model, batch, device, return_metrics=True, **kwargs)
            
            # 2. Backprop
            loss.backward()
            
            # 3. Step both optimizer and scheduler
            optimizer.step()
            scheduler.step()

            recent_total.append(metrics["loss_total"].item())
            recent_rank.append(metrics["loss_rank"].item())
            recent_entropy.append(metrics["loss_entropy_weighted"].item())
            
            step_count += 1
            if step_count % 10 == 0 or step_count == 1:
                print(
                    f"  [Step {step_count}/{total_steps}] "
                    f"Loss(total, avg{len(recent_total)}): {sum(recent_total) / len(recent_total):.4f} | "
                    f"rank(avg{len(recent_rank)}): {sum(recent_rank) / len(recent_rank):.4f} | "
                    f"beta*entropy(avg{len(recent_entropy)}): {sum(recent_entropy) / len(recent_entropy):.4f}"
                )



# =================
# Inference (K)
# =================
@torch.no_grad()
def predict_deltas(model, tokenizer, sequence, df, device, batch_size=64):
    """
    K-aware prediction for singles and multis.
    Returns: delta_sum [N], entropy [N], mut_sum [N], wt_sum [N]
    """
    vocab = tokenizer.get_vocab()
    data = prep_data_from_mutant(df, tokenizer, sequence, vocab)
    ds = ProteinDataset(data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    all_delta, all_ent, all_mut_sum, all_wt_sum = [], [], [], []

    for batch in loader:
        x    = batch["input_ids"].to(device)
        pos  = batch["pos_idx"].to(device).long()
        wt   = batch["wt_index"].to(device).long()
        mu   = batch["mut_index"].to(device).long()
        mask = batch["k_mask"].to(device)

        out   = model(x)
        logits= out.sequence_logits
        logp  = F.log_softmax(logits, dim=-1)

        # detect a prefix special token per-example
        first_tok = x[:, 0]
        prefix = torch.zeros(first_tok.size(0), dtype=torch.long, device=x.device)
        if getattr(tokenizer, "cls_token_id", None) is not None:
            prefix |= (first_tok == tokenizer.cls_token_id).long()
        if getattr(tokenizer, "bos_token_id", None) is not None:
            prefix |= (first_tok == tokenizer.bos_token_id).long()

        # shift positions to token indices
        if pos.dim() == 1:
            pos_tok = pos + prefix                      # [B]
        else:
            pos_tok = pos + prefix.unsqueeze(1)         # [B, K]

        # Multi-mutant batches use -1 padding for inactive mutation slots.
        # Clamp padded indices to 0 before indexing/gather; k_mask removes them afterward.
        pos_tok = pos_tok.clamp(min=0)
        wt = wt.clamp(min=0)
        mu = mu.clamp(min=0)

        B, L, V = logits.shape
        if pos_tok.max() >= L:
            raise ValueError(f"predict_deltas: pos_idx contains index >= L ({L})")
        

        bidx = torch.arange(B, device=device)[:, None].expand_as(pos_tok)
        logp_at_pos = logp[bidx, pos_tok]                            # [B,K,V]

        mut_lp = logp_at_pos.gather(-1, mu.unsqueeze(-1)).squeeze(-1) * mask
        wt_lp  = logp_at_pos.gather(-1, wt.unsqueeze(-1)).squeeze(-1) * mask

        mut_sum = mut_lp.sum(dim=1)                              # [B]
        wt_sum  = wt_lp.sum(dim=1)                               # [B]
        delta   = mut_sum - wt_sum                               # [B]

        probs_pos = F.softmax(logp_at_pos, dim=-1)               # [B,K,V]
        ent_k     = -(probs_pos * torch.log(probs_pos + 1e-8)).sum(-1)  # [B,K]
        denom     = mask.sum(dim=1).clamp(min=1.0)
        ent       = (ent_k * mask).sum(dim=1) / denom            # [B]

        all_delta.append(delta.cpu())
        all_ent.append(ent.cpu())
        all_mut_sum.append(mut_sum.cpu())
        all_wt_sum.append(wt_sum.cpu())

    return (
        torch.cat(all_delta).float().numpy(),
        torch.cat(all_ent).float().numpy(),
        torch.cat(all_mut_sum).float().numpy(),
        torch.cat(all_wt_sum).float().numpy(),
    )


# ==========================
# Misc helper functions
# ==========================
def evaluate_candidate_df(model, tokenizer, sequence, df, device, margin=0.1):
    deltas, _, mut_sum, wt_sum = predict_deltas(model, tokenizer, sequence, df, device)
    labels = np.sign(df["target"].values) if "target" in df.columns else np.zeros(len(df))
    valid  = labels != 0
    margin_loss = np.mean(np.maximum(0, margin - labels[valid] * (mut_sum[valid] - wt_sum[valid]))) if valid.sum() > 0 else 0.0
    corr = spearmanr(deltas, df["target"].values).correlation if ("target" in df.columns and df["target"].nunique() > 1) else np.nan
    return margin_loss, corr


def restore_wt_sequence(mutant_str, mutated_seq):
    wt_seq = list(mutated_seq)
    for mut in mutant_str.split(":"):
        wt, pos, new = mut[0], int(mut[1:-1]) - 1, mut[-1]
        if wt_seq[pos] != new:
            raise ValueError(f"Mismatch at {mut}: found {wt_seq[pos]}, expected {new}")
        wt_seq[pos] = wt
    return "".join(wt_seq)


def ensure_sequence_column(df, wt_seq):
    if "sequence" in df.columns:
        return df
    out = df.copy()

    def _mutate(mut_str):
        s = list(wt_seq)
        for m in mut_str.split(":"):
            wt, pos, aa = m[0], int(m[1:-1]) - 1, m[-1]
            if s[pos] != wt:
                raise ValueError(f"WT mismatch for {m}: expected {wt}, got {s[pos]} at pos {pos+1}")
            s[pos] = aa
        return "".join(s)

    out["sequence"] = out["mutant"].apply(_mutate)
    return out


@torch.no_grad()
def extract_mean_embeddings(df_seq, tokenizer, encoder, device, bs=64, max_length=2046):
    """
    Extract mean-pooled embeddings with OOM protection.
    On CUDA OOM, halves batch size (min=1) and retries from the same index.
    Never increases batch size once reduced.
    """
    seqs = df_seq["sequence"].tolist()
    if len(seqs) == 0:
        return np.zeros((0, getattr(encoder, "hidden_size", 0)), dtype=np.float32)

    feats = []
    encoder.eval().to(device)

    i = 0
    cur_bs = max(1, int(bs))
    use_autocast = (device.type == "cuda")

    while i < len(seqs):
        try:
            b = seqs[i : i + cur_bs]
            enc = tokenizer.batch_encode_plus(
                b, padding="longest", truncation=True, max_length=max_length, return_tensors="pt"
            )
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

            if use_autocast:
                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                ):
                    out = encoder.forward(enc["input_ids"], enc["attention_mask"])
            else:
                out = encoder.forward(enc["input_ids"], enc["attention_mask"])

            feats.append(out.embeddings.float().mean(dim=1).cpu().numpy())
            i += cur_bs  # advance only on success

        except RuntimeError as e:
            msg = str(e).lower()
            if ("out of memory" in msg or "cuda error: out of memory" in msg) and device.type == "cuda":
                # Reduce batch size and retry same slice
                cur_bs = max(1, cur_bs // 2)
                # Cleanup before retry
                try:
                    del enc, out
                except UnboundLocalError:
                    pass
                torch.cuda.empty_cache()
                if cur_bs == 1:
                    # If already at 1 and still OOM, fail clearly
                    continue  # try single-item again; if it keeps failing, loop will keep trying
            else:
                raise
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize()

    return np.concatenate(feats, axis=0)


def normalize_np(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

# ================
# MAIN
# ================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200) # Added fixed steps
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="esmc_300m")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="MutStacking_results")
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--zeroshot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Starting zero-shot + K-shot workflow for {args.protein}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.data_file)

    # if len(sequence) > 2046, break and warn (ESMC max length is 2048 including special tokens)
    # Ideally we would handle this more gracefully by truncating or using a sliding window, but for now just warn and exit
    if "mutated_sequence" in df.columns:
      seq_len = df["mutated_sequence"].str.len().max()
      if seq_len > 2046:
          print(f"Error: Maximum sequence length ({seq_len}) exceeds ESMC limit of 2046.")
          sys.exit(1)

    # Normalize targets if needed
    if "target" not in df.columns:
        if "DMS_score" in df.columns and "DMS_score_bin" in df.columns:
            df["target"] = df["DMS_score"]
            midpoint_scale_whole_df(df, target_col="target",
                                    bin_col="DMS_score_bin", out_col="target_norm")
            df["target"] = df["target_norm"]
            df.drop(columns=["target_norm"], inplace=True)
        else:
            print("Data must contain 'target' or both 'DMS_score' and 'DMS_score_bin'.")
            sys.exit(1)

    # Infer WT sequence
    if "mutated_sequence" in df.columns:
        seq = restore_wt_sequence(df["mutant"].iloc[0], df["mutated_sequence"].iloc[0])
    else:
        raise ValueError("Need 'mutated_sequence' column to infer WT sequence.")

    # -------------------------
    # Phase 1: Zero-shot ESMC
    # -------------------------

    # Load ESMC model and tokenizer once for zero-shot scoring, and reuse the same model
    model, tokenizer = load_model(args.model)
    model.to(device).eval()

    # zero_shot save name
    # save to output_dir but up one and then into a folder named "zeroshot_predictions" to keep things organized
    zeroshot_dir = os.path.join(os.path.dirname(args.output_dir), "zeroshot_predictions")
    # make sure the directory exists
    os.makedirs(zeroshot_dir, exist_ok=True)
    # set save name to be "{protein}_zero_shot_scores.csv" in the zeroshot_dir
    zero_shot_save_name = os.path.join(zeroshot_dir, f"{args.protein}_zero_shot_scores.csv")

    # Ensure df is only single mutants for candidate selection
    singles_df = df[df["mutant"].str.count(":") == 0].copy()
    multis_df = df[df["mutant"].str.count(":") > 0].copy()

    # if the file exists, load it and skip zero-shot scoring; otherwise run zero-shot scoring and save the results to this file
    if os.path.exists(zero_shot_save_name):
        df_sorted = pd.read_csv(zero_shot_save_name)
        # Restore columns from cache
        singles_df = singles_df.merge(
            df_sorted[['mutant', 'delta_esm', 'uncertainty', 'score']], 
            on='mutant', 
            how='left'
        )
    else:
        # if the file does not exist, run zero-shot scoring and save the results to this file
        print("Running initial Zero-Shot scoring to rank candidates...")
        z_delta, z_ent, _, _ = predict_deltas(model, tokenizer, seq, singles_df, device, batch_size=128)
        singles_df["delta_esm"] = z_delta
        singles_df["uncertainty"] = z_ent

        w = args.delta
        singles_df["score"] = (1 - w) * normalize_np(singles_df["delta_esm"].values) + w * normalize_np(singles_df["uncertainty"].values)
        
        # Sort ALL single mutants by zero-shot score to create the master ranking pool
        df_sorted = singles_df.sort_values("score", ascending=False).reset_index(drop=True)

        # Save the zero-shot scores for all single mutants to a CSV for future reference and to avoid re-computation
        df_sorted.to_csv(zero_shot_save_name, index=False)
        print(f"Zero-shot scoring complete. Saved scores to {zero_shot_save_name}")

    # k_values = [4, 8, 16, 32, 64, 128]
    k_values = [16, 32, 64, 128]
    # k_values = [4, 8, 16]

    vocab = tokenizer.get_vocab()

    # Iterate through all requested K sizes
    for current_k in k_values:
        print(f"\n{'='*50}")
        print(f"Starting evaluation for K={current_k}")
        print(f"{'='*50}")
        
        if len(singles_df) < current_k:
            print(f"Skipping K={current_k}: Not enough single mutants available ({len(singles_df)} total).")
            continue

        # Select topK mutants with unique mutation positions
        selected_positions = set()
        topK_indices = []
        pass_num = 1
        remaining_indices = set(df_sorted.index)

        while len(topK_indices) < current_k and remaining_indices:
            for idx in list(remaining_indices):
                if len(topK_indices) >= current_k:
                    break

                row = df_sorted.loc[idx]
                pos, = _parse_mutant_string(row["mutant"])[0]

                if pos in selected_positions:
                    continue

                selected_positions.add(pos)
                topK_indices.append(idx)
                remaining_indices.remove(idx)

            selected_positions.clear()
            pass_num += 1

            if not remaining_indices:
                break

        if len(topK_indices) < current_k:
            print(f"Warning: Could only find {len(topK_indices)} unique position mutants for K={current_k}.")
            continue # Or decide to proceed with the smaller subset

        topK_df = df_sorted.loc[topK_indices].copy()
        print(f"Successfully selected top {len(topK_df)} mutants.")


        # I WAS EVALUATING ON MY TEST SET... sigh
        # DON'T UNCOMMENT BELOW, IT'S HERE FOR POSTERITY TO REMEMBER NOT TO MAKE THE SAME MISTAKE AGAIN

        # Prepare the evaluation dataframe for this specific K
        # if multis_df.empty:
        #     current_eval_df = singles_df.drop(index=topK_df.index).copy()
        # else:
        #     current_eval_df = multis_df.copy()
        # Prepare the evaluation dataframe for this specific K


        if multis_df.empty:
            # Filter by the actual mutant strings to ensure correct exclusion
            current_eval_df = singles_df[~singles_df["mutant"].isin(topK_df["mutant"])].copy()
        else:
            current_eval_df = multis_df.copy()
        
        # -------------------------------
        # Phase 2: Fine-tune delta model
        # -------------------------------

        # Scale batch size with dataset size (K) to keep optimizer steps per epoch stable
        
        # vram_batch_size = min(current_k, 16)
        # accumulation_steps = max(1, effective_batch_size // vram_batch_size)

        batch_size = max(4, min(current_k // 2, 16))  # Start with half of K, capped at 16, and minimum of 4


        train_data = prep_data_from_mutant(topK_df, tokenizer, seq, vocab)
        train_ds = ProteinDataset(train_data)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)

        model_ft = deepcopy(model).to(device)
        
        # Apply LoRA instead of simple unfreezing
        # Target modules: Identify layers containing q, v, query, value.
        # In simple ESMC (UnifiedTransformerBlock), QKV is often in layernorm_qkv.1
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["layernorm_qkv.1", "query", "value", "q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=None,
        )
        model_ft = get_peft_model(model_ft, peft_config)
        
        # Ensure sequence head is unfrozen (LoRA freezes base model by default)
        # Access original model through base_model.model because get_peft_model wraps it
        for p in model_ft.base_model.model.sequence_head.parameters():
            p.requires_grad = True

        if not args.zeroshot:
            opt = torch.optim.AdamW(
                filter(
                    lambda p: p.requires_grad, 
                    model_ft.parameters()
                    )
                , lr=args.lr, 
                weight_decay=0.05)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)

            train_fixed_steps(
                model_ft, train_loader, opt, scheduler, device, 
                total_steps=args.steps, 
                kwargs={"beta": 0.1, "margin": 0.1}
                )
            # train(
            #     model_ft, train_loader, 
            #     opt, device, args.epochs, 
            #     kwargs={"beta": 0.1, "margin": 0.1}, 
            #     accumulation_steps=accumulation_steps
            #     )

        d_all, _, _, _ = predict_deltas(model_ft, tokenizer, seq, current_eval_df, device)
        current_eval_df["pred_my_method"] = d_all

        # ----------------------------------------
        # Phase 3: Random Forests
        # ----------------------------------------
        if not args.zeroshot:
            topK_df = ensure_sequence_column(topK_df, seq)
            current_eval_df = ensure_sequence_column(current_eval_df, seq)
            
            # Extract LoRA embeddings before deleting model_ft
            X_train_lora = extract_mean_embeddings(topK_df, tokenizer, model_ft, device)
            X_test_lora  = extract_mean_embeddings(current_eval_df, tokenizer, model_ft, device)
            y_train = normalize_np(topK_df["target"].values)

            # Train Random Forest on LoRA embeddings
            rf_lora = RandomForestRegressor(n_estimators=100, criterion="friedman_mse", n_jobs=40)
            rf_lora.fit(X_train_lora, y_train)
            current_eval_df["pred_epro_lora"] = rf_lora.predict(X_test_lora)

        # Cleanup memory from fine-tuning
        del model_ft
        torch.cuda.empty_cache()

        if not args.zeroshot:
            # Train Random Forest on Base embeddings
            clean_model, clean_tokenizer = load_model(args.model)
            clean_model.to(device).eval()

            X_train_base = extract_mean_embeddings(topK_df, clean_tokenizer, clean_model, device)
            X_test_base  = extract_mean_embeddings(current_eval_df, clean_tokenizer, clean_model, device)

            rf_base = RandomForestRegressor(n_estimators=100, criterion="friedman_mse", n_jobs=40)
            rf_base.fit(X_train_base, y_train)
            current_eval_df["pred_epro"] = rf_base.predict(X_test_base)
            
            del clean_model
            torch.cuda.empty_cache()

        # ----------------------------
        # Phase 4: Output
        # ----------------------------

        rho_my = spearmanr(current_eval_df["pred_my_method"], current_eval_df["target"]).correlation
        if "target" in current_eval_df.columns and current_eval_df["target"].nunique() > 1 and not args.zeroshot:
            rho_epr = spearmanr(current_eval_df["pred_epro"], current_eval_df["target"]).correlation
            rho_epr_lora = spearmanr(current_eval_df["pred_epro_lora"], current_eval_df["target"]).correlation
            
            if "delta_esm" in current_eval_df.columns and current_eval_df["delta_esm"].notna().any():
                rho_zeroshot = spearmanr(current_eval_df["delta_esm"], current_eval_df["target"]).correlation
                print(f"[Result K={current_k}] Spearman rho | SMITEpro: {rho_my:.4f}, EVOLVEpro(Base): {rho_epr:.4f}, EVOLVEpro(LoRA): {rho_epr_lora:.4f}, Zero-shot: {rho_zeroshot:.4f}")
            else:
                print(f"[Result K={current_k}] Spearman rho | SMITEpro: {rho_my:.4f}, EVOLVEpro(Base): {rho_epr:.4f}, EVOLVEpro(LoRA): {rho_epr_lora:.4f}")
        else:
            print(f"[Result K={current_k}] Spearman rho | SMITEpro: {rho_my:.4f}")

        out_path = os.path.join(args.output_dir, f"{args.protein}_final_predictions_shot{current_k}.csv")
        current_eval_df.to_csv(out_path, index=False)
        print(f"Saved dataset for K={current_k} → {out_path}")

    print("Run complete. Total time: %.1f minutes." % ((time.time() - start_time) / 60))
