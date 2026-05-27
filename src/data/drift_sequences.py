"""Drift sequence construction (thesis Ch 3.5).

Builds contiguous training/evaluation windows under three drift types:
1. Sudden: train on regime A, evaluate on regime B (single switch).
2. Gradual: train on mixed A+B with increasing B fraction, eval on B.
3. Recurring: train A, eval B, eval A, eval B (revisit).

Each window has min_size contiguous samples of the labeled regime.
"""
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd


def find_regime_blocks(regimes: pd.Series, min_size: int = 60) -> Dict[int, List[Tuple[int, int]]]:
    """Find contiguous blocks of constant regime, each ≥ min_size.
    
    Returns dict: regime_id -> list of (start_idx, end_idx) tuples (end exclusive).
    """
    blocks: Dict[int, List[Tuple[int, int]]] = {}
    if len(regimes) == 0:
        return blocks
    r = regimes.values
    start = 0
    cur = r[0]
    for t in range(1, len(r)):
        if r[t] != cur:
            if t - start >= min_size:
                blocks.setdefault(int(cur), []).append((start, t))
            start = t
            cur = r[t]
    # Tail
    if len(r) - start >= min_size:
        blocks.setdefault(int(cur), []).append((start, len(r)))
    return blocks


def build_sudden(regimes: pd.Series, src: int, dst: int,
                 train_size: int = 200, eval_size: int = 100,
                 min_block: int = 60):
    """Pick contiguous train block in src regime, then contiguous eval block in dst regime.
    
    Returns (train_idx, eval_idx) as numpy arrays, or None if no valid windows.
    """
    blocks = find_regime_blocks(regimes, min_size=min_block)
    if src not in blocks or dst not in blocks:
        return None
    # Find src block with ≥ train_size
    train_block = None
    for s, e in blocks[src]:
        if e - s >= train_size:
            train_block = (s, e)
            break
    if train_block is None:
        return None
    eval_block = None
    for s, e in blocks[dst]:
        if s >= train_block[1] and e - s >= eval_size:
            eval_block = (s, e)
            break
    if eval_block is None:
        return None
    train_idx = np.arange(train_block[0], train_block[0] + train_size)
    eval_idx = np.arange(eval_block[0], eval_block[0] + eval_size)
    return train_idx, eval_idx


def build_gradual(regimes: pd.Series, src: int, dst: int,
                  train_size: int = 200, eval_size: int = 100,
                  ramp_size: int = 100, min_block: int = 60):
    """Train on src, ramp (mixed), eval on dst. Returns (train_idx, eval_idx, ramp_idx) or None."""
    res = build_sudden(regimes, src, dst, train_size, eval_size, min_block)
    if res is None:
        return None
    train_idx, eval_idx = res
    ramp_start = train_idx[-1] + 1
    ramp_end = min(ramp_start + ramp_size, eval_idx[0])
    if ramp_end <= ramp_start:
        return None
    ramp_idx = np.arange(ramp_start, ramp_end)
    return train_idx, eval_idx, ramp_idx


def build_recurring(regimes: pd.Series, src: int, dst: int,
                    train_size: int = 200, eval_size: int = 100,
                    min_block: int = 60):
    """Train on src, eval on dst, then eval on src again (revisit).
    
    Returns (train_idx, eval_dst_idx, eval_src_revisit_idx) or None.
    """
    blocks = find_regime_blocks(regimes, min_size=min_block)
    if src not in blocks or dst not in blocks:
        return None
    src_blocks = blocks[src]
    dst_blocks = blocks[dst]
    if len(src_blocks) < 2:
        return None
    
    train_block = None
    for s, e in src_blocks:
        if e - s >= train_size:
            train_block = (s, e)
            break
    if train_block is None:
        return None
    eval_dst = None
    for s, e in dst_blocks:
        if s >= train_block[1] and e - s >= eval_size:
            eval_dst = (s, e)
            break
    if eval_dst is None:
        return None
    eval_src_revisit = None
    for s, e in src_blocks:
        if s >= eval_dst[1] and e - s >= eval_size:
            eval_src_revisit = (s, e)
            break
    if eval_src_revisit is None:
        return None
    return (
        np.arange(train_block[0], train_block[0] + train_size),
        np.arange(eval_dst[0], eval_dst[0] + eval_size),
        np.arange(eval_src_revisit[0], eval_src_revisit[0] + eval_size),
    )


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    r = pd.Series(np.repeat([0, 1, 2, 0, 1], 80))
    print('Blocks:', find_regime_blocks(r, min_size=60))
    print('Sudden 0->1:', build_sudden(r, 0, 1, 100, 50, 60))
