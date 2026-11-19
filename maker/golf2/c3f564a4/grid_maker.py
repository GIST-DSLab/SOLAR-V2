"""
ARC Task: c3f564a4 (RE-ARC) — LLM-generated grid_maker
"""
from __future__ import annotations

import inspect

import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

SOLAR_ROOT = Path(__file__).resolve().parents[3]
if str(SOLAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLAR_ROOT))

REARC_ROOT = SOLAR_ROOT / "re-arc"
_rs = str(REARC_ROOT)
while _rs in sys.path:
    sys.path.remove(_rs)
sys.path.insert(0, _rs)

from maker.base_grid_maker import BaseGridMaker

import importlib
for _m in ["utils", "dsl", "generators"]:
    if _m in sys.modules:
        del sys.modules[_m]
from utils import *  # noqa: F401,F403  (unifint, choice, sample, etc.)
from dsl import *    # noqa: F401,F403

# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
import random
import numpy as np
from collections import deque
from maker.sel_helpers import sel_of

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def sample_colors(num_examples=None) -> dict:
    fixc = random.choice(range(10))
    return {"fixc": fixc}


def generate(diff_lb, diff_ub, max_h, max_w, fixc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if fixc is None:
        fixc = random.choice(range(10))
    h = unifint(diff_lb, diff_ub, (7, max(7, max_h)))
    w = unifint(diff_lb, diff_ub, (7, max(7, max_w)))
    h = min(h, max_h)
    w = min(w, max_w)
    p = unifint(diff_lb, diff_ub, (2, min(9, min(h // 3, w // 3))))
    p = max(2, p)
    remcols = [c for c in range(10) if c != fixc]
    ccols = random.sample(remcols, p)
    random.shuffle(ccols)
    diag = random.choice((True, False))
    go = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            idx = (c + r) % p if diag else c % p
            go[r][c] = ccols[idx]
    gi = [row[:] for row in go]

    def occ_count(g):
        cnt = 0
        for r in range(h):
            for c in range(w - p + 1):
                if all(g[r][c + j] == ccols[j] for j in range(p)):
                    cnt += 1
                    if cnt > 1:
                        return cnt
        return cnt

    nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    maxtr = 4 * nsq
    tr = succ = 0
    while succ < nsq and tr < maxtr:
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (2, 5))
        loci = random.randint(0, h - oh)
        locj = random.randint(0, w - ow)
        tmpg = [row[:] for row in gi]
        for r in range(loci, loci + oh):
            for c in range(locj, locj + ow):
                tmpg[r][c] = fixc
        rows_ok = any(fixc not in row for row in tmpg)
        cols_ok = any(all(tmpg[r][c] != fixc for r in range(h)) for c in range(w))
        if occ_count(tmpg) > 1 and rows_ok and cols_ok:
            gi = tmpg
            succ += 1
        tr += 1
    gi = np.array(gi)
    go = np.array(go)
    if random.choice((True, False)):
        gi = np.rot90(gi, 3)
        go = np.rot90(go, 3)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    def line_period(line):
        n = len(line)
        for p in range(1, n):
            if all(line[c] == line[c + p] for c in range(n - p)):
                return p
        return n

    rows = [I[r].tolist() for r in range(hi)]
    colsl = [I[:, c].tolist() for c in range(wi)]
    row_p = [line_period(r) for r in rows]
    col_p = [line_period(c) for c in colsl]
    ph, pv = min(row_p), min(col_p)
    pattern_cols = set()
    for r in range(hi):
        if row_p[r] == ph:
            pattern_cols |= set(rows[r])
    for c in range(wi):
        if col_p[c] == pv:
            pattern_cols |= set(colsl[c])

    palette = sorted(set(I.flatten().tolist()))
    R, C = np.indices((hi, wi))
    cands = []
    for (a, b) in _DIRS:
        lat = a * R + b * C
        for p in range(2, 10):
            key = lat % p
            for k in palette:
                keep = I != k
                mapping = {}
                bad = False
                for cls in range(p):
                    u = np.unique(I[(key == cls) & keep])
                    if len(u) > 1:
                        bad = True
                        break
                    if len(u) == 1:
                        mapping[cls] = int(u[0])
                if bad or not mapping:
                    continue
                if len(set(mapping.values())) != len(mapping):
                    continue
                if len(np.unique(key[I == k])) < 2:
                    continue
                cands.append((k not in pattern_cols, len(mapping) == p,
                              int(keep.sum()), -p, k, (a, b), p, mapping))
    cands.sort(reverse=True, key=lambda t: t[:4])

    repairs = {}
    if cands:
        _, _, _, _, noise, (a, b), p, mapping = cands[0]
        key = (a * R + b * C) % p
        for r in range(hi):
            for c in range(wi):
                if int(I[r, c]) == noise:
                    tgt = mapping.get(int(key[r, c]))
                    if tgt is not None and tgt != noise:
                        repairs[(r, c)] = tgt

    if not repairs and (hi, wi) == (ho, wo) and not np.array_equal(I, O):
        for r in range(hi):
            for c in range(wi):
                if I[r, c] != O[r, c]:
                    repairs[(r, c)] = int(O[r, c])

    # One Color op per wallpaper colour: all occluded cells of that lattice class at once.
    groups = {}
    order = []
    for cell in sorted(repairs.keys()):
        v = repairs[cell]
        if v not in groups:
            groups[v] = []
            order.append(v)
        groups[v].append(cell)
    for v in order:
        ops.append(int(v))
        sels.append(sel_of(groups[v]))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
    return ops, sels


# ── GridMaker ─────────────────────────────────────────────────────────────────

class GridMaker(BaseGridMaker):

    def parse(self, **kwargs) -> List[Tuple[
        List[NDArray], List[NDArray],
        List[NDArray], List[NDArray],
        Dict[str, Any],
    ]]:
        num_samples  = kwargs.get("num_samples", 1)
        num_examples = kwargs.get("num_examples", 3)
        max_h, max_w = kwargs.get("max_grid_dim", [30, 30])
        dataset = []

        for _sn in range(num_samples):
            # Episode-level retry: if 10 attempts at some instance all fail, that's
            # transient (bad luck with the generator's randomness) — retry the WHOLE
            # episode from scratch (fresh colors/instance plan) up to 5 times, rather
            # than silently continuing with a partial episode (fewer examples than
            # requested, or a missing test instance with operations=[]/selections=[]
            # quietly appended as if it were a normal sample).
            for _episode_attempt in range(5):
                pr_in:  List[NDArray] = []
                pr_out: List[NDArray] = []
                ex_in:  List[NDArray] = []
                ex_out: List[NDArray] = []
                ops:  List[int]       = []
                sels: List[List[int]] = []

                # sample color roles once per episode → consistent across all instances
                # sample_colors() may optionally accept num_examples (to pre-plan
                # per-instance categories) — call it either way for compatibility
                # with grid_makers generated before this parameter existed.
                if "num_examples" in inspect.signature(sample_colors).parameters:
                    colors = sample_colors(num_examples=num_examples)
                else:
                    colors = sample_colors()

                # Plans are consumed by INDEX, not mutated: retries for instance j
                # must receive the same variant. category_plan is retained as a
                # backwards-compatible single-key form; new makers use kwargs dict entries.
                category_plan = colors.pop("category_plan", None) if isinstance(colors, dict) else None
                instance_plan = colors.pop("instance_plan", None) if isinstance(colors, dict) else None
                if category_plan is not None and instance_plan is not None:
                    raise ValueError(
                        "sample_colors must return only one of category_plan/instance_plan"
                    )
                if category_plan is not None and len(category_plan) != num_examples + 1:
                    # A wrong plan length is a deterministic bug in sample_colors(),
                    # not bad luck — retrying the episode won't fix it. Fail loudly
                    # instead of clamping the index and silently reusing an entry.
                    raise ValueError(
                        f"category_plan length {len(category_plan)} != "
                        f"num_examples+1 ({num_examples + 1}) for task c3f564a4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c3f564a4"
                        )
                    if any(not isinstance(entry, dict) for entry in instance_plan):
                        raise ValueError("every instance_plan entry must be a kwargs dict")
                    if instance_plan[-1] not in instance_plan[:-1]:
                        raise ValueError(
                            "instance_plan test variant must appear among the examples"
                        )

                try:
                    j = 0
                    while j < num_examples + 1:
                        ok = False
                        for _ in range(10):
                            try:
                                call_kwargs = dict(colors)
                                if instance_plan is not None:
                                    call_kwargs.update(instance_plan[j])
                                elif category_plan is not None:
                                    call_kwargs["category"] = category_plan[j]
                                r = generate(
                                    random.uniform(0.2, 0.5),
                                    random.uniform(0.5, 0.8),
                                    max_h, max_w,
                                    **call_kwargs,
                                )
                                I = np.array(r["input"],  dtype=np.uint8)
                                O = np.array(r["output"], dtype=np.uint8)
                                # enforce max_grid_dim — skip oversized grids
                                if I.shape[0] > max_h or I.shape[1] > max_w:
                                    continue
                                if O.shape[0] > max_h or O.shape[1] > max_w:
                                    continue
                                ok = True
                                break
                            except (IndexError, ValueError, KeyError):
                                continue
                        if not ok:
                            raise RuntimeError(
                                f"Failed to generate instance {j} after 10 attempts "
                                f"for task c3f564a4"
                            )
                        if j == num_examples:
                            pr_in.append(I)
                            pr_out.append(O)
                            ops, sels = derive_operations(I, O)
                        else:
                            ex_in.append(I)
                            ex_out.append(O)
                        j += 1
                    break  # episode complete
                except RuntimeError:
                    continue
            else:
                raise RuntimeError(
                    f"Failed to build a complete episode for task c3f564a4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c3f564a4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
