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
from maker.sel_helpers import sel_of

# Structural cases: the wallpaper is diagonal stripes or straight stripes, and the
# grid is either left as generated or rotated 90 degrees.
VARIANTS = [{"diag": True, "rot": False}, {"diag": True, "rot": True},
            {"diag": False, "rot": False}, {"diag": False, "rot": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    fixc = random.choice(cols)                       # the occluder colour
    ccols = [c for c in cols if c != fixc]           # wallpaper colour pool
    random.shuffle(ccols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"fixc": fixc, "ccols": ccols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, fixc, ccols, diag=None, rot=None, **kw):
    def unif(lo, hi):
        lo, hi = int(lo), int(hi)
        if hi <= lo:
            return lo
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, int(round(lo + d * (hi - lo)))))

    if diag is None:
        diag = random.choice([True, False])
    if rot is None:
        rot = random.choice([True, False])
    H, W = (max_w, max_h) if rot else (max_h, max_w)
    h = unif(7, min(30, H))
    w = unif(7, min(30, W))
    p = unif(2, min(9, min(h // 3, w // 3)))
    cc = list(ccols[:p])
    random.shuffle(cc)
    go = [[cc[((j + i) if diag else j) % p] for j in range(w)] for i in range(h)]
    gi = [row[:] for row in go]
    nsq = unif(1, max(1, (h * w) // 25))
    succ, tr = 0, 0
    while succ < nsq and tr < 4 * nsq:
        tr += 1
        oh, ow = unif(2, 5), unif(2, 5)
        li, lj = random.randint(0, h - oh), random.randint(0, w - ow)
        tmp = [row[:] for row in gi]
        for r in range(li, li + oh):
            for c in range(lj, lj + ow):
                tmp[r][c] = fixc
        flat = [v for row in tmp for v in row]
        if (all(c in flat for c in cc)
                and any(fixc not in row for row in tmp)
                and any(all(tmp[r][c] != fixc for r in range(h)) for c in range(w))):
            gi = tmp
            succ += 1
    gi, go = np.array(gi), np.array(go)
    if rot:
        gi, go = np.rot90(gi, 3), np.rot90(go, 3)
    return {"input": gi.tolist(), "output": go.tolist()}


_DIRS = [(1, 0), (0, 1), (1, 1), (1, -1)]


def derive_operations(I, O=None, examples=None):
    # Reading (shared with the golf entry): the whole grid is ONE wallpaper, made by
    # laying a single colour sequence along a lattice direction (rows, columns or
    # diagonals). Each stripe class of that lattice owns exactly one colour.
    # The maker repairs one occluding blob at a time. This version carves the
    # problem the way the golf entry does: it restores the wallpaper itself, one
    # stripe class at a time, walking the colour sequence in order. For each class,
    # every occluder-coloured cell on that class gets the class colour in a single
    # op. The selection "occluder cells lying on stripe class k" can be named
    # from I alone.
    # O is not read anywhere.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
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
            pattern_cols |= set(rows[r])        # colours on an unoccluded row
    for c in range(wi):
        if col_p[c] == pv:
            pattern_cols |= set(colsl[c])       # colours on an unoccluded column

    palette = sorted(set(I.flatten().tolist()))
    R, C = np.indices((hi, wi))
    cands = []
    for (a, b) in _DIRS:
        lat = a * R + b * C
        for p in range(2, 10):
            key = lat % p
            for k in palette:
                keep = I != k                   # try k as the occluding colour
                mapping, bad = {}, False
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

    if cands:
        _, _, _, _, noise, (a, b), p, mapping = cands[0]
        key = (a * R + b * C) % p
        for cls in range(p):                    # walk the colour sequence in order
            tgt = mapping.get(cls)
            if tgt is None or tgt == noise:
                continue
            cells = [(int(r), int(c))
                     for r, c in zip(*np.where((key == cls) & (I == noise)))]
            if cells:
                ops.append(int(tgt))
                sels.append(sel_of(cells))      # occluded cells of this stripe class

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
