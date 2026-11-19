"""
ARC Task: 5daaa586 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from itertools import combinations
import numpy as np
from maker.sel_helpers import sel_of

ROTS = [0, 1, 2, 3]

def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    c1, c2, c3, c4 = random.sample([c for c in cols if c != bgc], 4)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "instance_plan": plan}

def _unifint(lb, ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(lb, ub)
    return max(a, min(b, int(round(a + (b - a) * d + random.uniform(-0.5, 0.5)))))

def _nfrontiers(g):
    n = 0
    for row in g:
        if len(set(row.tolist())) == 1:
            n += 1
    for col in g.T:
        if len(set(col.tolist())) == 1:
            n += 1
    return n

def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2, c3, c4, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    hmax, wmax = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    hmax = max(7, min(30, hmax)); wmax = max(7, min(30, wmax))
    h = _unifint(diff_lb, diff_ub, (7, hmax))
    w = _unifint(diff_lb, diff_ub, (7, wmax))
    loci1 = random.randint(1, h - 4)
    locj1 = random.randint(1, w - 4)
    loci1 -= _unifint(diff_lb, diff_ub, (0, loci1 - 1))
    locj1 -= _unifint(diff_lb, diff_ub, (0, locj1 - 1))
    loci2 = _unifint(diff_lb, diff_ub, (loci1 + 2, h - 2))
    locj2 = _unifint(diff_lb, diff_ub, (locj1 + 2, w - 2))
    gi = np.full((h, w), bgc, dtype=int)
    fronts = [("r", loci1, c1), ("r", loci2, c2), ("c", locj1, c3), ("c", locj2, c4)]
    random.shuffle(fronts)
    for kind, idx, col in fronts:
        if kind == "r":
            gi[idx, :] = col
        else:
            gi[:, idx] = col
    cands = [(r, c) for r in range(h) for c in range(w) if gi[r, c] == bgc]
    nn = len(cands)
    for _ in range(200):
        nnoise = _unifint(diff_lb, diff_ub, (1, max(1, nn // 3)))
        noise = random.sample(cands, nnoise)
        g2 = gi.copy()
        for r, c in noise:
            g2[r, c] = c1
        if _nfrontiers(g2) <= 4:
            gi = g2
            break
    go = gi[loci1:loci2 + 1, locj1:locj2 + 1].copy()
    for r, c in zip(*np.where(go == c1)):
        go[:r, c] = c1
    gi = np.rot90(gi, -rot) if rot else gi
    go = np.rot90(go, -rot) if rot else go
    return {"input": gi.tolist(), "output": go.tolist()}

def _cands(I, axis, bgc):
    A = I if axis == 0 else I.T
    n = A.shape[1]
    res = []
    for i, row in enumerate(A):
        col, cnt = Counter(row.tolist()).most_common(1)[0]
        if cnt >= n - 2 and col != bgc:
            exc = frozenset(j for j in range(n) if row[j] != col)
            res.append((i, col, exc))
    return res

def _frame(I, bgc):
    R = _cands(I, 0, bgc); C = _cands(I, 1, bgc)
    best = None
    for ra, rb in combinations(R, 2):
        for ca, cb in combinations(C, 2):
            if len({ra[1], rb[1], ca[1], cb[1]}) != 4:
                continue
            rs = {ra[0], rb[0]}; cs = {ca[0], cb[0]}
            if not all(x[2] <= cs for x in (ra, rb)) or not all(x[2] <= rs for x in (ca, cb)):
                continue
            score = -(len(ra[2]) + len(rb[2]) + len(ca[2]) + len(cb[2]))
            if best is None or score > best[0]:
                best = (score, sorted([ra[:2], rb[:2]]), sorted([ca[:2], cb[:2]]))
    return best[1], best[2]

def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background is fixed per episode: read it from the demonstrations (pooled majority)
    pool = Counter(I.flatten().tolist())
    for ex in (examples or []):
        pool.update(np.asarray(ex[0]).flatten().tolist())
        pool.update(np.asarray(ex[1]).flatten().tolist())
    bgc = pool.most_common(1)[0][0]
    rows, colsl = _frame(I, bgc)
    (r1, rc1), (r2, rc2) = rows
    (k1, cc1), (k2, cc2) = colsl
    line_rows = {r1, r2}; line_cols = {k1, k2}
    off = Counter(int(I[r, c]) for r in range(hi) for c in range(wi)
                  if r not in line_rows and c not in line_cols)
    linecols = {rc1: "T", rc2: "B", cc1: "L", cc2: "R"}
    noise = [c for c in linecols if c in off]   # the line colour that also appears as noise
    ops, sels = [], []
    H, W = r2 - r1 + 1, k2 - k1 + 1
    # crop to the full rectangle framed by the four lines (bbox = exactly the intended region)
    ops.append(33); sels.append([r1, k1, H - 1, W - 1])
    G = I[r1:r2 + 1, k1:k2 + 1].copy()
    if noise:
        c1 = noise[0]
        side = linecols[c1]
        if side in "TB":
            for c in range(1, W - 1):
                rs = [r for r in range(1, H - 1) if G[r, c] == c1]
                if not rs:
                    continue
                span = range(1, max(rs) + 1) if side == "T" else range(min(rs), H - 1)
                cells = [(r, c) for r in span if G[r, c] != c1]
                if cells:
                    ops.append(c1); sels.append(sel_of(cells))
                    for p in cells: G[p] = c1
        else:
            for r in range(1, H - 1):
                cs = [c for c in range(1, W - 1) if G[r, c] == c1]
                if not cs:
                    continue
                span = range(1, max(cs) + 1) if side == "L" else range(min(cs), W - 1)
                cells = [(r, c) for c in span if G[r, c] != c1]
                if cells:
                    ops.append(c1); sels.append(sel_of(cells))
                    for p in cells: G[p] = c1
    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 5daaa586"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5daaa586"
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
                                f"for task 5daaa586"
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
                    f"Failed to build a complete episode for task 5daaa586 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5daaa586-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
