"""
ARC Task: d8c310e9 (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]  # np.rot90 k applied to the canonical grid


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": k} for k in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        return max(a, min(b, random.randint(min(lo, hi), max(lo, hi))))
    if rot is None:
        rot = random.choice(ROTS)
    if rot in (1, 3):
        hcap, wcap = max_w, max_h
    else:
        hcap, wcap = max_h, max_w
    hcap, wcap = min(30, hcap), min(30, wcap)
    h = unifint(diff_lb, diff_ub, (3, hcap))
    w = unifint(diff_lb, diff_ub, (10, wcap))
    p = unifint(diff_lb, diff_ub, (2, (w - 1) // 3))
    remcols = [c for c in range(10) if c != bgc]
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = random.sample(remcols, numc)
    obj = {}
    for j in range(p):
        numcells = unifint(diff_lb, diff_ub, (1, h - 1))
        for ii in range(h - 1, h - numcells - 1, -1):
            obj[(ii, j)] = random.choice(ccols)
    full = dict(obj)
    for (i, j), v in obj.items():
        full[(i, j + p)] = v
    leftshift = random.randint(0, random.randint(0, p))
    full = {(i, j - leftshift): v for (i, j), v in full.items()}
    gi = np.full((h, w), bgc, dtype=int)
    for (i, j), v in full.items():
        if 0 <= j < w:
            gi[i, j] = v
    go = gi.copy()
    for k in range(w // (2 * p) + 2):
        for (i, j), v in full.items():
            jj = j + k * 2 * p
            if 0 <= jj < w:
                go[i, jj] = v
    # reject instances whose shown block does not pin down its period: the
    # shortest self-matching slide must continue it exactly as the output does
    nzc = np.argwhere(gi != bgc)
    L = int(nzc[:, 1].max()) + 1
    P = L
    for cand in range(1, L):
        if np.array_equal(gi[:, cand:L], gi[:, :L - cand]):
            P = cand
            break
    ext = np.array([[gi[i, (j - L) % P + L - P] if j >= L else gi[i, j] for j in range(w)] for i in range(h)])
    if not np.array_equal(ext, go):
        return generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=rot)
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O=None, examples=None):
    """Each line of the striped block is its own repeating sequence: copy the
    line's last period and lay it down again and again until the line reaches the
    far edge.  Lines are handled one at a time; only I is read."""
    A = np.asarray(I, dtype=int)
    H, W = A.shape
    full = [0, 0, H - 1, W - 1]

    # canonical frame: the quarter turn that leaves top row and right column pure
    # background stands the block in the bottom-left corner, lines running rightward
    kk = None
    for k in (0, 3, 2, 1):
        C = np.rot90(A, k)
        if len(set(C[0].tolist())) == 1 and len(set(C[:, -1].tolist())) == 1:
            kk = k
            break
    if kk is None:
        return [34], [full]
    C = np.rot90(A, kk)
    idx = np.rot90(np.arange(H * W).reshape(H, W), kk)
    HC, WC = C.shape
    bgc = int(C[0, -1])

    def pos(i, j):
        return divmod(int(idx[i, j]), W)

    nz = np.argwhere(C != bgc)
    if len(nz) == 0:
        return [34], [full]
    c1 = int(nz[:, 1].max())
    rows = sorted(set(int(r) for r in nz[:, 0]))
    L = c1 + 1  # block spans canonical columns 0..c1

    # the block's period: shortest slide along the lines mapping the block onto itself
    P = L
    for cand in range(1, L):
        if all(np.array_equal(C[rows, j], C[rows, j - cand]) for j in range(cand, L)):
            P = cand
            break

    grid = A.copy()
    ops, sels = [], []
    clip_rect = None
    for r in rows:
        seg = C[r, :L]
        if np.all(seg == bgc):
            continue
        # this line's own period (a divisor of the block period, so it continues
        # the same way the block does)
        q = P
        for d in range(1, P + 1):
            if P % d == 0 and all(seg[j] == seg[j - d] for j in range(d, L)):
                q = d
                break
        zeros = []
        start = L
        while start < WC:
            n = min(q, WC - start)
            src = [pos(r, c1 - q + 1 + t) for t in range(n)]
            dst = [pos(r, start + t) for t in range(n)]
            vals = [int(A[a, b]) for a, b in src]
            rs = [a for a, _ in src]; cs = [b for _, b in src]
            rect = [min(rs), min(cs), max(rs) - min(rs), max(cs) - min(cs)]
            after = grid.copy()
            for (a, b), v in zip(dst, vals):
                if v != 0:
                    after[a, b] = v
            if not np.array_equal(after, grid):
                if clip_rect != rect:
                    ops.append(28); sels.append(rect)  # full rectangle: the line's last period
                    clip_rect = rect
                dr = min(a for a, _ in dst); dc = min(b for _, b in dst)
                ops.append(30); sels.append([dr, dc, 0, 0])
                grid = after
            zeros += [d for d, v in zip(dst, vals) if v == 0 and grid[d] != 0]
            start += q
        # clipboard skips colour 0: black the line's 0-cells the pastes left standing
        if zeros:
            ops.append(0); sels.append(sel_of(zeros))
            for a, b in zeros:
                grid[a, b] = 0
    ops.append(34); sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
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
                                f"for task d8c310e9"
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
                    f"Failed to build a complete episode for task d8c310e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d8c310e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
