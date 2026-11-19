"""
ARC Task: e21d9049 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    return {"bgc": random.choice(range(10))}


def _segments_ok(gi, go, bgc):
    try:
        return np.array_equal(_tile(np.asarray(gi)), np.asarray(go))
    except Exception:
        return False


def generate(diff_lb, diff_ub, max_h, max_w, bgc, **kw) -> dict:
    def unifint(lo, hi):
        a, b = lo, hi
        d = random.uniform(diff_lb, diff_ub)
        return min(b, max(a, int(round(a + (b - a) * d))))
    cols = list(range(10))
    remcols = [c for c in cols if c != bgc]
    for _ in range(200):
        h = unifint(10, max(10, max_h))
        w = unifint(10, max(10, max_w))
        ph = unifint(2, 9)
        pw = unifint(2, 9)
        locih = random.randint(0, h - ph); locjh = random.randint(0, w - 1)
        locjw = random.randint(0, w - pw); lociw = random.randint(0, h - 1)
        col = random.choice(remcols)
        hbar = {(locih + k, locjh): random.choice(remcols) for k in range(ph)}
        wbar = {(lociw, locjw + k): random.choice(remcols) for k in range(pw)}
        for (i, j) in hbar:
            if abs(i - lociw) % ph == 0:
                hbar[(i, j)] = col
        for (i, j) in wbar:
            if abs(j - locjh) % pw == 0:
                wbar[(i, j)] = col
        gi = [[bgc] * w for _ in range(h)]
        for d in (hbar, wbar):
            for (i, j), v in d.items():
                gi[i][j] = v
        go = [row[:] for row in gi]
        for k in range(h // ph + 1):
            for s in (k * ph, -k * ph):
                for (i, j), v in hbar.items():
                    if 0 <= i + s < h:
                        go[i + s][j] = v
        for k in range(w // pw + 1):
            for s in (k * pw, -k * pw):
                for (i, j), v in wbar.items():
                    if 0 <= j + s < w:
                        go[i][j + s] = v
        # keep only instances where the two runs read unambiguously from the input
        if _segments_ok(gi, go, bgc):
            return {"input": gi, "output": go}
    return {"input": gi, "output": go}


def _read(I):
    """bgc, and the two runs: (column, row_start, row_end) and (row, col_start, col_end)."""
    vals, cnts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(cnts)])
    fg = I != bgc
    cj = int(np.argmax(fg.sum(axis=0)))
    ri = int(np.argmax(fg.sum(axis=1)))

    def longest(line):
        best, cur = None, None
        for p, on in enumerate(list(line) + [False]):
            if on and cur is None:
                cur = p
            elif not on and cur is not None:
                if best is None or p - cur > best[1] - best[0] + 1:
                    best = (cur, p - 1)
                cur = None
        return best
    v0, v1 = longest(fg[:, cj])
    h0, h1 = longest(fg[ri, :])
    return bgc, (cj, v0, v1), (ri, h0, h1)


def _tile(I):
    I = np.asarray(I, dtype=int)
    _, (cj, v0, v1), (ri, h0, h1) = _read(I)
    G = I.copy()
    n = v1 - v0 + 1
    for r in range(I.shape[0]):
        G[r, cj] = I[v0 + (r - v0) % n, cj]
    n = h1 - h0 + 1
    for c in range(I.shape[1]):
        G[ri, c] = I[ri, h0 + (c - h0) % n]
    return G


def derive_operations(I, O=None, examples=None):
    """Reading: each run is a line that grows by copying ITSELF.  The unit copied
    is the whole stretch of the line laid down so far (not the original run), and
    the shift is the largest multiple of the run's period that the stretch spans,
    so the line roughly doubles with every CopyO/Paste until it hits the edge --
    first forward (down / right), then backward (up / left).  Everything is read
    from I; O is never inspected.  0-coloured run cells cannot ride a paste, so the
    cells of each landing zone whose source cell is 0 get a Color0 right after."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    _, (cj, v0, v1), (ri, h0, h1) = _read(I)
    G = I.copy()
    ops, sels = [], []

    def grow(axis, fixed, a, b, L):
        n = b - a + 1
        at = (lambda p: (p, fixed)) if axis == 0 else (lambda p: (fixed, p))

        def lay(src, dst0):
            # copy the stretch src (positions on the line), land it starting at dst0
            vals = [int(G[at(p)]) for p in src]
            dst = [dst0 + i for i in range(len(src))]
            pasted = [(d, v) for d, v in zip(dst, vals) if v != 0 and G[at(d)] != v]
            if pasted:
                ops.append(29); sels.append(sel_of([at(p) for p in src]))     # CopyO the grown line
                ops.append(30); sels.append(sel_of([at(dst0)]))               # Paste one shift further
                for d, v in zip(dst, vals):
                    if v != 0:
                        G[at(d)] = v
            zeros = [at(d) for d, v in zip(dst, vals) if v == 0 and G[at(d)] != 0]
            if zeros:
                ops.append(0); sels.append(sel_of(zeros))                     # the 0s a paste cannot carry
                for cell in zeros:
                    G[cell] = 0

        # forward: down / right
        while b < L - 1:
            S = ((b - a + 1) // n) * n
            hi_src = min(b, L - 1 - S)
            lay(list(range(b + 1 - S, hi_src + 1)), b + 1)
            b = min(b + S, L - 1)
        # backward: up / left
        while a > 0:
            S = ((b - a + 1) // n) * n
            if a - S >= 0:
                lay(list(range(a, a + S)), a - S)
                a -= S
            else:
                lay(list(range(S, S + a)), 0)
                a = 0

    grow(0, cj, v0, v1, h)
    grow(1, ri, h0, h1, w)
    ops.append(34); sels.append([0, 0, h - 1, w - 1])   # whole grid
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
                        f"num_examples+1 ({num_examples + 1}) for task e21d9049"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e21d9049"
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
                                f"for task e21d9049"
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
                    f"Failed to build a complete episode for task e21d9049 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e21d9049-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
