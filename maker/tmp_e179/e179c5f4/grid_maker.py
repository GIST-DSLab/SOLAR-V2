"""
ARC Task: e179c5f4 (RE-ARC) — LLM-generated grid_maker
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
import math
import random
from collections import Counter

import numpy as np

from maker.sel_helpers import sel_of

# Discrete structural variants: grid orientation (tall = ray steps row-by-row,
# wide = ray steps column-by-column) x number of emitting corners (1..4).
VARIANTS = [
    {"transpose": False, "numlins": 1},
    {"transpose": True,  "numlins": 2},
    {"transpose": False, "numlins": 4},
    {"transpose": True,  "numlins": 3},
]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]          # 8 is reserved as the output marker
    bgc, linc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
        if n_ex >= 2 and len({e["transpose"] for e in examples}) == 1:
            alt = [v for v in VARIANTS if v["transpose"] != examples[0]["transpose"]]
            examples[-1] = dict(random.choice(alt))
    plan = examples + [dict(random.choice(examples))]   # test case is one of the shown cases
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = int(math.ceil(a + (b - a) * diff_lb))
    hi = int(math.ceil(a + (b - a) * diff_ub))
    lo = max(a, min(lo, b))
    hi = max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _canonical_path(L, S):
    """Bouncing ray from bottom-left (L-1, 0) going up-right, reflecting off side walls."""
    r, c = L - 1, 0
    dc = 1
    path = [(r, c)]
    while True:
        r -= 1
        c += dc
        if not (0 <= r < L and 0 <= c < S):
            break
        path.append((r, c))
        if c == 0 or c == S - 1:
            dc = -dc
    return path


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, transpose=None, numlins=None) -> dict:
    if transpose is None or numlins is None:
        v = random.choice(VARIANTS)
        if transpose is None:
            transpose = v["transpose"]
        if numlins is None:
            numlins = v["numlins"]

    # L = long axis (the axis the ray steps along), S = short axis (the axis it bounces on)
    maxL = min(30, max_w if transpose else max_h)
    maxS = min(30, max_h if transpose else max_w)
    s_ub = min(10, maxS, maxL - 1)
    s_ub = max(2, s_ub)
    S = _unifint(diff_lb, diff_ub, (2, s_ub))
    S = max(2, min(S, maxS, maxL - 1))
    L = _unifint(diff_lb, diff_ub, (S + 1, max(S + 1, maxL)))

    gi = np.full((L, S), bgc, dtype=int)
    go = gi.copy()
    path = _canonical_path(L, S)
    gi[L - 1, 0] = linc
    for (r, c) in path:
        go[r, c] = linc

    if transpose:
        gi = np.ascontiguousarray(gi.T)
        go = np.ascontiguousarray(go.T)
    if random.random() < 0.5:
        gi = np.ascontiguousarray(np.fliplr(gi))
        go = np.ascontiguousarray(np.fliplr(go))
    if random.random() < 0.5:
        gi = np.ascontiguousarray(np.flipud(gi))
        go = np.ascontiguousarray(np.flipud(go))

    gix, gox = gi.copy(), go.copy()
    if numlins > 1:
        gi[np.flipud(gix) == linc] = linc
        go[np.flipud(gox) == linc] = linc
    if numlins > 2:
        gi[np.fliplr(gix) == linc] = linc
        go[np.fliplr(gox) == linc] = linc
    if numlins > 3:
        gi[np.flipud(np.fliplr(gix)) == linc] = linc
        go[np.flipud(np.fliplr(gox)) == linc] = linc

    go[go == bgc] = 8
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    tall = H > W                      # the ray always travels along the long axis
    L, S = (H, W) if tall else (W, H)

    def rc(l, s):                     # (long, short) -> grid (row, col)
        return (l, s) if tall else (s, l)

    # background: a non-corner cell; markers: corners holding the other colour
    bgc = int(I[rc(1, 0)])
    corners = {(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)}
    linc = [int(I[p]) for p in sorted(corners) if int(I[p]) != bgc][0]
    markers = sorted(p for p in corners if int(I[p]) == linc)

    P = 2 * (S - 1)                   # zigzag period along the long axis
    blk = min(P, L)                   # first tile: long indices 0..blk-1

    def ray_in_tile(m):
        ml, ms = (m if tall else (m[1], m[0]))
        cells = []
        for t in range(L):
            l = t if ml == 0 else L - 1 - t
            if l >= blk:
                continue
            k = t % P
            s = k if k <= S - 1 else P - k
            if ms != 0:
                s = S - 1 - s
            cells.append(rc(l, s))
        return cells

    def tile_sel(l0, n):              # full rectangle: long indices l0..l0+n-1, all short
        return [l0, 0, n - 1, W - 1] if tall else [0, l0, H - 1, n - 1]

    ops, sels = [], []
    cur = I.copy()

    # 1) inside the first tile, draw every marker's ray (one stroke per ray)
    for m in markers:
        cells = [p for p in ray_in_tile(m) if cur[p] != linc]
        if cells:
            ops.append(linc); sels.append(sel_of(cells))
            for p in cells:
                cur[p] = linc

    # 2) inside the first tile, the remaining background colour becomes 8
    tile_cells = [rc(l, s) for l in range(blk) for s in range(S)]
    bg_cells = [p for p in tile_cells if cur[p] == bgc]
    if bg_cells:
        ops.append(8); sels.append(sel_of(bg_cells))
        for p in bg_cells:
            cur[p] = 8

    # 3) the finished tile repeats along the long axis: copy it, paste it at every period
    if blk < L:
        if linc == 0:
            # 0 is transparent to Paste: lay a linc base on the rest first, then paste 8s on it
            ops.append(0); sels.append(tile_sel(blk, L - blk))   # full rectangle
            for l in range(blk, L):
                for s in range(S):
                    cur[rc(l, s)] = 0
        copied = None
        for l0 in range(P, L, P):
            n = min(P, L - l0)
            if copied != n:
                ops.append(29); sels.append(tile_sel(0, n))      # CopyO: full tile rectangle
                copied = n
            before = cur.copy()
            for l in range(n):
                for s in range(S):
                    v = cur[rc(l, s)]
                    if v != 0:
                        cur[rc(l0 + l, s)] = v
            if not np.array_equal(before, cur):
                ops.append(30); sels.append([l0, 0, 0, 0] if tall else [0, l0, 0, 0])
            else:
                cur = before

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
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
                                f"for task e179c5f4"
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
                    f"Failed to build a complete episode for task e179c5f4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e179c5f4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
