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
import random
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def _unif(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    return random.randint(min(lo, hi), max(lo, hi))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, linc=1) -> dict:
    big = min(30, max(max_h, max_w))
    for _ in range(200):
        w = _unif(diff_lb, diff_ub, (2, min(10, big - 1)))
        h = _unif(diff_lb, diff_ub, (w + 1, big))
        gi = np.full((h, w), bgc, dtype=int)
        gi[h - 1, 0] = linc
        go = gi.copy()
        r, c, d = h - 1, 0, 1
        while True:
            r, c = r - 1, c + d
            if r < 0:
                break
            if c == w - 1 or c == 0:
                d *= -1
            go[r, c] = linc
        mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T,
               np.fliplr, np.flipud, lambda g: np.rot90(g, 3),
               lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
        for fn in random.sample(mfs, random.choice((1, 2))):
            gi = np.ascontiguousarray(fn(gi))
            go = np.ascontiguousarray(fn(go))
        gix, gox = gi.copy(), go.copy()
        n = _unif(diff_lb, diff_ub, (1, 4))
        tfs = [np.flipud, np.fliplr, lambda g: np.flipud(np.fliplr(g))]
        for k in range(n - 1):
            gi[tfs[k](gix) == linc] = linc
            go[tfs[k](gox) == linc] = linc
        go[go == bgc] = 8
        if gi.shape[0] <= max_h and gi.shape[1] <= max_w:
            return {"input": gi.tolist(), "output": go.tolist()}
    raise ValueError("no fit")


def _trace(H, W, s):
    # ray from corner marker s along the long axis, bouncing between the
    # short-axis walls; returned as a list of straight strokes
    tr = W > H
    if tr:
        H, W = W, H
        s = (s[1], s[0])
    r, c = s
    dr = -1 if r == H - 1 else 1
    dc = 1 if c == 0 else -1
    strokes, cur = [], []
    while True:
        r, c = r + dr, c + dc
        if not (0 <= r < H):
            break
        cur.append((r, c))
        if c == 0 or c == W - 1:
            strokes.append(cur)
            cur = []
            dc = -dc
    if cur:
        strokes.append(cur)
    if tr:
        strokes = [[(b, a) for a, b in st] for st in strokes]
    return strokes


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    bseed = (1, 0) if H >= 3 else (0, 1)          # never a corner -> background
    bgc = int(I[bseed])
    fg = [int(v) for v in np.unique(I) if int(v) != bgc]
    linc = fg[0]
    seeds = sorted({(r, c) for r in (0, H - 1) for c in (0, W - 1) if I[r, c] == linc})

    ops, sels = [], []
    cur = I.copy()
    # 1) shoot each marker's ray, one straight stroke per bounce
    for s in seeds:
        for st in _trace(H, W, s):
            cells = [p for p in st if cur[p] != linc]
            if not cells:
                continue
            ops.append(linc)
            sels.append(sel_of(cells))
            for p in cells:
                cur[p] = linc
    # 2) the remaining background (all cells of the old background colour) becomes teal
    bg = [(r, c) for r in range(H) for c in range(W) if cur[r, c] == bgc]
    if bg:
        ops.append(8)
        sels.append(sel_of(bg))
        for p in bg:
            cur[p] = 8
    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])                # full-grid bbox for Submit
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
