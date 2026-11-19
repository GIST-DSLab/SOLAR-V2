"""
ARC Task: bd4472b8 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # fallback: mask given as explicit cell list
    def sel_of(cells):
        return {"cells": [(int(r), int(c)) for r, c in cells]}


ROTS = ["identity", "rot90", "rot180", "rot270"]


# ---------------------------------------------------------------- helpers ---
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    ilb = a + int((b - a) * diff_lb)
    iub = a + int((b - a) * diff_ub)
    if ilb > iub:
        ilb, iub = iub, ilb
    ilb = max(a, min(b, ilb))
    iub = max(a, min(b, iub))
    return random.randint(ilb, iub)


def _rot_cw(g):
    return [list(r) for r in zip(*g[::-1])]


def _rot_ccw(g):
    return [list(r) for r in zip(*g)][::-1]


def _rot180(g):
    return [list(r[::-1]) for r in g[::-1]]


def _apply_rot(g, rot):
    if rot == "identity":
        return [list(r) for r in g]
    if rot == "rot90":
        return _rot_ccw(g)
    if rot == "rot180":
        return _rot180(g)
    return _rot_cw(g)  # rot270


# ---------------------------------------------------------- sample_colors ---
def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    linc = random.choice(rem)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


# --------------------------------------------------------------- generate ---
def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, rot=None, **kwargs) -> dict:
    if rot is None:
        rot = random.choice(ROTS)

    # base grid is (h + 2, w); rot90/rot270 swap the axes
    if rot in ("identity", "rot180"):
        h_hi = min(28, max_h - 2)
        w_hi = min(8, max_w)
    else:
        h_hi = min(28, max_w - 2)
        w_hi = min(8, max_h)
    h_hi = max(1, h_hi)
    w_hi = max(2, w_hi)

    h = _unifint(diff_lb, diff_ub, (1, h_hi))
    w = _unifint(diff_lb, diff_ub, (2, w_hi))

    cols = list(range(1, 10))
    remcols = [c for c in cols if c != bgc]
    ccols = random.sample(remcols, w)

    gi = [list(ccols), [linc] * w] + [[bgc] * w for _ in range(h)]
    go = [row[:] for row in gi]
    for p in range(2, h + 2):
        go[p] = [ccols[(p - 2) % w]] * w

    gi = _apply_rot(gi, rot)
    go = _apply_rot(go, rot)
    return {"input": gi, "output": go}


# ------------------------------------------------------- derive_operations ---
def _edge_layout(H, W, edge):
    """Return (L, line_cells(p), along_ordered_cells(p)).

    line p = the grid line at distance p from `edge`, moving inward ("away").
    The along direction is the away direction rotated counter-clockwise, which
    is the handedness the strip/band correspondence keeps under any rotation.
    """
    if edge == "top":       # away = down, along = right
        L = H
        line = lambda p: [(p, c) for c in range(W)]
    elif edge == "bottom":  # away = up, along = left
        L = H
        line = lambda p: [(H - 1 - p, c) for c in range(W - 1, -1, -1)]
    elif edge == "left":    # away = right, along = up
        L = W
        line = lambda p: [(r, p) for r in range(H - 1, -1, -1)]
    else:                   # right: away = left, along = down
        L = W
        line = lambda p: [(r, W - 1 - p) for r in range(H)]
    return L, line


def _find_strip_edge(I, strict=True):
    H, W = I.shape
    for edge in ("top", "bottom", "left", "right"):
        L, line = _edge_layout(H, W, edge)
        if L < 3:
            continue
        strip = [int(I[r, c]) for r, c in line(0)]
        if len(strip) < 2:
            continue
        if strict:
            if len(set(strip)) != len(strip):
                continue
        else:
            if len(set(strip)) == 1:
                continue
        lin = [int(I[r, c]) for r, c in line(1)]
        if len(set(lin)) != 1:
            continue
        rest = set()
        for p in range(2, L):
            vals = {int(I[r, c]) for r, c in line(p)}
            if len(vals) != 1:
                rest = None
                break
            rest |= vals
        if rest is None or len(rest) != 1:
            continue
        bgc = next(iter(rest))
        if bgc == lin[0]:
            continue
        return edge, L, line, strip
    return None


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape

    ops, sels = [], []

    found = _find_strip_edge(I, strict=True) or _find_strip_edge(I, strict=False)

    if found is not None:
        edge, L, line, strip = found
        w = len(strip)
        # Draw the strip's colours as bands across the grid, starting at the
        # line just past the marker line and marching away from that edge,
        # cycling through the strip until the grid is filled.
        for p in range(2, L):
            colour = strip[(p - 2) % w]
            cells = line(p)
            ops.append(int(colour))
            sels.append(sel_of(cells))
    else:
        # Fallback: repaint each fully-changed line of the grid.
        for r in range(H):
            if not np.array_equal(I[r], O[r]) and len(set(O[r].tolist())) == 1:
                ops.append(int(O[r, 0]))
                sels.append(sel_of([(r, c) for c in range(W)]))

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])  # full-grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task bd4472b8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task bd4472b8"
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
                                f"for task bd4472b8"
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
                    f"Failed to build a complete episode for task bd4472b8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"bd4472b8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
