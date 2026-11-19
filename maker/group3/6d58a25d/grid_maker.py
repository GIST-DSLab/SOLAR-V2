"""
ARC Task: 6d58a25d (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of

_SHP_RAW = [(0, 0), (1, 0), (1, 1), (1, -1), (2, -1), (2, -2), (2, 1), (2, 2), (3, 3), (3, -3)]
_mc = min(c for _, c in _SHP_RAW)
CANON_SHAPE = frozenset((r, c - _mc) for r, c in _SHP_RAW)


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    v = a + int(round((b - a) * d))
    return max(a, min(b, v))


def _norm(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    c1 = random.choice(rem)
    c2 = random.choice([c for c in rem if c != c1])
    return {"bgc": bgc, "c1": c1, "c2": c2}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, c1=1, c2=2) -> dict:
    rot = random.choice([0, 1, 2, 3])
    if rot in (1, 3):
        hb, wb = min(30, max_w), min(30, max_h)
    else:
        hb, wb = min(30, max_h), min(30, max_w)
    if hb < 5 or wb < 8:
        rot = 0
        hb, wb = min(30, max_h), min(30, max_w)
    h = _unifint(diff_lb, diff_ub, (5, max(5, hb)))
    w = _unifint(diff_lb, diff_ub, (8, max(8, wb)))
    g = np.full((h, w), bgc, dtype=int)
    loci = random.randint(0, h - 4)
    locj = random.randint(0, w - 7)
    plcd = {(r + loci, c + locj) for r, c in CANON_SHAPE}
    rem = [(i, j) for i in range(h) for j in range(w) if (i, j) not in plcd]
    nn = _unifint(diff_lb, diff_ub, (1, max(1, len(rem) // 2 - 1)))
    nois = random.sample(rem, nn)
    for (i, j) in nois:
        g[i, j] = c2
    for (i, j) in plcd:
        g[i, j] = c1
    go = g.copy()
    for (i, j) in nois:
        colcells = [r for r, c in plcd if c == j]
        if colcells and any(r < i for r in colcells):
            for r in range(max(colcells) + 1, h):
                go[r, j] = c2
    gi = np.rot90(g, rot)
    go = np.rot90(go, rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Rule (from I alone): the fixed V template is drawn in c1, and the scattered noise is c2.
    A template column fires when some noise cell lies beyond the template's first cell
    in that column. Each firing column is filled with c2 from just past the template
    to the grid edge.
    Unit of work: an advancing WAVEFRONT. Instead of one ray per column, the fill is laid
    one line at a time across the ray direction, starting at the template's open side and
    sweeping toward the edge. Each op paints that line's cells that are in firing columns,
    past their column's start, and not already c2.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    flat_idx = np.arange(hi * wi).reshape(hi, wi)

    # find the orientation that puts the template in its canonical pose
    found = None
    for k in range(4):
        J = np.rot90(I, k)
        Jidx = np.rot90(flat_idx, k)
        for col in sorted(set(J.flatten().tolist())):
            if col == bgc:
                continue
            cells = [(int(r), int(c)) for r, c in zip(*np.where(J == col))]
            if len(cells) == len(CANON_SHAPE) and _norm(cells) == CANON_SHAPE:
                found = (J, Jidx, col, cells)
                break
        if found:
            break
    if found is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    J, Jidx, shape_col, shape_cells = found
    hJ, wJ = J.shape
    others = [c for c in sorted(set(J.flatten().tolist())) if c not in (bgc, shape_col)]
    if not others:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    noise_col = others[0]

    # firing columns and where each one's fill starts (just past the template)
    by_col = {}
    for r, c in shape_cells:
        by_col.setdefault(c, []).append(r)
    starts = {}
    for j, rs in by_col.items():
        if any(J[i, j] == noise_col for i in range(min(rs), hJ)):
            starts[j] = max(rs) + 1

    # sweep the front line by line, away from the template toward the edge
    if starts:
        for i in range(min(starts.values()), hJ):
            line = [(i, j) for j in sorted(starts)
                    if i >= starts[j] and J[i, j] != noise_col]
            if not line:
                continue
            pts = [divmod(int(Jidx[r, c]), wi) for r, c in line]
            ops.append(int(noise_col)); sels.append(sel_of(pts))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6d58a25d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6d58a25d"
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
                                f"for task 6d58a25d"
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
                    f"Failed to build a complete episode for task 6d58a25d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6d58a25d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
