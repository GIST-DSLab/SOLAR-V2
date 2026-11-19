"""
ARC Task: f8c80d96 (RE-ARC) — LLM-generated grid_maker
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

COLS = [0, 1, 2, 3, 4, 6, 7, 8, 9]


def unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def sample_colors(num_examples=None) -> dict:
    bgc, linc = random.sample(COLS, 2)
    return {"bgc": bgc, "linc": linc}


def _box(ui, uj, li, lj):
    cells = set()
    for j in range(uj, lj + 1):
        cells.add((ui, j)); cells.add((li, j))
    for i in range(ui, li + 1):
        cells.add((i, uj)); cells.add((i, lj))
    return cells


def _rings(h, w, ulci, ulcj, lrci, lrcj, hoffs, woffs):
    """The whole family of concentric boxes, clipped to the grid,
    keeping only the ones that touch it (same list the generator builds)."""
    out = []
    for k in range(max(h, w) // min(hoffs, woffs) + 1):
        b = _box(ulci - hoffs * k, ulcj - woffs * k, lrci + hoffs * k, lrcj + woffs * k)
        b = sorted((i, j) for i, j in b if 0 <= i < h and 0 <= j < w)
        if b:
            out.append(b)
    return out


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    if bgc is None or linc is None:
        bgc, linc = random.sample(COLS, 2)
    for _ in range(100):
        h = unifint(diff_lb, diff_ub, (8, max_h))
        w = unifint(diff_lb, diff_ub, (8, max_w))
        ow = random.randint(1, 3 if h > 10 else 2)
        oh = random.randint(1, 3 if w > 10 else 2)
        loci = random.randint(-oh + 1, h - 1)
        locj = random.randint(-ow + 1, w - 1)
        ulci, ulcj = loci - 1, locj - 1
        lrci, lrcj = loci + oh, locj + ow
        hoffs = random.randint(2, 4 if h > 12 else 3)
        woffs = random.randint(2, 4 if w > 12 else 3)
        lns = _rings(h, w, ulci, ulcj, lrci, lrcj, hoffs, woffs)
        nlns = len(lns)
        if nlns < 2:
            continue
        nmissing = unifint(diff_lb, diff_ub, (0, nlns - 2))
        npresent = nlns - nmissing
        gi = np.full((h, w), bgc, dtype=int)
        go = np.full((h, w), 5, dtype=int)
        for k in range(npresent):
            for i, j in lns[k]:
                gi[i, j] = linc
        for ln in lns:
            for i, j in ln:
                go[i, j] = linc
        # learnability: the visible rings must pin down the family uniquely
        fits = _all_fits(gi, linc, bgc)
        if len({tuple(map(tuple, f[0][f[1]:])) for f in fits}) != 1:
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise ValueError("no unambiguous instance")


def _all_fits(I, linc, bgc):
    """Every box family (innermost box, row step, col step) whose first
    `npresent` rings are exactly the visible linc cells."""
    h, w = I.shape
    L = (I == linc).astype(int)
    nL = int(L.sum())
    if nL == 0 or not np.all((I == linc) | (I == bgc)):
        return []
    R = np.zeros((h, w + 1), int); R[:, 1:] = np.cumsum(L, axis=1)
    C = np.zeros((h + 1, w), int); C[1:, :] = np.cumsum(L, axis=0)

    def box_ok(ui, uj, li, lj):
        # number of in-grid cells of the box if all are linc, else -1
        n = 0
        a, b = max(uj, 0), min(lj, w - 1)
        for i in {ui, li}:
            if 0 <= i < h and a <= b:
                if R[i, b + 1] - R[i, a] != b - a + 1:
                    return -1
                n += b - a + 1
        a, b = max(ui + 1, 0), min(li - 1, h - 1)
        for j in {uj, lj}:
            if 0 <= j < w and a <= b:
                if C[b + 1, j] - C[a, j] != b - a + 1:
                    return -1
                n += b - a + 1
        return n

    fits = []
    for oh in (1, 2, 3):
        for ow in (1, 2, 3):
            for loci in range(-oh + 1, h):
                for locj in range(-ow + 1, w):
                    ulci, ulcj, lrci, lrcj = loci - 1, locj - 1, loci + oh, locj + ow
                    n0 = box_ok(ulci, ulcj, lrci, lrcj)
                    if n0 <= 0:
                        continue
                    for hoffs in (2, 3, 4):
                        for woffs in (2, 3, 4):
                            cnt = n0
                            k = 1
                            while cnt < nL:
                                nk = box_ok(ulci - hoffs * k, ulcj - woffs * k,
                                            lrci + hoffs * k, lrcj + woffs * k)
                                if nk <= 0:
                                    break
                                cnt += nk
                                k += 1
                            if cnt == nL and k >= 2:
                                rings = _rings(h, w, ulci, ulcj, lrci, lrcj, hoffs, woffs)
                                if len(rings) >= k:
                                    fits.append((rings, k))
    return fits


def _fit(I):
    cols = sorted(set(I.flatten().tolist()))
    for linc in cols:
        for bgc in cols:
            if bgc == linc:
                continue
            fits = _all_fits(I, linc, bgc)
            if fits:
                return linc, bgc, fits[0]
    return None


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    linc, bgc, (rings, npresent) = _fit(I)
    # Reading: the visible concentric lines carry a fixed spacing; the family
    # is continued outward, each new ring one (row-step, col-step) further out
    # than the previous one, until the rings leave the grid.
    G = I.copy()
    for k in range(npresent, len(rings)):
        ops.append(int(linc)); sels.append(sel_of(rings[k]))
        for r, c in rings[k]:
            G[r, c] = linc
    # Then every cell still showing the background colour (i.e. off every
    # line, old or new) turns grey.
    rest = [(r, c) for r in range(h) for c in range(w) if G[r, c] == bgc]
    ops.append(5); sels.append(sel_of(rest))
    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f8c80d96"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f8c80d96"
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
                                f"for task f8c80d96"
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
                    f"Failed to build a complete episode for task f8c80d96 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f8c80d96-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
