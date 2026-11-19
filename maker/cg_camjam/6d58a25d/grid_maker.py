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


def _normalize_cells(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


CANON_SHAPE = _normalize_cells(_SHP_RAW)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    c1 = random.choice(rem)
    c2 = random.choice([c for c in rem if c != c1])
    return {"bgc": bgc, "c1": c1, "c2": c2}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = a + (b - a) * max(0.0, min(1.0, diff_lb))
    hi = a + (b - a) * max(0.0, min(1.0, diff_ub))
    v = int(round(lo + (hi - lo) * random.random()))
    return max(a, min(b, v))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, c1=None, c2=None, **kw) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if c1 is None:
        c1 = random.choice([c for c in cols if c != bgc])
    if c2 is None:
        c2 = random.choice([c for c in cols if c not in (bgc, c1)])

    def _fits(kk):
        a, b = (max_h, max_w) if kk % 2 == 0 else (max_w, max_h)
        return a >= 5 and b >= 8
    ks = [kk for kk in range(4) if _fits(kk)]
    if not ks:
        raise ValueError("grid bounds too small for the 4x7 template")
    k = random.choice(ks)                    # rotation applied at the end
    ch_max, cw_max = (max_h, max_w) if k % 2 == 0 else (max_w, max_h)
    ch_max, cw_max = min(30, ch_max), min(30, cw_max)

    h = _unifint(diff_lb, diff_ub, (5, max(5, ch_max)))
    w = _unifint(diff_lb, diff_ub, (8, max(8, cw_max)))
    shp = sorted(CANON_SHAPE)
    loci = random.randint(0, h - 4)
    locj = random.randint(0, w - 7)
    plcd = set((r + loci, c + locj) for r, c in shp)
    gi = np.full((h, w), bgc, dtype=int)
    rem = [(i, j) for i in range(h) for j in range(w) if (i, j) not in plcd]
    nnoise = _unifint(diff_lb, diff_ub, (1, max(1, len(rem) // 2 - 1)))
    nois = random.sample(rem, nnoise)
    for (i, j) in nois:
        gi[i, j] = c2
    for (i, j) in plcd:
        gi[i, j] = c1
    go = gi.copy()
    for (i, j) in nois:
        above = [r for (r, c) in plcd if c == j and r < i]
        if not above:
            continue
        start = max(r for (r, c) in plcd if c == j) + 1
        for r in range(start, h):
            go[r, j] = c2
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Rule (read from I alone):
      * the c1 object is the fixed 'V' template (4 x 7); the other non-background
        colour is scattered noise.
      * view the grid so the template is in canonical pose (point up, arms open
        downward); each template column whose far side holds a noise cell
        shoots ONE solid ray of the noise colour from just past the template's
        last cell in that column to the grid edge.
    The golf reading (find the template colour by its 7-column span, fill each of
    its columns past the last template cell with the colour found below) carves
    the task identically -- one column ray as the unit, columns in template
    order -- so the operations are the same: one Color op per ray.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    full = [(r, c) for r in range(hi) for c in range(wi)]

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    flat_idx = np.arange(hi * wi).reshape(hi, wi)

    found = None
    for k in range(4):
        J = np.rot90(I, k)
        Jidx = np.rot90(flat_idx, k)
        for col in sorted(set(J.flatten().tolist())):
            if col == bgc:
                continue
            cells = [(int(r), int(c)) for r, c in zip(*np.where(J == col))]
            if len(cells) != len(CANON_SHAPE):
                continue
            if _normalize_cells(cells) == CANON_SHAPE:
                found = (J, Jidx, col, cells)
                break
        if found is not None:
            break

    if found is None:
        ops.append(34); sels.append(sel_of(full))
        return ops, sels

    J, Jidx, shape_col, shape_cells = found
    hJ, wJ = J.shape
    noise_cols = [c for c in sorted(set(J.flatten().tolist())) if c not in (bgc, shape_col)]
    if not noise_cols:
        ops.append(34); sels.append(sel_of(full))
        return ops, sels
    noise_col = noise_cols[0]

    by_col = {}
    for (r, c) in shape_cells:
        by_col.setdefault(c, []).append(r)

    for j in sorted(by_col):
        last_r = max(by_col[j])
        start = last_r + 1
        if start > hJ - 1:
            continue
        fires = any(J[i, j] == noise_col for i in range(start, hJ))
        if not fires:
            continue
        cells = [(i, j) for i in range(start, hJ)]
        if all(J[i, j] == noise_col for i, _ in cells):
            continue  # ray already complete in I -> painting would be a no-op
        pts = [divmod(int(Jidx[r, c]), wi) for (r, c) in cells]
        ops.append(int(noise_col))
        sels.append(sel_of(pts))   # the whole ray: one straight segment to the edge

    ops.append(34)
    sels.append(sel_of(full))
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
