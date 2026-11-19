"""
ARC Task: 5c2c9af4 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    lo_i, hi_i = min(lo_i, hi_i), max(lo_i, hi_i)
    lo_i = max(a, lo_i)
    hi_i = min(b, hi_i)
    if lo_i > hi_i:
        lo_i = hi_i
    return random.randint(lo_i, hi_i)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kw) -> dict:
    if bgc is None:
        bgc = random.randint(0, 9)
    if fgc is None:
        fgc = random.choice([c for c in range(10) if c != bgc])
    h = _unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = _unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    boxhd = _unifint(diff_lb, diff_ub, (0, h // 2))
    boxwd = _unifint(diff_lb, diff_ub, (0, w // 2))
    boxh = random.choice((boxhd, h - boxhd))
    boxw = random.choice((boxwd, w - boxwd))
    if boxh % 2 == 0:
        boxh = random.choice((boxh - 1, boxh + 1))
    if boxw % 2 == 0:
        boxw = random.choice((boxw - 1, boxw + 1))
    boxh = min(max(1, boxh), h if h % 2 == 1 else h - 1)
    boxw = min(max(1, boxw), w if w % 2 == 1 else w - 1)
    loci = random.randint(0, h - boxh)
    locj = random.randint(0, w - boxw)
    cpi = loci + boxh // 2
    cpj = locj + boxw // 2
    A = (loci, locj)
    B = (loci + boxh - 1, locj + boxw - 1)
    cp = (cpi, cpj)
    pts = {A, B, cp}
    gi = np.full((h, w), bgc, dtype=int)
    for (r, c) in pts:
        gi[r, c] = fgc
    go = gi.copy()
    if len(pts) == 3 and len({p[0] for p in pts}) == 1:
        go[cpi, :] = fgc
    elif len(pts) == 3 and len({p[1] for p in pts}) == 1:
        go[:, cpj] = fgc
    else:
        k = 1
        while True:
            f1 = k * (boxh // 2)
            f2 = k * (boxw // 2)
            t, b, l, r = cpi - f1, cpi + f1, cpj - f2, cpj + f2
            go2 = go.copy()
            for rr in range(t, b + 1):
                for cc in range(l, r + 1):
                    if (rr in (t, b) or cc in (l, r)) and 0 <= rr < h and 0 <= cc < w:
                        go2[rr, cc] = fgc
            changed = not np.array_equal(go, go2)
            go = go2
            k += 1
            if not changed:
                break
    if random.choice((True, False)):
        gi = gi.T.copy()
        go = go.T.copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # Marker colour = the rare colour (3 markers, or 1 in the degenerate case).
    cnt = Counter(I.flatten().tolist())
    fgc = min(cnt.keys(), key=lambda c: (cnt[c], c))
    pts = [(int(r), int(c)) for r, c in zip(*np.where(I == fgc))]

    G = I.copy()

    def paint(r0, c0, r1, c1):
        # one straight edge segment (a full row/column run) of a ring
        if r0 > r1 or c0 > c1:
            return
        if np.all(G[r0:r1 + 1, c0:c1 + 1] == fgc):
            return
        G[r0:r1 + 1, c0:c1 + 1] = fgc
        ops.append(int(fgc))
        sels.append(sel_of([(r, c) for r in range(r0, r1 + 1)
                            for c in range(c0, c1 + 1)]))

    if len(pts) >= 2:
        ul_i = min(p[0] for p in pts); lr_i = max(p[0] for p in pts)
        ul_j = min(p[1] for p in pts); lr_j = max(p[1] for p in pts)
        cpi, cpj = (ul_i + lr_i) // 2, (ul_j + lr_j) // 2   # centre marker
        f1, f2 = (lr_i - ul_i) // 2, (lr_j - ul_j) // 2     # ring step (centre->corner)

        if f1 == 0 and f2 == 0:
            pass
        elif f1 == 0:
            paint(cpi, 0, cpi, w - 1)          # markers on one row -> whole row
        elif f2 == 0:
            paint(0, cpj, h - 1, cpj)          # markers on one column -> whole column
        else:
            # rings at (scaled) Chebyshev distance k*step from the centre, outward,
            # each ring drawn as its four edges, until a ring lies fully off-grid
            k = 1
            while True:
                t, b = cpi - k * f1, cpi + k * f1
                l, r = cpj - k * f2, cpj + k * f2
                cl, cr = max(0, l), min(w - 1, r)
                rt, rb = max(0, t + 1), min(h - 1, b - 1)
                inside = False
                if 0 <= t < h and cl <= cr:
                    inside = True; paint(t, cl, t, cr)
                if 0 <= b < h and cl <= cr:
                    inside = True; paint(b, cl, b, cr)
                if 0 <= l < w and rt <= rb:
                    inside = True; paint(rt, l, rb, l)
                if 0 <= r < w and rt <= rb:
                    inside = True; paint(rt, r, rb, r)
                if not inside:
                    break
                k += 1

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 5c2c9af4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5c2c9af4"
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
                                f"for task 5c2c9af4"
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
                    f"Failed to build a complete episode for task 5c2c9af4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5c2c9af4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
