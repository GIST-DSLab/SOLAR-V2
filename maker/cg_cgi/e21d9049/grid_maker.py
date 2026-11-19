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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def _unifint(lb, ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    return min(max(a, round(a + (b - a) * d)), b)


def _bg(I):
    return Counter(np.asarray(I).flatten().tolist()).most_common(1)[0][0]


def _segment(idx, cross):
    """The run's span along its line: the non-bg cells, minus a lone crossing cell."""
    idx = sorted(idx)
    if idx[-1] - idx[0] + 1 == len(idx):
        return idx[0], idx[-1]
    rest = [i for i in idx if i != cross]
    return rest[0], rest[-1]


def _read_runs(I):
    I = np.asarray(I, dtype=int)
    bgc = _bg(I)
    fg = I != bgc
    cj = int(np.argmax(fg.sum(axis=0)))
    ri = int(np.argmax(fg.sum(axis=1)))
    vr0, vr1 = _segment([r for r in range(I.shape[0]) if fg[r, cj]], ri)
    hc0, hc1 = _segment([c for c in range(I.shape[1]) if fg[ri, c]], cj)
    return cj, vr0, vr1, ri, hc0, hc1


def _tiled(I):
    I = np.asarray(I, dtype=int)
    G = I.copy()
    h, w = I.shape
    cj, vr0, vr1, ri, hc0, hc1 = _read_runs(I)
    n = vr1 - vr0 + 1
    for r in range(h):
        G[r, cj] = I[vr0 + (r - vr0) % n, cj]
    m = hc1 - hc0 + 1
    for c in range(w):
        G[ri, c] = I[ri, hc0 + (c - hc0) % m]
    return G


def sample_colors(num_examples=None) -> dict:
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    remcols = [c for c in range(10) if c != bgc]
    for _ in range(200):
        h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
        w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
        ph = _unifint(diff_lb, diff_ub, (2, 9))
        pw = _unifint(diff_lb, diff_ub, (2, 9))
        locih = random.randint(0, h - ph)
        locjh = random.randint(0, w - 1)
        locjw = random.randint(0, w - pw)
        lociw = random.randint(0, h - 1)
        vbar = {(locih + k, locjh): random.choice(remcols) for k in range(ph)}
        hbar = {(lociw, locjw + k): random.choice(remcols) for k in range(pw)}
        col = random.choice(remcols)
        for (i, j) in list(vbar):
            if abs(i - lociw) % ph == 0:
                vbar[(i, j)] = col
        for (i, j) in list(hbar):
            if abs(j - locjh) % pw == 0:
                hbar[(i, j)] = col
        gi = np.full((h, w), bgc, dtype=int)
        for (i, j), v in list(vbar.items()) + list(hbar.items()):
            gi[i, j] = v
        go = gi.copy()
        for k in range(h // ph + 1):
            for d in (k * ph, -k * ph):
                for (i, j), v in vbar.items():
                    if 0 <= i + d < h:
                        go[i + d, j] = v
        for k in range(w // pw + 1):
            for d in (k * pw, -k * pw):
                for (i, j), v in hbar.items():
                    if 0 <= j + d < w:
                        go[i, j + d] = v
        if gi.tolist() == go.tolist():
            continue
        try:
            if not np.array_equal(_tiled(gi), go):
                continue          # ambiguous layout (run touching a foreign crossing cell)
        except Exception:
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O=None, examples=None):
    """Reading taken from the golf entry: each line is a periodic sequence whose
    period is the run's own length, so every cell of the line takes the colour of
    the run cell in the same RESIDUE CLASS (position mod period).  The unit of work
    is therefore one residue class -- "every n-th cell of this column/row, in step
    with run cell i" -- painted with that run cell's colour by one Color op.
    Colour 0 needs no special handling: Color0 paints it like any other colour.
    O is never inspected."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    cj, vr0, vr1, ri, hc0, hc1 = _read_runs(I)
    G = I.copy()
    ops, sels = [], []

    lines = (
        # vertical run: column cj, period = its length
        (lambda p: (p, cj), vr0, vr1, h),
        # horizontal run: row ri, period = its length
        (lambda p: (ri, p), hc0, hc1, w),
    )
    for at, s0, s1, L in lines:
        n = s1 - s0 + 1
        for i in range(n):
            src = at(s0 + i)
            v = int(I[src])
            # the residue class of run cell i along the line, minus cells
            # already showing that colour (e.g. the run cell itself)
            cls = [at(p) for p in range(L) if (p - (s0 + i)) % n == 0]
            cells = [q for q in cls if G[q] != v]
            if not cells:
                continue
            ops.append(v)
            sels.append(sel_of(cells))
            for q in cells:
                G[q] = v

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # Submit: whole grid
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
