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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The rule only depends on the runs' structure; their colours are free.
    return {"bgc": random.choice(range(10))}


def _unif(diff_lb, diff_ub, lo, hi):
    hi = max(hi, lo)
    a = random.uniform(min(diff_lb, diff_ub), max(diff_lb, diff_ub))
    a = min(max(a, 0.0), 1.0)
    return lo + int(round(a * (hi - lo)))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kw) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    remcols = [c for c in range(10) if c != bgc]
    while True:
        h = _unif(diff_lb, diff_ub, 10, max(10, max_h))
        w = _unif(diff_lb, diff_ub, 10, max(10, max_w))
        ph = _unif(diff_lb, diff_ub, 2, 9)
        pw = _unif(diff_lb, diff_ub, 2, 9)
        locih = random.randint(0, h - ph)
        locjh = random.randint(0, w - 1)
        locjw = random.randint(0, w - pw)
        lociw = random.randint(0, h - 1)
        in_v = locih <= lociw < locih + ph
        in_h = locjw <= locjh < locjw + pw
        # keep the two runs separable: a bar that does not contain the crossing
        # cell must not end right next to it when the other bar does contain it
        if in_v and not in_h and (locjw + pw == locjh or locjw == locjh + 1):
            continue
        if in_h and not in_v and (locih + ph == lociw or locih == lociw + 1):
            continue
        break
    col = random.choice(remcols)
    vbar = {}
    for k in range(ph):
        r = locih + k
        vbar[(r, locjh)] = col if abs(r - lociw) % ph == 0 else random.choice(remcols)
    hbar = {}
    for k in range(pw):
        c = locjw + k
        hbar[(lociw, c)] = col if abs(c - locjh) % pw == 0 else random.choice(remcols)
    gi = [[bgc] * w for _ in range(h)]
    for (r, c), v in list(vbar.items()) + list(hbar.items()):
        gi[r][c] = v
    go = [row[:] for row in gi]
    for (r, c), v in vbar.items():
        for rr in range(h):
            if (rr - r) % ph == 0:
                go[rr][c] = v
    for (r, c), v in hbar.items():
        for cc in range(w):
            if (cc - c) % pw == 0:
                go[r][cc] = v
    return {"input": tuple(map(tuple, gi)), "output": tuple(map(tuple, go))}


def _segments(vals, bgc):
    segs, cur = [], []
    for i, v in enumerate(vals):
        if v != bgc:
            cur.append(i)
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return segs


def derive_operations(I, O=None, examples=None):
    """Reading (residue classes, not tiles): a line that carries a run is
    periodic with period = run length n.  Every cell of that line belongs to
    the residue class (index mod n) of exactly one run cell and takes its
    colour.  So: one Color op per run cell, selecting "every n-th cell of this
    line through that run cell".  O is never inspected."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # background: the colour every demonstration input (and I) is drawn on
    cnt = Counter(I.ravel().tolist())
    if examples:
        for ei, _ in examples:
            cnt.update(np.asarray(ei, dtype=int).ravel().tolist())
    bgc = cnt.most_common(1)[0][0]

    nb = I != bgc
    cj = int(np.argmax(nb.sum(axis=0)))   # the only column holding >= 2 run cells
    ri = int(np.argmax(nb.sum(axis=1)))   # the only row holding >= 2 run cells

    def run(vals):
        segs = [s for s in _segments(vals, bgc) if len(s) >= 2]
        return max(segs, key=len)

    vrun = run(I[:, cj].tolist())         # the vertical run in its column
    hrun = run(I[ri, :].tolist())         # the horizontal run in its row

    g = I.copy()
    ops, sels = [], []
    lines = [
        (vrun, h, lambda p: (p, cj)),     # the column first ...
        (hrun, w, lambda p: (ri, p)),     # ... then the row
    ]
    for seg, L, at in lines:
        n = len(seg)
        for p in seg:
            v = int(I[at(p)])
            cls = [at(q) for q in range(p % n, L, n)]   # every n-th cell of the line
            if all(g[c] == v for c in cls):
                continue                                  # would change nothing now
            for c in cls:
                g[c] = v
            ops.append(v)
            sels.append(sel_of(cls))
    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])     # bbox == the whole grid, exactly the cells meant
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
