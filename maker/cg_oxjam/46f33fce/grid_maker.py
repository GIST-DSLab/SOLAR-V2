"""
ARC Task: 46f33fce (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    hub = max(2, min(7, max_h // 4))
    wub = max(2, min(7, max_w // 4))
    h = unifint(diff_lb, diff_ub, (2, hub))
    w = unifint(diff_lb, diff_ub, (2, wub))
    nc = unifint(diff_lb, diff_ub, (0, max(0, (h * w) // 2 - 1)))
    go = [[bgc] * w for _ in range(h)]
    gi = [[bgc] * (2 * w) for _ in range(2 * h)]
    inds = [(i, j) for i in range(h) for j in range(w)]
    locs = random.sample(inds, nc)
    for (i, j) in locs:
        v = random.choice(remcols)
        go[i][j] = v
        gi[2 * i + 1][2 * j + 1] = v
    out = [[go[i // 4][j // 4] for j in range(4 * w)] for i in range(4 * h)]
    return {"input": gi, "output": out}


def derive_operations(I, O=None, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Reading: keep only the odd rows / odd columns of I (the sparse lattice the
    # pixels sit on); every kept column becomes 4 columns wide, every kept row
    # becomes a band of 4 identical rows.  Canvas therefore doubles.
    ho, wo = 2 * hi, 2 * wi

    # Background: the canvas colour every demonstration input is drawn on.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    if examples:
        cnt = Counter()
        for exI, _exO in examples:
            cnt.update(np.asarray(exI, dtype=int).flatten().tolist())
        cnt.update(I.flatten().tolist())
        bgc = cnt.most_common(1)[0][0]

    pixels = [(r, c, int(I[r, c])) for r in range(hi) for c in range(wi)
              if I[r, c] != bgc]

    # Lattice rows of I that carry at least one pixel, top to bottom.
    rows = sorted({r for r, _c, _v in pixels})

    def plan(clear_stale):
        ops, sels = [], []
        # 1. Canvas doubles (full canvas rectangle).
        ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
        # 2. Base layer: the whole canvas becomes background.
        if bgc != 0:
            ops.append(bgc); sels.append([0, 0, ho - 1, wo - 1])   # full canvas
        elif clear_stale and pixels:
            # zero padding already is background; only I's own pixels are stale
            ops.append(0); sels.append(sel_of([(r, c) for r, c, _v in pixels]))
        clip_key = None
        for r in rows:
            k = (r - 1) // 2
            top = 4 * k                      # first row of this lattice row's band
            row_px = [(c, v) for rr, c, v in pixels if rr == r]
            # 3. Horizontal stretch: each pixel on the band's top row becomes a
            #    1x4 strip (its lattice column widened to 4).
            for c, v in row_px:
                j = (c - 1) // 2
                ops.append(v); sels.append([top, 4 * j, 0, 3])
            # 4. Vertical stretch: the finished top row is repeated on the
            #    band's 3 remaining rows (copy the whole row, paste below).
            if any(v != 0 for _c, v in row_px):
                key = tuple(row_px)
                if key != clip_key:          # clipboard already holds this row otherwise
                    ops.append(29); sels.append([top, 0, 0, wo - 1])   # whole row
                    clip_key = key
                for d in (1, 2, 3):
                    ops.append(30); sels.append([top + d, 0, 0, 0])
            # 5. Paste is transparent to 0: a colour-0 strip can't be carried by
            #    the copy, so paint it down the band explicitly.
            if bgc != 0:
                for c, v in row_px:
                    if v == 0:
                        j = (c - 1) // 2
                        ops.append(0); sels.append([top + 1, 4 * j, 2, 3])
        return ops, sels

    ops, sels = plan(True)
    if bgc == 0 and pixels:
        # Clearing I's stale pixels only belongs in the trajectory if some stale
        # pixel would otherwise survive (decided by simulating our own ops on I).
        def sim(ops_, sels_):
            g = np.zeros((30, 30), dtype=int); g[:hi, :wi] = I
            dim = (hi, wi); clip = None
            for op, s in zip(ops_, sels_):
                if isinstance(s, dict):
                    cells = [tuple(x) for x in s["cells"]]
                else:
                    r0, c0, dh, dw = s
                    cells = [(a, b) for a in range(r0, r0 + dh + 1)
                             for b in range(c0, c0 + dw + 1)]
                if op <= 9:
                    for a, b in cells:
                        g[a, b] = op
                elif op == 33:
                    ng = np.zeros_like(g); h_, w_ = s[2] + 1, s[3] + 1
                    ng[:h_, :w_] = g[s[0]:s[0] + h_, s[1]:s[1] + w_]
                    g = ng; dim = (h_, w_)
                elif op == 29:
                    clip = g[s[0]:s[0] + s[2] + 1, s[1]:s[1] + s[3] + 1].copy()
                elif op == 30:
                    ch, cw = clip.shape
                    tgt = g[s[0]:s[0] + ch, s[1]:s[1] + cw]
                    src = clip[:tgt.shape[0], :tgt.shape[1]]
                    m = src != 0
                    tgt[m] = src[m]
            return g[:dim[0], :dim[1]].copy()
        o2, s2 = plan(False)
        if np.array_equal(sim(ops, sels), sim(o2, s2)):
            ops, sels = o2, s2

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 46f33fce"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 46f33fce"
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
                                f"for task 46f33fce"
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
                    f"Failed to build a complete episode for task 46f33fce "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"46f33fce-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
