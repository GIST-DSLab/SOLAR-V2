"""
ARC Task: c444b776 (RE-ARC) — LLM-generated grid_maker
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
    bgc, linc = random.sample(range(10), 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, round(random.uniform(lo, hi))))
    cols = list(range(10))
    if bgc is None or linc is None:
        bgc, linc = random.sample(cols, 2)
    remcols = [c for c in cols if c not in (bgc, linc)]
    for _ in range(200):
        h = unifint(diff_lb, diff_ub, (2, 9))
        w = unifint(diff_lb, diff_ub, (2, 9))
        nh = unifint(diff_lb, diff_ub, (1, 3))
        nw = unifint(diff_lb, diff_ub, (1 if nh > 1 else 2, 3))
        fullh = h * nh + (nh - 1)
        fullw = w * nw + (nw - 1)
        if fullh <= max_h and fullw <= max_w:
            break
    else:
        h, w, nh, nw = 2, 2, 1, 2
        fullh, fullw = 2, 5
    numcol = unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, numcol)
    numcels = unifint(diff_lb, diff_ub, (1, (h * w) // 2))
    inds = [(i, j) for i in range(h) for j in range(w)]
    cels = random.sample(inds, numcels)
    small = [[bgc] * w for _ in range(h)]
    for (i, j) in cels:
        small[i][j] = random.choice(ccols)
    gi = [[linc] * fullw for _ in range(fullh)]
    llocs = [(a, b) for a in range(0, fullh, h + 1) for b in range(0, fullw, w + 1)]
    src = random.choice(llocs)
    for (a, b) in llocs:
        for i in range(h):
            for j in range(w):
                gi[a + i][b + j] = small[i][j] if (a, b) == src else bgc
    go = [row[:] for row in gi]
    for (a, b) in llocs:
        for i in range(h):
            for j in range(w):
                go[a + i][b + j] = small[i][j]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared with the golf entry): the grid is periodic with period
    # (tile + divider); the content is overlaid onto itself by that period.
    # Done in two stages: first spread the source tile along its row band
    # (horizontal period), then treat the whole completed row band as the unit
    # and stamp it onto every other row band (vertical period).
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    grids = [I] + ([np.asarray(e[0], dtype=int) for e in examples] if examples else [])

    def line_colors(G):
        s = set()
        for r in range(G.shape[0]):
            if len(set(G[r].tolist())) == 1:
                s.add(int(G[r, 0]))
        for c in range(G.shape[1]):
            if len(set(G[:, c].tolist())) == 1:
                s.add(int(G[0, c]))
        return s

    # divider colour: the colour that forms full lines in every input of the episode
    cand = line_colors(I)
    for G in grids[1:]:
        cand2 = cand & line_colors(G)
        if cand2:
            cand = cand2
    tot = Counter()
    for G in grids:
        tot.update(G.flatten().tolist())

    def bg_for(lc):
        cc = Counter({k: v for k, v in tot.items() if k != lc})
        return cc.most_common(1)[0][0] if cc else lc

    linc = None
    if len(cand) == 1:
        linc = next(iter(cand))
    elif cand:
        top = Counter(I.flatten().tolist()).most_common(1)[0][0]
        linc = max(cand, key=lambda c: tot[c] if c != top else -1)
    # background: most common non-divider colour across the episode's inputs
    bgc = bg_for(linc)

    div_rows = [r for r in range(hi) if linc is not None and all(I[r] == linc)]
    div_cols = [c for c in range(wi) if linc is not None and all(I[:, c] == linc)]

    def bands(divs, n):
        out, prev = [], 0
        for d in divs:
            if d > prev:
                out.append((prev, d - 1))
            prev = d + 1
        if prev < n:
            out.append((prev, n - 1))
        return out

    rb = bands(div_rows, hi)
    cb = bands(div_cols, wi)

    src = None
    for (r0, r1) in rb:
        for (c0, c1) in cb:
            if np.any(I[r0:r1 + 1, c0:c1 + 1] != bgc):
                src = (r0, r1, c0, c1)
                break
        if src:
            break
    if src is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    sr0, sr1, sc0, sc1 = src
    th, tw = sr1 - sr0 + 1, sc1 - sc0 + 1
    tile = I[sr0:sr1 + 1, sc0:sc1 + 1]
    pastable = bool(np.any((tile != 0) & (tile != bgc)))
    # 0-coloured pattern cells: Paste can't write them, paint them explicitly
    zero_off = [(i, j) for i in range(th) for j in range(tw)
                if tile[i, j] == 0 and bgc != 0]

    # Stage 1: spread the source tile along its own row band.
    other_cols = [(c0, c1) for (c0, c1) in cb if c0 != sc0]
    if other_cols:
        if pastable:
            # whole tile rectangle is the unit being duplicated
            ops.append(28); sels.append([sr0, sc0, th - 1, tw - 1])
        for (c0, c1) in other_cols:
            if pastable:
                ops.append(30); sels.append([sr0, c0, 0, 0])
            if zero_off:
                ops.append(0); sels.append(sel_of([(sr0 + i, c0 + j) for i, j in zero_off]))

    # Stage 2: the completed full-width row band is the new unit; stamp it
    # onto every other row band.
    other_rows = [(r0, r1) for (r0, r1) in rb if r0 != sr0]
    if other_rows:
        if pastable:
            # full-width band rectangle, copied from the working grid as it now stands
            ops.append(29); sels.append([sr0, 0, th - 1, wi - 1])
        for (r0, r1) in other_rows:
            if pastable:
                ops.append(30); sels.append([r0, 0, 0, 0])
            if zero_off:
                ops.append(0); sels.append(sel_of([(r0 + i, c0 + j) for (c0, c1) in cb
                                                   for i, j in zero_off]))

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
                        f"num_examples+1 ({num_examples + 1}) for task c444b776"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c444b776"
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
                                f"for task c444b776"
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
                    f"Failed to build a complete episode for task c444b776 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c444b776-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
