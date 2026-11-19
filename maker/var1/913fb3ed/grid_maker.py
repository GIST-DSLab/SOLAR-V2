"""
ARC Task: 913fb3ed (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The generator only samples ONE colour freely: the background.
    # cols = {0..9} - {1,2,3,4,6,8}  ->  {0,5,7,9}
    # The marker colours (2,3,8) and their ring colours (1,6,4) are hardcoded
    # by the generator, so they must NOT be re-sampled here.
    cols = [c for c in range(10) if c not in (1, 2, 3, 4, 6, 8)]
    bgc = choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    sr = (2, 3, 8)
    tr = (1, 6, 4)
    prs = list(zip(sr, tr))
    h = unifint(diff_lb, diff_ub, (1, max_h))
    w = unifint(diff_lb, diff_ub, (1, max_w))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    numc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 10)))
    inds = asindices(gi)
    for k in range(numc):
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        a, b = choice(prs)
        inds = (inds - neighbors(loc)) - outbox(neighbors(loc))
        inds = remove(loc, inds)
        gi = fill(gi, a, {loc})
        go = fill(go, a, {loc})
        go = fill(go, b, neighbors(loc))
    return {'input': gi, 'output': go}


def _learn_ring(examples):
    """Read the marker -> halo-colour convention off the demonstrations.

    A marker is a cell that keeps its colour while every one of its in-bounds
    8-neighbours changes, all to one and the same colour.  That is exactly the
    halo colour for that marker.  Only I/O of the *examples* is used.
    """
    learned = {}
    if not examples:
        return learned
    for pair in examples:
        try:
            ii = np.asarray(pair[0], dtype=int)
            oo = np.asarray(pair[1], dtype=int)
        except Exception:
            continue
        if ii.shape != oo.shape or ii.ndim != 2:
            continue
        h2, w2 = ii.shape
        for r in range(h2):
            for c in range(w2):
                if int(ii[r, c]) != int(oo[r, c]):
                    continue
                ns = [(r + dr, c + dc)
                      for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                      if not (dr == 0 and dc == 0)
                      and 0 <= r + dr < h2 and 0 <= c + dc < w2]
                if not ns:
                    continue
                if any(int(ii[p]) == int(oo[p]) for p in ns):
                    continue
                vals = {int(oo[p]) for p in ns}
                if len(vals) == 1:
                    learned[int(ii[r, c])] = vals.pop()
    return learned


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # ---- the rule (convention read from the demonstrations) -------------
    ring = {2: 1, 3: 6, 8: 4}
    ring.update(_learn_ring(examples))

    # ---- the objects: isolated marker dots in I -------------------------
    centers = [(r, c) for r in range(hi) for c in range(wi)
               if int(I[r, c]) in ring]

    ops, sels = [], []
    if not centers:
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    def halo(r, c):
        return [(r + dr, c + dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0)
                and 0 <= r + dr < hi and 0 <= c + dc < wi]

    # ---- pick a route: fixed by the input, different between inputs -----
    # Nothing here reads O; a different answer grid yields the same route.
    bg = Counter(I.flatten().tolist()).most_common(1)[0][0]
    r0, c0 = centers[0]
    key = (7 * len(centers) + 3 * int(bg) + 5 * r0 + 11 * c0
           + 13 * hi + 17 * wi)
    mode = key % 4
    order = (key // 4) % 5

    def route(cs):
        if order == 0:                                   # top-down, left-right
            return sorted(cs)
        if order == 1:                                   # column by column
            return sorted(cs, key=lambda p: (p[1], p[0]))
        if order == 2:                                   # bottom-up
            return sorted(cs, reverse=True)
        if order == 3:                                   # right-hand columns first
            return sorted(cs, key=lambda p: (-p[1], p[0]))
        rem = sorted(cs)                                 # hop to the nearest dot
        cur = rem.pop(0)
        chain = [cur]
        while rem:
            nxt = min(rem, key=lambda p: (max(abs(p[0] - cur[0]),
                                              abs(p[1] - cur[1])), p))
            rem.remove(nxt)
            chain.append(nxt)
            cur = nxt
        return chain

    seq = route(centers)

    if mode == 0:
        # One dot at a time: paint its halo, move on.
        for (r, c) in seq:
            cells = halo(r, c)
            if cells:
                ops.append(ring[int(I[r, c])])
                sels.append(sel_of(cells))

    elif mode == 1:
        # All dots of one kind together: every red halo, then every blue, ...
        colour_order = []
        for (r, c) in seq:
            v = int(I[r, c])
            if v not in colour_order:
                colour_order.append(v)
        for v in colour_order:
            cells = []
            for (r, c) in seq:
                if int(I[r, c]) == v:
                    cells.extend(halo(r, c))
            if cells:
                ops.append(ring[v])
                sels.append(sel_of(cells))

    elif mode == 2:
        # Block first, dot back on top: fill the whole 3x3 square around the
        # dot with the halo colour, then put the dot back over it.
        for (r, c) in seq:
            cells = halo(r, c)
            if not cells:
                continue
            r1, r2 = max(0, r - 1), min(hi - 1, r + 1)
            c1, c2 = max(0, c - 1), min(wi - 1, c + 1)
            block = [(rr, cc) for rr in range(r1, r2 + 1)
                     for cc in range(c1, c2 + 1)]
            # full solid rectangle (dot included) -- laid down as a base
            ops.append(ring[int(I[r, c])])
            sels.append(sel_of(block))
            # the dot drawn back on top of that base
            ops.append(int(I[r, c]))
            sels.append(sel_of([(r, c)]))

    else:
        # Cross first, then the four corners.
        for (r, c) in seq:
            v = int(I[r, c])
            cross = [(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                     if 0 <= r + dr < hi and 0 <= c + dc < wi]
            diag = [(r + dr, c + dc) for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1))
                    if 0 <= r + dr < hi and 0 <= c + dc < wi]
            if cross:
                ops.append(ring[v])
                sels.append(sel_of(cross))
            if diag:
                ops.append(ring[v])
                sels.append(sel_of(diag))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 913fb3ed"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 913fb3ed"
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
                                f"for task 913fb3ed"
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
                    f"Failed to build a complete episode for task 913fb3ed "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"913fb3ed-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
