"""
ARC Task: a87f7484 (RE-ARC) — LLM-generated grid_maker
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
import random
from collections import Counter


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        a, b = b, a
    lo = a + int((b - a) * lb)
    hi = a + int(round((b - a) * ub))
    if hi < lo:
        hi = lo
    lo = max(lo, a)
    hi = min(hi, b)
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    # Rule depends only on the PRESENCE/pattern of the odd panel, not on which
    # colors are used -> only the shared background needs to be fixed per episode.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    cols = list(range(10))
    remcols = [c for c in cols if c != bgc]

    B = min(int(max_h), int(max_w))
    if B < 6:
        B = 6  # need room for num>=3 panels of width>=2

    # panel base size (h x w), number of panels = num
    h = _unifint(diff_lb, diff_ub, (2, 5))
    num_hi = max(3, min(9, B // 2))
    num = _unifint(diff_lb, diff_ub, (3, num_hi))
    w_hi = max(2, B // num)
    w = _unifint(diff_lb, diff_ub, (2, w_hi))

    ccols = random.sample(remcols, num)

    inds = [(r, c) for r in range(h) for c in range(w)]
    ncd = _unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc = random.choice((ncd, h * w - ncd))
    nc = min(max(1, nc), h * w - 1)

    origlocs = set(random.sample(inds, nc))
    canbrem = set(origlocs)
    canbeadd = set(inds) - set(origlocs)
    otherlocs = set(origlocs)

    nchangesinv = _unifint(diff_lb, diff_ub, (0, h * w - 1))
    nchanges = h * w - nchangesinv
    for _ in range(nchanges):
        if random.choice((True, False)):
            if len(canbrem) > 1:
                ch = random.choice(tuple(canbrem))
                otherlocs.discard(ch); canbrem.discard(ch)
            elif len(canbeadd) > 1:
                ch = random.choice(tuple(canbeadd))
                otherlocs.add(ch); canbeadd.discard(ch)
        else:
            if len(canbeadd) > 1:
                ch = random.choice(tuple(canbeadd))
                otherlocs.add(ch); canbeadd.discard(ch)
            elif len(canbrem) > 1:
                ch = random.choice(tuple(canbrem))
                otherlocs.discard(ch); canbrem.discard(ch)

    # Guarantee the odd panel is genuinely distinct so it is derivable from I.
    if otherlocs == set(origlocs):
        addable = set(inds) - otherlocs
        if addable:
            otherlocs.add(next(iter(addable)))
        elif len(otherlocs) > 1:
            otherlocs.discard(next(iter(otherlocs)))

    def build(color, locs):
        g = np.full((h, w), bgc, dtype=int)
        for (r, c) in locs:
            g[r, c] = color
        return g

    go = build(ccols[0], origlocs)          # the ODD panel
    grids = [go]
    for cc in ccols[1:]:
        grids.append(build(cc, otherlocs))  # all identical pattern
    random.shuffle(grids)

    gi = np.concatenate(grids, axis=1)      # panels merged horizontally
    go_out = go

    if random.choice((True, False)):
        gi = gi.T.copy()                    # dmirror (diagonal) -> row strips
        go_out = go_out.T.copy()

    return {"input": gi.tolist(), "output": go_out.tolist()}


def _check(panels):
    # bg = the single color common to EVERY panel (the shared canvas color).
    color_sets = [set(np.unique(p).tolist()) for p in panels]
    common = set(color_sets[0])
    for s in color_sets[1:]:
        common &= s
    if len(common) != 1:
        return None
    bg = next(iter(common))
    keys = [(p != bg).tobytes() for p in panels]
    cnt = Counter(keys)
    if len(cnt) != 2:
        return None
    if sorted(cnt.values())[0] != 1:
        return None
    odd_key = [k for k, v in cnt.items() if v == 1][0]
    return keys.index(odd_key)


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    candidates = []  # (num, orient, size, odd)

    # vertical strips (equal-width column panels)
    for num in range(3, wi + 1):
        if wi % num:
            continue
        w = wi // num
        panels = [I[:, k * w:(k + 1) * w] for k in range(num)]
        odd = _check(panels)
        if odd is not None:
            candidates.append((num, 'v', w, odd))

    # horizontal strips (equal-height row panels)
    for num in range(3, hi + 1):
        if hi % num:
            continue
        hh = hi // num
        panels = [I[k * hh:(k + 1) * hh, :] for k in range(num)]
        odd = _check(panels)
        if odd is not None:
            candidates.append((num, 'h', hh, odd))

    ops, sels = [], []

    if candidates:
        # prefer the finest genuine tiling (largest panel count)
        candidates.sort(key=lambda t: (t[0], t[1] == 'v'))
        num, orient, size, odd = candidates[-1]
        if orient == 'v':
            c0 = odd * size
            # crop the whole odd panel rectangle (bg included) -> output
            ops.append(33); sels.append([0, c0, hi - 1, size - 1])
            ho, wo = hi, size
        else:
            r0 = odd * size
            ops.append(33); sels.append([r0, 0, size - 1, wi - 1])
            ho, wo = size, wi
    else:
        ho, wo = O.shape

    ho2, wo2 = O.shape
    ops.append(34); sels.append([0, 0, ho2 - 1, wo2 - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task a87f7484"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a87f7484"
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
                                f"for task a87f7484"
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
                    f"Failed to build a complete episode for task a87f7484 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a87f7484-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
