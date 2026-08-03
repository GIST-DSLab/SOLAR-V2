"""
ARC Task: 9d9215db (RE-ARC) — LLM-generated grid_maker
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

ROTS = ("identity", "rot90", "rot180", "rot270")


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        exs = [{"rotf_name": r} for r in ROTS]
        exs += [{"rotf_name": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(exs)
    else:
        exs = [{"rotf_name": r} for r in random.sample(list(ROTS), n_ex)]
    plan = exs + [dict(random.choice(exs))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rotf_name=None) -> dict:
    if rotf_name is None:
        rotf_name = random.choice(list(ROTS))
    rotmap = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}
    rotf = rotmap[rotf_name]
    swap = rotf_name in ("rot90", "rot270")
    lim_h = max_w if swap else max_h
    lim_w = max_h if swap else max_w

    cols = interval(0, 10, 1)
    hub = max(5, min(14, (lim_h - 1) // 2))
    wub = max(5, min(14, (lim_w - 1) // 2))
    h = unifint(diff_lb, diff_ub, (5, hub))
    w = unifint(diff_lb, diff_ub, (5, wub))
    h = h * 2 + 1
    w = w * 2 + 1
    remcols = list(remove(bgc, cols))
    ub = min(h, w) // 4
    nrings = unifint(diff_lb, diff_ub, (1, ub))
    onlinesbase = tuple([(2 * k + 1, 2 * k + 1) for k in range(ub)])
    onlines = random.sample(list(onlinesbase), nrings)
    onlines = {(random.choice(remcols), ij) for ij in onlines}
    gi = canvas(bgc, (h, w))
    gi = paint(gi, onlines)
    linsbase = apply(rbind(add, (0, 2)), onlinesbase[:-1])
    nlines = unifint(diff_lb, diff_ub, (1, len(linsbase)))
    linesps = random.sample(list(linsbase), nlines)
    colors = [random.choice(remcols) for k in range(nlines)]
    dots = {(col, ij) for col, ij in zip(colors, linesps)}
    dots2 = {(col, ij[::-1]) for col, ij in zip(colors, linesps)}
    gi = paint(gi, dots | dots2)
    ff = lambda ij: ij[1] % 2 == 1
    ff2 = lambda ij: ij[0] % 2 == 1
    linesps2 = tuple(x[::-1] for x in linesps)
    lines = tuple(sfilter(connect(ij, (ij[0], w - ij[1] - 1)), ff) for ij in linesps)
    lines2 = tuple(sfilter(connect(ij, (h - ij[0] - 1, ij[1])), ff2) for ij in linesps2)
    lines = merge({recolor(col, l1 | l2) for col, (l1, l2) in zip(colors, zip(lines, lines2))})
    gobase = paint(gi, lines)
    go = paint(gobase, merge(fgpartition(vmirror(gobase))))
    go = paint(go, merge(fgpartition(hmirror(gobase))))
    go = paint(go, merge(fgpartition(vmirror(hmirror(gobase)))))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    # I holds seeds of concentric square "rings" (every other cell, odd rows/cols).
    # Ring k lives on rows {2k+1, H-2-2k} and cols {2k+1, W-2-2k}.
    # Each ring may carry: a corner dot (one of its 4 corners is seeded in I)
    # and a dotted edge colour (two seeds, one on a horizontal edge, one on a
    # vertical edge, each sitting right next to the anchor corner).
    # Rule: complete every ring -> 4 corner dots + all 4 dotted edges.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    ub = min(H, W) // 4

    ops, sels = [], []

    for k in range(ub):
        r0, r1 = 2 * k + 1, H - 2 - 2 * k
        c0, c1 = 2 * k + 1, W - 2 - 2 * k

        # ---- ring k: corner dots ----
        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        cseed = None
        for p in corners:
            if I[p[0], p[1]] != bgc:
                cseed = p
                break
        if cseed is not None:
            cval = int(I[cseed[0], cseed[1]])
            for p in corners:
                if p != cseed:
                    ops.append(cval)
                    sels.append([p[0], p[1], 0, 0])

        hcols = list(range(2 * k + 3, W - 2 * k - 3, 2))   # cols strictly inside ring corners
        vrows = list(range(2 * k + 3, H - 2 * k - 3, 2))   # rows strictly inside ring corners

        # ---- ring k: horizontal edges ----
        hseed = None
        for r in (r0, r1):
            for c in hcols:
                if I[r, c] != bgc:
                    hseed = (r, c)
                    break
            if hseed is not None:
                break
        if hseed is not None:
            rh, ch = hseed
            lval = int(I[rh, ch])
            # grow the dotted edge outward from its seed along the ring's row
            for c in sorted([c for c in hcols if c != ch], key=lambda x: abs(x - ch)):
                ops.append(lval)
                sels.append([rh, c, 0, 0])
            # mirror the finished edge onto the opposite side of the ring
            rm = r1 if rh == r0 else r0
            ops.append(29)
            sels.append([rh, hcols[0], 0, hcols[-1] - hcols[0]])
            ops.append(30)
            sels.append([rm, hcols[0], 0, 0])

        # ---- ring k: vertical edges ----
        vseed = None
        for c in (c0, c1):
            for r in vrows:
                if I[r, c] != bgc:
                    vseed = (r, c)
                    break
            if vseed is not None:
                break
        if vseed is not None:
            rv, cv = vseed
            lval = int(I[rv, cv])
            for r in sorted([r for r in vrows if r != rv], key=lambda x: abs(x - rv)):
                ops.append(lval)
                sels.append([r, cv, 0, 0])
            cm = c1 if cv == c0 else c0
            ops.append(29)
            sels.append([vrows[0], cv, vrows[-1] - vrows[0], 0])
            ops.append(30)
            sels.append([vrows[0], cm, 0, 0])

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9d9215db"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9d9215db"
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
                                f"for task 9d9215db"
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
                    f"Failed to build a complete episode for task 9d9215db "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9d9215db-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
