"""
ARC Task: 3618c87e (RE-ARC) — LLM-generated grid_maker
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

ROTS = ["identity", "rot90", "rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    """bgc / linc / dotc are the three colours the generator samples.
    Any of them may be 0 — the route has to cope with that.
    The four rotations are the task's discrete structural variants, so they are
    planned up front and handed to generate() one per instance."""
    cols = list(range(10))
    bgc, linc, dotc = random.sample(cols, 3)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "linc": linc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, dotc, rot=None) -> dict:
    def _unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = int(a + (b - a) * lb)
        hi = int(a + (b - a) * ub)
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    if rot is None:
        rot = random.choice(ROTS)

    # a 90/270 rotation swaps the axes, so build inside the swapped budget
    if rot in ("rot90", "rot270"):
        hb, wb = max_w, max_h
    else:
        hb, wb = max_h, max_w
    hb = max(4, min(30, int(hb)))
    wb = max(4, min(30, int(wb)))

    h = _unifint(diff_lb, diff_ub, (4, hb))
    w = _unifint(diff_lb, diff_ub, (4, wb))

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for c in range(w):
        gi[0][c] = linc
        go[0][c] = linc

    nlocs = _unifint(diff_lb, diff_ub, (1, w // 2))
    locs = []
    opts = list(range(w))
    for _ in range(nlocs):
        if not opts:
            break
        ch = random.choice(opts)
        locs.append(ch)
        opts = [o for o in opts if o not in (ch - 1, ch, ch + 1)]

    for j in locs:
        hh = random.randint(1, h - 3)
        for r in range(hh + 1):
            gi[r][j] = linc
            go[r][j] = linc
        gi[hh + 1][j] = dotc
        go[0][j] = dotc

    a = np.array(gi, dtype=int)
    b = np.array(go, dtype=int)
    if rot == "rot90":          # clockwise
        a, b = np.rot90(a, 3), np.rot90(b, 3)
    elif rot == "rot180":
        a, b = np.rot90(a, 2), np.rot90(b, 2)
    elif rot == "rot270":       # counter-clockwise
        a, b = np.rot90(a, 1), np.rot90(b, 1)

    return {"input": a.tolist(), "output": b.tolist()}


def derive_operations(I, O):
    """
    Picture: one whole border edge is a line (linc).  Perpendicular arms grow inward
    from it and each arm carries a single marker pixel (dotc) one cell past its tip.
    Rule: the marker is cleared from the tip and reappears on the line, at the base of
    its own arm.

    The rule is fixed, the route is not.  Three things are read off the input and decide
    how the work is carried out, so the same picture always produces the same route and
    different pictures produce different ones:
      * which end of the line to start from -> the end the deepest arm sits nearest,
      * how to group the work (arm by arm / by role / pick the marker up once and stamp
        it) -> a signature of the arms and the three colours,
      * which half of an arm's work comes first (clear the tip, or lay the marker down).
    Nothing here looks at O: the same operations come back whatever answer grid is given.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    full = [0, 0, h - 1, w - 1]

    def read(grid):
        gh, gw = grid.shape
        edges = [
            ([(0, c) for c in range(gw)], (1, 0)),
            ([(gh - 1, c) for c in range(gw)], (-1, 0)),
            ([(r, 0) for r in range(gh)], (0, 1)),
            ([(r, gw - 1) for r in range(gh)], (0, -1)),
        ]
        found = []
        for cells, (dr, dc) in edges:
            vals = set(int(grid[r, c]) for r, c in cells)
            if len(vals) != 1:
                continue
            linec = vals.pop()
            arms, beyond, flat, bad = [], set(), [], False
            for idx, (r, c) in enumerate(cells):
                rr, cc, d = r, c, 0
                while 0 <= rr < gh and 0 <= cc < gw and int(grid[rr, cc]) == linec:
                    d += 1
                    rr += dr
                    cc += dc
                inside = 0 <= rr < gh and 0 <= cc < gw
                if d < 2:
                    if inside:
                        flat.append(int(grid[rr, cc]))
                    continue
                if not inside:
                    bad = True
                    break
                beyond.add(int(grid[rr, cc]))
                arms.append({"pos": idx, "base": (r, c), "tip": (rr, cc), "depth": d})
            if bad or not arms or not flat or len(beyond) != 1:
                continue
            dotcol = beyond.pop()
            bgcol = Counter(flat).most_common(1)[0][0]
            if dotcol == linec or bgcol == linec or bgcol == dotcol:
                continue
            if len(arms) == len(cells):
                continue
            found.append({"line": linec, "dot": dotcol, "bg": bgcol,
                          "arms": arms, "span": len(cells)})
        if not found:
            return None
        found.sort(key=lambda f: len(f["arms"]))
        return found[0]

    info = read(I)
    if info is None:
        ops.append(34)
        sels.append(full)
        return ops, sels

    linec, dotc, bgc = info["line"], info["dot"], info["bg"]
    arms, span = info["arms"], info["span"]
    n = len(arms)

    # --- start from the end of the line the deepest arm reaches out of
    deepest = max(arms, key=lambda a: (a["depth"], a["pos"]))
    ascending = deepest["pos"] * 2 <= span - 1
    order = sorted(arms, key=lambda a: a["pos"], reverse=not ascending)

    # --- how to group the work, read off the arms and the palette
    routes = ["pairs", "roles"]
    if n >= 2 and dotc != 0:
        # colour 0 never travels in the clipboard, so stamping is only on the table
        # when the marker is a colour Paste can actually carry
        routes.append("stamp")
    sig = sum(a["pos"] + a["depth"] for a in arms) + linec + dotc + bgc + n
    route = "pairs" if n == 1 else routes[sig % len(routes)]
    tip_first = ((sig // len(routes)) % 2 == 0)

    if route == "pairs":
        # arm by arm: take the marker off the tip, put it back on the line
        for a in order:
            if tip_first:
                ops.append(int(bgc)); sels.append(sel_of([a["tip"]]))
                ops.append(int(dotc)); sels.append(sel_of([a["base"]]))
            else:
                ops.append(int(dotc)); sels.append(sel_of([a["base"]]))
                ops.append(int(bgc)); sels.append(sel_of([a["tip"]]))

    elif route == "roles":
        # by role: all the old markers at once, all the new ones at once
        tips = [a["tip"] for a in order]
        bases = [a["base"] for a in order]
        if tip_first:
            ops.append(int(bgc)); sels.append(sel_of(tips))
            ops.append(int(dotc)); sels.append(sel_of(bases))
        else:
            ops.append(int(dotc)); sels.append(sel_of(bases))
            ops.append(int(bgc)); sels.append(sel_of(tips))

    else:  # "stamp": the markers are all the same pixel, so pick one up and reuse it
        ops.append(28); sels.append(sel_of([order[0]["tip"]]))
        if tip_first:
            ops.append(int(bgc)); sels.append(sel_of([a["tip"] for a in order]))
            for a in order:
                ops.append(30); sels.append(sel_of([a["base"]]))
        else:
            for a in order:
                ops.append(int(bgc)); sels.append(sel_of([a["tip"]]))
                ops.append(30); sels.append(sel_of([a["base"]]))

    ops.append(34)
    sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task 3618c87e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3618c87e"
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
                                f"for task 3618c87e"
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
                    f"Failed to build a complete episode for task 3618c87e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3618c87e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
