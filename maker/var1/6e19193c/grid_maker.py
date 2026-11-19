"""
ARC Task: 6e19193c (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level colour role.
    # The rule ("shoot a diagonal ray out of the missing corner of each
    # 2x2-minus-one-cell block, in that block's own colour") depends on shape
    # only, never on which colour a block happens to have, so object colours
    # stay per-instance random.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    dirs = (
        ((0, 0), (-1, -1)),
        ((0, 1), (-1, 1)),
        ((1, 0), (1, -1)),
        ((1, 1), (1, 1)),
    )
    base = ((0, 0), (1, 0), (0, 1), (1, 1))

    h_ub = max(4, min(30, int(max_h)))
    w_ub = max(4, min(30, int(max_w)))
    remcols = [c for c in range(10) if c != bgc]

    gi = go = None
    for _attempt in range(12):
        h = unifint(diff_lb, diff_ub, (4, h_ub))
        w = unifint(diff_lb, diff_ub, (4, w_ub))
        num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 8)))

        raylen = h + w + 2
        cands = []
        for miss, d in dirs:
            obji = set(base) - {miss}
            ray = set()
            rr, cc = miss[0], miss[1]
            for _ in range(raylen):
                rr += d[0]
                cc += d[1]
                ray.add((rr, cc))
            objo = (set(base) | ray | {miss}) - {miss}
            cands.append((obji, objo))

        gi = [[bgc] * w for _ in range(h)]
        go = [[bgc] * w for _ in range(h)]
        fullinds = {(i, j) for i in range(h) for j in range(w)}
        inds = set(fullinds)

        kk, tr = 0, 0
        maxtrials = num * 4
        while kk < num and tr < maxtrials:
            if len(inds) == 0:
                break
            loc = random.choice(sorted(inds))
            obji, objo = random.choice(cands)
            obji = {(r + loc[0], c + loc[1]) for r, c in obji}
            objo = {(r + loc[0], c + loc[1]) for r, c in objo}
            objo = objo & fullinds
            # placement keeps every object's block AND ray disjoint from all
            # previous ones, and no block orthogonally touches another block
            if objo.issubset(inds) and obji.issubset(objo):
                col = random.choice(remcols)
                for r, c in obji:
                    gi[r][c] = col
                for r, c in objo:
                    go[r][c] = col
                nbs = set()
                for r, c in obji:
                    nbs |= {(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)}
                inds = (inds - objo) - nbs
                kk += 1
            tr += 1

        # every ray falling straight off the grid would make output == input
        if gi != go:
            break

    return {
        "input": tuple(tuple(row) for row in gi),
        "output": tuple(tuple(row) for row in go),
    }


def derive_operations(I, O, examples=None):
    """Read the rule off I alone, then pick a route the way a person would.

    Rule: every object is a 2x2 block with exactly ONE corner missing. The
    missing corner names a diagonal pointing away from the block body; the
    object's colour is shot along that diagonal, starting one cell beyond the
    missing corner, out to the grid edge.

    The rule is fixed; the ROUTE through it is not. Three choices, each read
    off the input (so one instance always yields one route, and the variety
    comes from instances differing):

      * sweep     <- grid size (h + w): which way the eye travels over the grid
      * grouping  <- how many arrows there are: one at a time / all the arrows
                     pointing the same way / all the arrows of one colour
      * stroke    <- the colour and position of the first arrow met: ray drawn
                     cell by cell away from its block, cell by cell inward from
                     the grid edge, or laid down as one whole line

    O is never consulted: hand this a different answer grid and it returns the
    same operations. No base-fill variant exists here because the background is
    already on the grid — laying bgc again would paint cells the colour they
    already hold. Rays are newly created content (nothing translates), so
    painting is the only honest op, and generator placement guarantees rays
    never overlap each other or any block, so every op paints fresh cells.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # Objects cover at most 3/8 of the canvas -> background is the majority colour.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- find the arrows: 4-connected single-colour 3-cell "2x2 minus a corner"
    seen = np.zeros((h, w), dtype=bool)
    arrows = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if len(cells) != 3:
                continue
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, c0 = min(rs), min(cs)
            if max(rs) - r0 != 1 or max(cs) - c0 != 1:
                continue
            body = set(cells)
            missing = [(y, x) for y in (r0, r0 + 1) for x in (c0, c0 + 1) if (y, x) not in body]
            if len(missing) != 1:
                continue
            my, mx = missing[0]
            dy = -1 if my == r0 else 1
            dx = -1 if mx == c0 else 1
            ray = []
            y, x = my + dy, mx + dx
            while 0 <= y < h and 0 <= x < w:
                ray.append((y, x))          # ordered outward, block -> edge
                y += dy
                x += dx
            if ray:                          # a ray leaving the grid at once adds nothing
                arrows.append({"col": col, "r": r0, "c": c0, "d": (dy, dx), "ray": ray})

    ops, sels = [], []

    if arrows:
        # --- choice 1: which way to sweep the grid (from its shape)
        sweep = (h + w) % 4
        if sweep == 0:
            key = lambda a: (a["r"], a["c"])      # top-left down, row by row
        elif sweep == 1:
            key = lambda a: (-a["r"], -a["c"])    # bottom-right up, row by row
        elif sweep == 2:
            key = lambda a: (a["c"], a["r"])      # column by column, left to right
        else:
            key = lambda a: (-a["c"], -a["r"])    # column by column, right to left
        ordered = sorted(arrows, key=key)

        # --- choice 2: what to batch together (from the number of arrows)
        grouping = len(arrows) % 3
        if grouping == 1:
            # all the arrows pointing the same way, biggest family first
            firsts = []
            for a in ordered:
                if a["d"] not in firsts:
                    firsts.append(a["d"])
            cnt = Counter(a["d"] for a in ordered)
            rank = {d: i for i, d in enumerate(firsts)}
            fam = sorted(firsts, key=lambda d: (-cnt[d], rank[d]))
            groups = [[a for a in ordered if a["d"] == d] for d in fam]
        elif grouping == 2:
            # all the arrows of one colour together, colours in the order met
            firsts = []
            for a in ordered:
                if a["col"] not in firsts:
                    firsts.append(a["col"])
            groups = [[a for a in ordered if a["col"] == k] for k in firsts]
        else:
            groups = [[a] for a in ordered]       # one arrow at a time

        # --- choice 3: how the ray itself is drawn (from the first arrow met)
        head = ordered[0]
        stroke = (head["col"] + head["r"] + head["c"]) % 3

        if grouping == 2 and stroke == 2:
            # every ray of a colour in a single stroke: "all the 2s go like this"
            for g in groups:
                ops.append(g[0]["col"])
                sels.append(sel_of([p for a in g for p in a["ray"]]))
        else:
            for g in groups:
                for a in g:
                    if stroke == 2:
                        # the whole line at once
                        ops.append(a["col"])
                        sels.append(sel_of(a["ray"]))
                    else:
                        # cell by cell: away from the block (0) or back from the edge (1)
                        path = a["ray"] if stroke == 0 else list(reversed(a["ray"]))
                        for cell in path:
                            ops.append(a["col"])
                            sels.append(sel_of([cell]))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # bbox = the whole grid, exactly what is submitted
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
                        f"num_examples+1 ({num_examples + 1}) for task 6e19193c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6e19193c"
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
                                f"for task 6e19193c"
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
                    f"Failed to build a complete episode for task 6e19193c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6e19193c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
