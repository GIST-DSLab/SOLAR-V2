"""
ARC Task: ecdecbb3 (RE-ARC) — LLM-generated grid_maker
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

# The generator ends with `if choice((True, False)): dmirror(...)` — the picture
# is either ruled with horizontal lines or with vertical ones. Plan it so both
# orientations are shown before a test instance can use either.
VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, dotc, linc = random.sample(cols, 3)          # generator: sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]  # test variant was demonstrated
    return {"bgc": bgc, "dotc": dotc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc, linc, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice((True, False))

    def unifint(lb, ub, bounds):
        a, b = bounds
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    def paint_nb(g, ri, cj, col, hh, ww):             # neighbors(...) fill
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = ri + dr, cj + dc
                if 0 <= rr < hh and 0 <= cc < ww:
                    g[rr][cc] = col

    # a transposed picture puts h on the horizontal axis -> swap the caps
    cap_h, cap_w = (max_w, max_h) if transpose else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (4, max(4, cap_h)))
    w = unifint(diff_lb, diff_ub, (4, max(4, cap_w)))

    def pick_lines(nl):
        inds = list(range(h))
        out = []
        for _ in range(nl):
            if not inds:
                break
            idx = random.choice(inds)
            out.append(idx)
            for d in (-2, -1, 0, 1, 2):
                if idx + d in inds:
                    inds.remove(idx + d)
        return sorted(out)

    def dot_rows(locs):
        """Rows where the beam is unambiguous: the two frontiers nearest the dot
        are exactly the line(s) its beam runs to (the grid edges count as
        frontiers too). Rows that fail this are the ones where the task's own
        verifier reads the picture differently, so they are left out."""
        excl = set(locs) | {e + 1 for e in locs} | {e - 1 for e in locs}
        cands = list(locs) + [-1, h]
        rows = []
        for i in range(h):
            if i in excl:
                continue
            if i < locs[0] or i > locs[-1]:
                rows.append(i)
                continue
            sp = max(e for e in locs if e < i)
            ep = min(e for e in locs if e > i)
            reach = max(i - sp, ep - i)
            if all(abs(i - f) > reach for f in cands if f != sp and f != ep):
                rows.append(i)
        return rows

    nl = unifint(diff_lb, diff_ub, (1, max(1, h // 4)))
    for attempt in range(24):
        locs = pick_lines(1 if attempt == 23 else nl)   # one line always leaves room
        iopts = dot_rows(locs)
        if iopts:
            break

    gi = [[bgc] * w for _ in range(h)]
    for ii in locs:
        for j in range(w):
            gi[ii][j] = linc

    jopts = list(range(w))
    ndots = unifint(diff_lb, diff_ub, (1, max(1, min(len(iopts), w // 2))))
    dlocs = []
    for _ in range(ndots):
        if not iopts or not jopts:
            break
        loci = random.choice(iopts)
        locj = random.choice(jopts)
        dlocs.append((loci, locj))
        for d in (-1, 0, 1):
            if locj + d in jopts:
                jopts.remove(locj + d)

    go = [row[:] for row in gi]
    for (loci, locj) in dlocs:
        if loci < locs[0]:
            for rr in range(loci, locs[0] + 1):
                go[rr][locj] = dotc
            paint_nb(go, locs[0], locj, linc, h, w)
        elif loci > locs[-1]:
            for rr in range(locs[-1], loci + 1):
                go[rr][locj] = dotc
            paint_nb(go, locs[-1], locj, linc, h, w)
        else:
            sp = max(e for e in locs if e < loci)
            ep = min(e for e in locs if e > loci)
            for rr in range(sp, ep + 1):
                go[rr][locj] = dotc
            paint_nb(go, sp, locj, linc, h, w)
            paint_nb(go, ep, locj, linc, h, w)
        gi[loci][locj] = dotc

    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Each dot shoots a ray along its column to the ruled line(s) it sits
    against, and a collar of line colour is stamped where the ray crosses a line.

    Everything below is read off I: the lines, their colour, the dot colour, and
    the route. Only Color ops are used, so a role that happens to be colour 0
    is laid down explicitly by Color0 (nothing is carried by Move/Copy/Paste,
    which would treat 0 as empty).

    The route itself is chosen from the picture, so two instances are worked
    differently while the rule performed is the same:
      * which end to start from — the dot lying nearest a border edge;
      * which way to sweep — across a wide picture, down a tall one;
      * lots of rays (more rays than lines + 1): draw every ray, then collar
        line by line; otherwise take one dot at a time, either drawing the ray
        outward from its dot and stamping the collars over its ends, or
        stamping the collars first and running the ray up to them.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background: the canvas colour — lines cover at most a quarter of the rows
    # and the dots are single cells, so it is the bulk of the grid
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # the ruling is full rows, or (dmirror case) full columns: work in a view
    # where the lines are rows, and map every cell back through m()
    row_lines = [r for r in range(hi) if len(set(I[r].tolist())) == 1 and I[r, 0] != bgc]
    flip = not row_lines
    G = (I.T if flip else I).copy()
    h, w = G.shape
    locs = [r for r in range(h) if len(set(G[r].tolist())) == 1 and G[r, 0] != bgc]
    if not locs:
        return [34], [[0, 0, hi - 1, wi - 1]]
    linc = int(G[locs[0], 0])
    others = [c for c in sorted(set(G.flatten().tolist())) if c != bgc and c != linc]
    if not others:
        return [34], [[0, 0, hi - 1, wi - 1]]
    dotc = int(others[0])

    def m(r, c):                        # working view -> real grid coords
        return (c, r) if flip else (r, c)

    cur = G.copy()                      # the grid as it stands after the ops so far

    def paint(col, cells):
        # a cell already holding col would not change, so it is not selected;
        # an op with nothing left to change is not emitted at all
        todo = [rc for rc in dict.fromkeys(cells) if cur[rc[0], rc[1]] != col]
        if not todo:
            return
        ops.append(int(col))
        sels.append(sel_of([m(r, c) for r, c in todo]))
        for r, c in todo:
            cur[r, c] = col

    def collar(t, c0):                  # ring of line colour around a crossing
        return [(r, c) for r in range(t - 1, t + 2) for c in range(c0 - 1, c0 + 2)
                if 0 <= r < h and 0 <= c < w and not (r == t and c == c0)]

    dots = [(r, c) for r in range(h) for c in range(w)
            if G[r, c] == dotc and r not in locs]

    rays = []
    for (dr, dc) in dots:
        if dr < locs[0]:                        # above the ruling: down to the first line
            r0, r1, hits = dr, locs[0], [locs[0]]
        elif dr > locs[-1]:                     # below it: up to the last line
            r0, r1, hits = locs[-1], dr, [locs[-1]]
        else:                                   # boxed in: line to line through the dot
            sp = max(e for e in locs if e < dr)
            ep = min(e for e in locs if e > dr)
            r0, r1, hits = sp, ep, [sp, ep]
        rays.append({"dot": (dr, dc), "r0": r0, "r1": r1, "hits": hits})

    nrays, nlines = len(rays), len(locs)

    # where to start and which way to go: sweep the long way across the picture,
    # beginning at whichever end has a dot closer to the border
    axis = 1 if w > h else 0
    span = w if axis == 1 else h
    lo_gap = min(ray["dot"][axis] for ray in rays)
    hi_gap = (span - 1) - max(ray["dot"][axis] for ray in rays)
    backwards = hi_gap < lo_gap
    rays.sort(key=lambda ray: (ray["dot"][axis], ray["dot"][1 - axis]), reverse=backwards)

    if nrays > nlines + 1:
        mode = "sweep"
    elif (nrays + nlines) % 2 == 0:
        mode = "collar_first"
    else:
        mode = "ray_first"

    def draw_ray(ray, blocked=(), from_dot=False):
        dr, dc = ray["dot"]
        if from_dot and len(ray["hits"]) == 2:
            # a boxed-in dot: the ray grows out of the dot, one way then the other
            paint(dotc, [(r, dc) for r in range(dr, ray["r0"] - 1, -1) if (r, dc) not in blocked])
            paint(dotc, [(r, dc) for r in range(dr, ray["r1"] + 1) if (r, dc) not in blocked])
        else:
            paint(dotc, [(r, dc) for r in range(ray["r0"], ray["r1"] + 1) if (r, dc) not in blocked])

    if mode == "sweep":
        for ray in rays:                         # all the rays first
            draw_ray(ray)
        for t in sorted(locs, reverse=backwards and axis == 0):
            cells = []                           # then the collars, line by line
            for ray in rays:
                if t in ray["hits"]:
                    cells += collar(t, ray["dot"][1])
            paint(linc, cells)
    elif mode == "ray_first":
        for ray in rays:                         # dot by dot: ray, then its collars
            draw_ray(ray, from_dot=True)
            for t in ray["hits"]:
                paint(linc, collar(t, ray["dot"][1]))
    else:
        for ray in rays:                         # dot by dot: collars first, ray up to them
            dc = ray["dot"][1]
            blocked = set()
            for t in ray["hits"]:
                paint(linc, collar(t, dc))
                blocked |= {(t - 1, dc), (t + 1, dc)}   # the collar already owns these
            draw_ray(ray, blocked=blocked)

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
                        f"num_examples+1 ({num_examples + 1}) for task ecdecbb3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ecdecbb3"
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
                                f"for task ecdecbb3"
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
                    f"Failed to build a complete episode for task ecdecbb3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ecdecbb3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
