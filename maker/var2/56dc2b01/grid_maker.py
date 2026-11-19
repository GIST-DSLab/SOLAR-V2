"""
ARC Task: 56dc2b01 (RE-ARC) — LLM-generated grid_maker
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

# The four structural cases the generator's mirror step can produce:
# the 2-bar is either a column or a row, and the object sits on either side of it.
VARIANTS = [
    {"transpose": False, "flip": False},   # vertical 2-bar, object to its right
    {"transpose": False, "flip": True},    # vertical 2-bar, object to its left
    {"transpose": True,  "flip": False},   # horizontal 2-bar, object below it
    {"transpose": True,  "flip": True},    # horizontal 2-bar, object above it
]


def sample_colors(num_examples=None) -> dict:
    # The generator samples bgc and objc from 0..9 minus {2, 8}; 2 (bar) and 8
    # (frontier) are hardcoded, so they are not episode colours.
    # 0 is a legal ordinary colour here for either role and is deliberately kept.
    cols = [c for c in range(10) if c not in (2, 8)]
    bgc, objc = random.sample(cols, 2)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]   # test case was demonstrated
    return {"bgc": bgc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc,
             transpose=None, flip=None, **kwargs) -> dict:
    if transpose is None:
        transpose = random.choice([False, True])
    if flip is None:
        flip = random.choice([False, True])

    # base grid is built with a VERTICAL bar (h rows, w cols); a transpose swaps
    # the dimensions, so cap them against the transposed limits.
    hcap = min(30, max_w if transpose else max_h)
    wcap = min(30, max_h if transpose else max_w)
    if hcap < 4 or wcap < 6:
        transpose = False
        hcap, wcap = min(30, max_h), min(30, max_w)
    hcap, wcap = max(4, hcap), max(6, wcap)

    h = unifint(diff_lb, diff_ub, (4, hcap))
    w = unifint(diff_lb, diff_ub, (6, wcap))
    oh = unifint(diff_lb, diff_ub, (1, h))
    ow = unifint(diff_lb, diff_ub, (1, max(1, (w - 1) // 2 - 1)))

    bb = asindices(canvas(-1, (oh, ow)))
    sp = random.choice(totuple(bb))
    obj = {sp}
    bb = remove(sp, bb)
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = random.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(0, ncells), oh * ow - 1)
    for k in range(ncells):
        obj.add(random.choice(totuple((bb - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)

    loci = random.randint(0, h - oh)
    locj = unifint(diff_lb, diff_ub, (1, w - ow))
    # keep the 8 frontier inside the grid so every instance shows the whole rule
    locj = max(1, min(locj, w - ow - 1))

    gi = canvas(bgc, (h, w))
    barlocji = unifint(diff_lb, diff_ub, (0, locj))
    barlocj = locj - barlocji
    barlocj = min(max(0, barlocj), locj - 1)
    gi = fill(gi, 2, connect((0, barlocj), (h - 1, barlocj)))
    go = fill(gi, objc, shift(obj, (loci, barlocj + 1)))
    go = fill(go, 8, connect((0, barlocj + ow + 1), (h - 1, barlocj + ow + 1)))
    gi = fill(gi, objc, shift(obj, (loci, locj)))

    if transpose:                       # bar becomes a row, object below it
        gi, go = dmirror(gi), dmirror(go)
    if flip:                            # object moves to the other side of the bar
        if transpose:
            gi, go = hmirror(gi), hmirror(go)
        else:
            gi, go = vmirror(gi), vmirror(go)

    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Slide the shape up against the 2-bar, then draw the 8 frontier just past
    its far edge.  The route (what to do first, how to tidy the vacated
    footprint) is chosen from features of I alone, so it is stable per instance
    and differs between instances.  O is never read."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    def bg_of(g):
        # object width < half the grid, so the canvas colour is a strict majority
        return Counter(np.asarray(g, dtype=int).flatten().tolist()).most_common(1)[0][0]

    def full_lines(g, bg):
        """colours whose cells are exactly one complete row or one complete column"""
        out = []
        h, w = g.shape
        for c in sorted(set(g.flatten().tolist())):
            if c == bg:
                continue
            m = (g == c)
            n = int(m.sum())
            if n == w and any(m[r].all() for r in range(h)):
                out.append(c)
            elif n == h and any(m[:, k].all() for k in range(w)):
                out.append(c)
        return out

    def one_blob(cells):
        cs = set(cells)
        if not cs:
            return False
        start = min(cs)
        seen, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in cs and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return len(seen) == len(cs)

    bgc = bg_of(I)
    cand_I = full_lines(I, bgc)

    # the marker colour and the frontier colour are conventions of the task: read
    # them off the demonstrations, fall back to the values every example shows.
    bar_c, front_c = 2, 8
    if examples:
        fronts, bars = Counter(), Counter()
        for ein, eout in examples:
            a = np.asarray(ein, dtype=int)
            b = np.asarray(eout, dtype=int)
            for c in set(b.flatten().tolist()) - set(a.flatten().tolist()):
                fronts[c] += 1
            abg = bg_of(a)
            for c in full_lines(a, abg):
                if np.array_equal(a == c, b == c):      # the marker never moves
                    bars[c] += 1
        if fronts:
            front_c = fronts.most_common(1)[0][0]
        for c, _ in bars.most_common():
            if c in cand_I:
                bar_c = c
                break
    if bar_c not in cand_I and len(cand_I) == 1:
        bar_c = cand_I[0]

    bar = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == bar_c]
    vertical = len(set(c for _, c in bar)) == 1

    obj = sorted((r, c) for r in range(hi) for c in range(wi)
                 if I[r, c] != bgc and I[r, c] != bar_c)
    objc = int(I[obj[0][0], obj[0][1]]) if obj else bgc

    if vertical:
        b = bar[0][1]
        lo = min(c for _, c in obj)
        hh = max(c for _, c in obj)
        span = hh - lo + 1
        if lo > b:                                  # object right of the bar -> slide left
            steps, dr, dc, move_op = lo - (b + 1), 0, -1, 23
            idx = b + span + 1
        else:                                       # object left of the bar -> slide right
            steps, dr, dc, move_op = (b - 1) - hh, 0, 1, 22
            idx = b - span - 1
        line = [(r, idx) for r in range(hi)] if 0 <= idx < wi else []
    else:
        b = bar[0][0]
        lo = min(r for r, _ in obj)
        hh = max(r for r, _ in obj)
        span = hh - lo + 1
        if lo > b:                                  # object below the bar -> slide up
            steps, dr, dc, move_op = lo - (b + 1), -1, 0, 20
            idx = b + span + 1
        else:                                       # object above the bar -> slide down
            steps, dr, dc, move_op = (b - 1) - hh, 1, 0, 21
            idx = b - span - 1
        line = [(idx, c) for c in range(wi)] if 0 <= idx < hi else []

    dst = [(r + dr * steps, c + dc * steps) for r, c in obj]
    overlap = bool(set(line) & set(obj))     # frontier runs through the shape's old body

    # ---- pick the route from the picture itself -------------------------------
    carries_zero = (objc == 0)               # ARCLE cannot Move a 0-coloured shape
    if carries_zero:
        cands = [["reloc", "fix", "front"],          # put it down, clear the leftovers
                 ["fix", "reloc", "front"]]          # wipe the old body first, draw on top
        if not overlap:
            cands.append(["front", "fix", "reloc"])  # frontier first
    else:
        cands = [["reloc", "front", "fix"],          # slide, rule the line, patch the hole
                 ["reloc", "fix", "front"]]          # slide, patch the hole, rule the line
        if not overlap:
            cands.append(["front", "reloc", "fix"])  # rule the line, then slide into place
    order = cands[(len(obj) + steps + span + b) % len(cands)]
    flood_hole = ((len(obj) + b + hi) % 2 == 1)      # flood the blank or paint it?

    G = I.copy()
    fix_first = order.index("fix") < order.index("reloc")

    for stage in order:
        if stage == "front":
            if not line:
                continue
            cells = [p for p in line if G[p[0], p[1]] != front_c]
            ops.append(int(front_c))
            sels.append(sel_of(cells))
            for r, c in cells:
                G[r, c] = front_c

        elif stage == "reloc":
            if steps == 0:
                continue
            if not carries_zero:
                # one grab, then empty selections: ARCLE keeps the shape held and
                # restores every cell it glides over.
                ops.append(move_op)
                sels.append(sel_of(obj))
                for _ in range(steps - 1):
                    ops.append(move_op)
                    sels.append(sel_of([]))
                for r, c in obj:                 # ARCLE blanks the grabbed cells
                    G[r, c] = 0
                for r, c in dst:
                    G[r, c] = objc
            else:
                # the shape's colour is 0, which no Move/Copy/Paste can carry:
                # lay it down where it lands with an explicit Color0.
                cells = [p for p in dst if G[p[0], p[1]] != 0]
                if cells:
                    ops.append(0)
                    sels.append(sel_of(cells))
                    for r, c in cells:
                        G[r, c] = 0

        else:  # "fix" — give the ground the shape left back to the background
            if steps == 0 or bgc == 0:
                continue
            scope = obj if fix_first else [p for p in obj if p not in set(dst)]
            cells = [p for p in scope if G[p[0], p[1]] == 0]
            if not cells:
                continue
            blank = [(r, c) for r in range(hi) for c in range(wi) if G[r, c] == 0]
            if flood_hole and set(blank) == set(cells) and one_blob(cells):
                seed = min(cells)
                ops.append(10 + int(bgc))             # FloodFill<bgc> from one cell
                sels.append(sel_of([seed]))
            else:
                ops.append(int(bgc))
                sels.append(sel_of(cells))
            for r, c in cells:
                G[r, c] = bgc

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])   # whole-grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 56dc2b01"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 56dc2b01"
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
                                f"for task 56dc2b01"
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
                    f"Failed to build a complete episode for task 56dc2b01 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"56dc2b01-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
