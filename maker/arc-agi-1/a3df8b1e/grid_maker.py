"""
ARC Task: a3df8b1e (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    g = globals().get("unifint")
    if g is not None:
        return g(diff_lb, diff_ub, bounds)
    a, b = bounds
    return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


# numlins is the one discrete structural knob: how many corners emit a ray.
VARIANTS = [{"numlins": 1}, {"numlins": 2}, {"numlins": 3}, {"numlins": 4}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, linc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    random.shuffle(ex)
    for i, e in enumerate(ex):                       # both orientations shown
        e["portrait"] = bool(i % 2 == 0) if n_ex >= 2 else bool(random.getrandbits(1))
    random.shuffle(ex)
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, linc: int, numlins=None, portrait=None) -> dict:
    cap = min(max_h, max_w)                          # either orientation must fit
    w = _unifint(diff_lb, diff_ub, (2, min(10, cap - 1)))
    h = _unifint(diff_lb, diff_ub, (w + 1, cap))
    c = canvas(bgc, (h, w))
    sp = (h - 1, 0)
    gi = fill(c, linc, {sp})
    go = tuple(e for e in gi)
    direc = 1
    while True:
        sp = add(sp, (-1, direc))
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        go2 = fill(go, linc, {sp})
        if go2 == go:
            break
        go = go2
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    if portrait is not None and (len(gi) > len(gi[0])) != bool(portrait):
        gi = dmirror(gi)
        go = dmirror(go)
    gix = tuple(e for e in gi)
    gox = tuple(e for e in go)
    if numlins is None:
        numlins = _unifint(diff_lb, diff_ub, (1, 4))
    if numlins > 1:
        gi = fill(gi, linc, ofcolor(hmirror(gix), linc))
        go = fill(go, linc, ofcolor(hmirror(gox), linc))
    if numlins > 2:
        gi = fill(gi, linc, ofcolor(vmirror(gix), linc))
        go = fill(go, linc, ofcolor(vmirror(gox), linc))
    if numlins > 3:
        gi = fill(gi, linc, ofcolor(hmirror(vmirror(gix)), linc))
        go = fill(go, linc, ofcolor(hmirror(vmirror(gox)), linc))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule, read off I alone: every corner carrying a mark is the start of a 45-degree
    ray that reflects off the two walls of the grid's SHORT side and runs to the far
    end of the LONG one.  Marked corners come in a mirror family, so the extra rays
    are mirror images of the first.

    Route.  The ray is drawn the way a billiard ball draws it: the first leg is
    painted out of the seed corner, and every later leg is that leg REFLECTED in the
    wall it just hit -- a FlipV/FlipH of the band straddling the bounce (the
    reflection axis is the bounce row/column, so the band is symmetric about it).
    ARCLE's flip moves what it grabs, so after each bounce the incoming leg is laid
    back down.  The other rays are the same reflection at grid scale: mirror the
    whole picture, then lay the rays that were already there back down.  A seed that
    a reflection would carry off its corner is cleared first and returns as the
    corner of its own ray; a seed no reflection touches is left alone.

    O is never read -- which region, which axis and how far all come from I.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    corner_list = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    cset = set(corner_list)

    # background = most common colour away from the corners (on the smallest grids
    # the corners can outnumber the rest); the mark colour is the other one in I
    cnt = Counter(int(I[r, c]) for r in range(h) for c in range(w) if (r, c) not in cset)
    bgc = cnt.most_common(1)[0][0]
    rest = sorted({int(v) for v in I.reshape(-1)} - {bgc})
    linc = rest[0] if rest else bgc
    marks = [q for q in corner_list if int(I[q[0], q[1]]) == linc]
    if not marks:
        return [34], [[0, 0, h - 1, w - 1]]

    rows_are_long = h > w                       # then the ray bounces off the sides

    def ray_cells(q):
        """the billiard path out of corner q, one cell per step"""
        r, c = q
        dr = 1 if r == 0 else -1
        dc = 1 if c == 0 else -1
        cells = [(r, c)]
        while True:
            nr, nc = r + dr, c + dc
            if rows_are_long:
                if nc < 0 or nc > w - 1:
                    dc = -dc
                    nc = c + dc
                if nr < 0 or nr > h - 1:
                    break
            else:
                if nr < 0 or nr > h - 1:
                    dr = -dr
                    nr = r + dr
                if nc < 0 or nc > w - 1:
                    break
            cells.append((nr, nc))
            r, c = nr, nc
        return cells

    def legs(cells):
        """straight runs, each keeping the bounce cell it starts and ends on"""
        out, start = [], 0
        for i in range(1, len(cells) - 1):
            d1 = (cells[i][0] - cells[i - 1][0], cells[i][1] - cells[i - 1][1])
            d2 = (cells[i + 1][0] - cells[i][0], cells[i + 1][1] - cells[i][1])
            if d1 != d2:
                out.append(cells[start:i + 1])
                start = i
        out.append(cells[start:])
        return out

    def ud(q):
        return (h - 1 - q[0], q[1])

    def lr(q):
        return (q[0], w - 1 - q[1])

    base = marks[0]
    ray = {q: ray_cells(q) for q in marks}
    rayset = {q: set(v) for q, v in ray.items()}

    # the picture the rule asks for, measured from I: bgc, with every marked
    # corner's ray on it.  It is what the route is checked against below; it is
    # never taken from O.
    target = np.full((h, w), bgc, dtype=int)
    for q in marks:
        for r, c in ray[q]:
            target[r, c] = linc

    def build(erase):
        """emit the whole route, clearing the seeds in `erase` first"""
        grid = I.copy()
        ops, sels = [], []

        def emit(op, sel, ng):
            if np.array_equal(ng, grid):        # never emit a step that shows nothing
                return False
            ops.append(int(op))
            sels.append(sel)
            grid[:, :] = ng
            return True

        def color_op(cells, val):
            pts = sorted({(int(r), int(c)) for r, c in cells if int(grid[r, c]) != val})
            if not pts:
                return False
            ng = grid.copy()
            for r, c in pts:
                ng[r, c] = val
            return emit(val, sel_of(pts), ng)

        def flip_rect(r0, c0, r1, c1, axis):
            """mirror a whole rectangle in place: the selection is every cell of it"""
            ng = grid.copy()
            sub = grid[r0:r1 + 1, c0:c1 + 1]
            ng[r0:r1 + 1, c0:c1 + 1] = np.flipud(sub) if axis == 'v' else np.fliplr(sub)
            return emit(27 if axis == 'v' else 26, [r0, c0, r1 - r0, c1 - c0], ng)

        def flip_obj(cells, axis):
            """mirror one ray about the grid's centre line: grab exactly its cells"""
            pts = sorted(cells)
            rs = [p[0] for p in pts]
            cs = [p[1] for p in pts]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            vals = {p: int(grid[p]) for p in pts}
            ng = grid.copy()
            for p in pts:
                ng[p] = 0                        # ARCLE zeroes what it grabs
            for (r, c), v in vals.items():
                nr, nc = (r0 + r1 - r, c) if axis == 'v' else (r, c0 + c1 - c)
                if v > 0:
                    ng[nr, nc] = v
            return emit(27 if axis == 'v' else 26, sel_of(pts), ng)

        if erase:
            color_op(erase, bgc)

        # ---- the base ray: first leg painted, every later leg a reflection ----
        segs = legs(ray[base])
        color_op(segs[0], linc)
        for m in range(1, len(segs)):
            seg = segs[m]
            k = len(seg) - 1                     # cells this leg adds
            src = segs[m - 1][-(k + 1):]         # incoming leg, bounce cell last
            pr, pc = seg[0]
            if rows_are_long:                    # the bounce row is the axis
                flip_rect(pr - k, min(c for _, c in src),
                          pr + k, max(c for _, c in src), 'v')
            else:                                # the bounce column is the axis
                flip_rect(min(r for r, _ in src), pc - k,
                          max(r for r, _ in src), pc + k, 'h')
            color_op(segs[m - 1], linc)          # lay the incoming leg back down

        # ---- the remaining rays are mirror images of the ones already drawn ----
        have = {base}
        for _ in range(4):
            drawn = {(r, c) for r in range(h) for c in range(w) if grid[r, c] == linc}
            have |= {q for q in marks if rayset[q] <= drawn}
            missing = [q for q in marks if q not in have]
            if not missing:
                break
            mirrored = False
            for axis, f in (('v', ud), ('h', lr)):
                img = {f(q) for q in have}
                if img <= set(marks) and img - have:
                    if flip_rect(0, 0, h - 1, w - 1, axis):   # mirror the picture
                        color_op([p for q in sorted(have) for p in ray[q]], linc)
                        have |= img
                        mirrored = True
                        break
            if mirrored:
                continue
            t = missing[0]
            src = None
            for q in sorted(have):
                for axis, f in (('v', ud), ('h', lr)):
                    if f(q) == t:
                        src = (q, axis)
                        break
                if src:
                    break
            if src is not None and linc != 0:     # grab that one ray and mirror it
                q, axis = src
                if flip_obj(ray[q], axis):
                    color_op(ray[q], linc)        # lay the ray it came from back down
                    have.add(t)
                    continue
            color_op(ray[t], linc)                # a 0-coloured ray cannot be grabbed
            have.add(t)

        return ops, sels, grid

    erase = [q for q in marks if q != base]
    ops, sels, grid = build(erase)
    for q in list(erase):                         # keep only the seeds that must go
        trial = [p for p in erase if p != q]
        t_ops, t_sels, t_grid = build(trial)
        if np.array_equal(t_grid, target):
            erase, ops, sels, grid = trial, t_ops, t_sels, t_grid

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a3df8b1e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a3df8b1e"
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
                                f"for task a3df8b1e"
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
                    f"Failed to build a complete episode for task a3df8b1e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a3df8b1e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
