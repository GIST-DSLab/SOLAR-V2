"""
ARC Task: b6afb2da (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- 1. colors
def sample_colors(num_examples=None) -> dict:
    # generator: cols = {0..9} \ {1,2,4};  bgc = choice(cols)
    # object colours are irrelevant to the rule (every non-bg rectangle is
    # redrawn as 2 / 4 / 1), so only the background needs fixing.
    cols = [c for c in range(10) if c not in (1, 2, 4)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


# ---------------------------------------------------------------- 2. generate
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 4))
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 9))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = randint(3, 7)
        ow = randint(3, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(remcols)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, bd)
            go = fill(go, 2, bd)
            go = fill(go, 4, box(bd))
            go = fill(go, 1, corners(bd))
            succ += 1
            indss = indss - bd
        tr += 1
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- 3. derive
def _components(G):
    """4-connected, single-colour components: list of (color, cells)."""
    h, w = G.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = int(G[r, c])
            cells = []
            dq = deque([(r, c)])
            seen[r, c] = True
            while dq:
                rr, cc = dq.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(G[nr, nc]) == col:
                        seen[nr, nc] = True
                        dq.append((nr, nc))
            comps.append((col, cells))
    return comps


def _is_full_rect(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    hh = max(rs) - min(rs) + 1
    ww = max(cs) - min(cs) + 1
    return len(cells) == hh * ww


def _bg_of(G):
    """Background = colour of a component that is NOT a perfect rectangle.

    Every planted shape is a solid rectangle by construction, so the only
    non-rectangular components are background ones (this is exactly what the
    task's own specification extracts).  Fallback: majority colour.
    """
    comps = _components(G)
    best = None
    for col, cells in comps:
        if not _is_full_rect(cells):
            if best is None or len(cells) > len(best[1]):
                best = (col, cells)
    if best is not None:
        return int(best[0]), True
    return int(Counter(G.flatten().tolist()).most_common(1)[0][0]), False


def _rects_of(G, bgc):
    """Every solid non-background rectangle, as (r0, c0, h, w)."""
    rects = []
    for col, cells in _components(G):
        if col == bgc:
            continue
        if not _is_full_rect(cells):
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        rects.append((r0, c0, max(rs) - r0 + 1, max(cs) - c0 + 1))
    rects.sort()
    return rects


def _parts(r0, c0, h, w):
    """The named pieces of one rectangle, all measured from I."""
    body = [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
    ring = [(r, c) for (r, c) in body
            if r in (r0, r0 + h - 1) or c in (c0, c0 + w - 1)]
    cnrs = sorted({(r0, c0), (r0, c0 + w - 1),
                   (r0 + h - 1, c0), (r0 + h - 1, c0 + w - 1)})
    ringset = set(ring)
    cset = set(cnrs)
    inner = [p for p in body if p not in ringset]
    edges = [p for p in ring if p not in cset]
    caps = [(r, c) for (r, c) in ring if r in (r0, r0 + h - 1)]        # top+bottom rows
    sides = [(r, c) for (r, c) in ring if r not in (r0, r0 + h - 1)]   # left+right columns
    return {"body": body, "ring": ring, "cnrs": cnrs,
            "inner": inner, "edges": edges, "caps": caps, "sides": sides}


def _layers(r0, c0, h, w, style):
    """One rectangle -> an ordered list of (colour, cells) strokes.

    Every style ends at the same picture (2 inside, 4 on the border, 1 on the
    corners); they differ the way two people drawing the same box differ --
    lay a flat base coat and draw over it, or put each piece straight onto the
    grid, and in which order.
    """
    p = _parts(r0, c0, h, w)
    if style == 0:      # flood it 2, then outline in 4, then dot the corners
        return [(2, p["body"]), (4, p["ring"]), (1, p["cnrs"])]
    if style == 1:      # flood it 4, hollow the middle back out to 2, corners last
        return [(4, p["body"]), (2, p["inner"]), (1, p["cnrs"])]
    if style == 2:      # no base coat, inside out: middle, border sides, corners
        return [(2, p["inner"]), (4, p["edges"]), (1, p["cnrs"])]
    if style == 3:      # no base coat, outside in: corners, sides, then the middle
        return [(1, p["cnrs"]), (4, p["edges"]), (2, p["inner"])]
    # style 4: base coat, then the outline drawn in two passes (rows, columns)
    return [(2, p["body"]), (4, p["caps"]), (4, p["sides"]), (1, p["cnrs"])]


def _order_rects(rects, order):
    if order == 0:
        return sorted(rects, key=lambda t: (t[0], t[1]))                 # top -> bottom
    if order == 1:
        return sorted(rects, key=lambda t: (t[1], t[0]))                 # left -> right
    if order == 2:
        return sorted(rects, key=lambda t: (-t[0], -t[1]))               # bottom -> top
    if order == 3:
        return sorted(rects, key=lambda t: (-t[1], -t[0]))               # right -> left
    return sorted(rects, key=lambda t: (-(t[2] * t[3]), t[0], t[1]))     # biggest first


def _route_seed(bgc, rects):
    """A number read off the input alone, so one input always draws one route."""
    s = 1009 * (bgc + 1) + 97 * len(rects)
    for i, (r0, c0, h, w) in enumerate(sorted(rects)):
        s += (i + 1) * (13 * r0 + 7 * c0 + 5 * h + 3 * w)
    return s % 1000003


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- background colour: measured from I; the demonstrations (same episode,
    #     same background) only break a degenerate tie. ---
    bgc, confident = _bg_of(I)
    if examples and not confident:
        votes = []
        for pair in examples:
            try:
                ex_bg, ex_ok = _bg_of(np.asarray(pair[0], dtype=int))
            except Exception:
                continue
            if ex_ok:
                votes.append(ex_bg)
        if votes:
            consensus = Counter(votes).most_common(1)[0][0]
            if consensus in set(I.flatten().tolist()):
                bgc = int(consensus)

    rects = _rects_of(I, bgc)

    # --- pick this input's route: which box first, how each box is drawn, and
    #     whether a finished box gets stamped onto its same-size twins. ---
    seed = _route_seed(bgc, rects)
    order = seed % 5
    style = (seed // 5) % 5
    want_stamp = (seed // 25) % 2 == 1

    todo = _order_rects(rects, order)
    sizes = Counter((h, w) for (_, _, h, w) in rects)
    use_stamp = want_stamp and any(n > 1 for n in sizes.values())

    cur = I.copy()
    ops, sels = [], []

    def stroke(colour, cells):
        """Paint `cells` colour -- unless every one of them already is."""
        cells = [(int(r), int(c)) for r, c in cells]
        if not cells:
            return
        if all(cur[r, c] == colour for r, c in cells):
            return
        ops.append(int(colour))
        sels.append(sel_of(cells))
        for r, c in cells:
            cur[r, c] = colour

    def draw(rect):
        for colour, cells in _layers(*rect, style=style):
            stroke(colour, cells)

    done = set()
    for rect in todo:
        if rect in done:
            continue
        draw(rect)
        done.add(rect)

        if not use_stamp:
            continue
        r0, c0, h, w = rect
        twins = [t for t in todo if t not in done and (t[2], t[3]) == (h, w)]
        if not twins:
            continue
        # this box is finished and others are exactly the same size: copy it
        # once and stamp it onto each twin.  The selection here IS the full
        # h x w rectangle, so the bbox form is the region meant.
        ops.append(29)
        sels.append([r0, c0, h - 1, w - 1])
        stamp = cur[r0:r0 + h, c0:c0 + w].copy()
        holes = [(r, c) for r in range(h) for c in range(w) if stamp[r, c] == 0]
        for (tr0, tc0, _, _) in twins:
            if np.array_equal(cur[tr0:tr0 + h, tc0:tc0 + w], stamp):
                continue
            ops.append(30)
            sels.append([tr0, tc0, 0, 0])
            nz = stamp > 0
            cur[tr0:tr0 + h, tc0:tc0 + w][nz] = stamp[nz]
            # a Paste carries nothing where the stamp holds colour 0, so those
            # cells are laid down by hand (emits nothing when there are none)
            stroke(0, [(tr0 + r, tc0 + c) for r, c in holes])
            done.add((tr0, tc0, h, w))

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
                        f"num_examples+1 ({num_examples + 1}) for task b6afb2da"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b6afb2da"
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
                                f"for task b6afb2da"
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
                    f"Failed to build a complete episode for task b6afb2da "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b6afb2da-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
