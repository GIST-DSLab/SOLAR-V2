"""
ARC Task: 3befdf3e (RE-ARC) — LLM-generated grid_maker
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
# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
import numpy as np
import random
from collections import Counter, deque

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(2, 9)
    ccols = random.sample(remcols, numcols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols=None) -> dict:
    cols = list(range(10))
    remcols = [c for c in cols if c != bgc]

    lo_h = min(10, max_h)
    lo_w = min(10, max_w)
    h = unifint(diff_lb, diff_ub, (lo_h, max_h))
    w = unifint(diff_lb, diff_ub, (lo_w, max_w))

    if ccols is None:
        ccols = random.sample(remcols, random.randint(2, 9))
    ccols = [c for c in ccols if c != bgc]
    if len(ccols) < 2:
        ccols = random.sample(remcols, 2)

    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 40)))
    nobjs = max(2, nobjs)

    # cover several ring sizes inside every instance so arm-length = core size is learnable
    shapes = [(1, 1), (1, 2), (2, 1), (2, 2)]
    random.shuffle(shapes)

    # some instances repeat a framed box verbatim (same size, same two colours) —
    # the original generator can do this by chance; here it happens often enough
    # that a solver meets the "I already drew this one" situation regularly.
    placed = []

    succ = 0
    maxtr = 5 * nobjs + 10
    tr = 0
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        if len(inds) == 0:
            break
        if placed and random.random() < 0.45:
            rh, rw, incol, outcol = random.choice(placed)
        else:
            rh, rw = shapes[succ % len(shapes)]
            incol, outcol = random.sample(ccols, 2)
        fullh = 2 + 3 * rh
        fullw = 2 + 3 * rw
        if fullh > h or fullw > w:
            continue
        cands = sfilter(inds, lambda ij: ij[0] <= h - fullh and ij[1] <= w - fullw)
        if len(cands) == 0:
            continue
        loc = random.choice(totuple(cands))
        loci, locj = loc
        fullobj = backdrop(frozenset({loc, (loci + fullh - 1, locj + fullw - 1)}))
        if fullobj.issubset(inds):
            succ += 1
            inds = inds - fullobj
            placed.append((rh, rw, incol, outcol))
            ofincol = backdrop(frozenset({(loci + rh + 1, locj + rw + 1), (loci + 2 * rh, locj + 2 * rw)}))
            ofoutcol = outbox(ofincol)
            gi = fill(gi, incol, ofincol)
            gi = fill(gi, outcol, ofoutcol)
            go = fill(go, outcol, ofincol)
            go = fill(go, incol, ofoutcol)
            ilocs = apply(first, ofoutcol)
            jlocs = apply(last, ofoutcol)
            ff = lambda ij: ij[0] in ilocs or ij[1] in jlocs
            addon = sfilter(fullobj - (ofincol | ofoutcol), ff)
            go = fill(go, outcol, addon)
    return {'input': gi, 'output': go}


# ── derive_operations ─────────────────────────────────────────────────────────
#
# Rule (read off the examples, then applied to I alone):
#   every object is a solid rectangle = a frame of colour `outcol` around a
#   rh x rw core of colour `incol`.  In the answer the two colours trade places
#   (frame -> incol, core -> outcol) and the frame grows four arms of `outcol`,
#   each exactly as long as the core is thick (rh up/down, rw left/right).
#
# The rule is fixed; the ROUTE through it is not.  Which box to start with,
# which way to sweep, whether to lay the outer colour down first and draw the
# frame back on top, whether to grow the arms in one stroke or one at a time,
# and whether to stamp an already-finished box onto its twin instead of
# redrawing it — all of that is decided from features of the INPUT (how many
# boxes, where the first one sits, which colours they use, how much ink is on
# the page), so one instance always takes the same route and two different
# instances rarely take the same one.  Nothing here ever looks at O.


def _components(I, bgc):
    """The framed boxes of I, as solid non-background rectangles."""
    hi, wi = I.shape
    seen = np.zeros((hi, wi), dtype=bool)
    boxes = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] != bgc:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0 = min(p[0] for p in cells); r1 = max(p[0] for p in cells)
            c0 = min(p[1] for p in cells); c1 = max(p[1] for p in cells)
            rh, rw = r1 - r0 - 1, c1 - c0 - 1      # core size == arm length
            if rh < 1 or rw < 1:
                continue
            outcol = int(I[r0, c0])                # frame colour
            incol = int(I[r0 + 1, c0 + 1])         # core colour
            ring = [(a, b) for a in range(r0, r1 + 1) for b in range(c0, c1 + 1)
                    if a in (r0, r1) or b in (c0, c1)]
            core = [(a, b) for a in range(r0 + 1, r1) for b in range(c0 + 1, c1)]
            arms = {
                "up":    [(a, b) for a in range(max(0, r0 - rh), r0) for b in range(c0, c1 + 1)],
                "down":  [(a, b) for a in range(r1 + 1, min(hi, r1 + 1 + rh)) for b in range(c0, c1 + 1)],
                "left":  [(a, b) for a in range(r0, r1 + 1) for b in range(max(0, c0 - rw), c0)],
                "right": [(a, b) for a in range(r0, r1 + 1) for b in range(min(wi, c1 + 1), min(wi, c1 + 1 + rw))],
            }
            boxes.append({
                "r0": r0, "c0": c0, "r1": r1, "c1": c1, "rh": rh, "rw": rw,
                "incol": incol, "outcol": outcol,
                "ring": ring, "core": core, "arms": arms,
                # the whole plot the finished figure occupies: the frame band
                # grown outward by the core size on all four sides
                "box_r": r0 - rh, "box_c": c0 - rw,
                "box_h": 3 * rh + 2, "box_w": 3 * rw + 2,
            })
    return boxes


def _arm_order(b, flip):
    """Which way a solver walks round the four arms."""
    if b["rw"] > b["rh"]:          # a wide box: the long arms first
        seq = ["left", "right", "up", "down"]
    else:
        seq = ["up", "down", "left", "right"]
    if flip:
        seq = seq[::-1]
    return seq


def _draw(b, style, flip, ops, sels):
    """Redraw one framed box in place. Every route paints exactly the same cells."""
    ring, core = b["ring"], b["core"]
    arms = b["arms"]
    incol, outcol = b["incol"], b["outcol"]
    order = _arm_order(b, flip)
    allarms = [cell for k in order for cell in arms[k]]

    if style == 0:
        # frame takes the core colour, then core + arms take the frame colour
        ops.append(incol);  sels.append(sel_of(ring))
        ops.append(outcol); sels.append(sel_of(core + allarms))
    elif style == 1:
        # lay the frame colour over the whole figure, then draw the frame back on top
        ops.append(outcol); sels.append(sel_of(ring + core + allarms))
        ops.append(incol);  sels.append(sel_of(ring))
    elif style == 2:
        # swap the two colours of the box, then grow the arms one at a time
        ops.append(incol);  sels.append(sel_of(ring))
        ops.append(outcol); sels.append(sel_of(core))
        for k in order:
            if arms[k]:
                ops.append(outcol); sels.append(sel_of(arms[k]))
    else:
        # swap the colours, then reach out along one band and then the other
        ops.append(incol);  sels.append(sel_of(ring))
        band_h = arms["left"] + core + arms["right"]
        band_v = arms["up"] + arms["down"]
        for band in ([band_v, band_h] if flip else [band_h, band_v]):
            if band:
                ops.append(outcol); sels.append(sel_of(band))


def _plot_clear(b, I, bgc):
    """True when the box has its whole plot to itself (it fits, nothing else in it)."""
    hi, wi = I.shape
    if not (b["box_r"] >= 0 and b["box_c"] >= 0
            and b["box_r"] + b["box_h"] <= hi and b["box_c"] + b["box_w"] <= wi):
        return False
    for a in range(b["box_r"], b["box_r"] + b["box_h"]):
        for c in range(b["box_c"], b["box_c"] + b["box_w"]):
            if b["r0"] <= a <= b["r1"] and b["c0"] <= c <= b["c1"]:
                continue
            if I[a, c] != bgc:
                return False
    return True


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the colour the canvas was painted with before the boxes landed
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    boxes = _components(I, bgc)
    ops, sels = [], []
    if not boxes:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    # ── route, decided from the input ────────────────────────────────────────
    n = len(boxes)
    ink = int((I != bgc).sum())                       # how much ink is on the page
    first = min(boxes, key=lambda b: (b["r0"], b["c0"]))
    hue = sum(b["incol"] + b["outcol"] for b in boxes)

    sweep = (n + first["c0"]) % 4       # which corner to start from / which way to go
    style = (ink + hue) % 4             # how a single box gets redrawn
    flip = ((first["r0"] + first["c0"]) % 2) == 1     # arm order within a box
    copy_twins = ((hue + n) % 2) == 0   # reuse a finished box on its twins?

    if sweep == 0:
        boxes.sort(key=lambda b: (b["r0"], b["c0"]))      # top to bottom
    elif sweep == 1:
        boxes.sort(key=lambda b: (-b["r0"], -b["c0"]))    # bottom to top
    elif sweep == 2:
        boxes.sort(key=lambda b: (b["c0"], b["r0"]))      # left to right
    else:
        boxes.sort(key=lambda b: (-b["c0"], -b["r0"]))    # right to left

    # twins = boxes identical in the input (same size, same two colours).
    # A twin can be stamped from a finished one only if the clipboard can carry
    # the whole figure: colour 0 reads as "nothing here" to Copy/Paste, so a box
    # that is painted in 0 never travels — it is drawn by hand instead, and its
    # 0 cells go down as an explicit Color0 on exactly the cells that need them.
    groups = {}
    for b in boxes:
        groups.setdefault((b["rh"], b["rw"], b["incol"], b["outcol"]), []).append(b)

    def stampable(key, g):
        incol, outcol = key[2], key[3]
        return (len(g) > 1 and incol != 0 and outcol != 0
                and all(_plot_clear(b, I, bgc) for b in g))

    use_copy = copy_twins and any(stampable(k, g) for k, g in groups.items())

    if use_copy:
        # handle each family of twins together: redraw the first one by hand,
        # then stamp it onto the others
        plan, done = [], set()
        for b in boxes:
            key = (b["rh"], b["rw"], b["incol"], b["outcol"])
            if key in done:
                continue
            done.add(key)
            plan.append((stampable(key, groups[key]), groups[key]))
    else:
        plan = [(False, [b]) for b in boxes]

    for stamp, family in plan:
        lead = family[0]
        _draw(lead, style, flip, ops, sels)
        rest = family[1:]
        if stamp:
            # pick up the finished figure — the whole plot it stands on,
            # background corners included (a plain rectangle) — and stamp it
            # onto every twin, which still shows the untouched box underneath
            ops.append(29)
            sels.append([lead["box_r"], lead["box_c"], lead["box_h"] - 1, lead["box_w"] - 1])
            for t in rest:
                ops.append(30)
                sels.append([t["box_r"], t["box_c"], 0, 0])
        else:
            for t in rest:
                _draw(t, style, flip, ops, sels)

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])   # full-grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 3befdf3e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3befdf3e"
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
                                f"for task 3befdf3e"
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
                    f"Failed to build a complete episode for task 3befdf3e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3befdf3e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
