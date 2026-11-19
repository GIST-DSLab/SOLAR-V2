"""
ARC Task: c8cbb738 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# 1. episode-level colours
#    The rule is colour independent (objects are identified structurally: the
#    4-corner marker defines the frame, every other colour group is a piece of
#    the ring).  Only the background and the corner/frame colour are fixed.
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    sc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "sc": sc}


# ----------------------------------------------------------------------------
# 2. generator (RE-ARC c8cbb738 with max_h / max_w bounds and injected colours)
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, sc) -> dict:
    cols = interval(0, 10, 1)
    gh_ub = max(3, min(10, max_h // 2))
    gw_ub = max(3, min(10, max_w // 2))
    gh = unifint(diff_lb, diff_ub, (3, gh_ub))
    gw = unifint(diff_lb, diff_ub, (3, gw_ub))
    h = unifint(diff_lb, diff_ub, (gh * 2, max(gh * 2, max_h)))
    w = unifint(diff_lb, diff_ub, (gw * 2, max(gw * 2, max_w)))

    remcols = remove(sc, remove(bgc, cols))
    ncols = unifint(diff_lb, diff_ub, (0, 8))
    ccols = list(sample(remcols, ncols)) if ncols > 0 else []

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (gh, gw))
    goinds = asindices(go)
    ring = box(goinds)
    crns = corners(ring)
    remring = ring - crns
    nrr = len(remring)

    go = fill(go, sc, crns)
    loci = randint(0, h - gh)
    locj = randint(0, w - gw)
    gi = fill(gi, sc, shift(crns, (loci, locj)))

    bL = connect((0, 0), (gh - 1, 0))
    bR = connect((0, gw - 1), (gh - 1, gw - 1))
    bT = connect((0, 0), (0, gw - 1))
    bB = connect((gh - 1, 0), (gh - 1, gw - 1))
    validpairs = [(bL, bT), (bL, bB), (bR, bT), (bR, bB)]

    for c in ccols:
        if len(remring) < 3:
            break
        nc = unifint(diff_lb, diff_ub, (3, max(3, min(len(remring), nrr // max(1, len(ccols))))))
        obj = set(sample(totuple(remring), min(nc, len(remring))))
        flag = False
        for b1, b2 in validpairs:
            if len(obj & b1) > 0 and len(obj & b2) > 0:
                flag = True
                break
        if flag:
            oh, ow = shape(obj)
            locs = ofcolor(gi, bgc)
            cands = sfilter(locs, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) > 0:
                objn = normalize(obj)
                cands2 = sfilter(cands, lambda ij: shift(objn, ij).issubset(locs))
                if len(cands2) > 0:
                    loc = choice(totuple(cands2))
                    gi = fill(gi, c, shift(objn, loc))
                    go = fill(go, c, obj)
                    remring -= obj

    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. derive_operations
#
#    Everything below is measured from I only:
#      * bgc            = dominant colour of I
#      * (gh, gw)       = max height / max width over the colour groups
#      * frame origin   = bbox of the colour group that is exactly the 4 corners
#                         of a gh x gw rectangle (the ring's corner marker)
#      * each other colour group is a fragment of the ring: its normalized shape
#        is placed at the offset inside the gh x gw frame that maximises the
#        number of its cells lying on the frame's border (ties broken by the
#        smaller manhattan distance of the shape's centre to the frame centre).
#    The pieces are then physically slid (Move) into the frame, and the frame is
#    cropped out.  O is never inspected.
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # --- background: dominant colour of the input canvas ---------------------
    counts = {}
    for v in I.reshape(-1).tolist():
        counts[int(v)] = counts.get(int(v), 0) + 1
    bgc = max(counts.items(), key=lambda kv: kv[1])[0]

    # --- colour groups (fgpartition) -----------------------------------------
    groups = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                groups.setdefault(v, []).append((r, c))

    ops, sels = [], []
    if not groups:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    # --- frame size = maximal group height / width ---------------------------
    gh = max(max(r for r, _ in cs) - min(r for r, _ in cs) + 1 for cs in groups.values())
    gw = max(max(c for _, c in cs) - min(c for _, c in cs) + 1 for cs in groups.values())

    # --- the corner marker: 4 cells = exact corners of a gh x gw rectangle ----
    frame_color, fr, fc = None, 0, 0
    for col, cs in groups.items():
        rs = [r for r, _ in cs]
        cls = [c for _, c in cs]
        r0, r1, c0, c1 = min(rs), max(rs), min(cls), max(cls)
        if (r1 - r0 + 1) == gh and (c1 - c0 + 1) == gw and len(cs) == 4 and \
           set(cs) == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            frame_color, fr, fc = col, r0, c0
            break
    if frame_color is None:                      # defensive fallback
        col = sorted(groups)[0]
        cs = groups[col]
        frame_color = col
        fr, fc = min(r for r, _ in cs), min(c for _, c in cs)

    # --- border cells of the gh x gw frame, and its centre --------------------
    border = set()
    for j in range(gw):
        border.add((0, j)); border.add((gh - 1, j))
    for i in range(gh):
        border.add((i, 0)); border.add((i, gw - 1))
    ctr_r, ctr_c = gh // 2, gw // 2
    weight = 2 * max(gh, gw)

    def placement(cells):
        rs = [r for r, _ in cells]
        cls = [c for _, c in cells]
        r0, c0 = min(rs), min(cls)
        norm = [(r - r0, c - c0) for r, c in cells]
        oh = max(a for a, _ in norm) + 1
        ow = max(b for _, b in norm) + 1
        best, best_score = (0, 0), None
        for i in range(gh):
            for j in range(gw):
                inter = 0
                for a, b in norm:
                    if (a + i, b + j) in border:
                        inter += 1
                dist = abs(i + oh // 2 - ctr_r) + abs(j + ow // 2 - ctr_c)
                score = weight * inter - dist
                if best_score is None or score > best_score:
                    best_score, best = score, (i, j)
        return best, norm, (r0, c0)

    frame_rect = {(r, c) for r in range(fr, fr + gh) for c in range(fc, fc + gw)}

    items = []
    for col in sorted(groups):
        if col == frame_color:
            continue
        cells = groups[col]
        (i, j), norm, (r0, c0) = placement(cells)
        dst = [(fr + i + a, fc + j + b) for a, b in norm]
        if set(dst) == set(cells):
            continue                                    # already in place
        items.append({
            "color": col,
            "src": sorted(cells),
            "dst": sorted(dst),
            "d": (fr + i - r0, fc + j - c0),
        })

    # ---------------- emission helpers ---------------------------------------
    def emit_move(o):
        dr, dc = o["d"]
        cur = list(o["src"])
        grabbed = False
        for step in range(abs(dr)):
            ops.append(20 if dr < 0 else 21)
            sels.append(sel_of(cur) if not grabbed else sel_of([]))
            grabbed = True
            cur = [(r + (-1 if dr < 0 else 1), c) for r, c in cur]
        for step in range(abs(dc)):
            ops.append(23 if dc < 0 else 22)
            sels.append(sel_of(cur) if not grabbed else sel_of([]))
            grabbed = True
            cur = [(r, c + (-1 if dc < 0 else 1)) for r, c in cur]
        # the grab zeroed the source footprint: restore the frame's background
        hole = sorted((set(o["src"]) - set(o["dst"])) & frame_rect)
        if bgc != 0 and hole:
            ops.append(bgc); sels.append(sel_of(hole))

    def emit_paint(o):
        # colour 0 pieces cannot be carried by Move (the object buffer keeps
        # only non-zero cells), so they are drawn at the destination instead
        ops.append(o["color"]); sels.append(sel_of(o["dst"]))
        rest = sorted((set(o["src"]) - set(o["dst"])) & frame_rect)
        if rest:
            ops.append(bgc); sels.append(sel_of(rest))

    # ---------------- order the pieces so none is overwritten -----------------
    pending = list(items)
    deferred = []
    while pending:
        chosen = None
        for idx, o in enumerate(pending):
            others = set()
            for k, p in enumerate(pending):
                if k != idx:
                    others |= set(p["src"])
            if not (set(o["dst"]) & others):
                chosen = idx
                break
        if chosen is None:
            # cyclic blockage: lift this piece out of the way first, draw later
            o = pending.pop(0)
            clear = sorted(set(o["src"]) & frame_rect)
            if clear:
                ops.append(bgc); sels.append(sel_of(clear))
            deferred.append(o)
        else:
            o = pending.pop(chosen)
            if o["color"] == 0:
                emit_paint(o)
            else:
                emit_move(o)

    for o in deferred:
        ops.append(o["color"]); sels.append(sel_of(o["dst"]))

    # ---------------- crop the completed frame -------------------------------
    # full rectangle on purpose: the whole gh x gw frame region, background included
    ops.append(33); sels.append([fr, fc, gh - 1, gw - 1])
    ops.append(34); sels.append([0, 0, gh - 1, gw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c8cbb738"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c8cbb738"
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
                                f"for task c8cbb738"
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
                    f"Failed to build a complete episode for task c8cbb738 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c8cbb738-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
