"""
ARC Task: 98cf29f8 (RE-ARC) — LLM-generated grid_maker
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
from itertools import permutations
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, objc, otherc = random.sample(cols, 3)
    return {"bgc": bgc, "objc": objc, "otherc": otherc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, objc: int, otherc: int) -> dict:
    # the final grid may be transposed/rotated, so bound both sides by the tighter limit
    lim = max(10, min(max_h, max_w))
    h = unifint(diff_lb, diff_ub, (10, lim))
    w = unifint(diff_lb, diff_ub, (10, lim))
    objh = unifint(diff_lb, diff_ub, (2, h - 5))
    objw = unifint(diff_lb, diff_ub, (2, w - 5))
    loci = random.randint(0, h - objh)
    locj = random.randint(0, w - objw)
    obj = backdrop(frozenset({(loci, locj), (loci + objh - 1, locj + objw - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, objc, obj)
    bmarg = h - (loci + objh)
    rmarg = w - (locj + objw)
    tmarg = loci
    lmarg = locj
    margs = (bmarg, rmarg, tmarg, lmarg)
    options = [idx for idx, marg in enumerate(margs) if marg > 2]
    pos = random.choice(options)
    for k in range(pos):
        gi = rot90(gi)
    h, w = shape(gi)
    ofc = ofcolor(gi, objc)
    locis = random.randint(lowermost(ofc) + 2, h - 2)
    locie = random.randint(locis + 1, h - 1)
    locjs = random.randint(0, min(w - 2, rightmost(ofc)))
    locje = random.randint(max(locjs + 1, leftmost(ofc)), w - 1)
    otherobj = backdrop(frozenset({(locis, locjs), (locie, locje)}))
    ub = min(rightmost(ofc), rightmost(otherobj))
    lb = max(leftmost(ofc), leftmost(otherobj))
    jloc = random.randint(lb, ub)
    ln = connect((lowermost(ofc) + 1, jloc), (uppermost(otherobj) - 1, jloc))
    gib = tuple(e for e in gi)
    gi = fill(gi, otherc, otherobj)
    gi = fill(gi, otherc, ln)
    go = fill(gib, otherc, shift(otherobj, (-len(ln), 0)))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = random.choice((1, 2))
    for fn in random.sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """A solid anchor rectangle, a block, and a 1-cell-wide connector running from the
    block to the anchor.  The block slides down the connector until it touches the
    anchor; the connector itself is scaffolding and goes back to background.

    The rule is fixed, but the ROUTE through it is not, and the input picks which one
    this instance takes: clear the connector first and slide the block onto the cleared
    lane, or slide first and tidy only what the block left showing; and when two
    separate patches need background, which one is done first and whether they go down
    as one stroke or two.  Nothing here reads O.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    submit_sel = [0, 0, hi - 1, wi - 1]

    counts = Counter(I.flatten().tolist())
    colors = sorted(counts)

    def parse(anchor_c, block_c):
        A = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == anchor_c]
        B = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == block_c]
        if len(A) < 4 or len(B) < 5:
            return None
        ar0 = min(p[0] for p in A); ar1 = max(p[0] for p in A)
        ac0 = min(p[1] for p in A); ac1 = max(p[1] for p in A)
        if len(A) != (ar1 - ar0 + 1) * (ac1 - ac0 + 1):
            return None                      # the anchor is a full rectangle
        rowc = Counter(p[0] for p in B)
        colc = Counter(p[1] for p in B)
        thin_r = {r for r, n in rowc.items() if n == 1}
        thin_c = {c for c, n in colc.items() if n == 1}
        # the block is at least 2x2, so only the connector can leave a 1-cell line/column
        if thin_r and not thin_c:
            stem = [p for p in B if p[0] in thin_r]
            axis = 0
        elif thin_c and not thin_r:
            stem = [p for p in B if p[1] in thin_c]
            axis = 1
        else:
            return None
        sset = set(stem)
        rect = [p for p in B if p not in sset]
        if not rect:
            return None
        br0 = min(p[0] for p in rect); br1 = max(p[0] for p in rect)
        bc0 = min(p[1] for p in rect); bc1 = max(p[1] for p in rect)
        if len(rect) != (br1 - br0 + 1) * (bc1 - bc0 + 1):
            return None                      # the block is a full rectangle too
        n = len(stem)
        if axis == 0:
            js = {p[1] for p in stem}
            rs = sorted(p[0] for p in stem)
            if len(js) != 1 or rs[-1] - rs[0] + 1 != n:
                return None
            j = js.pop()
            if not (bc0 <= j <= bc1 and ac0 <= j <= ac1):
                return None
            if rs[-1] == br0 - 1 and ar1 == rs[0] - 1:
                step = (-1, 0)               # connector leaves the block upward
            elif rs[0] == br1 + 1 and ar0 == rs[-1] + 1:
                step = (1, 0)
            else:
                return None
        else:
            iss = {p[0] for p in stem}
            cs = sorted(p[1] for p in stem)
            if len(iss) != 1 or cs[-1] - cs[0] + 1 != n:
                return None
            i0 = iss.pop()
            if not (br0 <= i0 <= br1 and ar0 <= i0 <= ar1):
                return None
            if cs[-1] == bc0 - 1 and ac1 == cs[0] - 1:
                step = (0, -1)
            elif cs[0] == bc1 + 1 and ac0 == cs[-1] + 1:
                step = (0, 1)
            else:
                return None
        return {"rect": rect, "stem": stem, "step": step, "nsteps": n,
                "block": block_c, "bbox": (br0, bc0, br1, bc1)}

    found = None
    for ca, cb in permutations(colors, 2):
        res = parse(ca, cb)
        if res is None:
            continue
        others = [c for c in colors if c != ca and c != cb]
        if not others:
            continue
        res["bg"] = max(others, key=lambda c: counts[c])
        if found is None or counts[res["bg"]] > counts[found["bg"]]:
            found = res

    if found is None:
        return [34], [submit_sel]

    bg = int(found["bg"])
    blockc = int(found["block"])
    rect = sorted(found["rect"])
    stem = sorted(found["stem"])
    dr, dc = found["step"]
    n = found["nsteps"]
    br0, bc0, br1, bc1 = found["bbox"]

    dst = set((r + dr * n, c + dc * n) for r, c in rect)
    hole = sorted(set(rect) - dst)          # footprint the block no longer covers
    leftover = sorted(set(stem) - dst)      # connector the block does not end up burying

    # which route this instance takes — read off the figure's own geometry
    lay_base = ((br0 + bc0 + br1 + bc1 + n) % 2 == 0)
    hole_first = (((br1 - br0) + n) % 2 == 0)
    one_stroke = ((hi + wi + n) % 2 == 0)

    def tidy(regions):
        """Put background back over the listed patches, in the order this route chose,
        as one stroke or as separate ones."""
        regions = [rg for rg in regions if rg]
        if not regions:
            return
        if not hole_first:
            regions = regions[::-1]
        if one_stroke and len(regions) > 1:
            cells = sorted(set().union(*[set(rg) for rg in regions]))
            ops.append(bg); sels.append(sel_of(cells))
        else:
            for rg in regions:
                ops.append(bg); sels.append(sel_of(rg))

    if blockc != 0:
        move_op = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]

        def slide():
            ops.append(move_op); sels.append(sel_of(rect))   # first Move grabs the block
            for _ in range(n - 1):
                ops.append(move_op); sels.append(sel_of([]))  # empty -> same object keeps going

        if lay_base:
            # wipe the whole connector back to background, then slide the block
            # down the lane that was just cleared
            ops.append(bg); sels.append(sel_of(stem))
            slide()
            # ARCLE leaves the block's old footprint at 0; only patch it when 0 isn't bg
            tidy([hole if bg != 0 else []])
        else:
            # slide straight away, then tidy whatever the block left showing:
            # its vacated footprint and any connector it did not cover
            slide()
            tidy([hole if bg != 0 else [], leftover])
    else:
        # a colour-0 block is "nothing there" to ARCLE's object buffer — it cannot be
        # carried by a Move, so the block (and its equally-0 connector) is put down
        # with an explicit Color0 where it lands
        dst_cells = sorted(dst)
        old_rect = sorted(set(rect) - dst)
        old_stem = sorted(set(stem) - dst)
        if lay_base:
            # sweep the whole old figure back to background, then set the block down
            ops.append(bg); sels.append(sel_of(sorted(set(rect) | set(stem))))
            ops.append(0); sels.append(sel_of(dst_cells))
        else:
            # set the block down in its new place first, then clear what still shows
            ops.append(0); sels.append(sel_of(dst_cells))
            tidy([old_rect, old_stem])

    ops.append(34)
    sels.append(submit_sel)
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
                        f"num_examples+1 ({num_examples + 1}) for task 98cf29f8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 98cf29f8"
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
                                f"for task 98cf29f8"
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
                    f"Failed to build a complete episode for task 98cf29f8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"98cf29f8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
