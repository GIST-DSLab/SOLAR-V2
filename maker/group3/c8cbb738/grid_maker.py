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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    cols = interval(0, 10, 1)
    gh = unifint(diff_lb, diff_ub, (3, max(3, min(10, max_h // 2))))
    gw = unifint(diff_lb, diff_ub, (3, max(3, min(10, max_w // 2))))
    h = unifint(diff_lb, diff_ub, (gh * 2, max(gh * 2, max_h)))
    w = unifint(diff_lb, diff_ub, (gw * 2, max(gw * 2, max_w)))
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (gh, gw))
    goinds = asindices(go)
    ring = box(goinds)
    crns = corners(ring)
    remring = ring - crns
    nrr = len(remring)
    sc = ccols[0]
    go = fill(go, sc, crns)
    loci = randint(0, h - gh)
    locj = randint(0, w - gw)
    gi = fill(gi, sc, shift(crns, (loci, locj)))
    ccols = ccols[1:]
    bL = connect((0, 0), (gh - 1, 0))
    bR = connect((0, gw - 1), (gh - 1, gw - 1))
    bT = connect((0, 0), (0, gw - 1))
    bB = connect((gh - 1, 0), (gh - 1, gw - 1))
    validpairs = [(bL, bT), (bL, bB), (bR, bT), (bR, bB)]
    for c in ccols:
        if len(remring) < 3:
            break
        obj = set(sample(totuple(remring), unifint(diff_lb, diff_ub, (3, max(3, min(len(remring), nrr // len(ccols)))))))
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


def derive_operations(I, O=None, examples=None):
    """Rule: one colour is four lone cells on the corners of a gh x gw frame
    (gh / gw = tallest / widest colour group). Every other colour is a fragment of
    that frame's border and lands where it covers the border best (ties -> nearest
    the frame centre).  Route: crop the canvas to the frame FIRST, wipe whatever
    stray colour happened to lie inside it, then copy each colour fragment out of
    the INPUT and paste it at its border slot.  Measured from I only."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    icols = sorted(int(v) for v in np.unique(I) if int(v) != bgc)
    cells = {c: sorted((int(r), int(q)) for r, q in zip(*np.where(I == c))) for c in icols}

    def bbox(cs):
        rs = [r for r, _ in cs]
        qs = [q for _, q in cs]
        return min(rs), min(qs), max(rs) - min(rs) + 1, max(qs) - min(qs) + 1

    gh = max(bbox(cells[c])[2] for c in icols)
    gw = max(bbox(cells[c])[3] for c in icols)

    fcol, R0, C0 = None, 0, 0
    for c in icols:
        r0, c0, hh, ww = bbox(cells[c])
        if len(cells[c]) == 4 and (hh, ww) == (gh, gw) and set(cells[c]) == {
                (r0, c0), (r0, c0 + ww - 1), (r0 + hh - 1, c0), (r0 + hh - 1, c0 + ww - 1)}:
            fcol, R0, C0 = c, r0, c0
            break

    ring = {(r, q) for r in range(gh) for q in range(gw)
            if r in (0, gh - 1) or q in (0, gw - 1)}
    K = 2 * max(gh, gw)
    cr, cc = gh // 2, gw // 2

    # frame-local destination of every fragment colour
    dest = {}
    for c in icols:
        if c == fcol:
            continue
        r0, c0, oh, ow = bbox(cells[c])
        norm = [(r - r0, q - c0) for r, q in cells[c]]
        best, bshift = None, None
        for i in range(gh):
            for j in range(gw):
                sh = {(a + i, b + j) for a, b in norm}
                sc = K * len(sh & ring) - (abs(i + oh // 2 - cr) + abs(j + ow // 2 - cc))
                if best is None or sc > best:
                    best, bshift = sc, (i, j)
        dest[c] = (bshift, sorted((bshift[0] + a, bshift[1] + b) for a, b in norm))

    # 1. crop the canvas to the frame window (a full rectangle: the frame itself)
    ops.append(33); sels.append([R0, C0, gh - 1, gw - 1])
    win = I[R0:R0 + gh, C0:C0 + gw].copy()   # crop is transparent: 0 stays 0

    # fragments that already sit exactly on their slot inside the window stay put
    settled = set()
    for c in dest:
        loc = sorted((r, q) for r, q in zip(*np.where(win == c)))
        if len(loc) == len(cells[c]) and loc == dest[c][1]:
            settled.add(c)

    # 2. wipe each stray colour caught inside the window (all its cells at once).
    #    A stray piece lying wholly on border slots that incoming fragments will
    #    fill is simply pasted over, so it needs no separate wipe.
    slots = set(p for c in dest if c not in settled for p in dest[c][1])
    for c in sorted(dest):
        if c in settled:
            continue
        stray = [(int(r), int(q)) for r, q in zip(*np.where(win == c))]
        if stray and c != bgc and not set(stray) <= slots:
            ops.append(int(bgc)); sels.append(sel_of(stray))
            for p in stray:
                win[p] = bgc

    # 3. bring every fragment colour from the input to its slot on the border
    for c in sorted(dest):
        if c in settled:
            continue
        (si, sj), tgt = dest[c]
        if c == 0:
            # 0 is invisible to Copy/Paste: paint the slot with Color0 instead
            ops.append(0); sels.append(sel_of(tgt))
        else:
            ops.append(28); sels.append(sel_of(cells[c]))   # CopyI: this colour's cells
            ops.append(30); sels.append([si, sj, 0, 0])     # Paste at slot top-left
        for p in tgt:
            win[p] = c

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
