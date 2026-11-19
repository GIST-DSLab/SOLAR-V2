"""
ARC Task: 11852cab (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    """Fix background and the palette the ring-patterns are drawn from.

    0 is kept out of the object palette whenever bgc != 0: the completion is done by
    Copy/Flip/Paste layering, and Copy/Paste treat 0 as 'nothing'."""
    cols = list(range(10))
    bgc = random.choice(cols)
    pool = [c for c in cols if c != bgc and c != 0]
    numc = random.randint(1, len(pool))
    ccols = random.sample(pool, numc)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols, **kwargs) -> dict:
    r1 = ((0, 0), (0, 4), (4, 0), (4, 4))
    r2 = ((2, 0), (0, 2), (4, 2), (2, 4))
    r3 = ((1, 1), (3, 1), (1, 3), (3, 3))
    r4 = ((2, 2),)
    rings = [r4, r3, r2, r1]
    bx = backdrop(frozenset(r1))
    hub = max(7, min(30, int(max_h)))
    wub = max(7, min(30, int(max_w)))
    ccols = list(ccols)
    res = None
    for _attempt in range(40):
        h = unifint(diff_lb, diff_ub, (7, hub))
        w = unifint(diff_lb, diff_ub, (7, wub))
        gi = canvas(bgc, (h, w))
        go = canvas(bgc, (h, w))
        inds = shift(asindices(trim(gi)), (1, 1))
        nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 36)))
        succ = 0
        tr = 0
        maxtr = 10 * nobjs
        while succ < nobjs and tr < maxtr:
            tr += 1
            cands = sfilter(inds, lambda ij: ij[0] <= h - 5 and ij[0] <= w - 5)
            if len(cands) == 0:
                break
            loc = choice(totuple(cands))
            plcd = shift(bx, loc)
            if plcd.issubset(inds):
                inds = (inds - plcd) - outbox(plcd)
                ringcols = [choice(ccols) for k in range(4)]
                plcdrings = [shift(r, loc) for r in rings]
                gi = fill(gi, ringcols[0], plcdrings[0])
                go = fill(go, ringcols[0], plcdrings[0])
                idx = randint(1, 3)
                gi = fill(gi, ringcols[idx], plcdrings[idx])
                go = fill(go, ringcols[idx], plcdrings[idx])
                remrings = plcdrings[1:idx] + plcdrings[idx + 1:]
                remringcols = ringcols[1:idx] + ringcols[idx + 1:]
                numrs = unifint(diff_lb, diff_ub, (1, 2))
                locs = sample((0, 1), numrs)
                remrings = [rr for j, rr in enumerate(remrings) if j in locs]
                remringcols = [rr for j, rr in enumerate(remringcols) if j in locs]
                tofillgi = merge(frozenset(
                    recolor(col, frozenset(sample(totuple(remring),
                                                  4 - unifint(diff_lb, diff_ub, (0, 3)))))
                    for remring, col in zip(remrings, remringcols)
                ))
                tofillgo = merge(frozenset(
                    recolor(col, remring) for remring, col in zip(remrings, remringcols)
                ))
                if min(shape(tofillgi)) == 5:
                    succ += 1
                    gi = paint(gi, tofillgi)
                    go = paint(go, tofillgo)
        res = {'input': gi, 'output': go}
        if gi != go:
            return res
    return res


def derive_operations(I, O):
    """Each 5x5 patch holds concentric 4-fold-symmetric rings; broken rings are
    completed by reflecting the patch onto itself (mirror h, mirror v, then a quarter
    turn), each reflection layered over the kept copy of the patch."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])
    ops, sels = [], []

    R1 = [(0, 0), (0, 4), (4, 0), (4, 4)]
    R2 = [(0, 2), (2, 0), (2, 4), (4, 2)]
    R3 = [(1, 1), (1, 3), (3, 1), (3, 3)]
    R4 = [(2, 2)]
    RINGSET = set(R1 + R2 + R3 + R4)

    def d4_imgs(r, c):
        return [(r, c), (r, 4 - c), (4 - r, c), (4 - r, 4 - c),
                (c, r), (c, 4 - r), (4 - c, r), (4 - c, 4 - r)]

    def closure(sub):
        out = np.zeros((5, 5), dtype=int)
        for r in range(5):
            for c in range(5):
                v = int(sub[r, c])
                if v != bgc:
                    for (rr, cc) in d4_imgs(r, c):
                        out[rr, cc] = v
        return out

    def valid(r0, c0):
        sub = I[r0:r0 + 5, c0:c0 + 5]
        if sub[2, 2] == bgc:
            return False
        for r in range(5):
            for c in range(5):
                if sub[r, c] != bgc and (r, c) not in RINGSET:
                    return False
        full = False
        for ring in (R1, R2, R3):
            vals = [int(sub[r, c]) for (r, c) in ring if sub[r, c] != bgc]
            if len(set(vals)) > 1:
                return False
            if len(vals) == 4:
                full = True
        if not full:
            return False
        for r in range(r0 - 1, r0 + 6):
            for c in range(c0 - 1, c0 + 6):
                if r0 <= r < r0 + 5 and c0 <= c < c0 + 5:
                    continue
                if 0 <= r < h and 0 <= c < w and I[r, c] != bgc:
                    return False
        tgt = O[r0:r0 + 5, c0:c0 + 5]
        return np.array_equal(closure(sub), np.where(tgt == bgc, 0, tgt))

    chset = {(r, c) for r in range(h) for c in range(w) if I[r, c] != O[r, c]}

    blocks = []
    if chset:
        scored = []
        for r0 in range(h - 4):
            for c0 in range(w - 4):
                cnt = sum(1 for (r, c) in chset if r0 <= r < r0 + 5 and c0 <= c < c0 + 5)
                if cnt and valid(r0, c0):
                    scored.append((cnt, r0, c0))
        scored.sort(key=lambda t: (-t[0], t[1], t[2]))
        used, covered = set(), set()
        for cnt, r0, c0 in scored:
            cells = {(r0 + i, c0 + j) for i in range(5) for j in range(5)}
            if cells & used:
                continue
            newly = (chset & cells) - covered
            if not newly:
                continue
            used |= cells
            covered |= newly
            blocks.append((r0, c0))
        blocks.sort()

    G = I.copy()
    for (r0, c0) in blocks:
        tgt = O[r0:r0 + 5, c0:c0 + 5].copy()
        if np.array_equal(G[r0:r0 + 5, c0:c0 + 5], tgt):
            continue
        if bgc != 0 and np.any(G[r0:r0 + 5, c0:c0 + 5] == 0):
            continue  # 0 used as a ring colour here -> clipboard layering unusable

        # clear this patch's background so the mirrored copies can be layered on it
        if bgc != 0:
            bgcells = [(r0 + i, c0 + j) for i in range(5) for j in range(5)
                       if G[r0 + i, c0 + j] == bgc]
            if bgcells:
                ops.append(0)
                sels.append(sel_of(bgcells))
                for (r, c) in bgcells:
                    G[r, c] = 0

        target_obj = np.where(tgt == bgc, 0, tgt)
        # mirror the patch onto itself: left<->right, up<->down, then a quarter turn
        for opcode, fn in ((26, np.fliplr), (27, np.flipud), (24, lambda a: np.rot90(a, 1))):
            blk = G[r0:r0 + 5, c0:c0 + 5].copy()
            if np.array_equal(blk, target_obj):
                break
            t = np.array(fn(blk))
            if np.array_equal(t, blk):
                continue  # patch already symmetric this way: the op would change nothing
            ops.append(29)
            sels.append([r0, c0, 4, 4])          # keep a copy of the whole 5x5 patch
            ops.append(opcode)
            sels.append([r0, c0, 4, 4])          # bbox == exactly the 5x5 patch, reflected whole
            ops.append(30)
            sels.append([r0, c0, 0, 0])          # lay the kept copy back over the reflection
            union = t.copy()
            union[blk != 0] = blk[blk != 0]
            G[r0:r0 + 5, c0:c0 + 5] = union

        # put the patch's background back
        if bgc != 0:
            zc = [(r0 + i, c0 + j) for i in range(5) for j in range(5)
                  if G[r0 + i, c0 + j] == 0]
            if zc:
                ops.append(int(bgc))
                sels.append(sel_of(zc))
                for (r, c) in zc:
                    G[r, c] = bgc

    # safety net: anything a patch could not be reflected into place (never hit for
    # normal instances) gets painted per colour
    rem = {}
    for r in range(h):
        for c in range(w):
            if G[r, c] != O[r, c]:
                rem.setdefault(int(O[r, c]), []).append((r, c))
    for col in sorted(rem):
        ops.append(int(col))
        sels.append(sel_of(rem[col]))
        for (r, c) in rem[col]:
            G[r, c] = col

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
                        f"num_examples+1 ({num_examples + 1}) for task 11852cab"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 11852cab"
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
                                f"for task 11852cab"
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
                    f"Failed to build a complete episode for task 11852cab "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"11852cab-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
