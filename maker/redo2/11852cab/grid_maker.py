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
    """Background and the palette every ring is drawn from, fixed for the episode.

    The rule (complete each broken ring by reflecting its patch onto itself) is
    colour-independent, so only the roles have to stay stable across instances."""
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    numc = random.randint(1, len(rem))
    ccols = random.sample(rem, numc)
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
    fallback = None
    for _attempt in range(60):
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
                # the inner ring has to be one of the broken ones unless it is the
                # complete one: a figure with nothing on it is not a ring figure
                # any more, and the task's own verifier leaves such a patch alone
                if idx != 1:
                    numrs = 2
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
        # Two figures two cells apart can line up so that a window straddling
        # them reads as a ring figure of its own; completing that phantom is
        # not the rule anyone would infer, so such a grid is redrawn.
        gia = np.array(gi, dtype=int)
        if not np.array_equal(_complete_rings(gia, bgc), np.array(go, dtype=int)):
            continue
        res = {'input': gi, 'output': go}
        # a patch whose rings are all complete already leaves nothing to reflect
        if gi != go:
            return res
        fallback = res
    if fallback is None:
        raise ValueError('no clean ring layout')
    return fallback


# ── the ring lattice of a 5x5 patch ───────────────────────────────────────────
_BOX5 = frozenset((i, j) for i in range(5) for j in range(5)
                  if i in (0, 4) or j in (0, 4))
_INBOX5 = frozenset((i, j) for i in range(1, 4) for j in range(1, 4)
                    if (i, j) != (2, 2))


def _ring_patches(I, bgc):
    """Every 5x5 patch of I that holds a concentric-ring figure.

    Measured on I alone: the figure sits on the even (i+j) lattice of the patch,
    its centre is lit, it reaches the patch border and the ring just inside it,
    it spans the patch in both directions, and the one-cell frame around the
    patch is clear — which is what keeps the figure a self-contained object.
    """
    h, w = I.shape
    found = []
    for r0 in range(1, h - 5):
        for c0 in range(1, w - 5):
            P = I[r0:r0 + 5, c0:c0 + 5]
            cont = [(i, j) for i in range(5) for j in range(5) if P[i, j] != bgc]
            if not cont:
                continue
            if any((i + j) % 2 for i, j in cont):
                continue
            if P[2, 2] == bgc:
                continue
            if not any((i, j) in _BOX5 for i, j in cont):
                continue
            if not any((i, j) in _INBOX5 for i, j in cont):
                continue
            rs = [i for i, _ in cont]
            cs = [j for _, j in cont]
            if min(rs) or max(rs) != 4 or min(cs) or max(cs) != 4:
                continue
            frame = [(r, c) for r in range(r0 - 1, r0 + 6)
                     for c in (c0 - 1, c0 + 5)]
            frame += [(r, c) for c in range(c0 - 1, c0 + 6)
                      for r in (r0 - 1, r0 + 5)]
            if any(I[r, c] != bgc for r, c in frame):
                continue
            found.append((r0, c0))
    return found


def _d4_closure(patch, bgc):
    """The patch with every ring it holds completed: each drawn cell reflected
    into the whole orbit its position has under the patch's own symmetries."""
    out = patch.copy()
    for i in range(5):
        for j in range(5):
            v = int(patch[i, j])
            if v == bgc:
                continue
            for (a, b) in ((i, j), (i, 4 - j), (4 - i, j), (4 - i, 4 - j),
                           (j, i), (j, 4 - i), (4 - j, i), (4 - j, 4 - i)):
                if out[a, b] == bgc:
                    out[a, b] = v
    return out


def _complete_rings(grid, bgc):
    """The rule itself, on a whole grid — used to keep generate() honest."""
    out = grid.copy()
    for (r0, c0) in _ring_patches(grid, bgc):
        out[r0:r0 + 5, c0:c0 + 5] = _d4_closure(grid[r0:r0 + 5, c0:c0 + 5], bgc)
    return out


def derive_operations(I, O):
    """Complete every broken ring by reflecting its own patch onto itself.

    Each figure is a set of concentric 4-fold-symmetric rings on the even cells
    of a 5x5 patch, and the ones that are broken have kept enough of themselves
    to say where the rest goes. So for each patch found in I: keep a copy of what
    is drawn there, mirror the whole patch left-right, and lay the kept copy back
    on top — the mirror image supplies the cells the ring was missing on the far
    side and the copy keeps the ones it had. The same over up-down, then a
    quarter turn, and the figure is symmetric under every reflection its rings
    have. Which patches, and how many of the three reflections each one needs,
    are read off I; O is never consulted.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])
    ops, sels = [], []
    G = I.copy()

    for (r0, c0) in _ring_patches(I, bgc):
        cells5 = [(i, j) for i in range(5) for j in range(5)]
        patch = G[r0:r0 + 5, c0:c0 + 5].copy()
        whole = _d4_closure(patch, bgc)
        if np.array_equal(whole, patch):
            continue                                  # every ring here is already whole

        # Copy/Paste carry nothing at colour 0, so a ring that is broken and
        # drawn in 0 over a non-0 background is parked on a spare colour for as
        # long as it is being reflected.
        spare = None
        if bgc != 0 and not np.array_equal(patch == 0, whole == 0):
            zeros = [(i, j) for (i, j) in cells5 if G[r0 + i, c0 + j] == 0]
            if zeros:
                taken = set(G[r0:r0 + 5, c0:c0 + 5].flatten().tolist()) | {0, bgc}
                spare = next(k for k in range(1, 10) if k not in taken)
                ops.append(spare)
                sels.append(sel_of([(r0 + i, c0 + j) for (i, j) in zeros]))
                for (i, j) in zeros:
                    G[r0 + i, c0 + j] = spare

        for opcode, mirror in ((26, np.fliplr), (27, np.flipud),
                               (24, lambda a: np.rot90(a, 1))):
            blk = G[r0:r0 + 5, c0:c0 + 5].copy()
            drawn = [(i, j) for (i, j) in cells5 if blk[i, j] != bgc]
            mirrored = np.array(mirror(blk))
            union = mirrored.copy()
            for (i, j) in drawn:                      # the kept copy, laid back on top
                union[i, j] = blk[i, j]
            if np.array_equal(union, blk):
                continue                              # this reflection adds nothing here
            ops.append(29)                            # keep what is drawn on the patch
            sels.append(sel_of([(r0 + i, c0 + j) for (i, j) in drawn]))
            ops.append(opcode)                        # bbox == exactly the 5x5 patch
            sels.append([r0, c0, 4, 4])               # reflected whole, background included
            ops.append(30)                            # lay the kept copy back over it
            sels.append([r0, c0, 0, 0])
            G[r0:r0 + 5, c0:c0 + 5] = union

        if spare is not None:                         # give the parked ring its 0 back
            back = [(i, j) for (i, j) in cells5 if G[r0 + i, c0 + j] == spare]
            ops.append(0)
            sels.append(sel_of([(r0 + i, c0 + j) for (i, j) in back]))
            for (i, j) in back:
                G[r0 + i, c0 + j] = 0

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
