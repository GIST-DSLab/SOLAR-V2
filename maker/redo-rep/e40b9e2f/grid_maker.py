"""
ARC Task: e40b9e2f (RE-ARC) — LLM-generated grid_maker
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
    """Episode-level colours.

    The generator samples: bgc (canvas colour) and ccols (the pattern's colours).
    The rule itself is purely geometric (complete the 4-fold rotational symmetry),
    but we still fix the palette so every instance of the episode looks alike.

    Colour 0 is kept out of the foreground palette whenever bgc != 0, so that
    'empty' and 'pattern' stay distinguishable for the copy/paste overlay.
    """
    cols = list(range(10))
    bgc = random.choice(cols)
    if bgc == 0:
        avail = [c for c in cols if c != 0]
    else:
        avail = [c for c in cols if c != 0 and c != bgc]
    palette = avail[:]
    random.shuffle(palette)
    return {"bgc": bgc, "palette": palette}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int = 0, palette=None, **kwargs) -> dict:
    if palette is None:
        palette = [c for c in range(10) if c != bgc and (bgc == 0 or c != 0)]
    pal = [c for c in palette if c != bgc and (bgc == 0 or c != 0)]
    if not pal:
        pal = [c for c in range(10) if c != bgc]

    hub = max(6, min(30, int(max_h)))
    wub = max(6, min(30, int(max_w)))

    h = unifint(diff_lb, diff_ub, (6, hub))
    w = unifint(diff_lb, diff_ub, (6, wub))
    d = unifint(diff_lb, diff_ub, (4, min(h, w) - 2))
    loci = randint(0, h - d)
    locj = randint(0, w - d)
    loc = (loci, locj)

    numcols = unifint(diff_lb, diff_ub, (1, len(pal)))
    ccols = pal[:numcols]

    subg = canvas(bgc, (d, d))
    inds = asindices(subg)
    if d % 2 == 0:
        q = sfilter(inds, lambda ij: ij[0] < d // 2 and ij[1] < d // 2)
        cp = {(d // 2 - 1, d // 2 - 1), (d // 2, d // 2 - 1),
              (d // 2 - 1, d // 2), (d // 2, d // 2)}
    else:
        q = sfilter(inds, lambda ij: ij[0] < d // 2 and ij[1] <= d // 2)
        cp = {(d // 2, d // 2)} | set(ineighbors((d // 2, d // 2)))

    nrings = unifint(diff_lb, diff_ub, (1, max(1, (d - 2) // 2)))
    rings = set()
    for k in range(nrings):
        ring = box({(k, k), (d - k - 1, d - k - 1)})
        rings = rings | set(ring)

    q = set(q)
    qin = q - rings
    qout = rings & q

    ntailobjcells = unifint(diff_lb, diff_ub, (1, len(q)))
    tailobjcells = sample(totuple(frozenset(q)), ntailobjcells)
    tailobjcells = set(tailobjcells) | {choice(totuple(frozenset(qin)))} | {choice(totuple(frozenset(qout)))}
    tailobj = frozenset({(choice(ccols), ij) for ij in tailobjcells})
    guard = 0
    while hmirror(tailobj) == tailobj and vmirror(tailobj) == tailobj and guard < 100:
        guard += 1
        ntailobjcells = unifint(diff_lb, diff_ub, (1, len(q)))
        tailobjcells = sample(totuple(frozenset(q)), ntailobjcells)
        tailobjcells = set(tailobjcells) | {choice(totuple(frozenset(qin)))} | {choice(totuple(frozenset(qout)))}
        tailobj = frozenset({(choice(ccols), ij) for ij in tailobjcells})

    for k in range(4):
        subg = paint(subg, tailobj)
        subg = rot90(subg)

    fxobj = recolor(choice(ccols), frozenset(cp))
    subg = paint(subg, fxobj)
    subgi = subg
    subgo = tuple(e for e in subgi)
    subgi = fill(subgi, bgc, frozenset(rings))

    nsplits = unifint(diff_lb, diff_ub, (1, 4))
    splits = [set() for k in range(nsplits)]
    for idx, cel in enumerate(tailobj):
        splits[idx % nsplits].add(cel)
    for jj in range(4):
        if jj < len(splits):
            subgi = paint(subgi, frozenset(splits[jj]))
        subgi = rot90(subgi)
    subgi = paint(subgi, fxobj)

    rotf = choice((identity, rot90, rot180, rot270))
    subgi = rotf(subgi)
    subgo = rotf(subgo)

    gi = paint(canvas(bgc, (h, w)), shift(asobject(subgi), loc))
    go = paint(canvas(bgc, (h, w)), shift(asobject(subgo), loc))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """The pattern lives in a square region and is invariant under quarter turns.
    Route: clear that square's background so it becomes 'empty', copy the visible
    fragment, then three times {quarter-turn the whole square, lay the original
    fragment back on top}.  That replicates the fragment into all four orientations.
    Finally restore the background in whatever cells stayed empty."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # --- background colour: any fully uniform row/col of I must be the canvas colour
    bgc = None
    for r in range(hi):
        if np.all(I[r] == I[r, 0]):
            bgc = int(I[r, 0])
            break
    if bgc is None:
        for c in range(wi):
            if np.all(I[:, c] == I[0, c]):
                bgc = int(I[0, c])
                break
    if bgc is None:
        bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # --- the square the pattern occupies (a 4-fold symmetric figure has a square extent)
    fg = np.argwhere(O != bgc)
    if len(fg) == 0:
        return [34], [[0, 0, ho - 1, wo - 1]]
    r0 = int(fg[:, 0].min()); r1 = int(fg[:, 0].max())
    c0 = int(fg[:, 1].min()); c1 = int(fg[:, 1].max())
    n = max(r1 - r0 + 1, c1 - c0 + 1)
    n = min(n, hi, wi)
    r0 = min(max(0, r0 - (n - (r1 - r0 + 1)) // 2), hi - n)
    c0 = min(max(0, c0 - (n - (c1 - c0 + 1)) // 2), wi - n)
    block = [r0, c0, n - 1, n - 1]   # bbox == exactly the whole square region we rotate/copy

    ops, sels = [], []
    work = I.copy()

    # 1. empty out the square's background so the overlays can be transparent
    if bgc != 0:
        bgcells = [(r, c) for r in range(r0, r0 + n) for c in range(c0, c0 + n)
                   if int(work[r, c]) == bgc]
        if bgcells:
            ops.append(0); sels.append(sel_of(bgcells))
            for r, c in bgcells:
                work[r, c] = 0

    # 2. take the visible fragment of the pattern
    ops.append(28 if bgc == 0 else 29); sels.append(list(block))

    A = {(r, c): int(work[r, c]) for r in range(r0, r0 + n) for c in range(c0, c0 + n)
         if int(work[r, c]) != 0}
    cur = dict(A)

    # 3. three quarter-turns of the square, each followed by laying the fragment back on
    for _ in range(3):
        ops.append(25); sels.append(list(block))            # turn the whole square
        cur = {(r0 + (c - c0), c0 + (n - 1 - (r - r0))): v for (r, c), v in cur.items()}
        ops.append(30); sels.append([r0, c0, 0, 0])         # overlay the original fragment
        cur.update(A)

    for r in range(r0, r0 + n):
        for c in range(c0, c0 + n):
            work[r, c] = 0
    for (r, c), v in cur.items():
        work[r, c] = v

    # 4. cells of the square that stayed empty go back to the canvas colour
    if bgc != 0:
        empt = [(r, c) for r in range(r0, r0 + n) for c in range(c0, c0 + n)
                if int(work[r, c]) == 0]
        if empt:
            ops.append(bgc); sels.append(sel_of(empt))
            for r, c in empt:
                work[r, c] = bgc

    # safety net (should be empty)
    diff = np.argwhere(work != O)
    if len(diff):
        by = {}
        for r, c in diff:
            by.setdefault(int(O[r, c]), []).append((int(r), int(c)))
        for v, cells in by.items():
            ops.append(v); sels.append(sel_of(cells))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e40b9e2f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e40b9e2f"
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
                                f"for task e40b9e2f"
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
                    f"Failed to build a complete episode for task e40b9e2f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e40b9e2f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
