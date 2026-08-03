"""
ARC Task: 2dd70a9a (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- 1. colors
VARIANTS = [{"case": True}, {"case": False}]   # generator's two structural branches


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 3)]
    bgc, fgc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "fgc": fgc, "instance_plan": plan}


# ---------------------------------------------------------------- 2. generate
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc, fgc, case=None) -> dict:
    if case is None:
        case = choice((True, False))

    hub = max(10, max_h)
    wub = max(10, max_w)
    h = unifint(diff_lb, diff_ub, (10, hub))
    w = unifint(diff_lb, diff_ub, (10, wub))

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))

    if case:
        oh = unifint(diff_lb, diff_ub, (5, h - 2))
        ow = unifint(diff_lb, diff_ub, (3, w - 2))
        loci = randint(1, h - oh - 1)
        locj = randint(1, w - ow - 1)
        hli = randint(loci + 2, loci + oh - 3)
        sp = {(loci + oh - 1, locj), (loci + oh - 2, locj)}
        ep = {(loci, locj + ow - 1), (loci + 1, locj + ow - 1)}
        bp1 = (hli - 1, locj)
        bp2 = (hli, locj + ow)
        ln1 = connect((loci + oh - 1, locj), (hli, locj))
        ln2 = connect((hli, locj), (hli, locj + ow - 1))
        ln3 = connect((hli, locj + ow - 1), (loci + 2, locj + ow - 1))
    else:
        oh = unifint(diff_lb, diff_ub, (3, h - 2))
        ow = unifint(diff_lb, diff_ub, (3, w - 2))
        loci = randint(1, h - oh - 1)
        locj = randint(1, w - ow - 1)
        if choice((True, False)):
            sp1j = randint(locj, locj + ow - 3)
            ep1j = locj
        else:
            ep1j = randint(locj, locj + ow - 3)
            sp1j = locj
        sp = {(loci, sp1j), (loci, sp1j + 1)}
        ep = {(loci + oh - 1, ep1j), (loci + oh - 1, ep1j + 1)}
        bp1 = (loci, locj + ow)
        bp2 = (loci + oh, locj + ow - 1)
        ln1 = connect((loci, sp1j + 2), (loci, locj + ow - 1))
        ln2 = connect((loci, locj + ow - 1), (loci + oh - 1, locj + ow - 1))
        ln3 = connect((loci + oh - 1, ep1j + 2), (loci + oh - 1, locj + ow - 1))

    gi = fill(gi, 3, sp)
    gi = fill(gi, 2, ep)
    go = fill(go, 3, sp)
    go = fill(go, 2, ep)
    lns = ln1 | ln2 | ln3
    bps = {bp1, bp2}
    gi = fill(gi, fgc, bps)
    go = fill(go, fgc, bps)
    go = fill(go, 3, lns)
    inds = ofcolor(go, bgc)
    namt = unifint(diff_lb, diff_ub, (0, len(inds) // 2))
    noise = sample(totuple(inds), namt)
    gi = fill(gi, fgc, noise)
    go = fill(go, fgc, noise)

    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)

    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- 3. ops
def derive_operations(I, O):
    """
    Rule (measured from I only):
      * a 3-coloured domino and a 2-coloured domino sit on a noisy background;
      * shoot a ray OUT of the free end of the 3-domino, along the domino's own
        axis, until it is stopped by a non-background pixel  -> corner1;
      * turn perpendicular, toward the 2-domino, and run until stopped -> corner2
        (this always lands on the line the 2-domino points along);
      * turn again and run along that line straight into the 2-domino.
      The three straight runs are painted with colour 3.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    m3 = [tuple(p) for p in np.argwhere(I == 3)]
    m2 = [tuple(p) for p in np.argwhere(I == 2)]
    set2 = set(m2)

    def axis(cells):
        rs = {r for r, _ in cells}
        return (1, 0) if len(rs) > 1 else (0, 1)

    def inside(r, c):
        return 0 <= r < h and 0 <= c < w

    def ray(start, d):
        """walk from `start` (exclusive) in direction d over background cells.
        returns (cells, blocker) ; blocker is None if we ran off the grid."""
        cells = []
        r, c = start[0] + d[0], start[1] + d[1]
        while inside(r, c) and I[r, c] == bgc:
            cells.append((r, c))
            r += d[0]
            c += d[1]
        blocker = (r, c) if inside(r, c) else None
        return cells, blocker

    ops, sels = [], []
    path = None

    if len(m3) >= 2 and len(m2) >= 2:
        a3 = axis(m3)
        a2 = axis(m2)
        line2 = m2[0][0] if a2 == (0, 1) else m2[0][1]     # the row/col the 2-domino points along
        perps = [(0, 1), (0, -1)] if a3 == (1, 0) else [(1, 0), (-1, 0)]

        for d in (a3, (-a3[0], -a3[1])):
            # free end of the 3-domino in direction d
            head = max(m3, key=lambda p: p[0] * d[0] + p[1] * d[1])
            seg1, blk1 = ray(head, d)
            if blk1 is None or not seg1:
                continue
            corner1 = seg1[-1]
            for e in perps:
                seg2, blk2 = ray(corner1, e)
                if blk2 is None or not seg2:
                    continue
                corner2 = seg2[-1]
                on_line = (corner2[0] == line2) if a2 == (0, 1) else (corner2[1] == line2)
                if not on_line:
                    continue
                # final run: along the 2-domino's own axis, into the 2-domino
                tgt = min(m2, key=lambda p: abs(p[0] - corner2[0]) + abs(p[1] - corner2[1]))
                if a2 == (0, 1):
                    f = (0, 1 if tgt[1] > corner2[1] else -1)
                else:
                    f = (1 if tgt[0] > corner2[0] else -1, 0)
                seg3, blk3 = ray(corner2, f)
                if blk3 is None or blk3 not in set2:
                    continue
                path = (seg1, seg2, seg3)
                break
            if path:
                break

    if path is None:
        # degenerate instance: fall back to the changed cells, kept as one region
        diff = [(r, c) for r in range(h) for c in range(w) if I[r, c] != O[r, c]]
        if diff:
            ops.append(3)
            sels.append(sel_of(diff))
    else:
        seg1, seg2, seg3 = path
        # 1) ray out of the 3-domino
        ops.append(3)
        sels.append(sel_of(seg1))
        # 2) turn, ray up to the 2-domino's frontier
        ops.append(3)
        sels.append(sel_of(seg2))
        # 3) turn, ray into the 2-domino
        if seg3:
            ops.append(3)
            sels.append(sel_of(seg3))

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 2dd70a9a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2dd70a9a"
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
                                f"for task 2dd70a9a"
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
                    f"Failed to build a complete episode for task 2dd70a9a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2dd70a9a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
