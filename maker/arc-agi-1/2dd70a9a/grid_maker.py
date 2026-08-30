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
from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colors / plan
# Discrete structural variant of this task: the two 2-cell markers are either a
# HORIZONTAL pair (the connecting path runs H-V-H and can be drawn straight away)
# or a VERTICAL pair (the verifier's `vline -> dmirror` branch: the grid has to be
# transposed first, the path drawn, and the grid transposed back).
VARIANTS = [{"vertical_markers": True}, {"vertical_markers": False}]


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


# ---------------------------------------------------------------- generator
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, fgc=None, vertical_markers=None) -> dict:
    if vertical_markers is None:
        vertical_markers = choice((True, False))
    if bgc is None or fgc is None:
        cols = difference(interval(0, 10, 1), (2, 3))
        bgc, fgc = sample(cols, 2)

    # the final random mirrors may transpose the grid, so keep both dims bounded
    hb = max(10, min(max_h, max_w))
    h = unifint(diff_lb, diff_ub, (10, hb))
    w = unifint(diff_lb, diff_ub, (10, hb))

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))

    if choice((True, False)):
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

    # force the planned marker orientation (the discrete variant of this task)
    twos = [(i, j) for i, row in enumerate(gi) for j, v in enumerate(row) if v == 2]
    is_vert = (twos[0][1] == twos[1][1])
    if is_vert != bool(vertical_markers):
        gi = dmirror(gi)
        go = dmirror(go)

    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- trajectory
def derive_operations(I, O):
    """
    Rule: the 3-marker and the 2-marker each shoot a ray along their own line; the
    rays are joined by a perpendicular connector, and the whole elbow is drawn in 3.

    The rays run along ROWS only when the marker pairs lie horizontally.  When the
    2-marker is a vertical pair the grid is first REFLECTED onto its main diagonal
    (dmirror = rotate CCW + flip up-down, padded to a square when the grid is not
    square), the elbow is drawn there, and the same reflection carries the result
    back.  That reflection is performed by real ops; only the elbow is painted.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    two_cells = [(int(r), int(c)) for r, c in zip(*np.where(I == 2))]
    vertical = len({c for _, c in two_cells}) == 1

    def transpose_ops(h, w):
        """dmirror of the whole canvas; every selection is a FULL rectangle."""
        o, s = [], []
        if h == w:
            full = [0, 0, h - 1, w - 1]          # whole grid rectangle
            o.append(24); s.append(full)         # rotate CCW
            o.append(27); s.append(full)         # flip up-down -> transpose
        else:
            sq = max(h, w)
            o.append(33); s.append([0, 0, sq - 1, sq - 1])   # pad canvas to square
            o.append(24); s.append([0, 0, sq - 1, sq - 1])   # rotate CCW
            o.append(27); s.append([0, 0, sq - 1, sq - 1])   # flip up-down
            o.append(33); s.append([0, 0, w - 1, h - 1])     # crop back to w x h
        return o, s

    if vertical:
        o, s = transpose_ops(hi, wi)
        ops += o
        sels += s
        A, B = I.T, O.T
    else:
        A, B = I, O

    hA, wA = A.shape
    diff = [(r, c) for r in range(hA) for c in range(wA) if A[r, c] != B[r, c]]

    r3 = int(np.where(A == 3)[0][0])          # row of the 3 marker pair
    r2 = int(np.where(A == 2)[0][0])          # row of the 2 marker pair

    ray3 = sorted([p for p in diff if p[0] == r3], key=lambda p: p[1])
    ray2 = sorted([p for p in diff if p[0] == r2], key=lambda p: p[1])
    stem = sorted([p for p in diff if p[0] != r3 and p[0] != r2])

    # draw the elbow: out from the 3 marker, along the connector, into the 2 marker
    for seg in (ray3, stem, ray2):
        if seg:
            ops.append(3)
            sels.append(sel_of(seg))

    if vertical:
        o, s = transpose_ops(hA, wA)          # reflect back to the original frame
        ops += o
        sels += s

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
