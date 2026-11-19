"""
ARC Task: 4093f84a (RE-ARC) — LLM-generated grid_maker
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

ORIENTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    bgc, barc, dotc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTS):
        ex = [dict(v) for v in ORIENTS]
        ex += [dict(random.choice(ORIENTS)) for _ in range(n_ex - len(ORIENTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(ORIENTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "barc": barc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barc, dotc, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice((True, False))
    hb, wb = (max_w, max_h) if transpose else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (7, max(7, min(30, hb))))
    w = unifint(diff_lb, diff_ub, (7, max(7, min(30, wb))))
    loci1, loci2 = sorted(sample(interval(2, h - 2, 1), 2))
    gi = canvas(bgc, (h, w))
    for ii in range(loci1, loci2 + 1, 1):
        gi = fill(gi, barc, connect((ii, 0), (ii, w - 1)))
    go = tuple(e for e in gi)
    opts = interval(0, w, 1)
    num1 = unifint(diff_lb, diff_ub, (1, w // 2))
    num2 = unifint(diff_lb, diff_ub, (1, w // 2))
    locs1 = sample(opts, num1)
    locs2 = sample(opts, num2)
    for l1 in locs1:
        k = unifint(diff_lb, diff_ub, (1, loci1 - 1))
        locsx = sample(interval(0, loci1, 1), k)
        gi = fill(gi, dotc, apply(rbind(astuple, l1), locsx))
        go = fill(go, barc, connect((loci1 - 1, l1), (loci1 - k, l1)))
    for l2 in locs2:
        k = unifint(diff_lb, diff_ub, (1, h - loci2 - 2))
        locsx = sample(interval(loci2 + 1, h, 1), k)
        gi = fill(gi, dotc, apply(rbind(astuple, l2), locsx))
        go = fill(go, barc, connect((loci2 + 1, l2), (loci2 + k, l2)))
    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (taken from the golf entry, which repeatedly pulls each dot toward
    the band): the band is a wall, and every dot FALLS toward it along its own
    line. Dots pile up against the band edge (or against dots already resting
    there) and take on the band's colour.
    The maker counted the dots and painted a bar. Here each line's dots are
    instead recoloured, then slid with gravity Moves: grab every dot still in
    the air, slide them together until the nearest one lands, and repeat with
    the rest. Afterwards the cells the grabs emptied go back to background.
    Exception: when the band colour is 0, a cell a grab empties already reads
    as band, so the Moves cannot be shown to matter. In that case the line's
    dots become a bar grown out of the band edge, as in the maker.
    Everything is measured from I; O is used only for the submit bbox.
    """
    Ia = np.asarray(I, dtype=int)
    Oa = np.asarray(O, dtype=int)
    hi, wi = Ia.shape
    colors = sorted(set(Ia.flatten().tolist()))

    # band colour: the colour living only in complete, contiguous full lines
    barc, vertical = None, False
    for c in colors:
        rs = [r for r in range(hi) if (Ia[r] == c).any()]
        if rs and len(rs) == rs[-1] - rs[0] + 1 and all((Ia[r] == c).all() for r in rs):
            barc, vertical = c, False
            break
    if barc is None:
        for c in colors:
            cs = [j for j in range(wi) if (Ia[:, j] == c).any()]
            if cs and len(cs) == cs[-1] - cs[0] + 1 and all((Ia[:, j] == c).all() for j in cs):
                barc, vertical = c, True
                break

    A = Ia.T if vertical else Ia
    h, w = A.shape
    band = [r for r in range(h) if (A[r] == barc).all()]
    r1, r2 = band[0], band[-1]

    rest = [c for c in colors if c != barc]
    # background outnumbers the dots: every line keeps >=1 background cell
    # per side and at most half the lines carry dots
    cnt = Counter(Ia.flatten().tolist())
    bgc = max(rest, key=lambda c: cnt[c])
    dots_cols = [c for c in rest if c != bgc]
    dotc = dots_cols[0] if dots_cols else None

    def g(r, c):  # canonical (row, line) -> real cell
        return (c, r) if vertical else (r, c)

    # move ops in canonical frame: toward smaller row / larger row
    if vertical:
        TOWARD_LOW, TOWARD_HIGH = 23, 22  # left / right
    else:
        TOWARD_LOW, TOWARD_HIGH = 20, 21  # up / down

    ops, sels = [], []

    def fall(srcs, tgts, c, op):
        """Gravity: grab every dot still in the air on this line and slide them
        together until the nearest one lands; repeat with the ones left over."""
        cur = list(srcs)
        grabbed = set()
        i = 0
        while i < len(cur):
            if cur[i] == tgts[i]:
                i += 1
                continue
            n = abs(cur[i] - tgts[i])
            step = 1 if tgts[i] > cur[i] else -1
            ops.append(op); sels.append(sel_of([g(r, c) for r in cur[i:]]))
            grabbed.update(cur[i:])
            for _ in range(n - 1):
                ops.append(op); sels.append(sel_of([]))
            cur[i:] = [r + step * n for r in cur[i:]]
            i += 1
        # cells a grab emptied and no dot came to rest on
        return sorted(grabbed - set(cur))

    if dotc is None:
        ops.append(34); sels.append([0, 0, Oa.shape[0] - 1, Oa.shape[1] - 1])
        return ops, sels

    for c in range(w):                       # one line at a time
        for side in (0, 1):
            if side == 0:                    # above the band: fall downward
                srcs = sorted([r for r in range(r1) if A[r, c] == dotc], reverse=True)
                tgts = [r1 - 1 - i for i in range(len(srcs))]
                op = TOWARD_HIGH
            else:                            # below the band: fall upward
                srcs = sorted([r for r in range(r2 + 1, h) if A[r, c] == dotc])
                tgts = [r2 + 1 + i for i in range(len(srcs))]
                op = TOWARD_LOW
            if not srcs:
                continue
            if barc != 0:
                # dots take the band colour, then fall onto it (gravity, in groups)
                ops.append(int(barc)); sels.append(sel_of([g(r, c) for r in srcs]))
                hole = fall(srcs, tgts, c, op)
                if bgc != 0 and hole:
                    ops.append(int(bgc)); sels.append(sel_of([g(r, c) for r in hole]))
            else:
                # band colour 0: a grabbed dot leaves a 0 that already reads as
                # band, so Moves cannot be shown to matter; the line's dots
                # instead become a bar of that many band cells grown out of the
                # band edge, and the dots outside that bar vanish
                ops.append(int(barc)); sels.append(sel_of([g(t, c) for t in tgts]))
                gone = [r for r in srcs if r not in set(tgts)]
                if gone:
                    ops.append(int(bgc)); sels.append(sel_of([g(r, c) for r in gone]))
                continue
    ops.append(34); sels.append([0, 0, Oa.shape[0] - 1, Oa.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 4093f84a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4093f84a"
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
                                f"for task 4093f84a"
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
                    f"Failed to build a complete episode for task 4093f84a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4093f84a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
