"""
ARC Task: 890034e9 (RE-ARC) — LLM-generated grid_maker
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
from random import randint, choice, sample
import numpy as np
from maker.sel_helpers import sel_of
from dsl import *
from utils import unifint


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    markercol = random.choice(cols)
    rem = [c for c in cols if c != markercol]
    numbgc = random.randint(1, 8)
    bgcols = random.sample(rem, numbgc)
    return {"markercol": markercol, "bgcols": bgcols}


def generate(diff_lb, diff_ub, max_h, max_w, markercol, bgcols) -> dict:
    bgcols = tuple(bgcols)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    oh = randint(2, h // 4)
    ow = randint(2, w // 4)
    gi = canvas(0, (h, w))
    inds = asindices(gi)
    obj = {(choice(bgcols), ij) for ij in inds}
    gi = paint(gi, obj)
    numbl = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    blacks = sample(totuple(inds), numbl)
    gi = fill(gi, 0, blacks)
    patt = asindices(canvas(-1, (oh, ow)))
    tocover = set()
    for occ in occurrences(gi, recolor(0, patt)):
        tocover.add(choice(totuple(shift(patt, occ))))
    tocover = {(choice(bgcols), ij) for ij in tocover}
    gi = paint(gi, tocover)
    noccs = unifint(diff_lb, diff_ub, (2, (h * w) // ((oh + 2) * (ow + 2))))
    tr = 0
    succ = 0
    maxtr = 5 * noccs
    go = tuple(e for e in gi)
    while tr < maxtr and succ < noccs:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        bd = shift(patt, loc)
        plcd = outbox(bd)
        if plcd.issubset(inds):
            succ += 1
            inds = inds - plcd
            gi = fill(gi, 0, bd)
            go = fill(go, 0, bd)
            if succ == 1:
                gi = fill(gi, markercol, plcd)
            go = fill(go, markercol, plcd)
            loci, locj = loc
            ln1 = connect((loci - 1, locj), (loci - 1, locj + ow - 1))
            ln2 = connect((loci + oh, locj), (loci + oh, locj + ow - 1))
            ln3 = connect((loci, locj - 1), (loci + oh - 1, locj - 1))
            ln4 = connect((loci, locj + ow), (loci + oh - 1, locj + ow))
            if succ > 1:
                fixxer = {
                    (choice(bgcols), choice(totuple(xx))) for xx in [ln1, ln2, ln3, ln4]
                }
                gi = paint(gi, fixxer)
    # reject instances where the hidden boxes are ambiguous: the only empty
    # oh x ow windows in the input must be the placed (framed) ones
    A = np.array(gi)
    nwin = sum(1 for r in range(h - oh + 1) for c in range(w - ow + 1)
               if not A[r:r + oh, c:c + ow].any())
    if succ < 2 or nwin != succ:
        raise ValueError('ambiguous instance')
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    # Reading: the marker is a stamp. Its hollow (all-0) interior is the shape
    # being searched for; wherever that empty shape occurs elsewhere in the
    # grid, the marker frame itself is copied and stamped around it.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    def ring(r0, c0, r1, c1):
        s = set()
        for c in range(c0, c1 + 1):
            s.add((r0, c)); s.add((r1, c))
        for r in range(r0, r1 + 1):
            s.add((r, c0)); s.add((r, c1))
        return s

    # marker = colour whose cells are exactly one hollow box with an all-0
    # interior of at least 2x2; prefer the rarest such colour
    cands = []
    for col in range(1, 10):
        rs, cs = np.nonzero(I == col)
        if len(rs) == 0:
            continue
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        if set(zip(rs.tolist(), cs.tolist())) != ring(r0, c0, r1, c1):
            continue
        if (I[r0 + 1:r1, c0 + 1:c1] != 0).any():
            continue
        cands.append((len(rs), col, int(r0), int(c0), int(r1), int(c1)))
    if cands:
        cands.sort()
        _, mcol, mr0, mc0, mr1, mc1 = cands[0]
        oh, ow = mr1 - mr0 - 1, mc1 - mc0 - 1
        # every empty oh x ow window with room for a frame around it
        holes = [(r, c) for r in range(1, hi - oh) for c in range(1, wi - ow)
                 if (r, c) != (mr0 + 1, mc0 + 1)
                 and not (I[r:r + oh, c:c + ow] != 0).any()]
        if holes:
            # copy the whole marker box (full rectangle intended; its 0
            # interior is transparent, so only the frame reaches the clipboard)
            ops.append(28); sels.append([mr0, mc0, mr1 - mr0, mc1 - mc0])
            g = I.copy()
            # frames never overlap one another (nor the marker's own frame):
            # scan windows in reading order, keep each one whose frame is free
            used = ring(mr0, mc0, mr1, mc1)
            for (r, c) in holes:
                rr = ring(r - 1, c - 1, r + oh, c + ow)
                if rr & used:
                    continue
                used |= rr
                if all(g[a, b] == mcol for a, b in rr):
                    continue
                for a, b in rr:
                    g[a, b] = mcol
                ops.append(30); sels.append([r - 1, c - 1, 0, 0])  # stamp frame
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
                        f"num_examples+1 ({num_examples + 1}) for task 890034e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 890034e9"
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
                                f"for task 890034e9"
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
                    f"Failed to build a complete episode for task 890034e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"890034e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
