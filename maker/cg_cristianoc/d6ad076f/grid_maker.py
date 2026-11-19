"""
ARC Task: d6ad076f (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 8]
    bgc, c1, c2 = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants)) for _ in range(n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "c1": c1, "c2": c2, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, c1=None, c2=None,
             transpose=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if bgc is None or c1 is None or c2 is None:
        bgc, c1, c2 = random.sample([c for c in range(10) if c != 8], 3)
    if transpose is None:
        transpose = random.choice([True, False])

    lim = min(max_h, max_w) if transpose else None
    H = lim if transpose else max_h
    W = lim if transpose else max_w
    h = unifint(diff_lb, diff_ub, (4, max(4, H)))
    w = unifint(diff_lb, diff_ub, (3, max(3, W)))
    inh = unifint(diff_lb, diff_ub, (4, h))
    inw = unifint(diff_lb, diff_ub, (3, w))
    itv = list(range(inh))
    loci2i = unifint(diff_lb, diff_ub, (2, inh - 1))
    loci2 = itv[loci2i]
    itv2 = itv[:loci2i - 1][::-1]
    loci1i = unifint(diff_lb, diff_ub, (0, len(itv2) - 1))
    loci1 = itv2[loci1i]
    cp = random.randint(1, inw - 2)
    ajs = random.randint(0, cp - 1)
    aje = random.randint(cp + 1, inw - 1)
    bjs = random.randint(0, cp - 1)
    bje = random.randint(cp + 1, inw - 1)

    loci = random.randint(0, h - inh)
    locj = random.randint(0, w - inw)
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + loci1 + 1, locj + ajs:locj + aje + 1] = c1
    gi[loci + loci2:loci + inh, locj + bjs:locj + bje + 1] = c2
    go = gi.copy()
    r0, r1 = loci + loci1 + 1, loci + loci2 - 1
    q0, q1 = locj + max(ajs, bjs) + 1, locj + min(aje, bje) - 1
    go[r0:r1 + 1, q0:q1 + 1] = 8
    if transpose:
        gi = gi.T.copy()
        go = go.T.copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    colors = [int(c) for c in np.unique(I)]

    # bbox and solidity of each colour
    info = {}
    for col in colors:
        rs, cs = np.where(I == col)
        r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
        solid = len(rs) == (r1 - r0 + 1) * (c1 - c0 + 1)
        info[col] = (r0, r1, c0, c1, solid)

    # background: the colour that is not a solid rectangle; if all three are
    # solid (full-span bars), the background is the one sandwiched between.
    non_rect = [c for c in colors if not info[c][4]]
    if len(non_rect) == 1:
        bgc = non_rect[0]
    else:
        def disjoint(i0, i1):
            spans = sorted((info[c][i0], info[c][i1], c) for c in colors)
            return all(spans[k][1] < spans[k + 1][0] for k in range(len(spans) - 1)), spans
        ok, spans = disjoint(0, 1)
        if not ok:
            ok, spans = disjoint(2, 3)
        bgc = spans[1][2]

    objs = [c for c in colors if c != bgc]
    a, b = info[objs[0]][:4], info[objs[1]][:4]
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b

    ops, sels = [], []
    # Reading: rays shoot from one rectangle across the background toward the
    # other, one ray per inner line of their shared span (outermost shared
    # line on each side excluded). Each ray is one line of cells, painted 8.
    if ar1 < br0 or br1 < ar0:
        top, bot = (a, b) if ar1 < br0 else (b, a)
        gr0, gr1 = top[1] + 1, bot[0] - 1
        gc0, gc1 = max(ac0, bc0) + 1, min(ac1, bc1) - 1
        for c in range(gc0, gc1 + 1):          # vertical ray in column c
            ops.append(8)
            sels.append(sel_of([(r, c) for r in range(gr0, gr1 + 1)]))
    else:
        left, right = (a, b) if ac1 < bc0 else (b, a)
        gc0, gc1 = left[3] + 1, right[2] - 1
        gr0, gr1 = max(ar0, br0) + 1, min(ar1, br1) - 1
        for r in range(gr0, gr1 + 1):          # horizontal ray in row r
            ops.append(8)
            sels.append(sel_of([(r, c) for c in range(gc0, gc1 + 1)]))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task d6ad076f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d6ad076f"
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
                                f"for task d6ad076f"
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
                    f"Failed to build a complete episode for task d6ad076f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d6ad076f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
