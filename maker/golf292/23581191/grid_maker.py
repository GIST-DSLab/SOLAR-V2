"""
ARC Task: 23581191 (RE-ARC) — LLM-generated grid_maker
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
    colopts = [c for c in range(10) if c != 2]
    bgc = random.choice(colopts)
    rem = [c for c in colopts if c != bgc]
    acol = random.choice(rem)
    bcol = random.choice([c for c in rem if c != acol])
    return {"bgc": bgc, "acol": acol, "bcol": bcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, acol=None, bcol=None) -> dict:
    colopts = [c for c in range(10) if c != 2]
    if bgc is None:
        bgc = random.choice(colopts)
    rem = [c for c in colopts if c != bgc]
    if acol is None:
        acol = random.choice(rem)
    if bcol is None:
        bcol = random.choice([c for c in rem if c != acol])

    def unif(lo, hi):
        lo_v = lo + int(round(diff_lb * (hi - lo)))
        hi_v = lo + int(round(diff_ub * (hi - lo)))
        lo_v = max(lo, min(hi, lo_v))
        hi_v = max(lo_v, min(hi, hi_v))
        return random.randint(lo_v, hi_v)

    h = unif(3, max(3, max_h))
    w = unif(3, max(3, max_w))
    inds = [(i, j) for i in range(h) for j in range(w)]
    ub = max(1, (h * w) // 4)
    na = unif(1, ub)
    nb = unif(1, ub)
    a = random.sample(inds, na)
    aset = set(a)
    b = random.sample([x for x in inds if x not in aset], nb)

    gi = [[bgc] * w for _ in range(h)]
    for i, j in a:
        gi[i][j] = acol
    for i, j in b:
        gi[i][j] = bcol

    fa = {i for i, _ in a}; la = {j for _, j in a}
    fb = {i for i, _ in b}; lb = {j for _, j in b}
    go = [[bgc] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            ina = i in fa or j in la
            inb = i in fb or j in lb
            if ina and inb:
                go[i][j] = 2
            elif ina:
                go[i][j] = acol
            elif inb:
                go[i][j] = bcol
    for i, j in a:
        go[i][j] = acol
    for i, j in b:
        go[i][j] = bcol
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background: canvas colour (always >= half the cells by construction)
    cnt = Counter(I.flatten().tolist())
    bgc = int(cnt.most_common(1)[0][0])
    others = sorted(int(c) for c in cnt if c != bgc)

    # each point colour's lines: every row and column containing one of its points
    fams = []
    for col in others[:2]:
        rows = set(np.where((I == col).any(axis=1))[0].tolist())
        cols = set(np.where((I == col).any(axis=0))[0].tolist())
        fams.append((col, rows, cols))

    def on_lines(fam, r, c):
        return r in fam[1] or c in fam[2]

    bg_cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == bgc]

    if len(fams) == 2:
        A, B = fams
        only_a = [p for p in bg_cells if on_lines(A, *p) and not on_lines(B, *p)]
        only_b = [p for p in bg_cells if on_lines(B, *p) and not on_lines(A, *p)]
        both = [p for p in bg_cells if on_lines(A, *p) and on_lines(B, *p)]
        # background cells crossed only by colour-A lines -> A colour
        if only_a:
            ops.append(A[0]); sels.append(sel_of(only_a))
        # background cells crossed only by colour-B lines -> B colour
        if only_b:
            ops.append(B[0]); sels.append(sel_of(only_b))
        # background cells where the two line families cross -> 2
        if both:
            ops.append(2); sels.append(sel_of(both))
    elif len(fams) == 1:
        A = fams[0]
        only_a = [p for p in bg_cells if on_lines(A, *p)]
        if only_a:
            ops.append(A[0]); sels.append(sel_of(only_a))

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
                        f"num_examples+1 ({num_examples + 1}) for task 23581191"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 23581191"
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
                                f"for task 23581191"
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
                    f"Failed to build a complete episode for task 23581191 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"23581191-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
