"""
ARC Task: 93b581b8 (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, **kwargs) -> dict:
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    remcols = [c for c in range(10) if c != bgc]
    numcols = _unifint(diff_lb, diff_ub, (1, 9))
    ccols = random.sample(remcols, numcols)
    numocc = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 50)))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ, tr, maxtr = 0, 0, 10 * numocc
    while tr < maxtr and succ < numocc:
        tr += 1
        cands = [ij for ij in inds if ij[0] <= h - 2 and ij[1] <= w - 2]
        if not cands:
            break
        lr, lc = random.choice(sorted(cands))
        c1, c2, c3, c4 = [random.choice(ccols) for _ in range(4)]
        inobj = [(c1, (0, 0)), (c2, (0, 1)), (c3, (1, 0)), (c4, (1, 1))]
        q = [(0, 0), (0, 1), (1, 0), (1, 1)]
        outobj = list(inobj)
        for col, (dr, dc) in ((c4, (-2, -2)), (c3, (-2, 2)), (c2, (2, -2)), (c1, (2, 2))):
            outobj += [(col, (a + dr, b + dc)) for a, b in q]
        outp = [(col, (a + lr, b + lc)) for col, (a, b) in outobj
                if 0 <= a + lr < h and 0 <= b + lc < w]
        outi = {ij for _, ij in outp}
        if outi.issubset(inds):
            succ += 1
            inp = [(col, (a + lr, b + lc)) for col, (a, b) in inobj]
            nb = set()
            for _, (a, b) in inp:
                nb |= {(a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)}
            inds = (inds - outi) - nb
            for col, (a, b) in inp:
                gi[a, b] = col
            for col, (a, b) in outp:
                go[a, b] = col
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: the canvas colour the generator paints first; objects cover <10% of cells
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # find the 2x2 objects (4-connected non-bg components)
    seen = set()
    objs = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and (r, c) not in seen:
                comp, st = [], [(r, c)]
                seen.add((r, c))
                while st:
                    a, b = st.pop()
                    comp.append((a, b))
                    for na, nb in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                        if 0 <= na < hi and 0 <= nb < wi and (na, nb) not in seen and I[na, nb] != bgc:
                            seen.add((na, nb))
                            st.append((na, nb))
                objs.append(comp)
    cur = I.copy()
    ops, sels = [], []
    for comp in objs:
        r0 = min(a for a, _ in comp); c0 = min(b for _, b in comp)
        r1 = max(a for a, _ in comp); c1 = max(b for _, b in comp)
        # each corner's colour is stamped as a 2x2 block at the diagonally opposite side
        spec = [
            (I[r1, c1], r0 - 2, c0 - 2),   # lower-right colour -> upper-left block
            (I[r1, c0], r0 - 2, c1 + 1),   # lower-left colour  -> upper-right block
            (I[r0, c1], r1 + 1, c0 - 2),   # upper-right colour -> lower-left block
            (I[r0, c0], r1 + 1, c1 + 1),   # upper-left colour  -> lower-right block
        ]
        for col, br, bc in spec:
            cells = [(a, b) for a in range(br, br + 2) for b in range(bc, bc + 2)
                     if 0 <= a < hi and 0 <= b < wi]
            if not cells or all(cur[a, b] == col for a, b in cells):
                continue
            ops.append(int(col)); sels.append(sel_of(cells))
            for a, b in cells:
                cur[a, b] = col
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 93b581b8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 93b581b8"
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
                                f"for task 93b581b8"
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
                    f"Failed to build a complete episode for task 93b581b8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"93b581b8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
