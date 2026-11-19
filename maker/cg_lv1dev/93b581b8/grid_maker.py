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
from collections import Counter, deque
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round((b - a) * diff_lb))
    hi = a + int(round((b - a) * diff_ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if lo > hi:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(1, 9)
    ccols = random.sample(remcols, numcols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    if not ccols:
        rem = [c for c in range(10) if c != bgc]
        ccols = random.sample(rem, random.randint(1, 9))
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    h = min(h, max_h)
    w = min(w, max_w)
    numocc = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 50)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ, tr, maxtr = 0, 0, 10 * numocc
    q = [(0, 0), (0, 1), (1, 0), (1, 1)]
    while tr < maxtr and succ < numocc:
        tr += 1
        cands = [ij for ij in inds if ij[0] <= h - 2 and ij[1] <= w - 2]
        if not cands:
            break
        lr, lc = random.choice(cands)
        c1, c2, c3, c4 = [random.choice(ccols) for _ in range(4)]
        inobj = [(c1, (0, 0)), (c2, (0, 1)), (c3, (1, 0)), (c4, (1, 1))]
        outobj = list(inobj)
        for (dr, dc, col) in [(-2, -2, c4), (-2, 2, c3), (2, -2, c2), (2, 2, c1)]:
            outobj += [(col, (a + dr, b + dc)) for a, b in q]
        inpl = [(v, (a + lr, b + lc)) for v, (a, b) in inobj]
        outpl = [(v, (a + lr, b + lc)) for v, (a, b) in outobj]
        outpl = [(v, (a, b)) for v, (a, b) in outpl if 0 <= a < h and 0 <= b < w]
        outi = {ij for _, ij in outpl}
        if outi.issubset(inds):
            succ += 1
            nb = set()
            for _, (a, b) in inpl:
                nb |= {(a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)}
            inds = (inds - outi) - nb
            for v, (a, b) in inpl:
                gi[a][b] = v
            for v, (a, b) in outpl:
                go[a][b] = v
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: the canvas colour every example is drawn on (blocks are sparse)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    visited = set()
    blocks = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and (r, c) not in visited:
                dq = deque([(r, c)])
                visited.add((r, c))
                cells = []
                while dq:
                    a, b = dq.popleft()
                    cells.append((a, b))
                    for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        na, nb = a + da, b + db
                        if 0 <= na < hi and 0 <= nb < wi and (na, nb) not in visited \
                                and I[na, nb] != bgc:
                            visited.add((na, nb))
                            dq.append((na, nb))
                blocks.append((min(x[0] for x in cells), min(x[1] for x in cells)))

    # Each block cell projects its colour through the block to the 2x2 region
    # diagonally beyond the opposite corner.
    for (r, c) in blocks:
        for (sr, sc, dr, dc) in [(1, 1, -2, -2), (1, 0, -2, 2), (0, 1, 2, -2), (0, 0, 2, 2)]:
            col = int(I[r + sr, c + sc])
            cells = [(r + dr + a, c + dc + b) for a in range(2) for b in range(2)
                     if 0 <= r + dr + a < hi and 0 <= c + dc + b < wi]
            if cells:
                ops.append(col)
                sels.append(sel_of(cells))

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
