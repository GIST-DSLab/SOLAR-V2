"""
ARC Task: 0962bcdd (RE-ARC) — LLM-generated grid_maker
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
    # Rule depends only on plus presence/pattern; per-plus colors are read
    # dynamically in derive_operations. Only bgc must be fixed per episode.
    cols = [c for c in range(10) if c not in (3, 4)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, **color_kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        return int(min(b, max(a, round(a + (b - a) * random.uniform(lb, ub)))))

    cols = [c for c in range(10) if c not in (3, 4)]
    hub = max(10, min(30, max_h))
    wub = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (10, hub))
    w = unifint(diff_lb, diff_ub, (10, wub))

    remcols = [c for c in cols if c != bgc]
    numc = unifint(diff_lb, diff_ub, (2, 7))
    numc = min(numc, len(remcols))
    ccols = random.sample(remcols, max(2, numc))

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    oh, ow = 5, 5

    indss = set((i, j) for i in range(h) for j in range(w))
    subs = [(i, j) for (i, j) in indss if i < h - oh and j < w - ow]

    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if not indss:
            break
        if not subs:
            tr += 1
            continue
        loci, locj = random.choice(subs)
        bd = set((loci + di, locj + dj) for di in range(5) for dj in range(5))
        if bd.issubset(indss):
            ca, cb = random.sample(ccols, 2)  # ca=center(least), cb=arms(most)
            ci, cj = loci + 2, locj + 2
            # input plus: center ca, 4 orthogonal neighbors cb
            gi[ci][cj] = ca
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                gi[ci + di][cj + dj] = cb
            # output stamp: cross (cb) then X-diagonals (ca) on top
            for j in range(locj, locj + 5):
                go[loci + 2][j] = cb
            for i in range(loci, loci + 5):
                go[i][locj + 2] = cb
            for k in range(5):
                go[loci + k][locj + k] = ca
                go[loci + k][locj + 4 - k] = ca
            succ += 1
            indss -= bd
        tr += 1

    return {"input": np.array(gi, dtype=int), "output": np.array(go, dtype=int)}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # detect plus centers: cell whose 4 orthogonal neighbors are all one
    # non-bg color cb (equal), and center is a different non-bg color ca.
    centers = []
    for r in range(1, hi - 1):
        for c in range(1, wi - 1):
            v = int(I[r, c])
            if v == bgc:
                continue
            up = int(I[r - 1, c]); dn = int(I[r + 1, c])
            lf = int(I[r, c - 1]); rt = int(I[r, c + 1])
            if up == dn == lf == rt and up != bgc and up != v:
                centers.append((r, c, v, up))  # ca=v, cb=up

    ops, sels = [], []

    def inb(r, c):
        return 0 <= r < hi and 0 <= c < wi

    for (r, c, ca, cb) in centers:
        # cross tips (extend + arms to radius 2) -> cb  (all bg in I now)
        cross_new = [(r - 2, c), (r + 2, c), (r, c - 2), (r, c + 2)]
        cross_new = [(rr, cc) for (rr, cc) in cross_new if inb(rr, cc)]
        if cross_new:
            ops.append(int(cb)); sels.append(sel_of(cross_new))
        # X diagonals (inner + outer corners) -> ca (all bg in I now)
        diag_new = [
            (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1),
            (r - 2, c - 2), (r - 2, c + 2), (r + 2, c - 2), (r + 2, c + 2),
        ]
        diag_new = [(rr, cc) for (rr, cc) in diag_new if inb(rr, cc)]
        if diag_new:
            ops.append(int(ca)); sels.append(sel_of(diag_new))

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
                        f"num_examples+1 ({num_examples + 1}) for task 0962bcdd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0962bcdd"
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
                                f"for task 0962bcdd"
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
                    f"Failed to build a complete episode for task 0962bcdd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0962bcdd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
