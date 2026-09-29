"""
ARC Task: ded97339 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    from dsl import (interval, canvas, asindices, totuple, neighbors, fill,
                     connect, dmirror)
    from utils import unifint
    cols = interval(0, 10, 1)
    if bgc is None or linc is None:
        bgc, linc = random.sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (5, max_h))
    w = unifint(diff_lb, diff_ub, (5, max_w))
    gi = canvas(bgc, (h, w))
    ndots = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 9)))
    inds = asindices(gi)
    dots = set()
    if random.choice((True, False)):
        idxi = random.randint(0, h - 1)
        locj1 = random.randint(0, w - 3)
        locj2 = random.randint(locj1 + 2, w - 1)
        dots.add((idxi, locj1))
        dots.add((idxi, locj2))
    else:
        idxj = random.randint(0, w - 1)
        loci1 = random.randint(0, h - 3)
        loci2 = random.randint(loci1 + 2, h - 1)
        dots.add((loci1, idxj))
        dots.add((loci2, idxj))
    for k in range(ndots - 2):
        if len(inds) == 0:
            break
        loc = random.choice(totuple(inds))
        dots.add(loc)
        inds = (inds - {loc}) - neighbors(loc)
    gi = fill(gi, linc, dots)
    go = tuple(e for e in gi)
    for ii, r in enumerate(gi):
        if r.count(linc) > 1:
            a = r.index(linc)
            b = w - r[::-1].index(linc) - 1
            go = fill(go, linc, connect((ii, a), (ii, b)))
    go = dmirror(go)
    gi = dmirror(gi)
    for ii, r in enumerate(gi):
        if r.count(linc) > 1:
            a = r.index(linc)
            b = h - r[::-1].index(linc) - 1
            go = fill(go, linc, connect((ii, a), (ii, b)))
    go = dmirror(go)
    gi = dmirror(gi)
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(counts)])
    linc_candidates = [int(v) for v in vals if int(v) != bgc]

    ops, sels = [], []
    G = I.copy()  # current working grid, tracked to skip ops with no visible effect

    if linc_candidates:
        linc = linc_candidates[0]

        # horizontal segments between the outermost dots of each row
        for ii in range(hi):
            cs = [jj for jj in range(wi) if I[ii, jj] == linc]
            if len(cs) >= 2:
                a, b = min(cs), max(cs)
                if np.all(G[ii, a:b + 1] == linc):
                    continue  # already fully painted: op would change nothing
                ops.append(int(linc))
                # full 1-row rectangle = the segment exactly
                sels.append([int(ii), int(a), 0, int(b - a)])
                G[ii, a:b + 1] = linc

        # vertical segments between the outermost dots of each column
        for jj in range(wi):
            rs = [ii for ii in range(hi) if I[ii, jj] == linc]
            if len(rs) >= 2:
                a, b = min(rs), max(rs)
                if np.all(G[a:b + 1, jj] == linc):
                    continue  # already covered by earlier horizontal lines
                ops.append(int(linc))
                # full 1-column rectangle = the segment exactly
                sels.append([int(a), int(jj), int(b - a), 0])
                G[a:b + 1, jj] = linc

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
                        f"num_examples+1 ({num_examples + 1}) for task ded97339"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ded97339"
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
                                f"for task ded97339"
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
                    f"Failed to build a complete episode for task ded97339 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ded97339-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
