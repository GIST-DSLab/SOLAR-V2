"""
ARC Task: 6773b310 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 1]
    bgc, linc, fgc = random.sample(cols, 3)
    return {"bgc": bgc, "linc": linc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, fgc=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    if bgc is None or linc is None or fgc is None:
        bgc, linc, fgc = random.sample(cols, 3)

    def unif(lo, hi):
        d = random.uniform(diff_lb, diff_ub)
        return int(round(lo + d * (hi - lo)))

    for _ in range(100):
        h = unif(2, 5)
        w = unif(2, 5)
        nh = unif(2, 5)
        nw = unif(2, 5)
        fullh = h * nh + (nh - 1)
        fullw = w * nw + (nw - 1)
        if fullh <= max_h and fullw <= max_w:
            break
    else:
        h = w = nh = nw = 2
        fullh = fullw = 5

    gi = [[linc] * fullw for _ in range(fullh)]
    llocs = [(a, b) for a in range(0, fullh, h + 1) for b in range(0, fullw, w + 1)]
    nbldev = unif(0, (nh * nw) // 2)
    nbl = random.choice((nbldev, nh * nw - nbldev))
    nbl = min(max(1, nbl), nh * nw - 1)
    bluelocs = random.sample(llocs, nbl)
    blueset = set(bluelocs)
    inds = [(i, j) for i in range(h) for j in range(w)]
    go = [[bgc] * nw for _ in range(nh)]
    for (a, b) in llocs:
        k = 2 if (a, b) in blueset else 1
        marks = set(random.sample(inds, k))
        for (i, j) in inds:
            gi[a + i][b + j] = fgc if (i, j) in marks else bgc
        if (a, b) in blueset:
            go[a // (h + 1)][b // (w + 1)] = 1
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # lattice lines: rows/cols made of a single colour
    sep_rows = [r for r in range(hi) if len(set(I[r, :].tolist())) == 1]
    sep_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    linc = int(I[sep_rows[0], 0])
    nh, nw = len(sep_rows) + 1, len(sep_cols) + 1
    bh, bw = sep_rows[0], sep_cols[0]

    cnt = Counter(I.flatten().tolist())
    cnt.pop(linc, None)
    fgc = min(cnt.items(), key=lambda kv: kv[1])[0]
    bgc = [c for c in cnt if c != fgc][0]

    counts = np.zeros((nh, nw), dtype=int)
    for i in range(nh):
        for j in range(nw):
            r0, c0 = i * (bh + 1), j * (bw + 1)
            counts[i, j] = int((I[r0:r0 + bh, c0:c0 + bw] == fgc).sum())
    blue = counts == counts.max()

    ops, sels = [], []
    # shrink canvas to one cell per lattice block (transparent crop of top-left corner)
    ops.append(33); sels.append([0, 0, nh - 1, nw - 1])
    cur = np.where(I[:nh, :nw] != 0, I[:nh, :nw], 0)

    # work one output row (= one row of lattice blocks) at a time
    for i in range(nh):
        stray = [(i, j) for j in range(nw) if not blue[i, j] and cur[i, j] != bgc]
        if stray:
            ops.append(int(bgc)); sels.append(sel_of(stray))
            for (r, c) in stray:
                cur[r, c] = bgc
        winners = [(i, j) for j in range(nw) if blue[i, j]]
        if winners:
            ops.append(1); sels.append(sel_of(winners))
            for (r, c) in winners:
                cur[r, c] = 1

    ops.append(34); sels.append([0, 0, nh - 1, nw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6773b310"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6773b310"
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
                                f"for task 6773b310"
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
                    f"Failed to build a complete episode for task 6773b310 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6773b310-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
