"""
ARC Task: 06df4c85 (RE-ARC) — LLM-generated grid_maker
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
    bgc, linc = random.sample(range(10), 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, **kw) -> dict:
    from random import randint, choice, sample

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        return randint(min(lo, hi), max(lo, hi))

    cols = list(range(10))
    oh = unifint(diff_lb, diff_ub, (1, 4))
    ow = unifint(diff_lb, diff_ub, (1, 4))
    numh = unifint(diff_lb, diff_ub, (3, max(3, (max_h + 1) // (oh + 1))))
    numw = unifint(diff_lb, diff_ub, (3, max(3, (max_w + 1) // (ow + 1))))
    remcols = [c for c in cols if c not in (bgc, linc)]
    ncols = unifint(diff_lb, diff_ub, (1, min(8, (numh * numh) // 3)))
    ccols = sample(remcols, ncols)
    fullh = numh * oh + numh - 1
    fullw = numw * ow + numw - 1
    gi = np.full((fullh, fullw), linc, dtype=int)
    for a in range(numh):
        for b in range(numw):
            gi[a * (oh + 1):a * (oh + 1) + oh, b * (ow + 1):b * (ow + 1) + ow] = bgc
    go = gi.copy()

    def free(g):
        return {(a * (oh + 1), b * (ow + 1)) for a in range(numh) for b in range(numw)
                if np.all(g[a * (oh + 1):a * (oh + 1) + oh, b * (ow + 1):b * (ow + 1) + ow] == bgc)}

    def fl(g, col, loc):
        g[loc[0]:loc[0] + oh, loc[1]:loc[1] + ow] = col

    for col in ccols:
        inds = free(go)
        if not inds:
            break
        loc = choice(sorted(inds))
        narms = randint(1, 4)
        armdirs = sample([(-1, 0), (1, 0), (0, -1), (0, 1)], narms)
        for x, y in armdirs:
            arm = []
            for k in range(1, max(numh, numw)):
                nl = (loc[0] + k * x * (oh + 1), loc[1] + k * y * (ow + 1))
                if nl not in inds:
                    break
                arm.append(nl)
            if len(arm) < 2:
                continue
            aidx = unifint(diff_lb, diff_ub, (1, len(arm) - 1))
            fl(gi, col, arm[aidx])
            for p in arm[:aidx + 1]:
                fl(go, col, p)
        fl(gi, col, loc)
        fl(go, col, loc)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # A column running the full height must cross separator rows -> it is a separator line.
    linc = None
    for c in range(wi):
        if len(set(I[:, c].tolist())) == 1:
            linc = int(I[0, c])
            break
    line_rows = [r for r in range(hi) if np.all(I[r] == linc)]
    line_cols = [c for c in range(wi) if np.all(I[:, c] == linc)]
    oh, ow = line_rows[0], line_cols[0]
    rst = list(range(0, hi, oh + 1))
    cst = list(range(0, wi, ow + 1))
    nh, nw = len(rst), len(cst)
    B = [[int(I[rst[a], cst[b]]) for b in range(nw)] for a in range(nh)]

    # Background block colour: the colour the demonstrations draw corridors over.
    bgc = None
    if examples:
        cnt = Counter()
        for ei, eo in examples:
            ei = np.asarray(ei)
            eo = np.asarray(eo)
            if ei.shape == eo.shape:
                cnt.update(ei[ei != eo].tolist())
        if cnt:
            cand = cnt.most_common(1)[0][0]
            if any(B[a][b] == cand for a in range(nh) for b in range(nw)):
                bgc = cand
    if bgc is None:
        bgc = Counter(B[a][b] for a in range(nh) for b in range(nw)).most_common(1)[0][0]

    cur = [row[:] for row in B]
    ops, sels = [], []

    def block_cells(a, b):
        return [(r, c) for r in range(rst[a], rst[a] + oh) for c in range(cst[b], cst[b] + ow)]

    def fill(blocks, col):
        # paint the still-background blocks of one corridor (separator lines untouched)
        todo = [(a, b) for a, b in blocks if cur[a][b] == bgc]
        if not todo:
            return
        cells = []
        for a, b in todo:
            cells += block_cells(a, b)
            cur[a][b] = col
        ops.append(col)
        sels.append(sel_of([(int(r), int(c)) for r, c in cells]))

    # Raster over blocks: each coloured block reaches to the next same-coloured block
    # to its right and the next one below, filling the background blocks between them.
    for a in range(nh):
        for b in range(nw):
            col = B[a][b]
            if col == bgc:
                continue
            for b2 in range(b + 1, nw):
                if B[a][b2] == col:
                    fill([(a, x) for x in range(b + 1, b2)], col)
                    break
            for a2 in range(a + 1, nh):
                if B[a2][b] == col:
                    fill([(y, b) for y in range(a + 1, a2)], col)
                    break

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
                        f"num_examples+1 ({num_examples + 1}) for task 06df4c85"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 06df4c85"
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
                                f"for task 06df4c85"
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
                    f"Failed to build a complete episode for task 06df4c85 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"06df4c85-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
