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
    cols = list(range(10))
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, linc=None) -> dict:
    from random import choice, randint, sample as rsample

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))

    cols = list(range(10))
    if bgc is None or linc is None:
        bgc, linc = random.sample(cols, 2)
    oh = unifint(diff_lb, diff_ub, (1, 4))
    ow = unifint(diff_lb, diff_ub, (1, 4))
    numh = unifint(diff_lb, diff_ub, (3, (max_h + 1) // (oh + 1)))
    numw = unifint(diff_lb, diff_ub, (3, (max_w + 1) // (ow + 1)))
    remcols = [c for c in cols if c not in (bgc, linc)]
    ncols = unifint(diff_lb, diff_ub, (1, min(8, (numh * numh) // 3)))
    ccols = rsample(remcols, ncols)
    fullh = numh * oh + numh - 1
    fullw = numw * ow + numw - 1
    gi = np.full((fullh, fullw), linc, dtype=int)
    for a in range(numh):
        for b in range(numw):
            gi[a * (oh + 1):a * (oh + 1) + oh, b * (ow + 1):b * (ow + 1) + ow] = bgc
    go = gi.copy()

    def isbg(g, r, c):
        return r + oh <= fullh and c + ow <= fullw and bool(np.all(g[r:r + oh, c:c + ow] == bgc))

    for col in ccols:
        inds = {(r, c) for r in range(0, fullh, oh + 1) for c in range(0, fullw, ow + 1) if isbg(go, r, c)}
        if not inds:
            break
        loc = choice(sorted(inds))
        narms = randint(1, 4)
        armdirs = rsample([(-1, 0), (1, 0), (0, -1), (0, 1)], narms)
        for (x, y) in armdirs:
            arm = []
            for k in range(1, max(numh, numw)):
                nl = (loc[0] + k * x * (oh + 1), loc[1] + k * y * (ow + 1))
                if nl not in inds:
                    break
                arm.append(nl)
            if len(arm) < 2:
                continue
            aidx = unifint(diff_lb, diff_ub, (1, len(arm) - 1))
            er, ec = arm[aidx]
            gi[er:er + oh, ec:ec + ow] = col
            for (r, c) in arm[:aidx + 1]:
                go[r:r + oh, c:c + ow] = col
        gi[loc[0]:loc[0] + oh, loc[1]:loc[1] + ow] = col
        go[loc[0]:loc[0] + oh, loc[1]:loc[1] + ow] = col
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading (shared with the golf entry): a lattice block takes colour k when
    # colour k occurs on BOTH sides of it along its block-row or block-column.
    # The unit is therefore a lattice LINE: for each colour on that line, the
    # whole corridor between its first and last block is painted in one stroke
    # (blocks only -- separator cells are never selected).
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # separator colour: a fully uniform row can only be a separator line
    linc = None
    for r in range(hi):
        if len(set(I[r].tolist())) == 1:
            linc = int(I[r, 0])
            break
    line_rows = [r for r in range(hi) if bool(np.all(I[r] == linc))]
    line_cols = [c for c in range(wi) if bool(np.all(I[:, c] == linc))]
    oh, ow = line_rows[0], line_cols[0]
    rstarts = list(range(0, hi, oh + 1))
    cstarts = list(range(0, wi, ow + 1))
    nh, nw = len(rstarts), len(cstarts)

    B = [[int(I[rstarts[a], cstarts[b]]) for b in range(nw)] for a in range(nh)]
    bgc = Counter(v for row in B for v in row).most_common(1)[0][0]
    cur = [row[:] for row in B]  # block colours as the working grid currently stands

    def block_cells(a, b):
        r0, c0 = rstarts[a], cstarts[b]
        return [(r, c) for r in range(r0, r0 + oh) for c in range(c0, c0 + ow)]

    ops, sels = [], []
    lines = [[(a, b) for b in range(nw)] for a in range(nh)] + \
            [[(a, b) for a in range(nh)] for b in range(nw)]
    for line in lines:
        seen = []
        for (a, b) in line:
            v = B[a][b]
            if v != bgc and v not in seen:
                seen.append(v)
        for col in seen:
            idx = [i for i, (a, b) in enumerate(line) if B[a][b] == col]
            if len(idx) < 2:
                continue
            # corridor = blocks strictly between the outermost two `col` blocks
            # on this line, skipping those already holding `col` right now
            corridor = [line[i] for i in range(idx[0] + 1, idx[-1])
                        if cur[line[i][0]][line[i][1]] != col]
            if not corridor:
                continue
            cells = []
            for (a, b) in corridor:
                cells += block_cells(a, b)
                cur[a][b] = col
            ops.append(col)
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(hi) for c in range(wi)]))
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
