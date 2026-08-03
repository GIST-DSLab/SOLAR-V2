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


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, linc: int) -> dict:
    cols = interval(0, 10, 1)
    oh_ub = max(1, min(4, (max_h + 1) // 3 - 1))
    ow_ub = max(1, min(4, (max_w + 1) // 3 - 1))
    oh = unifint(diff_lb, diff_ub, (1, oh_ub))
    ow = unifint(diff_lb, diff_ub, (1, ow_ub))
    numh = unifint(diff_lb, diff_ub, (3, max(3, (max_h + 1) // (oh + 1))))
    numw = unifint(diff_lb, diff_ub, (3, max(3, (max_w + 1) // (ow + 1))))
    remcols = difference(cols, (bgc, linc))
    ncols = unifint(diff_lb, diff_ub, (1, min(8, max(1, (numh * numh) // 3))))
    ccols = sample(remcols, ncols)
    fullh = numh * oh + numh - 1
    fullw = numw * ow + numw - 1
    gi = canvas(linc, (fullh, fullw))
    sgi = asindices(canvas(bgc, (oh, ow)))
    for a in range(numh):
        for b in range(numw):
            gi = fill(gi, bgc, shift(sgi, (a * (oh + 1), b * (ow + 1))))
    go = tuple(e for e in gi)
    sinds = asindices(canvas(-1, (oh, ow)))
    for col in ccols:
        inds = occurrences(go, recolor(bgc, sinds))
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        narms = randint(1, 4)
        armdirs = sample(totuple(dneighbors((0, 0))), narms)
        for armdir in armdirs:
            x, y = armdir
            arm = []
            for k in range(1, max(numh, numw)):
                nextloc = add(loc, (k * x * (oh + 1), k * y * (ow + 1)))
                if nextloc not in inds:
                    break
                arm.append(nextloc)
            if len(arm) < 2:
                continue
            aidx = unifint(diff_lb, diff_ub, (1, len(arm) - 1))
            endp = arm[aidx]
            gi = fill(gi, col, shift(sinds, endp))
            go = fill(go, col, mapply(lbind(shift, sinds), set(arm[:aidx + 1])))
        gi = fill(gi, col, shift(sinds, loc))
        go = fill(go, col, shift(sinds, loc))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # --- recover the lattice: a fully-uniform row can only be the separator color
    linc = None
    for r in range(hi):
        if len(set(I[r].tolist())) == 1:
            linc = int(I[r, 0])
            break
    line_rows = [r for r in range(hi) if bool(np.all(I[r] == linc))]
    line_cols = [c for c in range(wi) if bool(np.all(I[:, c] == linc))]
    oh = line_rows[0]          # block height
    ow = line_cols[0]          # block width
    rstarts = list(range(0, hi, oh + 1))
    cstarts = list(range(0, wi, ow + 1))
    nh, nw = len(rstarts), len(cstarts)

    # block-color lattice
    B = [[int(I[rstarts[a], cstarts[b]]) for b in range(nw)] for a in range(nh)]
    bgc = Counter([B[a][b] for a in range(nh) for b in range(nw)]).most_common(1)[0][0]

    # colors in first-appearance order
    order = []
    for a in range(nh):
        for b in range(nw):
            c = B[a][b]
            if c != bgc and c not in order:
                order.append(c)

    emitted = set()
    for col in order:
        cells = [(a, b) for a in range(nh) for b in range(nw) if B[a][b] == col]
        cells.sort()
        # each aligned pair of same-colored blocks fills the corridor between them
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                (a1, b1), (a2, b2) = cells[i], cells[j]
                between = []
                if a1 == a2:
                    for b in range(min(b1, b2) + 1, max(b1, b2)):
                        between.append((a1, b))
                elif b1 == b2:
                    for a in range(min(a1, a2) + 1, max(a1, a2)):
                        between.append((a, b1))
                for (a, b) in between:
                    if B[a][b] == col or (a, b) in emitted:
                        continue
                    emitted.add((a, b))
                    r0, c0 = rstarts[a], cstarts[b]
                    ops.append(col)
                    # whole block is a solid rectangle -> bbox selection is exact
                    sels.append([r0, c0, oh - 1, ow - 1])

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
