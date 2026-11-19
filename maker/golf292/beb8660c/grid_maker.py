"""
ARC Task: beb8660c (RE-ARC) — LLM-generated grid_maker
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

ROTS = ["identity", "rot90", "rot180", "rot270"]


def _rot(g, name):
    a = np.array(g)
    if name == "identity":
        r = a
    elif name == "rot90":
        r = np.rot90(a, 3)
    elif name == "rot180":
        r = np.rot90(a, 2)
    else:
        r = np.rot90(a, 1)
    return tuple(tuple(int(x) for x in row) for row in r)


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rotf": r} for r in ROTS]
        ex += [{"rotf": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rotf": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rotf=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        return random.randint(min(lo, hi), max(lo, hi))

    if rotf is None:
        rotf = random.choice(ROTS)
    cap = min(max_h, max_w, 30)
    cols = [c for c in range(10) if c != 8]
    w = unifint(diff_lb, diff_ub, (3, cap))
    h = unifint(diff_lb, diff_ub, (w, cap))
    remcols = [c for c in cols if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    k = min(8, w - 1)
    k = unifint(diff_lb, diff_ub, (1, k))
    co = random.sample(remcols, k)
    wds = sorted(random.sample(list(range(1, w)), k))
    for j, (c, l) in enumerate(zip(co, wds)):
        jj = h - k - 1 + j
        for x in range(l):
            gi[jj][x] = c
    gi[h - 1] = [8] * w
    go = [list(reversed(r)) for r in gi]
    rows = [list(r) for r in gi[:-1]]
    random.shuffle(rows)
    rows = rows + [list(go[-1])]
    gif = []
    for r in rows:
        nbc = r.count(bgc)
        ofs = random.randint(0, nbc)
        if ofs == 0:
            gif.append(list(r))
        else:
            gif.append(r[-ofs:] + r[:-ofs])
    gi = [list(reversed(r)) for r in gif]
    return {"input": _rot(gi, rotf), "output": _rot(go, rotf)}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # locate the full 8-edge
    if np.all(I[0, :] == 8):
        edge = 'top'
    elif np.all(I[hi - 1, :] == 8):
        edge = 'bottom'
    elif np.all(I[:, 0] == 8):
        edge = 'left'
    else:
        edge = 'right'
    horizontal = edge in ('top', 'bottom')

    # background = the colour present in every non-8 slice parallel to the 8-edge
    if horizontal:
        slices = [set(I[r, :].tolist()) for r in range(hi) if not np.all(I[r, :] == 8)]
    else:
        slices = [set(I[:, c].tolist()) for c in range(wi) if not np.all(I[:, c] == 8)]
    common = set.intersection(*slices) - {8}
    if len(common) == 1:
        bgc = next(iter(common))
    else:
        cnt = Counter(I.flatten().tolist())
        bgc = max(common or set(cnt), key=lambda c: cnt[c])

    # each line = all cells of one non-bg, non-8 colour
    line_cells = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc and v != 8:
                line_cells.setdefault(v, []).append((r, c))
    order = sorted(line_cells, key=lambda col: (-len(line_cells[col]), col))

    # destination of each line: stacked outward from the 8-edge, longest first,
    # aligned to the corner fixed by the edge orientation
    dests = {}
    for i, col in enumerate(order):
        L = len(line_cells[col])
        if edge == 'top':
            cells = [(1 + i, c) for c in range(L)]
        elif edge == 'bottom':
            cells = [(hi - 2 - i, c) for c in range(wi - L, wi)]
        elif edge == 'left':
            cells = [(r, 1 + i) for r in range(hi - L, hi)]
        else:
            cells = [(r, wi - 2 - i) for r in range(L)]
        dests[col] = cells

    to_move = [col for col in order if set(line_cells[col]) != set(dests[col])]

    # 1. clear every line that is out of place (one stroke)
    erase = sorted(cell for col in to_move for cell in line_cells[col])
    if erase:
        ops.append(int(bgc)); sels.append(sel_of(erase))
    # 2. lay each line in its slot, one line = one stroke
    for col in to_move:
        ops.append(int(col)); sels.append(sel_of(dests[col]))

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
                        f"num_examples+1 ({num_examples + 1}) for task beb8660c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task beb8660c"
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
                                f"for task beb8660c"
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
                    f"Failed to build a complete episode for task beb8660c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"beb8660c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
