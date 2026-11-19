"""
ARC Task: f15e1fac (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 2]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))
    cols = [c for c in range(10) if c != 2]
    if bgc is None or linc is None:
        bgc, linc = random.sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (4, max(4, max_h)))
    w = unifint(diff_lb, diff_ub, (4, max(4, max_w)))
    nsps = unifint(diff_lb, diff_ub, (1, (w - 1) // 2))
    ngps = unifint(diff_lb, diff_ub, (1, (h - 1) // 2))
    spsj = sorted(random.sample(range(1, w - 1), nsps))
    gpsi = sorted(random.sample(range(1, h - 1), ngps))
    gi = np.full((h, w), bgc, dtype=int)
    for jj in spsj:
        gi[0, jj] = linc
    for ii in gpsi:
        gi[ii, 0] = 2
    go = gi.copy()
    ofs = 0
    for a, b in zip([0] + gpsi, [x - 1 for x in gpsi] + [h - 1]):
        for jj in spsj:
            if jj + ofs < w:
                go[a:b + 1, jj + ofs] = linc
        ofs += 1
    mfs = [
        lambda g: g,
        lambda g: g.T,                       # dmirror
        lambda g: np.rot90(g, 2).T,          # cmirror
        lambda g: g[:, ::-1],                # vmirror
        lambda g: g[::-1, :],                # hmirror
        lambda g: np.rot90(g, -1),           # rot90
        lambda g: np.rot90(g, 2),            # rot180
        lambda g: np.rot90(g, 1),            # rot270
    ]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O=None, examples=None):
    # Reading: the stripe pattern is a ROW (cross-section perpendicular to the
    # marker edge) that is carried forward one row at a time from the seed row.
    # Each next row receives a copy of the row before it; when that next row
    # carries a 2-marker the copy lands one cell further along (lines step).
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    linc = None
    for c in np.unique(I).tolist():
        if c != bgc and c != 2:
            linc = c
    full = [0, 0, hi - 1, wi - 1]
    if linc is None:
        return [34], [full]

    # canonical view: markers on the left column, seeds on the top row
    idx = np.arange(hi * wi).reshape(hi, wi)
    C = IX = None
    for k in range(4):
        for flip in (False, True):
            g = np.rot90(I, k)
            ix = np.rot90(idx, k)
            if flip:
                g = np.fliplr(g)
                ix = np.fliplr(ix)
            r2, c2 = np.where(g == 2)
            rl, cl = np.where(g == linc)
            if r2.size and rl.size and np.all(c2 == 0) and np.all(rl == 0):
                C, IX = g, ix
                break
        if C is not None:
            break
    if C is None:
        return [34], [full]

    hc, wc = C.shape

    def actual(r, c):
        return divmod(int(IX[r, c]), wi)

    def bbox(r, c0, c1):  # canonical row segment -> actual bbox
        a = actual(r, c0)
        b = actual(r, c1)
        return [min(a[0], b[0]), min(a[1], b[1]),
                abs(a[0] - b[0]), abs(a[1] - b[1])]

    cur = C.copy()  # working grid in canonical coordinates
    ops, sels = [], []
    for r in range(hc - 1):
        s = 1 if C[r + 1, 0] == 2 else 0
        c_end = wc - 1 - s               # last source column that stays on grid
        src = [c for c in range(1, c_end + 1) if cur[r, c] == linc]
        if not src:
            break                        # every line has left the grid
        tgt = [(r + 1, c + s) for c in src]
        if all(cur[t] == linc for t in tgt):
            continue
        if linc != 0:
            # copy this row's stripe band forward onto the next row
            ops.append(29); sels.append(bbox(r, 1, c_end))
            ops.append(30)
            tb = bbox(r + 1, 1 + s, c_end + s)
            sels.append([tb[0], tb[1], 0, 0])
            for c in range(1, c_end + 1):
                v = cur[r, c]
                if v != 0:
                    cur[r + 1, c + s] = v
        else:
            # colour 0 cannot travel through the clipboard: paint the
            # carried-forward line cells directly
            ops.append(linc); sels.append(sel_of([actual(*t) for t in tgt]))
            for t in tgt:
                cur[t] = linc
    ops.append(34); sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task f15e1fac"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f15e1fac"
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
                                f"for task f15e1fac"
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
                    f"Failed to build a complete episode for task f15e1fac "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f15e1fac-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
