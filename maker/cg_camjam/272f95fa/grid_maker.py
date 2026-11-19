"""
ARC Task: 272f95fa (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2, 3, 4, 6)]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    from random import choice, sample, shuffle
    if bgc is None or linc is None:
        cols = [c for c in range(10) if c not in (1, 2, 3, 4, 6)]
        bgc, linc = random.sample(cols, 2)

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))

    def dmirror(g):
        return [list(r) for r in zip(*g)]

    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    gi = [[bgc] * 5 for _ in range(5)]
    for i in (1, 3):
        for j in range(5):
            gi[i][j] = linc
            gi[j][i] = linc
    hdist = [0, 0, 0]
    wdist = [0, 0, 0]
    idx = 0
    for _ in range(h - 2):
        hdist[idx] += 1
        idx = (idx + 1) % 3
    for _ in range(w - 2):
        wdist[idx] += 1
        idx = (idx + 1) % 3
    shuffle(hdist)
    shuffle(wdist)
    d1 = unifint(diff_lb, diff_ub, (0, hdist[0] - 1))
    hdist[0] -= d1; hdist[1] += d1
    d2 = unifint(diff_lb, diff_ub, (0, min(hdist[1], hdist[2]) - 1))
    d2 = choice((d2, -d2))
    hdist[1] += d2; hdist[2] -= d2
    d1 = unifint(diff_lb, diff_ub, (0, wdist[0] - 1))
    wdist[0] -= d1; wdist[1] += d1
    d2 = unifint(diff_lb, diff_ub, (0, min(wdist[1], wdist[2]) - 1))
    d2 = choice((d2, -d2))
    wdist[1] += d2; wdist[2] -= d2

    def expand(g, dist):
        out = []
        out += [list(g[0]) for _ in range(dist[0])]
        out.append(list(g[1]))
        out += [list(g[2]) for _ in range(dist[1])]
        out.append(list(g[3]))
        out += [list(g[4]) for _ in range(dist[2])]
        return out

    gi = expand(gi, hdist)
    gi = dmirror(gi)
    gi = expand(gi, wdist)
    gi = dmirror(gi)

    def vm(g): return [list(r[::-1]) for r in g]
    def hm(g): return [list(r) for r in g[::-1]]
    def cm(g): return vm(hm(dmirror(vm(hm(g)))))
    def r90(g): return [list(r) for r in zip(*g[::-1])]
    def r180(g): return hm(vm(g))
    def r270(g): return r90(r180(g))
    mfs = (lambda g: g, dmirror, cm, vm, hm, r90, r180, r270)
    for fn in sample(mfs, choice((1, 2))):
        gi = fn(gi)

    H, W = len(gi), len(gi[0])
    lr = [r for r in range(H) if all(v == linc for v in gi[r])]
    lc = [c for c in range(W) if all(gi[r][c] == linc for r in range(H))]
    r1, r2 = min(lr), max(lr)
    c1, c2 = min(lc), max(lc)
    go = [list(r) for r in gi]

    def paint(col, ra, rb, ca, cb):
        for r in range(ra, rb):
            for c in range(ca, cb):
                go[r][c] = col

    paint(6, r1 + 1, r2, c1 + 1, c2)
    paint(4, r1 + 1, r2, 0, c1)
    paint(3, r1 + 1, r2, c2 + 1, W)
    paint(2, 0, r1, c1 + 1, c2)
    paint(1, r2 + 1, H, c1 + 1, c2)
    return {"input": [list(r) for r in gi], "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # The two line rows / two line cols are the only uniform rows/cols
    # (every other row/col is crossed by line cells).
    line_rows = [r for r in range(hi) if len(set(I[r, :].tolist())) == 1]
    line_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    r1, r2 = min(line_rows), max(line_rows)
    c1, c2 = min(line_cols), max(line_cols)

    # The lines carve a 3x3 lattice of compartments: row band k in {top, middle, bottom},
    # column band b in {left, middle, right}. Walk the lattice in raster order; each
    # compartment that gets a colour is an exact full rectangle -> bbox == true cell set.
    row_bands = [(0, r1 - 1), (r1 + 1, r2 - 1), (r2 + 1, hi - 1)]
    col_bands = [(0, c1 - 1), (c1 + 1, c2 - 1), (c2 + 1, wi - 1)]
    table = [[None, 2, None],
             [4,    6, 3],
             [None, 1, None]]

    ops, sels = [], []
    for k, (ra, rb) in enumerate(row_bands):
        for b, (ca, cb) in enumerate(col_bands):
            col = table[k][b]
            if col is None:
                continue  # corner compartments keep the background
            ops.append(col)
            sels.append([ra, ca, rb - ra, cb - ca])

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
                        f"num_examples+1 ({num_examples + 1}) for task 272f95fa"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 272f95fa"
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
                                f"for task 272f95fa"
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
                    f"Failed to build a complete episode for task 272f95fa "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"272f95fa-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
