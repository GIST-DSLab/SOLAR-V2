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
    # The rule only depends on the grid structure, so only the background and
    # line colours are fixed for the episode. The marker colours can change.
    bgc, linc = random.sample(range(10), 2)
    return {"bgc": bgc, "linc": linc}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + round((b - a) * max(0.0, min(1.0, lb)))
    hi = a + round((b - a) * max(0.0, min(1.0, ub)))
    return random.randint(min(lo, hi), max(lo, hi))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, **kw) -> dict:
    if bgc is None or linc is None:
        bgc, linc = random.sample(range(10), 2)
    remcols = [c for c in range(10) if c not in (bgc, linc)]
    gi = go = None
    for _ in range(100):
        oh = _unifint(diff_lb, diff_ub, (1, max(1, min(4, (max_h + 1) // 3 - 1))))
        ow = _unifint(diff_lb, diff_ub, (1, max(1, min(4, (max_w + 1) // 3 - 1))))
        numh = _unifint(diff_lb, diff_ub, (3, min(31, max_h + 1) // (oh + 1)))
        numw = _unifint(diff_lb, diff_ub, (3, min(31, max_w + 1) // (ow + 1)))
        ncols = _unifint(diff_lb, diff_ub, (1, min(8, (numh * numh) // 3)))
        ccols = random.sample(remcols, ncols)
        fullh = numh * oh + numh - 1
        fullw = numw * ow + numw - 1
        gi = [[linc] * fullw for _ in range(fullh)]
        for a in range(numh):
            for b in range(numw):
                for i in range(oh):
                    for j in range(ow):
                        gi[a * (oh + 1) + i][b * (ow + 1) + j] = bgc
        go = [row[:] for row in gi]

        def fillcell(g, col, loc):
            for i in range(oh):
                for j in range(ow):
                    g[loc[0] + i][loc[1] + j] = col

        for col in ccols:
            inds = set()
            for a in range(numh):
                for b in range(numw):
                    r, c = a * (oh + 1), b * (ow + 1)
                    if all(go[r + i][c + j] == bgc for i in range(oh) for j in range(ow)):
                        inds.add((r, c))
            if not inds:
                break
            loc = random.choice(sorted(inds))
            narms = random.randint(1, 4)
            armdirs = random.sample([(-1, 0), (1, 0), (0, -1), (0, 1)], narms)
            for x, y in armdirs:
                arm = []
                for k in range(1, max(numh, numw)):
                    nl = (loc[0] + k * x * (oh + 1), loc[1] + k * y * (ow + 1))
                    if nl not in inds:
                        break
                    arm.append(nl)
                if len(arm) < 2:
                    continue
                aidx = _unifint(diff_lb, diff_ub, (1, len(arm) - 1))
                fillcell(gi, col, arm[aidx])
                for p in arm[:aidx + 1]:
                    fillcell(go, col, p)
            fillcell(gi, col, loc)
            fillcell(go, col, loc)
        if gi != go:
            return {"input": gi, "output": go}
    return {"input": gi, "output": go}


def _structure(I):
    # Separator rows/cols are the fully uniform ones; cell height/width = index of first separator.
    h, w = I.shape
    line_rows = [r for r in range(h) if len(set(I[r].tolist())) == 1]
    line_cols = [c for c in range(w) if len(set(I[:, c].tolist())) == 1]
    oh = line_rows[0] if line_rows else h
    ow = line_cols[0] if line_cols else w
    return oh, ow


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    oh, ow = _structure(I)
    numh = (h + 1) // (oh + 1)
    numw = (w + 1) // (ow + 1)

    def cellcolors(G):
        G = np.asarray(G, dtype=int)
        a, b = _structure(G)
        nh = (G.shape[0] + 1) // (a + 1)
        nw = (G.shape[1] + 1) // (b + 1)
        return [int(G[i * (a + 1), j * (b + 1)]) for i in range(nh) for j in range(nw)]

    # Background cell colour (fixed per episode): read it from the demonstrations.
    cnt = Counter()
    if examples:
        for ei, _eo in examples:
            cnt.update(cellcolors(ei))
    else:
        cnt.update(cellcolors(I))
    bgc = cnt.most_common(1)[0][0]

    grid = np.array([[int(I[a * (oh + 1), b * (ow + 1)]) for b in range(numw)]
                     for a in range(numh)])
    cur = grid.copy()
    ops, sels = [], []

    def cells_of(blocks):
        out = []
        for a, b in blocks:
            for i in range(oh):
                for j in range(ow):
                    out.append((a * (oh + 1) + i, b * (ow + 1) + j))
        return out

    # For every colour, fill the cell blocks between same-coloured cells
    # that share a cell row or cell column. The separator lines stay as they are.
    colors = sorted(set(grid.flatten().tolist()) - {bgc})
    for col in colors:
        segs = []
        for a in range(numh):
            bs = [b for b in range(numw) if grid[a, b] == col]
            if len(bs) >= 2:
                segs.append([(a, b) for b in range(min(bs) + 1, max(bs))])
        for b in range(numw):
            as_ = [a for a in range(numh) if grid[a, b] == col]
            if len(as_) >= 2:
                segs.append([(a, b) for a in range(min(as_) + 1, max(as_))])
        for seg in segs:
            blocks = [p for p in seg if cur[p] != col]
            if not blocks:
                continue
            ops.append(col)
            sels.append(sel_of(cells_of(blocks)))
            for p in blocks:
                cur[p] = col
    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
