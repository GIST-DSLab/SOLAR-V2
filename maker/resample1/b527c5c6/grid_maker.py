"""
ARC Task: b527c5c6 (RE-ARC) — LLM-generated grid_maker
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
    # The rule only depends on the structure (rectangle + marker dot), not on the
    # foreground colours, so only the background is fixed per episode.
    return {"bgc": random.choice(range(10))}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    return max(a, min(b, int(round(a + (b - a) * d))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    cols = list(range(10))
    h = _unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = _unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    remcols = [c for c in cols if c != bgc]
    ncols = _unifint(diff_lb, diff_ub, (2, 9))
    ccols = random.sample(remcols, ncols)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    inds = {(i, j) for i in range(h) for j in range(w)}
    noccs = _unifint(diff_lb, diff_ub, (1, 10))
    tr, succ, maxtr = 0, 0, 10 * noccs
    while succ < noccs and tr < maxtr:
        tr += 1
        d1 = random.randint(3, random.randint(3, min(h, w) // 2 - 1))
        d2 = random.randint(d1 * 2 + 1, random.randint(d1 * 2 + 1, min(h, w) - 1))
        oh, ow = random.sample([d1, d2], 2)
        cands = [ij for ij in inds if 1 <= ij[0] <= h - oh - 1 and 1 <= ij[1] <= w - ow - 1]
        if not cands:
            continue
        loci, locj = random.choice(cands)
        bd = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
        if ow < oh:
            lrflag = True
            dc1 = [(i, locj) for i in range(loci + ow - 1, loci + oh - ow + 1)]
            dc2 = [(i, j + ow - 1) for i, j in dc1]
        else:
            lrflag = False
            dc1 = [(loci, j) for j in range(locj + oh - 1, locj + ow - oh + 1)]
            dc2 = [(i + oh - 1, j) for i, j in dc1]
        loc = random.choice(dc1 + dc2)
        sgn = -1 if loc in dc1 else 1
        direc = (sgn * (0 if lrflag else 1), sgn * (1 if lrflag else 0))
        ln = {(loc[0] + k * direc[0], loc[1] + k * direc[1]) for k in range(43)}
        t = min(oh, ow) - 1
        r0 = min(a for a, _ in ln) - t; r1 = max(a for a, _ in ln) + t
        c0 = min(b for _, b in ln) - t; c1 = max(b for _, b in ln) + t
        shell = {(i, j) for i in range(r0, r1 + 1) for j in range(c0, c1 + 1)} - ln
        sqc, dotc = random.sample(ccols, 2)
        goobj = {}
        for c in (bd | shell) - ln:
            goobj[c] = sqc
        for c in ln:
            goobj[c] = dotc
        goobj = {c: v for c, v in goobj.items() if 0 <= c[0] < h and 0 <= c[1] < w}
        if set(goobj).issubset(inds):
            succ += 1
            nb = set()
            for (i, j) in bd:
                nb |= {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
            inds = (inds - set(goobj)) - nb
            for (i, j) in bd:
                gi[i, j] = sqc
            gi[loc] = dotc
            for (i, j), v in goobj.items():
                go[i, j] = v
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Generator keeps a 1-cell margin around every rectangle -> border is pure background.
    bgc = int(I[0, 0])
    seen = np.zeros_like(I, dtype=bool)
    objs = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            stack = [(r, c)]; seen[r, c] = True; cells = []
            while stack:
                a, b = stack.pop(); cells.append((a, b))
                for na, nb in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                    if 0 <= na < hi and 0 <= nb < wi and not seen[na, nb] and I[na, nb] != bgc:
                        seen[na, nb] = True; stack.append((na, nb))
            objs.append(cells)
    ops, sels = [], []
    for cells in objs:
        cnt = {}
        for a, b in cells:
            cnt[int(I[a, b])] = cnt.get(int(I[a, b]), 0) + 1
        if len(cnt) != 2:
            continue
        dotc = min(cnt, key=cnt.get); sqc = max(cnt, key=cnt.get)
        dr, dc = next((a, b) for a, b in cells if I[a, b] == dotc)
        rs = [a for a, _ in cells]; cs = [b for _, b in cells]
        top, bot, lef, rig = min(rs), max(rs), min(cs), max(cs)
        t = min(bot - top + 1, rig - lef + 1) - 1   # band half-width = short side - 1
        if dr == top:
            line = [(r, dc) for r in range(0, top)]
            strip = [(r, c) for r in range(0, top) for c in range(dc - t, dc + t + 1)]
        elif dr == bot:
            line = [(r, dc) for r in range(bot + 1, hi)]
            strip = [(r, c) for r in range(bot + 1, hi) for c in range(dc - t, dc + t + 1)]
        elif dc == lef:
            line = [(dr, c) for c in range(0, lef)]
            strip = [(r, c) for r in range(dr - t, dr + t + 1) for c in range(0, lef)]
        else:
            line = [(dr, c) for c in range(rig + 1, wi)]
            strip = [(r, c) for r in range(dr - t, dr + t + 1) for c in range(rig + 1, wi)]
        strip = [(r, c) for r, c in strip if 0 <= r < hi and 0 <= c < wi]
        # extend the rectangle's body outward as a band of width 2t+1 around the dot
        ops.append(sqc); sels.append(sel_of(strip))
        # then shoot the dot's colour through the middle of that band to the grid edge
        ops.append(dotc); sels.append(sel_of(line))
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
                        f"num_examples+1 ({num_examples + 1}) for task b527c5c6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b527c5c6"
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
                                f"for task b527c5c6"
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
                    f"Failed to build a complete episode for task b527c5c6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b527c5c6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
