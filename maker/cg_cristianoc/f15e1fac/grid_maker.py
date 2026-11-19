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


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))
    if bgc is None or linc is None:
        cols = [c for c in range(10) if c != 2]
        bgc, linc = random.sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
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
    mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T,
           np.fliplr, np.flipud, lambda g: np.rot90(g, 3),
           lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O, examples=None):
    # Reading (golf solution): every seed shoots a straight ray across the
    # grid; then at each 2-marker, the whole part of the rays beyond that
    # marker is shifted one cell sideways (away from the marker edge).
    # Shifts accumulate marker by marker.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = np.asarray(O).shape
    submit = ([34], [[0, 0, ho - 1, wo - 1]])
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    linc = None
    for c in np.unique(I).tolist():
        if c != bgc and c != 2:
            linc = c
    if linc is None:
        return submit
    # canonical view: markers on left column, seeds on top row
    idx = np.arange(hi * wi).reshape(hi, wi)
    C = IX = None
    for k in range(4):
        for flip in (False, True):
            g = np.rot90(I, k); ix = np.rot90(idx, k)
            if flip:
                g = np.fliplr(g); ix = np.fliplr(ix)
            r2, c2 = np.where(g == 2)
            rl, cl = np.where(g == linc)
            if r2.size and rl.size and np.all(c2 == 0) and np.all(rl == 0):
                C, IX = g, ix
                break
        if C is not None:
            break
    if C is None:
        return submit
    hc, wc = C.shape

    def orig(cells):
        return [divmod(int(IX[r, c]), wi) for r, c in cells]

    gps = sorted(set(np.where(C == 2)[0].tolist()))
    sps = sorted(np.where(C[0] == linc)[0].tolist())
    ops, sels = [], []

    if linc == 0:
        # Move cannot carry colour-0 objects: paint each band's shifted
        # segments (all rays of one band as one unit) directly.
        starts = [0] + gps
        ends = [x - 1 for x in gps] + [hc - 1]
        for k, (a, b) in enumerate(zip(starts, ends)):
            cells = [(r, j + k) for j in sps if j + k < wc
                     for r in range(a, b + 1) if C[r, j + k] != linc]
            if cells:
                ops.append(linc); sels.append(sel_of(orig(cells)))
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    # 1) straight rays, one line per seed, to the far edge
    ray = set()
    for j in sps:
        cells = [(r, j) for r in range(1, hc)]
        ops.append(linc); sels.append(sel_of(orig(cells)))
        ray |= set(cells) | {(0, j)}
    # direction of canonical +col in original coordinates
    (r0, c0), (r1, c1) = orig([(0, 0), (0, 1)])
    mop = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(r1 - r0, c1 - c0)]
    # 2) at each marker, shift all ray cells beyond it one step sideways
    for g in gps:
        part = sorted(p for p in ray if p[0] >= g)   # visible line cells past marker row
        if not part:
            continue
        ops.append(mop); sels.append(sel_of(orig(part)))
        moved = {(r, c + 1) for r, c in part if c + 1 < wc}   # off-grid clipped
        hole = sorted(set(part) - moved)
        ray = (ray - set(part)) | moved
        if bgc != 0 and hole:
            ops.append(bgc); sels.append(sel_of(orig(hole)))
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
