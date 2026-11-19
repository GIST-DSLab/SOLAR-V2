"""
ARC Task: b230c067 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, **kw) -> dict:
    def unifint(lo, hi):
        d = random.uniform(diff_lb, diff_ub)
        return min(max(lo, int(round(lo + (hi - lo) * d))), hi)

    def nbrs(p):
        i, j = p
        return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    def norm(s):
        mi = min(i for i, _ in s); mj = min(j for _, j in s)
        return {(i - mi, j - mj) for i, j in s}

    def hgt(s): return max(i for i, _ in s) - min(i for i, _ in s) + 1
    def wid(s): return max(j for _, j in s) - min(j for _, j in s) + 1

    H = max(10, min(30, max_h)); W = max(10, min(30, max_w))
    while True:
        h = unifint(10, H); w = unifint(10, W)
        oh = unifint(2, max(2, h // 3 - 1)); ow = unifint(2, max(2, w // 3 - 1))
        numcd = unifint(0, (oh * ow) // 2)
        numc = random.choice((numcd, oh * ow - numcd))
        numca = min(max(2, numc), oh * ow - 2)
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        sp = random.choice(sorted(bounds))
        shp = {sp}
        for _ in range(numca):
            cand = set()
            for p in shp: cand |= nbrs(p)
            cand = sorted((bounds - shp) & cand)
            shp.add(random.choice(cand))
        shpa = norm(shp); shpb = set(shpa)
        mxnch = oh * ow - len(shpa)
        nchinv = unifint(1, mxnch)
        nch = min(max(1, mxnch - nchinv), mxnch)
        for _ in range(nch):
            cand = set()
            for p in shpb: cand |= nbrs(p)
            cand = sorted((bounds - shpb) & cand)
            shpb.add(random.choice(cand))
        shpb = norm(shpb)
        if random.choice((True, False)):
            shpa, shpb = shpb, shpa
        gi = [[bgc] * w for _ in range(h)]
        go = [[bgc] * w for _ in range(h)]
        inds = {(i, j) for i in range(h) for j in range(w)}
        ha, wa = hgt(shpa), wid(shpa)
        acands = sorted(p for p in inds if p[0] <= h - ha and p[1] <= w - wa)
        ai, aj = random.choice(acands)
        apl = {(i + ai, j + aj) for i, j in shpa}
        for i, j in apl:
            gi[i][j] = fgc; go[i][j] = 2
        nb = set()
        for p in apl: nb |= nbrs(p)
        inds = inds - apl - nb
        hb, wb = hgt(shpb), wid(shpb)
        inds = {p for p in inds if p[0] <= h - hb and p[1] <= w - wb}
        succ = tr = 0
        while succ < 2 and tr <= 10:
            if not inds:
                break
            li, lj = random.choice(sorted(inds))
            pl = {(i + li, j + lj) for i, j in shpb}
            if pl <= inds:
                for i, j in pl:
                    gi[i][j] = fgc; go[i][j] = 1
                succ += 1
                nb = set()
                for p in pl: nb |= nbrs(p)
                inds = inds - pl - nb
            tr += 1
        if succ == 2:
            return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    from collections import Counter, deque
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # objects are small (<= h/3 x w/3, three of them) -> background is the majority colour
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Reading (as in the golf entry): every foreground cell is one unit and is
    # coloured 1 in a single stroke; then the odd-shaped object (the 8-connected
    # shape that occurs only once) is overpainted with 2.
    fg = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]

    seen = set()
    comps = []
    for p in fg:
        if p in seen:
            continue
        seen.add(p)
        q = deque([p])
        cells = []
        while q:
            y, x = q.popleft()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    n = (y + dy, x + dx)
                    if 0 <= n[0] < hi and 0 <= n[1] < wi and n not in seen and I[n] != bgc:
                        seen.add(n)
                        q.append(n)
        comps.append(cells)

    def key(cells):
        mr = min(y for y, _ in cells); mc = min(x for _, x in cells)
        return tuple(sorted((y - mr, x - mc) for y, x in cells))

    groups = {}
    for cells in comps:
        groups.setdefault(key(cells), []).append(cells)
    rare = min(groups, key=lambda k: len(groups[k]))

    ops, sels = [], []
    ops.append(1); sels.append(sel_of(fg))            # all objects -> colour 1
    for cells in groups[rare]:
        ops.append(2); sels.append(sel_of(cells))     # the odd-shape object -> colour 2
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
                        f"num_examples+1 ({num_examples + 1}) for task b230c067"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b230c067"
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
                                f"for task b230c067"
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
                    f"Failed to build a complete episode for task b230c067 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b230c067-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
