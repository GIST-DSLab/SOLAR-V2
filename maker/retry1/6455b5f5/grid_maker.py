"""
ARC Task: 6455b5f5 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 8)]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, fgc=None, **kw) -> dict:
    cols = [c for c in range(10) if c not in (1, 8)]
    if bgc is None:
        bgc = random.choice(cols)
    if fgc is None:
        fgc = random.choice([c for c in cols if c != bgc])

    def unif(lo, hi):
        if hi < lo:
            hi = lo
        d = random.uniform(diff_lb, diff_ub)
        v = int(round(lo + d * (hi - lo)))
        return max(lo, min(hi, v))

    def comps8(g):
        h, w = g.shape
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for r in range(h):
            for c in range(w):
                if g[r, c] == bgc and not seen[r, c]:
                    q = deque([(r, c)]); seen[r, c] = True; cells = []
                    while q:
                        y, x = q.popleft(); cells.append((y, x))
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] == bgc:
                                    seen[ny, nx] = True; q.append((ny, nx))
                    out.append(cells)
        return out

    while True:
        h = unif(6, max(6, min(30, max_h)))
        w = unif(6, max(6, min(30, max_w)))
        gi = np.full((h, w), bgc, dtype=int)
        ub = int((h * w) ** 0.5 * 1.5)
        num = unif(1, ub)
        for _ in range(num):
            objs = comps8(gi)
            elig = []
            for o in objs:
                rs = [p[0] for p in o]; cs = [p[1] for p in o]
                oh = max(rs) - min(rs) + 1; ow = max(cs) - min(cs) + 1
                if oh > 2 or ow > 2:
                    elig.append((o, min(rs), max(rs), min(cs), max(cs), oh, ow))
            if not elig:
                break
            if random.choice((True, False)):
                ro = max(elig, key=lambda t: len(t[0]))
            else:
                ro = random.choice(elig)
            _, u, d, l, rr, oh, ow = ro
            if random.choice((True, False)):
                vfr = oh < ow
            else:
                vfr = random.choice((True, False))
            if vfr and ow < 3:
                vfr = False
            if (not vfr) and oh < 3:
                vfr = True
            if vfr:
                j = random.randint(l + 1, rr - 1)
                gi[u:d + 1, j] = fgc
            else:
                j = random.randint(u + 1, d - 1)
                gi[j, l:rr + 1] = fgc
        objs = comps8(gi)
        if objs:
            sizes = [len(o) for o in objs]
            if min(sizes) != max(sizes):
                break
    go = gi.copy()
    mn, mx = min(sizes), max(sizes)
    for o, s in zip(objs, sizes):
        if s == mn:
            for (r, c) in o:
                go[r, c] = 8
    for o, s in zip(objs, sizes):
        if s == mx:
            for (r, c) in o:
                go[r, c] = 1
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # Background = corner colour of I. Lines are always drawn strictly inside a
    # region, so they never reach a grid corner; all four corners are background
    # (this is the verifier's mostcolor(corners) rule, read from I alone).
    corners = [int(I[0, 0]), int(I[0, wi - 1]), int(I[hi - 1, 0]), int(I[hi - 1, wi - 1])]
    bgc = max(set(corners), key=corners.count)

    # Background regions of I (8-connectivity, as the generator labels them).
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; cells = []
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx]
                                    and I[ny, nx] == bgc):
                                seen[ny, nx] = True; q.append((ny, nx))
                comps.append(cells)

    if comps:
        sizes = [len(c) for c in comps]
        maxs, mins = max(sizes), min(sizes)
        if maxs != mins:
            # One size class at a time: paint every largest region at once,
            # then every smallest region at once.
            big = sorted(cell for cells, s in zip(comps, sizes) if s == maxs for cell in cells)
            small = sorted(cell for cells, s in zip(comps, sizes) if s == mins for cell in cells)
            ops.append(1); sels.append(sel_of(big))
            ops.append(8); sels.append(sel_of(small))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6455b5f5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6455b5f5"
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
                                f"for task 6455b5f5"
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
                    f"Failed to build a complete episode for task 6455b5f5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6455b5f5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
