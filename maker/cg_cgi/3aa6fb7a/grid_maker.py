"""
ARC Task: 3aa6fb7a (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    base = ((0, 0), (0, 1), (1, 0), (1, 1))
    cols = [c for c in range(10) if c != 1]
    if bgc is None:
        bgc = random.choice(cols)
    if fgc is None:
        fgc = random.choice([c for c in cols if c != bgc])

    def unif(lo, hi):
        lo_v = lo + int(round(diff_lb * (hi - lo)))
        hi_v = lo + int(round(diff_ub * (hi - lo)))
        lo_v = max(lo, min(hi, lo_v))
        hi_v = max(lo_v, min(hi, hi_v))
        return random.randint(lo_v, hi_v)

    h = unif(3, max(3, max_h))
    w = unif(3, max(3, max_w))
    gi = [[bgc] * w for _ in range(h)]
    inds = set((r, c) for r in range(h) for c in range(w))
    maxnum = max(1, ((h * w) // 2) // 3)
    num = unif(1, maxnum)
    kk, tr = 0, 0
    maxtrials = num * 2
    binds = set()
    while kk < num and tr < maxtrials:
        loc = random.choice(sorted(inds)) if inds else (0, 0)
        ooo = random.choice(base)
        oo = [b for b in base if b != ooo]
        oop = [(loc[0] + a, loc[1] + b) for a, b in oo]
        if inds and set(oop).issubset(inds):
            rem = set(oop)
            for (a, b) in oop:
                for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rem.add((a + da, b + db))
            inds -= rem
            for (a, b) in oop:
                gi[a][b] = fgc
            binds.add((loc[0] + ooo[0], loc[1] + ooo[1]))
            kk += 1
        tr += 1
    go = [row[:] for row in gi]
    for (a, b) in binds:
        if 0 <= a < h and 0 <= b < w:
            go[a][b] = 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared with the golf entry): the rule is one sweep per orientation.
    # For each of the four ways an L-tromino can sit in its 2x2 box (which corner is
    # missing), every hole of that orientation is filled with 1 at once. So the unit
    # of action is "all holes of L-shapes missing corner X" -- one Color(1) per
    # orientation class, each selection nameable from I alone.
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            try:
                ei = ex[0] if not isinstance(ex, dict) else ex["input"]
                cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
            except Exception:
                pass
    bgc = cnt.most_common(1)[0][0]

    seen = np.zeros((h, w), dtype=bool)
    groups = {}
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            comp = []
            dq = deque([(r, c)])
            seen[r, c] = True
            while dq:
                y, x = dq.popleft()
                comp.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == I[y, x]:
                        seen[ny, nx] = True
                        dq.append((ny, nx))
            cells = set(comp)
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if len(comp) != 3 or (r1 - r0) != 1 or (c1 - c0) != 1:
                continue
            missing = [(y, x) for y in (r0, r1) for x in (c0, c1) if (y, x) not in cells]
            if len(missing) != 1:
                continue
            my, mx = missing[0]
            groups.setdefault((my == r0, mx == c0), []).append((my, mx))

    ops, sels = [], []
    painted = set()
    # one sweep per orientation: missing top-left, top-right, bottom-left, bottom-right
    for key in ((True, True), (True, False), (False, True), (False, False)):
        cells = sorted(set(groups.get(key, [])) - painted)  # shared hole already 1 -> skip
        if not cells:
            continue
        painted.update(cells)
        ops.append(1)
        sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 3aa6fb7a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3aa6fb7a"
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
                                f"for task 3aa6fb7a"
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
                    f"Failed to build a complete episode for task 3aa6fb7a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3aa6fb7a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
