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
from collections import Counter, deque

import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Colour 1 is reserved for the filled holes, so background and shape colour never use it.
    cols = [c for c in range(10) if c != 1]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return min(b, max(a, int(round(a + d * (b - a)))))

    cols = [c for c in range(10) if c != 1]
    if bgc is None:
        bgc = random.choice(cols)
    if fgc is None:
        fgc = random.choice([c for c in cols if c != bgc])

    base = [(0, 0), (0, 1), (1, 0), (1, 1)]
    h = unifint(diff_lb, diff_ub, (3, max(3, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (3, max(3, min(30, max_w))))
    gi = [[bgc] * w for _ in range(h)]
    inds = {(r, c) for r in range(h) for c in range(w)}
    maxnum = max(1, ((h * w) // 2) // 3)
    num = unifint(diff_lb, diff_ub, (1, maxnum))
    kk, tr = 0, 0
    maxtrials = num * 2
    binds = set()
    while kk < num and tr < maxtrials:
        loc = random.choice(sorted(inds)) if inds else (0, 0)
        ooo = random.choice(base)
        oo = [p for p in base if p != ooo]
        oop = [(loc[0] + a, loc[1] + b) for a, b in oo]
        if inds and set(oop).issubset(inds):
            rem = set(oop)
            for (r, c) in oop:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rem.add((r + dr, c + dc))
            inds -= rem
            for (r, c) in oop:
                gi[r][c] = fgc
            binds.add((loc[0] + ooo[0], loc[1] + ooo[1]))
            kk += 1
        tr += 1
    go = [row[:] for row in gi]
    for (r, c) in binds:
        if 0 <= r < h and 0 <= c < w:
            go[r][c] = 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared with the golf entry): a hole is a background cell that is closed in
    # by the shape on one horizontal side AND one vertical side, i.e. the missing corner of
    # an L. The golf entry treats every such cell as one unit under a single cell-local test.
    # Here they are gathered into one set named from I and painted 1 in a single Color op.
    # Guard: the horizontal neighbour, the vertical neighbour and the diagonal between them
    # must all be foreground cells of the SAME 3-cell, 2x2-bbox component. This way two
    # separate shapes that happen to touch a background cell from two sides do not make a
    # false hole.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # Background colour is shared across the episode, so count the demonstration inputs too.
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            try:
                ei = ex["input"] if isinstance(ex, dict) else ex[0]
                cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
            except Exception:
                pass
    bgc = cnt.most_common(1)[0][0]

    # Label the 4-connected same-colour components (used only by the guard).
    comp_id = -np.ones((h, w), dtype=int)
    comp_cells = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or comp_id[r, c] >= 0:
                continue
            k = len(comp_cells)
            cells = []
            dq = deque([(r, c)])
            comp_id[r, c] = k
            while dq:
                y, x = dq.popleft()
                cells.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w and comp_id[ny, nx] < 0
                            and I[ny, nx] == I[y, x]):
                        comp_id[ny, nx] = k
                        dq.append((ny, nx))
            comp_cells.append(cells)

    def is_L(k):
        cells = comp_cells[k]
        if len(cells) != 3:
            return False
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        return max(rs) - min(rs) == 1 and max(cs) - min(cs) == 1

    holes = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc:
                continue
            closed = False
            for dc in (-1, 1):          # horizontal side
                for dr in (-1, 1):      # vertical side
                    hr, hc = r, c + dc
                    vr, vc = r + dr, c
                    dgr, dgc = r + dr, c + dc
                    if not (0 <= hc < w and 0 <= vr < h):
                        continue
                    a, b, d = comp_id[hr, hc], comp_id[vr, vc], comp_id[dgr, dgc]
                    if a >= 0 and a == b == d and is_L(a):
                        closed = True
            if closed:
                holes.append((r, c))

    ops, sels = [], []
    if holes:
        ops.append(1)
        sels.append(sel_of(holes))   # every closed-in corner cell, painted 1 together
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
