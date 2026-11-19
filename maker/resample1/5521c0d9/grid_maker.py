"""
ARC Task: 5521c0d9 (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]  # number of np.rot90 k (CCW quarter-turns)

def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc and c != 0]
    ncols = random.randint(2, len(remcols))
    ccols = random.sample(remcols, ncols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": k} for k in ex]
    plan.append({"rot": random.choice(ex)})
    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}

def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols, rot=None) -> dict:
    from random import choice, sample
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        return random.randint(min(lo, hi), max(lo, hi))
    if rot is None:
        rot = random.choice(ROTS)
    if rot % 2 == 0:
        hmax, wmax = max_h, max_w
    else:
        hmax, wmax = max_w, max_h
    h = unifint(diff_lb, diff_ub, (4, min(30, hmax)))
    w = unifint(diff_lb, diff_ub, (6, min(30, wmax)))
    inds = list(range(w))
    while True:
        nobjs = unifint(diff_lb, diff_ub, (1, w // 3))
        speps = sample(inds, nobjs * 2)
        if 0 not in speps and w - 1 not in speps:
            break
    speps = sorted(speps)
    starts = speps[::2]
    ends = speps[1::2]
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    forb = -1
    for sp, ep in zip(starts, ends):
        col = choice([c for c in ccols if c != forb])
        forb = col
        hdev = unifint(diff_lb, diff_ub, (0, h // 2))
        hei = choice((hdev, h - hdev))
        hei = min(max(1, hei), h - 1)
        gi[h - hei:h, sp:ep + 1] = col
        top = max(0, h - 2 * hei)
        go[top:h - hei, sp:ep + 1] = col
    gi = np.rot90(gi, rot)
    go = np.rot90(go, rot)
    return {"input": gi.tolist(), "output": go.tolist()}

def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: dominant colour on the grid perimeter (bars never cover it)
    border = np.concatenate([I[0, :], I[-1, :], I[:, 0], I[:, -1]]).tolist()
    bgc = max(set(border), key=border.count)
    # 4-connected univalued components of non-background cells = the bars
    seen = np.zeros_like(I, dtype=bool)
    objs = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for na, nb in ((a+1, b), (a-1, b), (a, b+1), (a, b-1)):
                    if 0 <= na < hi and 0 <= nb < wi and not seen[na, nb] and I[na, nb] == col:
                        seen[na, nb] = True
                        stack.append((na, nb))
            objs.append((int(col), sorted(cells)))
    # base edge: the edge every bar is attached to
    def touches(cells, e):
        if e == 'B': return any(r == hi - 1 for r, _ in cells)
        if e == 'T': return any(r == 0 for r, _ in cells)
        if e == 'L': return any(c == 0 for _, c in cells)
        return any(c == wi - 1 for _, c in cells)
    edge = None
    for e in ('B', 'T', 'L', 'R'):
        if objs and all(touches(cl, e) for _, cl in objs):
            edge = e
            break
    ops, sels = [], []
    for col, cells in objs:
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        oh = max(rs) - min(rs) + 1; ow = max(cs) - min(cs) + 1
        # each bar slides away from its base edge by its own length
        if edge == 'B': op, d, dr, dc = 20, oh, -1, 0
        elif edge == 'T': op, d, dr, dc = 21, oh, 1, 0
        elif edge == 'L': op, d, dr, dc = 22, ow, 0, 1
        else: op, d, dr, dc = 23, ow, 0, -1
        dest = [(r + dr * d, c + dc * d) for r, c in cells]
        dest = [(r, c) for r, c in dest if 0 <= r < hi and 0 <= c < wi]
        if col == 0:
            # colour 0 is invisible to ARCLE's object buffer: it cannot be moved
            ops.append(bgc); sels.append(sel_of(cells))
            ops.append(0); sels.append(sel_of(dest))
            continue
        ops.append(op); sels.append(sel_of(cells))  # grab bar, first step
        for _ in range(d - 1):
            ops.append(op); sels.append(sel_of([]))  # continue same grab
        hole = sorted(set(cells) - set(dest))
        if bgc != 0 and hole:
            ops.append(bgc); sels.append(sel_of(hole))  # repair vacated footprint
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
                        f"num_examples+1 ({num_examples + 1}) for task 5521c0d9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5521c0d9"
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
                                f"for task 5521c0d9"
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
                    f"Failed to build a complete episode for task 5521c0d9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5521c0d9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
