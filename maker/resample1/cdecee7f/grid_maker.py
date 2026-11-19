"""
ARC Task: cdecee7f (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ccols = random.sample(remcols, random.randint(2, 6))
    n_ex = num_examples if num_examples else 3
    groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    vals = [random.choice(g) for g in groups]
    if n_ex >= 3:
        examples = [{"numc": v} for v in vals]
        examples += [{"numc": random.randint(1, 9)} for _ in range(n_ex - 3)]
    else:
        picks = [vals[2]] + random.sample(vals[:2], max(0, n_ex - 1))
        examples = [{"numc": v} for v in picks[:max(1, n_ex)]]
    random.shuffle(examples)
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, numc=None) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = choice(cols)
    remcols = [c for c in cols if c != bgc]
    if ccols is None:
        numcols = unifint(diff_lb, diff_ub, (1, 9))
        ccols = sample(remcols, numcols)
    ccols = [c for c in ccols if c != bgc] or [choice(remcols)]
    max_h = max(3, int(max_h))
    max_w = max(3, int(max_w))
    h = unifint(diff_lb, diff_ub, (3, max_h))
    if numc is None:
        w = unifint(diff_lb, diff_ub, (3, max_w))
        numc = unifint(diff_lb, diff_ub, (1, min(9, w)))
    else:
        numc = max(1, min(9, int(numc)))
        wlb = max(3, min(numc, max_w))
        w = unifint(diff_lb, diff_ub, (wlb, max_w))
        numc = min(numc, w)
    locs = sorted(sample(list(range(w)), numc))
    gi = canvas(bgc, (h, w))
    go = []
    for j in locs:
        iloc = randint(0, h - 1)
        col = choice(ccols)
        gi = fill(gi, col, {(iloc, j)})
        go.append(col)
    go = go + [bgc] * (9 - len(go))
    go = tuple(go)
    go = tuple([go[:3], go[3:6][::-1], go[6:]])
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: the canvas colour; marks are at most one per column in an
    # h>=3 grid, so the canvas colour is the majority colour
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    marks = [(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) != bgc]
    marks.sort(key=lambda rc: rc[1])          # read the marks left to right
    marks = marks[:9]
    # snake order: row 0 L->R, row 1 R->L, row 2 L->R
    slots = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2)]

    # candidate 3x3 patches to build the answer in, fewest marks inside first
    blocks = []
    for r0 in range(hi - 2):
        for c0 in range(wi - 2):
            cnt = sum(1 for (r, c) in marks if r0 <= r < r0 + 3 and c0 <= c < c0 + 3)
            blocks.append((cnt, r0, c0))
    blocks.sort()

    def paint_plan(R0, C0):
        ops, sels = [], []
        if any(int(I[R0 + a, C0 + b]) != bgc for a in range(3) for b in range(3)):
            ops.append(bgc); sels.append([R0, C0, 2, 2])  # whole patch = base
        for k, (r, c) in enumerate(marks):
            col = int(I[r, c])
            a, b = slots[k]
            ops.append(col); sels.append(sel_of([(R0 + a, C0 + b)]))
        return ops, sels

    def move_plan(R0, C0):
        inblock = lambda r, c: R0 <= r < R0 + 3 and C0 <= c < C0 + 3
        g = I.copy()
        ops, sels = [], []
        visited = {g.tobytes()}
        tgt = {k: (R0 + slots[k][0], C0 + slots[k][1]) for k in range(len(marks))}
        pend = {k: marks[k] for k in range(len(marks))}
        holes = set()
        placed = set()

        def try_slide(k):
            # unit-step slide of mark k to its slot; reject if any step would
            # leave the grid unchanged or revisit an earlier grid state
            r, c = pend[k]
            tr, tc = tgt[k]
            col = int(g[r, c])
            H, W = g.shape

            def leg(a, b, axis):
                if axis == 0:
                    return [(1 if b > a else -1, 0)] * abs(b - a)
                return [(0, 1 if b > a else -1)] * abs(b - a)

            # direct L-shaped routes first; if gliding over same-coloured cells
            # would stall, detour through another row / column
            routes = [leg(r, tr, 0) + leg(c, tc, 1), leg(c, tc, 1) + leg(r, tr, 0)]
            det = []
            for rv in range(H):
                if rv not in (r, tr):
                    det.append(leg(r, rv, 0) + leg(c, tc, 1) + leg(rv, tr, 0))
            for cv in range(W):
                if cv not in (c, tc):
                    det.append(leg(c, cv, 1) + leg(r, tr, 0) + leg(cv, tc, 1))
            det.sort(key=len)
            routes += det
            for seq in routes:
                snap = g.copy(); snap[r, c] = 0
                rr, cc = r, c
                seen = set(visited)
                cur = g
                ok = True
                for dr, dc in seq:
                    rr, cc = rr + dr, cc + dc
                    cur = snap.copy(); cur[rr, cc] = col
                    key = cur.tobytes()
                    if key in seen:
                        ok = False; break
                    seen.add(key)
                if ok:
                    return seq, cur, seen
            return None

        while pend:
            pos_of = {p: k for k, p in pend.items()}
            cands = [k for k in sorted(pend) if tgt[k] == pend[k] or tgt[k] not in pos_of]
            if not cands:
                return None
            chosen = False
            for k in cands:
                r, c = pend[k]
                tr, tc = tgt[k]
                col = int(g[r, c])
                if (r, c) == (tr, tc):
                    pend.pop(k); placed.add((tr, tc)); chosen = True
                    break
                if col == 0:
                    # a 0-coloured mark cannot be grabbed (0 = empty): paint it
                    if g[tr, tc] != 0:   # target may already read 0 (a vacated start)
                        ng = g.copy(); ng[tr, tc] = 0
                        if ng.tobytes() in visited:
                            continue
                        ops.append(0); sels.append(sel_of([(tr, tc)]))
                        g = ng; visited.add(g.tobytes())
                    pend.pop(k); placed.add((tr, tc))
                    holes.add((r, c))   # erased later with the other vacated cells
                    chosen = True
                    break
                res = try_slide(k)
                if res is None:
                    continue
                seq, cur, seen = res
                for i, step in enumerate(seq):
                    ops.append({(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[step])
                    sels.append(sel_of([(r, c)]) if i == 0 else sel_of([]))
                g = cur; visited = seen
                pend.pop(k); holes.add((r, c)); placed.add((tr, tc))
                chosen = True
                break
            if not chosen:
                return None
        # repair vacated start cells that lie inside the answer patch
        fix = sorted(p for p in holes if inblock(*p) and g[p] == 0 and p not in placed)
        if bgc != 0 and fix:
            ops.append(bgc); sels.append(sel_of(fix))
        return ops, sels

    res = None
    for _, R0, C0 in blocks:
        res = move_plan(R0, C0)
        if res is not None:
            break
    if res is None:
        _, R0, C0 = blocks[0]
        res = paint_plan(R0, C0)
    ops, sels = res
    if not (hi == 3 and wi == 3 and R0 == 0 and C0 == 0):
        ops.append(33); sels.append([R0, C0, 2, 2])   # keep exactly the 3x3 patch
    ops.append(34); sels.append([0, 0, 2, 2])
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
                        f"num_examples+1 ({num_examples + 1}) for task cdecee7f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task cdecee7f"
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
                                f"for task cdecee7f"
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
                    f"Failed to build a complete episode for task cdecee7f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"cdecee7f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
