"""
ARC Task: 2bee17df (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    import random
    cols = [c for c in range(10) if c != 3]          # 3 is reserved for the marked lines
    bgc = random.choice(cols)                        # interior fill color
    rem = [c for c in cols if c != bgc]
    cola = random.choice(rem)                        # first perimeter arc color
    colb = random.choice([c for c in rem if c != cola])   # second perimeter arc color
    return {"bgc": bgc, "cola": cola, "colb": colb}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, cola=None, colb=None) -> dict:
    import random
    from collections import Counter

    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            a, b = b, a
        ba = min(a + int((b - a) * lb), b)
        bb = min(a + int((b - a) * ub), b)
        if ba > bb:
            ba, bb = bb, ba
        return random.randint(ba, bb)

    cols_all = [c for c in range(10) if c != 3]
    if bgc is None:
        bgc = random.choice(cols_all)
    rem = [c for c in cols_all if c != bgc]
    if cola is None:
        cola = random.choice(rem)
    if colb is None:
        colb = random.choice([c for c in rem if c != cola])

    hi_h = max(7, min(30, int(max_h)))
    hi_w = max(7, min(30, int(max_w)))

    last = None
    for _attempt in range(300):
        h = unifint(diff_lb, diff_ub, (7, hi_h))
        w = unifint(diff_lb, diff_ub, (7, hi_w))
        g = [[bgc] * w for _ in range(h)]

        # perimeter index order (top row, right col, bottom row, left col)
        indord = [(0, j) for j in range(w)]
        indord += [(i, w - 1) for i in range(1, h - 1)]
        indord += [(h - 1, j) for j in range(w - 1, 0, -1)]
        indord += [(i, 0) for i in range(h - 1, 0, -1)]
        k = len(indord)
        sp = random.randint(0, k)
        arr = indord[sp:] + indord[:sp]
        ep = random.randint(k // 2 - 3, k // 2 + 1)
        ep = max(0, min(k, ep))
        for (i, j) in arr[:ep]:
            g[i][j] = cola
        for (i, j) in arr[ep:]:
            g[i][j] = colb

        nr = unifint(diff_lb, diff_ub, (1, min(4, min(h, w) // 2)))
        for kk in range(nr):
            r1, c1, r2, c2 = 1 + kk, 1 + kk, h - 1 - kk, w - 1 - kk
            if r2 < r1 or c2 < c1:
                continue
            ring = set()
            for j in range(c1, c2 + 1):
                ring.add((r1, j))
                ring.add((r2, j))
            for i in range(r1, r2 + 1):
                ring.add((i, c1))
                ring.add((i, c2))
            for br in (cola, colb):
                nbrs = set()
                for i in range(h):
                    for j in range(w):
                        if g[i][j] == br:
                            nbrs.update([(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)])
                bcands = [p for p in ring if g[p[0]][p[1]] == bgc and p in nbrs]
                jj = len(bcands)
                lo = max(0, jj // 2 - 2)
                hi = min(jj, jj // 2 + 1)
                if hi < lo:
                    lo, hi = hi, lo
                jj2 = random.randint(lo, hi) if jj > 0 else 0
                for (i, j) in random.sample(bcands, jj2):
                    g[i][j] = br

        # frontiers of trim(g): full uniform rows/cols of the trimmed grid, shifted by (1,1)
        H, W = h - 2, w - 2
        inner = [[g[r + 1][c + 1] for c in range(W)] for r in range(H)]
        flat = [v for row in inner for v in row]
        mc = Counter(flat).most_common(1)[0][0]
        urows = [r for r in range(H) if len(set(inner[r])) == 1]
        ucols = [c for c in range(W) if len(set(inner[r][c] for r in range(H))) == 1]
        # keep verifier (mostcolor-based) and generator (frontiers) in agreement
        ok = all(inner[r][0] == mc for r in urows) and all(inner[0][c] == mc for c in ucols)
        if not ok:
            continue
        if not urows and not ucols:
            last = (g, [row[:] for row in g])
            continue
        go = [row[:] for row in g]
        for r in urows:
            for c in range(W):
                go[r + 1][c + 1] = 3
        for c in ucols:
            for r in range(H):
                go[r + 1][c + 1] = 3
        return {"input": g, "output": go}

    if last is None:
        g = [[bgc] * 7 for _ in range(7)]
        for j in range(7):
            g[0][j] = cola
            g[6][j] = colb
        for i in range(7):
            g[i][0] = cola
            g[i][6] = colb
        go = [row[:] for row in g]
        for r in range(1, 6):
            for c in range(1, 6):
                go[r][c] = 3
        return {"input": g, "output": go}
    return {"input": last[0], "output": last[1]}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for (r, c) in cells]}

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    def rect(r0, c0, nh, nw):
        return [(r, c) for r in range(r0, r0 + nh) for c in range(c0, c0 + nw)]

    # ---- rule: inside the trimmed grid (border stripped), every row/column that is
    # entirely the interior fill colour becomes a line of 3s.
    inner = I[1:h - 1, 1:w - 1]
    H, W = inner.shape
    bgc = Counter(inner.flatten().tolist()).most_common(1)[0][0]
    rows = [r for r in range(H) if bool(np.all(inner[r, :] == bgc))]
    cols = [c for c in range(W) if bool(np.all(inner[:, c] == bgc))]

    if not rows and not cols:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # 1. TRIM: reframe the canvas onto the interior the rule lives in.
    #    (full rectangle -> the exact cells of the trimmed region)
    ops.append(33); sels.append(sel_of(rect(1, 1, H, W)))
    cur = inner.copy()

    # 2. the uniform rows become identical lines of 3 -> paint one, replicate it.
    if rows:
        r0 = rows[0]
        ops.append(3); sels.append(sel_of(rect(r0, 0, 1, W)))
        cur[r0, :] = 3
        if len(rows) > 1:
            ops.append(29); sels.append(sel_of(rect(r0, 0, 1, W)))   # CopyO the 3-line
            for r in rows[1:]:
                ops.append(30); sels.append(sel_of([(r, 0)]))        # Paste at each row
                cur[r, :] = 3

    # 3. same for the uniform columns (skipped if the rows already covered everything)
    if cols and len(rows) < H:
        c0 = cols[0]
        ops.append(3); sels.append(sel_of(rect(0, c0, H, 1)))
        cur[:, c0] = 3
        if len(cols) > 1:
            ops.append(29); sels.append(sel_of(rect(0, c0, H, 1)))   # CopyO the 3-column
            for c in cols[1:]:
                ops.append(30); sels.append(sel_of([(0, c)]))        # Paste at each column
                cur[:, c] = 3

    # 4. carry the marked interior on the clipboard, restore the untrimmed input,
    #    and paste the interior back at its (1,1) offset.
    ops.append(29); sels.append(sel_of(rect(0, 0, H, W)))            # CopyO marked interior
    ops.append(31); sels.append(sel_of(rect(0, 0, H, W)))            # CopyInput: un-trim
    ops.append(30); sels.append(sel_of([(1, 1)]))                    # Paste at (1,1)

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 2bee17df"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2bee17df"
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
                                f"for task 2bee17df"
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
                    f"Failed to build a complete episode for task 2bee17df "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2bee17df-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
