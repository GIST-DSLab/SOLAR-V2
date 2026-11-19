"""
ARC Task: 239be575 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of

VARIANTS = [{"connected": True}, {"connected": False}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    markcol, sqcol = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"markcol": markcol, "sqcol": sqcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, markcol=None, sqcol=None, connected=None) -> dict:
    def unifint(lb, ub):
        if ub < lb:
            ub = lb
        d = random.uniform(diff_lb, diff_ub)
        return max(lb, min(ub, lb + int(round(d * (ub - lb)))))

    if markcol is None or sqcol is None:
        markcol, sqcol = random.sample(range(1, 10), 2)
    if connected is None:
        connected = random.choice([True, False])

    def comps(g, h, w, col):
        seen = set()
        out = []
        for r in range(h):
            for c in range(w):
                if g[r][c] != col or (r, c) in seen:
                    continue
                st = [(r, c)]
                seen.add((r, c))
                comp = set()
                while st:
                    a, b = st.pop()
                    comp.add((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and (na, nb) not in seen and g[na][nb] == col:
                            seen.add((na, nb))
                            st.append((na, nb))
                out.append(comp)
        return out

    def halo(cells):
        o = set()
        for (r, c) in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                o.add((r + dr, c + dc))
        return o - set(cells)

    hmax = max(6, min(30, max_h))
    wmax = max(6, min(30, max_w))
    while True:
        h = unifint(6, hmax)
        w = unifint(6, wmax)
        g = [[0] * w for _ in range(h)]
        full = [(r, c) for r in range(h - 1) for c in range(w - 1)]
        mindist = unifint(3, min(h, w) - 3)
        a = random.choice(full)
        b = random.choice([x for x in full if x != a])
        tries = 0
        while not (abs(a[0] - b[0]) + abs(a[1] - b[1]) > mindist):
            a = random.choice(full)
            b = random.choice([x for x in full if x != a])
            tries += 1
            if tries > 2000:
                break
        if not (abs(a[0] - b[0]) + abs(a[1] - b[1]) > mindist):
            continue
        aset = {(a[0] + i, a[1] + j) for i in (0, 1) for j in (0, 1)}
        bset = {(b[0] + i, b[1] + j) for i in (0, 1) for j in (0, 1)}
        for (r, c) in aset | bset:
            g[r][c] = sqcol
        cands = [(r, c) for r in range(h) for c in range(w) if g[r][c] == 0]
        num = unifint(int(0.25 * len(cands)), int(0.75 * len(cands)))
        for (r, c) in random.sample(cands, num):
            g[r][c] = markcol
        ha, hb = halo(aset), halo(bset)

        def nbridge():
            return sum(1 for o in comps(g, h, w, markcol) if (o & ha) and (o & hb))

        ss = nbridge()
        if connected and ss == 0:
            while ss == 0:
                opts = [(r, c) for r in range(h) for c in range(w) if g[r][c] == 0]
                if not opts:
                    break
                r, c = random.choice(opts)
                g[r][c] = markcol
                ss = nbridge()
        elif not connected and ss > 0:
            while ss > 0:
                opts = [(r, c) for r in range(h) for c in range(w) if g[r][c] == markcol]
                if not opts:
                    break
                r, c = random.choice(opts)
                g[r][c] = 0
                ss = nbridge()
        pal = {v for row in g for v in row}
        if len(pal) != 3:
            continue
        if (ss > 0) != connected:
            continue
        oc = markcol if connected else 0
        return {"input": g, "output": [[oc]]}


def derive_operations(I, O, examples=None):
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def components(color):
        cells = {(r, c) for r in range(hi) for c in range(wi) if I[r, c] == color}
        seen = set()
        out = []
        for cell in sorted(cells):
            if cell in seen:
                continue
            stack = [cell]
            seen.add(cell)
            comp = set()
            while stack:
                r, c = stack.pop()
                comp.add((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (r + dr, c + dc)
                    if n in cells and n not in seen:
                        seen.add(n)
                        stack.append(n)
            out.append(comp)
        return out

    def is_2x2(comp):
        if len(comp) != 4:
            return False
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        return max(rs) - min(rs) == 1 and max(cs) - min(cs) == 1

    def halo(cells):
        out = set()
        for (r, c) in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                out.add((r + dr, c + dc))
        return out - set(cells)

    # background is 0 (generator paints canvas 0)
    colors = sorted({int(v) for v in I.flatten().tolist() if v != 0})
    sqcol, squares = None, None
    for col in colors:
        cs = components(col)
        if len(cs) == 2 and all(is_2x2(x) for x in cs):
            sqcol, squares = col, cs
            break
    if sqcol is None:
        sqcol = colors[0]
        squares = components(sqcol)[:2]
    markcol = next((c for c in colors if c != sqcol), sqcol)

    halo_a = halo(squares[0])
    halo_b = halo(squares[1])
    bridged = any((comp & halo_a) and (comp & halo_b) for comp in components(markcol))

    ops, sels = [], []
    if bridged:
        # Flood one square into the marker colour: it merges with the marker
        # object that reaches the other square -- the square is absorbed.
        r, c = min(squares[0])
        ops.append(10 + markcol); sels.append([r, c, 0, 0])
        # Crop the canvas to that absorbed square cell (single-cell rectangle).
        ops.append(33); sels.append([r, c, 0, 0])
    else:
        # No marker object bridges the squares: the answer is a background cell.
        zeros = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == 0]
        r, c = zeros[0]
        ops.append(33); sels.append([r, c, 0, 0])
    ops.append(34); sels.append([0, 0, 0, 0])
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
                        f"num_examples+1 ({num_examples + 1}) for task 239be575"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 239be575"
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
                                f"for task 239be575"
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
                    f"Failed to build a complete episode for task 239be575 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"239be575-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
