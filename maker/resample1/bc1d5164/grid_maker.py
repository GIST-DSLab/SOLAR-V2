"""
ARC Task: bc1d5164 (RE-ARC) — LLM-generated grid_maker
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
import itertools
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    # object colour kept non-zero so the quadrant objects can be grabbed by Move
    objc = random.choice([c for c in range(1, 10) if c != bgc])
    return {"bgc": bgc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, objc=1) -> dict:
    hub = max(3, min(15, (max_h + 1) // 2))
    wub = max(2, min(14, (max_w - 1) // 2))
    h = unifint(diff_lb, diff_ub, (3, hub))
    w = unifint(diff_lb, diff_ub, (2, wub))
    fullh = 2 * h - 1
    fullw = 2 * w + 1
    inds = asindices(canvas(-1, (h, w)))
    nA = randint(1, (h - 1) * (w - 1) - 1) if (h - 1) * (w - 1) > 1 else 1
    nB = randint(1, (h - 1) * (w - 1) - 1) if (h - 1) * (w - 1) > 1 else 1
    nC = randint(1, (h - 1) * (w - 1) - 1) if (h - 1) * (w - 1) > 1 else 1
    nD = randint(1, (h - 1) * (w - 1) - 1) if (h - 1) * (w - 1) > 1 else 1
    A = sample(totuple(sfilter(inds, lambda ij: ij[0] < h - 1 and ij[1] < w - 1)), nA)
    B = sample(totuple(sfilter(inds, lambda ij: ij[0] < h - 1 and ij[1] > 0)), nB)
    C = sample(totuple(sfilter(inds, lambda ij: ij[0] > 0 and ij[1] < w - 1)), nC)
    D = sample(totuple(sfilter(inds, lambda ij: ij[0] > 0 and ij[1] > 0)), nD)
    gi = canvas(bgc, (fullh, fullw))
    gi = fill(gi, objc, A)
    gi = fill(gi, objc, shift(B, (0, fullw - w)))
    gi = fill(gi, objc, shift(C, (fullh - h, 0)))
    gi = fill(gi, objc, shift(D, (fullh - h, fullw - w)))
    go = canvas(bgc, (h, w))
    go = fill(go, objc, set(A) | set(B) | set(C) | set(D))
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    h = (H + 1) // 2
    w = (W - 1) // 2
    # background = colour of the empty separator column (generator guarantees it)
    bgc = int(I[h - 1, w])
    objc_vals = [int(v) for v in np.unique(I) if int(v) != bgc]
    objc = objc_vals[0] if objc_vals else None
    origins = {"A": (0, 0), "B": (0, w + 1), "C": (h - 1, 0), "D": (h - 1, w + 1)}

    if objc is None:
        return [33, 34], [[0, 0, h - 1, w - 1], [0, 0, h - 1, w - 1]]

    def quad(name):
        r0, c0 = origins[name]
        return [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)
                if I[r, c] == objc]

    qcells = {n: quad(n) for n in origins}

    lift_ops, lift_sels = [], []
    base = I.copy()
    mobj = objc
    if objc == 0:
        # colour 0 cannot be carried by Move: lift object cells to a stand-in colour
        mobj = next(c for c in range(1, 10) if c != bgc)
        zc = [(r, c) for r in range(H) for c in range(W) if I[r, c] == 0]
        lift_ops, lift_sels = [mobj], [sel_of(zc)]
        base[I == 0] = mobj

    MV = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}

    def simulate(ops_, sels_, tr, tc):
        g = I.copy()
        obj, bg, active = None, None, False
        for op, sl in zip(ops_, sels_):
            if 20 <= op <= 23:
                cells = sl["cells"]
                if cells:
                    obj = [(r, c, int(g[r, c])) for r, c in cells]
                    bg = g.copy()
                    for r, c in cells:
                        bg[r, c] = 0
                    active = True
                elif not active:
                    continue
                d = MV[op]
                obj = [(r + d[0], c + d[1], v) for r, c, v in obj]
                g = bg.copy()
                for r, c, v in obj:
                    if v > 0 and 0 <= r < H and 0 <= c < W:
                        g[r, c] = v
            elif op <= 9:
                for r, c in sl["cells"]:
                    g[r, c] = op
        return g[tr:tr + h, tc:tc + w].copy()

    def paths(dr, dc):
        v = [(20 if dr > 0 else 21)] * abs(dr)
        hz = [(23 if dc > 0 else 22)] * abs(dc)
        alt = []
        for k in range(max(len(v), len(hz))):
            if k < len(v): alt.append(v[k])
            if k < len(hz): alt.append(hz[k])
        out = [v + hz]
        if v and hz:
            out += [hz + v, alt]
        return out

    def comps(cells):
        left = set(cells); out = []
        while left:
            st = [left.pop()]; comp = list(st)
            while st:
                a, b = st.pop()
                for nb in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                    if nb in left:
                        left.discard(nb); st.append(nb); comp.append(nb)
            out.append(sorted(comp))
        return sorted(out)

    def plan(target, choice):
        tr, tc = origins[target]
        grid = base.copy()
        ops, sels = list(lift_ops), list(lift_sels)
        seen = {I.tobytes(), base.tobytes()}
        for name, mode, pi in choice:
            qr, qc = origins[name]
            dr, dc = qr - tr, qc - tc
            # cells of this quadrant that add something not yet shown at the destination
            new = [(r, c) for r, c in qcells[name] if grid[r - dr, c - dc] != mobj]
            if not new:
                continue
            groups = comps(new) if mode == 2 else [qcells[name] if mode else new]
            steps = paths(dr, dc)[pi]
            for mover in groups:
                snap = grid.copy()
                for r, c in mover:
                    snap[r, c] = 0
                cur = list(mover)
                for k, op in enumerate(steps):
                    ops.append(op)
                    sels.append(sel_of(cur) if k == 0 else sel_of([]))   # grab once
                    sr, sc = MV[op]
                    cur = [(r + sr, c + sc) for r, c in cur]
                    g = snap.copy()
                    for r, c in cur:
                        if 0 <= r < H and 0 <= c < W:
                            g[r, c] = mobj
                    if g.tobytes() in seen:
                        return None
                    seen.add(g.tobytes())
                    grid = g
        return ops, sels, grid

    def candidates():
        for level in range(2):
            for target in "ABCD":
                others = [n for n in "ABCD" if n != target]
                for order in itertools.permutations(others):
                    opts = []
                    for n in order:
                        qr, qc = origins[n]
                        tr, tc = origins[target]
                        npaths = len(paths(qr - tr, qc - tc))
                        modes = (0, 1) if level == 0 else (2,)
                        opts.append([(n, m, p) for p in range(npaths) for m in modes])
                    for combo in itertools.product(*opts):
                        yield target, combo

    def all_needed(ops_, sels_, tr, tc):
        ref = simulate(ops_, sels_, tr, tc)
        for k in range(len(ops_)):
            if np.array_equal(simulate(ops_[:k] + ops_[k + 1:], sels_[:k] + sels_[k + 1:], tr, tc), ref):
                return False
        return True

    chosen, fallback = None, None
    for target, combo in candidates():
        res = plan(target, combo)
        if res is None:
            continue
        tr, tc = origins[target]
        if fallback is None:
            fallback = (target, res)
        if all_needed(res[0], res[1], tr, tc):
            chosen = (target, res)
            break
    if chosen is None:
        chosen = fallback
    target, (ops, sels, fg) = chosen
    tr, tc = origins[target]

    if objc == 0 and len(ops) == len(lift_ops):
        # nothing to overlay: no lift needed
        ops, sels, fg = [], [], I.copy()
        objc = -1

    # crop the canvas to the quadrant everything was overlaid onto
    ops.append(33); sels.append([tr, tc, h - 1, w - 1])
    if objc == 0:
        # flood each lifted component back to colour 0 (one seed per component)
        sub = fg[tr:tr + h, tc:tc + w]
        for comp in comps([(r, c) for r in range(h) for c in range(w) if sub[r, c] == mobj]):
            r, c = comp[0]
            ops.append(10); sels.append([r, c, 0, 0])
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
                        f"num_examples+1 ({num_examples + 1}) for task bc1d5164"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task bc1d5164"
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
                                f"for task bc1d5164"
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
                    f"Failed to build a complete episode for task bc1d5164 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"bc1d5164-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
