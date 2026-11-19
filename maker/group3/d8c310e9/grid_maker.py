"""
ARC Task: d8c310e9 (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "instance_plan": plan}


def _unif(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    return random.randint(lo, max(lo, hi))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None, **kw) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    m = min(max_h, max_w)  # grid may be rotated, so both sides must fit either bound
    h = _unif(diff_lb, diff_ub, (3, m))
    w = _unif(diff_lb, diff_ub, (10, m))
    p = _unif(diff_lb, diff_ub, (2, (w - 1) // 3))
    remcols = [c for c in range(10) if c != bgc]
    numc = _unif(diff_lb, diff_ub, (1, 9))
    ccols = random.sample(remcols, numc)
    obj = set()
    for j in range(p):
        numcells = _unif(diff_lb, diff_ub, (1, h - 1))
        for ii in range(h - 1, h - numcells - 1, -1):
            obj.add((random.choice(ccols), (ii, j)))
    minobj = obj | {(v, (i, j + p)) for v, (i, j) in obj}
    addonw = random.randint(0, p)
    addon = {(v, (i, j)) for v, (i, j) in obj if j < addonw}
    full = minobj | addon
    ls = random.randint(0, addonw)
    full = {(v, (i, j - ls)) for v, (i, j) in full}
    gi = np.full((h, w), bgc, dtype=int)
    for v, (i, j) in full:
        if 0 <= i < h and 0 <= j < w:
            gi[i, j] = v
    go = gi.copy()
    for k in range(w // (2 * p) + 2):
        for v, (i, j) in full:
            jj = j + k * 2 * p
            if 0 <= i < h and 0 <= jj < w:
                go[i, jj] = v
    kk = {0: 0, 1: 3, 2: 2, 3: 1}[rot]  # identity, rot90, rot180, rot270
    gi = np.rot90(gi, kk)
    go = np.rot90(go, kk)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """Extend the periodic block one grid LINE at a time: every line beyond the block
    (a column in the block's own frame) is a copy of the input line a whole number of
    periods back. Unit of work = one line, not the whole block. O is never read."""
    A = np.asarray(I, dtype=int)
    H, W = A.shape
    full = [0, 0, H - 1, W - 1]

    # frame: the quarter turn that stands the block in the bottom-left corner
    kk = None
    for k in (0, 1, 2, 3):
        C = np.rot90(A, k)
        if len(set(C[0].tolist())) == 1 and len(set(C[:, -1].tolist())) == 1 \
                and len(set(C[-1].tolist())) > 1:
            kk = k
            break
    if kk is None:
        return [34], [full]
    C = np.rot90(A, kk)
    idx = np.rot90(np.arange(H * W).reshape(H, W), kk)  # canonical -> original flat index
    bgc = int(C[0, -1])
    cells = np.argwhere(C != bgc)
    if len(cells) == 0:
        return [34], [full]

    # period: shortest horizontal slide mapping the block onto itself (both directions)
    colors = {(int(r), int(c)): int(C[r, c]) for r, c in cells}
    minc, maxc = int(cells[:, 1].min()), int(cells[:, 1].max())
    per = maxc - minc + 1
    for cand in range(1, maxc - minc + 1):
        ok = all(colors.get((r, c - cand)) == v
                 for (r, c), v in colors.items() if c - cand >= minc) and \
             all(colors.get((r, c + cand)) == v
                 for (r, c), v in colors.items() if c + cand <= maxc)
        if ok:
            per = cand
            break

    def line(j):
        """original-grid cells of canonical column j, in canonical row order"""
        return [divmod(int(x), W) for x in idx[:, j]]

    def rect(pts):
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        return min(rs), min(cs), max(rs) - min(rs), max(cs) - min(cs)

    g = A.copy()
    ops, sels = [], []
    # march outward from the block, one line at a time
    for t in range(maxc + 1, C.shape[1]):
        s = t - per * (-(-(t - maxc) // per))  # same phase, inside the block
        src, dst = line(s), line(t)
        new = g.copy()
        for (a, b), (x, y) in zip(src, dst):
            if A[a, b] != 0:
                new[x, y] = A[a, b]
        if not np.array_equal(new, g):
            r0, c0, hh, ww = rect(src)
            ops.append(28); sels.append([r0, c0, hh, ww])  # full rectangle: one whole grid line
            d0, e0, _, _ = rect(dst)
            ops.append(30); sels.append([d0, e0, 0, 0])
            g = new
        # clipboard treats 0 as empty: blacken the landed cells whose source is black
        if bgc != 0:
            black = [(x, y) for (a, b), (x, y) in zip(src, dst)
                     if A[a, b] == 0 and g[x, y] != 0]
            if black:
                ops.append(0); sels.append(sel_of(black))
                for x, y in black:
                    g[x, y] = 0
    ops.append(34); sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
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
                                f"for task d8c310e9"
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
                    f"Failed to build a complete episode for task d8c310e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d8c310e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
