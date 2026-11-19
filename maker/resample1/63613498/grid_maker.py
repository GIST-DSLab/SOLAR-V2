"""
ARC Task: 63613498 (RE-ARC) — LLM-generated grid_maker
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
    bgc, sepc = random.sample(cols, 2)
    remcols = [c for c in cols if c not in (bgc, sepc)]
    ncols = random.randint(1, 8)
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "sepc": sepc, "ccols": ccols}


def _unif(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round((b - a) * lb))
    hi = a + int(round((b - a) * ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _dn(obj):
    s = set()
    for (i, j) in obj:
        s |= {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
    return s


def _norm(obj):
    mi = min(i for i, j in obj)
    mj = min(j for i, j in obj)
    return {(i - mi, j - mj) for i, j in obj}


def _grow(objh, objw, ncells):
    bounds = {(i, j) for i in range(objh) for j in range(objw)}
    obj = {random.choice(sorted(bounds))}
    for _ in range(ncells - 1):
        obj.add(random.choice(sorted((bounds - obj) & _dn(obj))))
    return obj, bounds


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sepc=None, ccols=None, **kw) -> dict:
    if bgc is None or sepc is None or ccols is None:
        d = sample_colors()
        bgc, sepc, ccols = d["bgc"], d["sepc"], d["ccols"]
    h = _unif(diff_lb, diff_ub, (10, max_h))
    w = _unif(diff_lb, diff_ub, (10, max_w))
    objh = _unif(diff_lb, diff_ub, (1, h // 3))
    objw = _unif(diff_lb, diff_ub, (1, w // 3))
    obj, bounds = _grow(objh, objw, _unif(diff_lb, diff_ub, (1, objh * objw)))
    gi = np.full((h, w), bgc, dtype=int)
    objc = random.choice(ccols)
    for i, j in obj:
        gi[i, j] = objc
    gi[objh + 1, 0:objw + 2] = sepc
    gi[0:objh + 2, objw + 1] = sepc
    inds = {(i, j) for i in range(h) for j in range(w)
            if not (i <= objh + 1 and j <= objw + 1)}
    nobjs = _unif(diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    baseobj = _norm(obj)
    obj = _norm(obj)
    go = gi.copy()
    while (succ < nobjs and tr < maxtr) or succ == 0:
        tr += 1
        if tr > 2000:
            break
        oh = max(i for i, j in obj) + 1
        ow = max(j for i, j in obj) + 1
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            break
        loc = random.choice(sorted(cands))
        plcd = {(i + loc[0], j + loc[1]) for i, j in obj}
        if plcd.issubset(inds):
            col = random.choice(ccols)
            for i, j in plcd:
                gi[i, j] = col
                go[i, j] = sepc if succ == 0 else col
            succ += 1
            inds = (inds - plcd) - _dn(plcd)
        objh = random.randint(1, h // 3)
        objw = random.randint(2 if objh == 1 else 1, w // 3)
        if random.choice((True, False)):
            objh, objw = objw, objh
        obj, bounds = _grow(objh, objw, _unif(diff_lb, diff_ub, (1, objh * objw)))
        obj = _norm(obj)
        if obj == baseobj:
            if len(obj) < objh * objw:
                obj.add(random.choice(sorted((bounds - obj) & _dn(obj))))
            else:
                mi = max(i for i, j in obj)
                mj = max(j for i, j in obj)
                obj.discard(random.choice([(0, 0), (0, mj), (mi, 0), (mi, mj)]))
        obj = _norm(obj)
    if succ == 0:
        raise ValueError("no placement")
    k = random.randint(0, 3)
    return {"input": np.rot90(gi, k).tolist(), "output": np.rot90(go, k).tolist()}


def _components(I, bgc):
    H, W = I.shape
    seen = np.zeros((H, W), bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]
            st = [(r, c)]
            seen[r, c] = True
            cells = []
            while st:
                a, b = st.pop()
                cells.append((a, b))
                for x, y in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                    if 0 <= x < H and 0 <= y < W and not seen[x, y] and I[x, y] == col:
                        seen[x, y] = True
                        st.append((x, y))
            comps.append((int(col), cells))
    return comps


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    comps = _components(I, bgc)
    # Separator: L-shaped line (size h+w-1 of its bbox), touching the border,
    # whose enclosed corner region holds background + exactly one other colour.
    best = None
    for col, cells in comps:
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh + bw - 1 != len(cells):
            continue
        if not (r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1):
            continue
        cset = set(cells)
        delta = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                 if (r, c) not in cset]
        if not delta:
            continue
        if len(set(int(I[r, c]) for r, c in delta)) != 2:
            continue
        if best is None or len(cells) > len(best[1]):
            best = (col, cells, delta)
    sepc, sepcells, delta = best
    ref = [(r, c) for r, c in delta if I[r, c] != bgc]
    refn = _norm(ref)
    refset = set(ref)
    target = None
    for col, cells in comps:
        if set(cells) <= refset or cells == sepcells:
            continue
        if _norm(cells) == refn:
            target = cells
            break
    ops, sels = [], []
    # Recolour the object outside that matches the reference shape to the separator colour.
    ops.append(sepc)
    sels.append(sel_of(sorted(target)))
    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 63613498"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 63613498"
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
                                f"for task 63613498"
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
                    f"Failed to build a complete episode for task 63613498 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"63613498-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
