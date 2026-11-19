"""
ARC Task: d0f5fe59 (RE-ARC) — LLM-generated grid_maker
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
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    if bgc is None:
        bgc = random.randint(0, 9)
    if fgc is None:
        fgc = random.choice([c for c in range(10) if c != bgc])

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    cap = max(1, min(30, max_h, max_w, (h * w) // 9))
    nobjs = unifint(diff_lb, diff_ub, (1, cap))

    gi = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    nfound, trials, maxtrials = 0, 0, nobjs * 5
    while trials < maxtrials and nfound < nobjs:
        oh = unifint(diff_lb, diff_ub, (1, 5))
        ow = unifint(diff_lb, diff_ub, (1, 5))
        bx = {(i, j) for i in range(oh) for j in range(ow)}
        sp = random.choice(sorted(bx))
        shp = {sp}
        dev = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = random.choice((dev, oh * ow - dev))
        ncells = min(max(1, ncells), oh * ow - 1)
        for _ in range(ncells):
            nb = set()
            for (a, b) in shp:
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb.add((a + da, b + db))
            cands = sorted((bx - shp) & nb)
            if not cands:
                break
            shp.add(random.choice(cands))
        mi = min(a for a, _ in shp)
        mj = min(b for _, b in shp)
        shp = {(a - mi, b - mj) for a, b in shp}
        if len(inds) == 0:
            break
        li, lj = random.choice(sorted(inds))
        plcd = {(a + li, b + lj) for a, b in shp}
        if plcd.issubset(inds):
            for a, b in plcd:
                gi[a][b] = fgc
            rem = set(plcd)
            for a, b in plcd:
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        rem.add((a + da, b + db))
            inds = inds - rem
            nfound += 1
        trials += 1
    n = max(1, nfound)
    go = [[bgc] * n for _ in range(n)]
    for i in range(n):
        go[i][i] = fgc
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    fg = [c for c in cnt if c != bgc]
    fgc = min(fg, key=lambda c: (cnt[c], c)) if fg else bgc

    def components(g, skip):
        h, w = g.shape
        vis = np.zeros((h, w), bool)
        out = []
        for r in range(h):
            for c in range(w):
                if vis[r, c] or (skip is not None and g[r, c] == skip):
                    continue
                col = g[r, c]
                comp, stack = [], [(r, c)]
                vis[r, c] = True
                while stack:
                    y, x = stack.pop()
                    comp.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not vis[ny, nx] and g[ny, nx] == col:
                            vis[ny, nx] = True
                            stack.append((ny, nx))
                out.append(comp)
        return out

    # count the objects of I: each connected non-background component is one object
    n = len(components(I, bgc))
    n = max(1, min(30, n))

    ops, sels = [], []

    # 1. canvas becomes n x n (full rectangle, transparent crop)
    ops.append(33)
    sels.append([0, 0, n - 1, n - 1])
    state = np.zeros((n, n), dtype=int)
    rr, cc = min(n, hi), min(n, wi)
    state[:rr, :cc] = I[:rr, :cc]

    # 2. clear every leftover object fragment / zero padding region back to background,
    #    one connected region at a time (each FloodFill acts on one named region)
    for comp in components(state, bgc):
        col = state[comp[0]]
        if col == fgc and all(y == x for (y, x) in comp):
            continue  # fragment lying entirely on the diagonal in the line colour
        r, c = comp[0]
        ops.append(10 + bgc)
        sels.append(sel_of([(r, c)]))
        for (y, x) in comp:
            state[y, x] = bgc

    # 3. draw the main diagonal in the object colour
    diag = [(i, i) for i in range(n)]
    if any(state[i, i] != fgc for i in range(n)):
        ops.append(fgc)
        sels.append(sel_of(diag))
        for i in range(n):
            state[i, i] = fgc

    ops.append(34)
    sels.append([0, 0, n - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task d0f5fe59"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d0f5fe59"
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
                                f"for task d0f5fe59"
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
                    f"Failed to build a complete episode for task d0f5fe59 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d0f5fe59-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
