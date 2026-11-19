"""
ARC Task: 890034e9 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(1, 10))
    markercol = random.choice(cols)
    remcols = [c for c in cols if c != markercol]
    numbgc = random.randint(1, 8)
    bgcols = random.sample(remcols, numbgc)
    return {"markercol": markercol, "bgcols": bgcols}


def _outbox(cells):
    rs = [r for r, c in cells]; cs = [c for r, c in cells]
    r0, r1, c0, c1 = min(rs) - 1, max(rs) + 1, min(cs) - 1, max(cs) + 1
    out = set()
    for c in range(c0, c1 + 1):
        out.add((r0, c)); out.add((r1, c))
    for r in range(r0, r1 + 1):
        out.add((r, c0)); out.add((r, c1))
    return out


def generate(diff_lb, diff_ub, max_h, max_w, markercol=None, bgcols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))
    if markercol is None:
        markercol = random.randint(1, 9)
    if bgcols is None:
        rem = [c for c in range(1, 10) if c != markercol]
        bgcols = random.sample(rem, random.randint(1, 8))
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    oh = random.randint(2, h // 4)
    ow = random.randint(2, w // 4)
    gi = np.array([[random.choice(bgcols) for _ in range(w)] for _ in range(h)], dtype=int)
    allinds = [(i, j) for i in range(h) for j in range(w)]
    numbl = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    for (i, j) in random.sample(allinds, numbl):
        gi[i, j] = 0
    occs = [(i, j) for i in range(h - oh + 1) for j in range(w - ow + 1)
            if (gi[i:i + oh, j:j + ow] == 0).all()]
    for (i, j) in occs:
        a = random.randint(0, oh - 1); b = random.randint(0, ow - 1)
        gi[i + a, j + b] = random.choice(bgcols)
    noccs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // ((oh + 2) * (ow + 2)))))
    go = gi.copy()
    inds = set(allinds)
    succ = 0
    tr = 0
    while tr < 5 * noccs and succ < noccs:
        tr += 1
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            break
        loci, locj = random.choice(cands)
        bd = [(loci + a, locj + b) for a in range(oh) for b in range(ow)]
        plcd = _outbox(bd)
        if plcd.issubset(inds):
            succ += 1
            inds -= plcd
            for (i, j) in bd:
                gi[i, j] = 0; go[i, j] = 0
            for (i, j) in plcd:
                if succ == 1:
                    gi[i, j] = markercol
                go[i, j] = markercol
            if succ > 1:
                lines = [
                    [(loci - 1, locj + b) for b in range(ow)],
                    [(loci + oh, locj + b) for b in range(ow)],
                    [(loci + a, locj - 1) for a in range(oh)],
                    [(loci + a, locj + ow) for a in range(oh)],
                ]
                for ln in lines:
                    i, j = random.choice(ln)
                    gi[i, j] = random.choice(bgcols)
    if succ < 2:
        raise ValueError("not enough boxes")
    return {"input": gi.tolist(), "output": go.tolist()}


def _components(I):
    h, w = I.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = I[r, c]
            st = [(r, c)]; seen[r, c] = True; cells = []
            while st:
                y, x = st.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True; st.append((ny, nx))
            comps.append((int(col), cells))
    return comps


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # find box-shaped univalued objects with min dimension > 2
    cands = []
    for col, cells in _components(I):
        rs = [r for r, c in cells]; cs = [c for r, c in cells]
        bh = max(rs) - min(rs) + 1; bw = max(cs) - min(cs) + 1
        if min(bh, bw) <= 2:
            continue
        inner = {(r, c) for r in range(min(rs) + 1, max(rs)) for c in range(min(cs) + 1, max(cs))}
        box = {(r, c) for r in range(min(rs), max(rs) + 1) for c in range(min(cs), max(cs) + 1)} - inner
        if set(cells) == box:
            cands.append(col)
    cnt = Counter(cands)
    mc = min(cnt, key=lambda k: cnt[k])
    mcells = [(r, c) for r in range(h) for c in range(w) if I[r, c] == mc]
    rs = [r for r, c in mcells]; cs = [c for r, c in mcells]
    ih = max(rs) - min(rs) - 1
    iw = max(cs) - min(cs) - 1

    G = I.copy()
    for i in range(h - ih + 1):
        for j in range(w - iw + 1):
            if not (I[i:i + ih, j:j + iw] == 0).all():
                continue
            interior = [(i + a, j + b) for a in range(ih) for b in range(iw)]
            ring = [(r, c) for (r, c) in _outbox(interior) if 0 <= r < h and 0 <= c < w]
            if all(G[r, c] == mc for r, c in ring):
                continue
            ops.append(mc); sels.append(sel_of(ring))
            for r, c in ring:
                G[r, c] = mc

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
                        f"num_examples+1 ({num_examples + 1}) for task 890034e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 890034e9"
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
                                f"for task 890034e9"
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
                    f"Failed to build a complete episode for task 890034e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"890034e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
