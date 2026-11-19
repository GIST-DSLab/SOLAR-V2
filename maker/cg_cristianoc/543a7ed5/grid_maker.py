"""
ARC Task: 543a7ed5 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (3, 4)]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numc = random.randint(1, 7)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(lo, hi)

    cols = [c for c in range(10) if c not in (3, 4)]
    if bgc is None:
        bgc = random.choice(cols)
    if ccols is None:
        remcols = [c for c in cols if c != bgc]
        ccols = random.sample(remcols, unifint(diff_lb, diff_ub, (1, 7)))
    ccols = [c for c in ccols if c != bgc] or [c for c in cols if c != bgc][:1]

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    def fill(g, col, cells):
        for (i, j) in cells:
            if 0 <= i < h and 0 <= j < w:
                g[i][j] = col

    def rect(r0, c0, r1, c1):
        return {(i, j) for i in range(r0, r1 + 1) for j in range(c0, c1 + 1)}

    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    indss = {(i, j) for i in range(h) for j in range(w)}
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = random.randint(4, 8)
        ow = random.randint(4, 8)
        subs = [ij for ij in indss if ij[0] < h - oh and ij[1] < w - ow]
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = random.choice(subs)
        bd = rect(loci, locj, loci + oh - 1, locj + ow - 1)
        col = random.choice(ccols)
        if bd.issubset(indss):
            ulci, ulcj = loci + 1, locj + 1
            lrci, lrcj = loci + oh - 2, locj + ow - 2
            bdibd = rect(ulci, ulcj, lrci, lrcj)
            fill(go, col, bdibd)
            box = {(i, j) for (i, j) in bd if i in (loci, loci + oh - 1) or j in (locj, locj + ow - 1)}
            fill(go, 3, box)
            fill(gi, col, bdibd)
            if oh > 5 and ow > 5 and random.randint(1, 10) != 1:
                aa = random.randint(ulci + 1, lrci - 1)
                aa = random.randint(ulci + 1, aa)
                bb = random.randint(ulcj + 1, lrcj - 1)
                bb = random.randint(ulcj + 1, bb)
                cc = random.randint(aa, lrci - 1)
                dd = random.randint(bb, lrcj - 1)
                cc = random.randint(cc, lrci - 1)
                dd = random.randint(dd, lrcj - 1)
                ins = rect(aa, bb, cc, dd)
                fill(go, 4, ins)
                fill(gi, bgc, ins)
            succ += 1
            indss = indss - bd
        tr += 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the canvas colour shared by every input of the episode.
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex_in, _ in examples:
            cnt.update(np.asarray(ex_in, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    def components(mask):
        seen = np.zeros(mask.shape, dtype=bool)
        comps = []
        H, W = mask.shape
        for i in range(H):
            for j in range(W):
                if mask[i, j] and not seen[i, j]:
                    stack = [(i, j)]
                    seen[i, j] = True
                    comp = []
                    while stack:
                        r, c = stack.pop()
                        comp.append((r, c))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            u, v = r + dr, c + dc
                            if 0 <= u < H and 0 <= v < W and mask[u, v] and not seen[u, v]:
                                seen[u, v] = True
                                stack.append((u, v))
                    comps.append(comp)
        return comps

    # Objects: univalued, 4-connected, non-background components of I.
    objs = []
    for col in sorted(set(I.flatten().tolist()) - {bgc}):
        for comp in components(I == col):
            objs.append(comp)
    objs.sort(key=lambda cells: (min(r for r, _ in cells), min(c for _, c in cells)))

    ops, sels = [], []
    for cells in objs:
        rmin = min(r for r, _ in cells); rmax = max(r for r, _ in cells)
        cmin = min(c for _, c in cells); cmax = max(c for _, c in cells)

        # 1. Frame the object: its whole outbox ring painted 3 as one unit.
        ring = []
        for c in range(cmin - 1, cmax + 2):
            for r in (rmin - 1, rmax + 1):
                if 0 <= r < hi and 0 <= c < wi:
                    ring.append((r, c))
        for r in range(rmin, rmax + 1):
            for c in (cmin - 1, cmax + 1):
                if 0 <= r < hi and 0 <= c < wi:
                    ring.append((r, c))
        ring = sorted(set(ring))
        if ring:
            ops.append(3); sels.append(sel_of(ring))

        # 2. Fill each enclosed hole (background region inside the object's bbox)
        #    with 4 via a single flood-fill seed per hole.
        sub = np.zeros_like(I, dtype=bool)
        sub[rmin:rmax + 1, cmin:cmax + 1] = (I[rmin:rmax + 1, cmin:cmax + 1] == bgc)
        for hole in components(sub):
            seed = min(hole)
            ops.append(14); sels.append([seed[0], seed[1], 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 543a7ed5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 543a7ed5"
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
                                f"for task 543a7ed5"
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
                    f"Failed to build a complete episode for task 543a7ed5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"543a7ed5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
