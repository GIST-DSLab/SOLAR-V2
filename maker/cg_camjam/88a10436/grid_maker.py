"""
ARC Task: 88a10436 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return int(round(a + (b - a) * d))


def sample_colors(num_examples=None) -> dict:
    bgc, fgc = random.sample(range(10), 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None or fgc is None:
        bgc, fgc = random.sample(cols, 2)
    for _attempt in range(200):
        h = _unifint(diff_lb, diff_ub, (8, max_h))
        w = _unifint(diff_lb, diff_ub, (8, max_w))
        objh = _unifint(diff_lb, diff_ub, (0, 2))
        objw = _unifint(diff_lb, diff_ub, (0 if objh > 0 else 1, 2))
        objh = objh * 2 + 1
        objw = objw * 2 + 1
        bb = {(i, j) for i in range(objh) for j in range(objw)}
        sp = (objh // 2, objw // 2)
        obj = {sp}
        bb.discard(sp)

        def frontier():
            nb = set()
            for (i, j) in obj:
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb.add((i + di, j + dj))
            return sorted((bb - obj) & nb)

        ncells = _unifint(diff_lb, diff_ub, (max(objh, objw), objh * objw))
        for _ in range(ncells - 1):
            obj.add(random.choice(frontier()))

        def hw(o):
            rs = [i for i, _ in o]; cs = [j for _, j in o]
            return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
        while hw(obj) != (objh, objw):
            obj.add(random.choice(frontier()))
        remcols = [c for c in cols if c not in (bgc, fgc)]
        ncols = _unifint(diff_lb, diff_ub, (1, 8))
        ccols = random.sample(remcols, ncols)
        cobj = [(random.choice(ccols), ij) for ij in sorted(obj)]
        gi = [[bgc] * w for _ in range(h)]
        go = [[bgc] * w for _ in range(h)]
        if objh > h or objw > w:
            continue
        loci = random.randint(0, h - objh)
        locj = random.randint(0, w - objw)
        plcd = [(c, (i + loci, j + locj)) for c, (i, j) in cobj]
        for c, (i, j) in plcd:
            gi[i][j] = c
            go[i][j] = c
        pidx = {ij for _, ij in plcd}
        neigh = set()
        for (i, j) in pidx:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    neigh.add((i + di, j + dj))
        inds = {(i, j) for i in range(h) for j in range(w)} - pidx - neigh
        nobjs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * ncells))))
        maxtrials = 4 * nobjs
        tr = 0
        succ = 0
        while succ < nobjs and tr <= maxtrials:
            if len(inds) == 0:
                break
            li, lj = random.choice(sorted(inds))
            p2 = [(c, (i + li, j + lj)) for c, (i, j) in cobj]
            p2i = {ij for _, ij in p2}
            if p2i.issubset(inds):
                for c, (i, j) in p2:
                    go[i][j] = c
                gi[li + objh // 2][lj + objw // 2] = fgc
                succ += 1
                dn = set()
                for (i, j) in p2i:
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        dn.add((i + di, j + dj))
                inds = inds - p2i - dn
            tr += 1
        if succ >= 1:
            return {"input": gi, "output": go}
    raise RuntimeError("generation failed")


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # background: the canvas colour; confirmed against the demonstrations' inputs
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex_i, _ in examples:
            cnt.update(np.asarray(ex_i, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # 4-connected non-background components of I
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] \
                                and I[nr, nc] != bgc:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(sorted(cells))
    if not comps:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    # the stamp is the one multi-cell object; markers are the lone cells
    stamp = max(comps, key=len)
    r0 = min(r for r, _ in stamp); c0 = min(c for _, c in stamp)
    oh = max(r for r, _ in stamp) - r0 + 1
    ow = max(c for _, c in stamp) - c0 + 1
    ar, ac = oh // 2, ow // 2                      # stamp's centre within its bbox
    markers = sorted(cl[0] for cl in comps if len(cl) == 1 and cl is not stamp)

    # stamp cells whose colour is 0: Copy/Paste cannot carry them
    zero_offs = [(r - r0, c - c0) for (r, c) in stamp if I[r, c] == 0]
    has_nonzero = any(I[r, c] != 0 for (r, c) in stamp)

    G = I.copy()
    if has_nonzero and markers:
        # pick up the whole stamp (its own cells only) as one rigid unit
        ops.append(28); sels.append(sel_of(stamp))
    for (mr, mc) in markers:
        tr, tc = mr - ar, mc - ac                  # stamp origin centred on the marker
        if has_nonzero:
            ops.append(30); sels.append([tr, tc, 0, 0])
            for (r, c) in stamp:
                v = int(I[r, c])
                if v != 0:
                    G[tr + r - r0, tc + c - c0] = v
        # the 0-coloured parts of this stamp copy, painted explicitly
        zc = [(tr + dr, tc + dc) for dr, dc in zero_offs]
        zc = [(r, c) for r, c in zc if 0 <= r < hi and 0 <= c < wi and G[r, c] != 0]
        if zc:
            ops.append(0); sels.append(sel_of(zc))
            for r, c in zc:
                G[r, c] = 0

    # full-grid rectangle: submitting the whole canvas
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 88a10436"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 88a10436"
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
                                f"for task 88a10436"
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
                    f"Failed to build a complete episode for task 88a10436 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"88a10436-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
