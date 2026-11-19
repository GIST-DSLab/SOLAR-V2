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


def sample_colors(num_examples=None) -> dict:
    bgc, fgc = random.sample(range(10), 2)
    return {"bgc": bgc, "fgc": fgc}


def _unifint(lb, ub, rng):
    a, b = rng
    lo = a + round((b - a) * lb)
    hi = a + round((b - a) * ub)
    lo, hi = min(lo, hi), max(lo, hi)
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, fgc=1, **kw) -> dict:
    h = _unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = _unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    objh = _unifint(diff_lb, diff_ub, (0, 2))
    objw = _unifint(diff_lb, diff_ub, (0 if objh > 0 else 1, 2))
    objh = objh * 2 + 1
    objw = objw * 2 + 1
    bb = {(i, j) for i in range(objh) for j in range(objw)}
    sp = (objh // 2, objw // 2)
    obj = {sp}
    bb.discard(sp)

    def frontier():
        out = set()
        for (i, j) in obj:
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                out.add((i + di, j + dj))
        return sorted((bb - obj) & out)

    ncells = _unifint(diff_lb, diff_ub, (max(objh, objw), objh * objw))
    for _ in range(ncells - 1):
        obj.add(random.choice(frontier()))
    while (max(i for i, _ in obj) - min(i for i, _ in obj) + 1 != objh or
           max(j for _, j in obj) - min(j for _, j in obj) + 1 != objw):
        obj.add(random.choice(frontier()))
    remcols = [c for c in range(10) if c not in (bgc, fgc)]
    ncols = _unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, ncols)
    mi = min(i for i, _ in obj); mj = min(j for _, j in obj)
    obj = [(random.choice(ccols), (i - mi, j - mj)) for (i, j) in sorted(obj)]
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    loci = random.randint(0, h - objh)
    locj = random.randint(0, w - objw)
    plcd = [(c, (i + loci, j + locj)) for c, (i, j) in obj]
    for c, (i, j) in plcd:
        gi[i][j] = c
        go[i][j] = c
    pidx = {ij for _, ij in plcd}
    nb8 = set()
    for (i, j) in pidx:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                nb8.add((i + di, j + dj))
    inds = {(i, j) for i in range(h) for j in range(w)} - pidx - nb8
    nobjs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * ncells))))
    maxtrials = 4 * nobjs
    tr = 0
    succ = 0
    while succ < nobjs and tr <= maxtrials:
        if len(inds) == 0:
            break
        loc = random.choice(sorted(inds))
        plcd = [(c, (i + loc[0], j + loc[1])) for c, (i, j) in obj]
        plcdi = {ij for _, ij in plcd}
        if plcdi.issubset(inds):
            for c, (i, j) in plcd:
                go[i][j] = c
            gi[loc[0] + objh // 2][loc[1] + objw // 2] = fgc
            succ += 1
            dn = set()
            for (i, j) in plcdi:
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    dn.add((i + di, j + dj))
            inds = (inds - plcdi) - dn
        tr += 1
    return {"input": gi, "output": go}


def _components(I, bgc):
    H, W = I.shape
    seen = np.zeros((H, W), bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or I[r, c] == bgc:
                continue
            stack = [(r, c)]; seen[r, c] = True; comp = []
            while stack:
                a, b = stack.pop(); comp.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < H and 0 <= y < W and not seen[x, y] and I[x, y] != bgc:
                        seen[x, y] = True; stack.append((x, y))
            comps.append(comp)
    return comps


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    # background: the colour every demonstration input is painted on
    if examples:
        cnt = Counter()
        for ei, _ in examples:
            cnt.update(np.asarray(ei).flatten().tolist())
        cnt.update(I.flatten().tolist())
        bgc = cnt.most_common(1)[0][0]
        if bgc not in I:
            bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    ops, sels = [], []
    comps = _components(I, bgc)
    if not comps:
        return [34], [[0, 0, H - 1, W - 1]]
    tmpl = max(comps, key=len)                       # the multicolour template
    markers = sorted(c[0] for c in comps if len(c) == 1)  # single-pixel markers
    if not markers:
        return [34], [[0, 0, H - 1, W - 1]]
    tr0 = min(r for r, _ in tmpl); tc0 = min(c for _, c in tmpl)
    th = max(r for r, _ in tmpl) - tr0 + 1; tw = max(c for _, c in tmpl) - tc0 + 1
    ch, cw = th // 2, tw // 2
    rel = [(r - tr0, c - tc0) for r, c in tmpl]
    nz = [p for p in rel if I[p[0] + tr0, p[1] + tc0] != 0]
    zc = [p for p in rel if I[p[0] + tr0, p[1] + tc0] == 0]
    tset = set(tmpl)
    blank = []
    if nz:
        bbox_cells = [(r, c) for r in range(tr0, tr0 + th) for c in range(tc0, tc0 + tw)]
        blank = [p for p in bbox_cells if p not in tset and I[p] != 0]
        if blank:
            # isolate the template inside its bbox so the copy carries only its cells
            ops.append(0); sels.append(sel_of(blank))
            ops.append(29)
        else:
            ops.append(28)
        sels.append([tr0, tc0, th - 1, tw - 1])  # full template bbox (transparent copy)
    covered = set()
    for (mr, mc) in markers:
        r0, c0 = mr - ch, mc - cw
        if nz:
            ops.append(30); sels.append([r0, c0, 0, 0])
        if zc:
            # colour-0 template cells are transparent to Paste: paint them explicitly
            ops.append(0); sels.append(sel_of([(r0 + a, c0 + b) for a, b in zc]))
        covered |= {(r0 + a, c0 + b) for a, b in rel}
    if blank:
        # restore the template's surrounding background cells not taken by a copy
        restore = [p for p in blank if p not in covered]
        if restore:
            ops.append(bgc); sels.append(sel_of(restore))
    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
