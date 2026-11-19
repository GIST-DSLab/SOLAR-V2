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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, fgc = random.sample(range(10), 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, round(a + (b - a) * d)))

    if bgc is None or fgc is None:
        bgc, fgc = random.sample(range(10), 2)
    h = unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    objh = unifint(diff_lb, diff_ub, (0, 2))
    objw = unifint(diff_lb, diff_ub, (0 if objh > 0 else 1, 2))
    objh = objh * 2 + 1
    objw = objw * 2 + 1
    bb = {(i, j) for i in range(objh) for j in range(objw)}
    sp = (objh // 2, objw // 2)
    obj = {sp}
    bb.discard(sp)

    def dn(c):
        return {(c[0] - 1, c[1]), (c[0] + 1, c[1]), (c[0], c[1] - 1), (c[0], c[1] + 1)}

    def nb(c):
        return {(c[0] + a, c[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    def frontier():
        s = set()
        for c in obj:
            s |= dn(c)
        return sorted((bb - obj) & s)

    ncells = unifint(diff_lb, diff_ub, (max(objh, objw), objh * objw))
    for _ in range(ncells - 1):
        obj.add(random.choice(frontier()))
    while (max(r for r, _ in obj) - min(r for r, _ in obj) + 1 != objh or
           max(c for _, c in obj) - min(c for _, c in obj) + 1 != objw):
        obj.add(random.choice(frontier()))
    remcols = [c for c in range(10) if c not in (bgc, fgc)]
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, ncols)
    mr = min(r for r, _ in obj); mc = min(c for _, c in obj)
    obj = [(random.choice(ccols), (r - mr, c - mc)) for r, c in obj]
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    loci = random.randint(0, h - objh)
    locj = random.randint(0, w - objw)
    for col, (r, c) in obj:
        gi[r + loci, c + locj] = col
        go[r + loci, c + locj] = col
    pl = {(r + loci, c + locj) for _, (r, c) in obj}
    inds = {(i, j) for i in range(h) for j in range(w)} - pl
    for c in pl:
        inds -= nb(c)
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * ncells))))
    maxtrials = 4 * nobjs
    tr = succ = 0
    while succ < nobjs and tr <= maxtrials:
        if not inds:
            break
        li, lj = random.choice(sorted(inds))
        plcd = [(col, (r + li, c + lj)) for col, (r, c) in obj]
        plcdi = {ij for _, ij in plcd}
        if plcdi.issubset(inds):
            for col, (r, c) in plcd:
                go[r, c] = col
            rs = [r for r, _ in plcdi]; cs = [c for _, c in plcdi]
            gi[(min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2] = fgc
            succ += 1
            inds -= plcdi
            for c in plcdi:
                inds -= dn(c)
        tr += 1
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    # background: the canvas colour (sparse task, clearly the majority colour);
    # markers are isolated single cells, the stamp is the one multi-cell component.
    vals, cnts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(cnts)])

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
    if comps:
        stamp = max(comps, key=len)
        r0 = min(r for r, _ in stamp); c0 = min(c for _, c in stamp)
        oh = max(r for r, _ in stamp) - r0 + 1
        ow = max(c for _, c in stamp) - c0 + 1
        ar, ac = oh // 2, ow // 2
        markers = sorted(cl[0] for cl in comps if len(cl) == 1)
        nz = [(r, c) for r, c in stamp if I[r, c] != 0]
        zc = [(r, c) for r, c in stamp if I[r, c] == 0]
        if markers and nz:
            # copy the stamp's own cells once (mask-limited, so no bbox background),
            # then stamp it centred on each marker
            ops.append(28); sels.append(sel_of(stamp))
            for (mr, mc) in markers:
                ops.append(30); sels.append([mr - ar, mc - ac, 0, 0])
        if markers and zc:
            # Paste is transparent for colour 0: paint the stamp's 0-cells at every copy
            cells = [(mr + r - r0 - ar, mc + c - c0 - ac)
                     for (mr, mc) in markers for (r, c) in zc]
            ops.append(0); sels.append(sel_of(cells))
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
