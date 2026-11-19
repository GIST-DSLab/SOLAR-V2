"""
ARC Task: 6aa20dc0 (RE-ARC) — LLM-generated grid_maker
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
    bgc, fgc, c1, c2 = random.sample(range(10), 4)
    return {"bgc": bgc, "fgc": fgc, "c1": c1, "c2": c2}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, c1, c2) -> dict:
    from dsl import (interval, canvas, asindices, totuple, dneighbors, recolor,
                     dmirror, hmirror, vmirror, cmirror, identity, shift, paint,
                     backdrop, outbox, compose, normalize, upscale, shape, sfilter,
                     toindices)
    from utils import unifint
    from random import randint, sample, choice
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    od = unifint(diff_lb, diff_ub, (2, 4))
    ncellsextra = randint(1, max(1, (od ** 2 - 2) // 2))
    sinds = asindices(canvas(-1, (od, od)))
    extracells = set(sample(totuple(sinds - {(0, 0), (od - 1, od - 1)}), ncellsextra))
    extracells.add(choice(totuple(dneighbors((0, 0)) & sinds)))
    extracells.add(choice(totuple(dneighbors((od - 1, od - 1)) & sinds)))
    extracells = frozenset(extracells)
    obj = frozenset({(c1, (0, 0)), (c2, (od - 1, od - 1))}) | recolor(fgc, extracells)
    obj = obj | dmirror(obj)
    if choice((True, False)):
        obj = hmirror(obj)
    gi = canvas(bgc, (h, w))
    loci = randint(0, h - od)
    locj = randint(0, w - od)
    plcd = shift(obj, (loci, locj))
    gi = paint(gi, plcd)
    go = tuple(e for e in gi)
    inds = asindices(gi)
    inds = inds - backdrop(outbox(plcd))
    nocc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (od ** 2 * 2))))
    succ = 0
    tr = 0
    maxtr = 4 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = randint(1, 4)
        mf1 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        mf2 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        mf = compose(mf2, mf1)
        cobj = normalize(upscale(mf(obj), fac))
        ohx, owx = shape(cobj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - ohx and ij[1] <= w - owx)
        if len(cands) == 0:
            continue
        locc = choice(totuple(cands))
        cobjo = shift(cobj, locc)
        cobji = sfilter(cobjo, lambda cij: cij[0] != fgc)
        cobjoi = toindices(cobjo)
        if cobjoi.issubset(inds):
            succ += 1
            inds = inds - backdrop(outbox(cobjoi))
            gi = paint(gi, cobji)
            go = paint(go, cobjo)
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    # Uses only I: the template, its body colour and every site are all read from the input.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 1) template = 8-connected component(s) with the most distinct colours
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            stack, comp = [(r, c)], []
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] != bgc:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(comp)
    if not comps:
        return [34], [[0, 0, h - 1, w - 1]]
    ncol = [len({int(I[r, c]) for r, c in comp}) for comp in comps]
    mx = max(ncol)
    tcells = [p for comp, n in zip(comps, ncol) if n == mx for p in comp]
    tr0 = min(p[0] for p in tcells); tr1 = max(p[0] for p in tcells)
    tc0 = min(p[1] for p in tcells); tc1 = max(p[1] for p in tcells)
    BH, BW = tr1 - tr0 + 1, tc1 - tc0 + 1
    base = {}
    for r in range(tr0, tr1 + 1):
        for c in range(tc0, tc1 + 1):
            if I[r, c] != bgc:
                base[(r - tr0, c - tc0)] = int(I[r, c])
    if not base:
        return [34], [[0, 0, h - 1, w - 1]]
    fgc = Counter(base.values()).most_common(1)[0][0]
    body_keys = sorted(k for k, v in base.items() if v == fgc)

    # 2) every legal stamp; each stamp cell remembers which template cell it came from
    def mirror(cells, H, W, mf):
        if mf == 'identity':
            return dict(cells), H, W
        if mf == 'dmirror':
            return {(c, r): v for (r, c), v in cells.items()}, W, H
        if mf == 'cmirror':
            return {(W - 1 - c, H - 1 - r): v for (r, c), v in cells.items()}, W, H
        if mf == 'hmirror':
            return {(H - 1 - r, c): v for (r, c), v in cells.items()}, H, W
        return {(r, W - 1 - c): v for (r, c), v in cells.items()}, H, W

    units = []
    for f in range(1, 5):
        up = {}
        for (r, c), v in base.items():
            for dr in range(f):
                for dc in range(f):
                    up[(r * f + dr, c * f + dc)] = (v, (r, c))
        for mf in ('identity', 'dmirror', 'cmirror', 'hmirror', 'vmirror'):
            units.append(mirror(up, BH * f, BW * f, mf))

    # 3) find occurrences: marker cells present in I, body still background, isolated
    images = {k: [] for k in body_keys}   # template body cell -> its image at every site
    seen_sets = set()
    for cells, UH, UW in units:
        marks = {p: v for p, (v, _) in cells.items() if v != fgc}
        body = {p: src for p, (v, src) in cells.items() if v == fgc}
        if not marks or not body:
            continue
        mset = set(marks)
        ring = set()
        for (r, c) in mset:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if (r + dr, c + dc) not in mset:
                    ring.add((r + dr, c + dc))
        for r0 in range(h - UH + 1):
            for c0 in range(w - UW + 1):
                if any(I[r0 + r, c0 + c] != v for (r, c), v in marks.items()):
                    continue
                if any(I[r0 + r, c0 + c] != bgc for (r, c) in body):
                    continue
                if any(0 <= r0 + r < h and 0 <= c0 + c < w and I[r0 + r, c0 + c] != bgc
                       for (r, c) in ring):
                    continue
                filled = frozenset((r0 + r, c0 + c) for (r, c) in body)
                if filled in seen_sets:
                    continue
                seen_sets.add(filled)
                for (r, c), src in body.items():
                    images[src].append((r0 + r, c0 + c))

    # 4) unit of work = one template body cell: paint its counterpart (scaled block,
    #    mirrored position) at every site in one stroke
    for k in body_keys:
        cells = sorted(set(images[k]))
        if cells:
            ops.append(int(fgc)); sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 6aa20dc0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6aa20dc0"
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
                                f"for task 6aa20dc0"
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
                    f"Failed to build a complete episode for task 6aa20dc0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6aa20dc0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
