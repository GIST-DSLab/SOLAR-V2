"""
ARC Task: 447fd412 (RE-ARC) — LLM-generated grid_maker
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
    bgc, indic, mainc = random.sample(range(10), 3)
    return {"bgc": bgc, "indic": indic, "mainc": mainc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, a + int(round(d * (b - a)))))


def _nbrs8(cell):
    r, c = cell
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, indic=None, mainc=None, **kw) -> dict:
    if bgc is None or indic is None or mainc is None:
        bgc, indic, mainc = random.sample(range(10), 3)
    h = _unifint(diff_lb, diff_ub, (12, max(12, max_h)))
    w = _unifint(diff_lb, diff_ub, (12, max(12, max_w)))
    oh = _unifint(diff_lb, diff_ub, (1, 4))
    ow = _unifint(diff_lb, diff_ub, (1, 4))
    if oh * ow < 3:
        if random.choice((True, False)):
            oh = _unifint(diff_lb, diff_ub, (3, 4))
        else:
            ow = _unifint(diff_lb, diff_ub, (3, 4))
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    ncells = _unifint(diff_lb, diff_ub, (3, oh * ow))
    obj = {random.choice(sorted(bounds))}
    for _ in range(ncells - 1):
        nb = set()
        for cc in obj:
            nb |= _nbrs8(cc)
        cand = sorted((bounds - obj) & nb)
        obj.add(random.choice(cand))
    r0 = min(i for i, j in obj); c0 = min(j for i, j in obj)
    obj = {(i - r0, j - c0) for i, j in obj}
    oh = max(i for i, j in obj) + 1
    ow = max(j for i, j in obj) + 1
    objt = sorted(obj)
    kk = len(obj)
    nindic = random.randint(1, kk // 2 if kk % 2 == 1 else kk // 2 - 1)
    indicobj = set(random.sample(objt, nindic))
    mainobj = obj - indicobj
    cobj = [(indic, ij) for ij in indicobj] + [(mainc, ij) for ij in mainobj]
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    for col, (i, j) in cobj:
        gi[i + loci, j + locj] = col
        go[i + loci, j + locj] = col
    plcd_idx = {(i + loci, j + locj) for i, j in obj}
    nb = set()
    for cc in plcd_idx:
        nb |= _nbrs8(cc)
    inds = {(i, j) for i in range(h) for j in range(w) if gi[i, j] == bgc} - nb
    noccs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (4 * len(plcd_idx)))))
    tr, maxtr, succ = 0, 5 * noccs, 0
    while succ < noccs and tr < maxtr:
        tr += 1
        fac = random.randint(1, min(5, min(h, w) // max(oh, ow) - 1))
        outobj = [(col, (i * fac + a, j * fac + b)) for col, (i, j) in cobj
                  for a in range(fac) for b in range(fac)]
        inobj = [(col, ij) for col, ij in outobj if col == indic]
        hh, ww = oh * fac, ow * fac
        cands = sorted(ij for ij in inds if ij[0] <= h - hh and ij[1] <= w - ww)
        if not cands:
            continue
        li, lj = random.choice(cands)
        outp = [(col, (i + li, j + lj)) for col, (i, j) in outobj
                if 0 <= i + li < h and 0 <= j + lj < w]
        outpi = {ij for _, ij in outp}
        if outpi.issubset(inds):
            succ += 1
            innb = set()
            for _, (i, j) in inobj:
                innb |= _nbrs8((i + li, j + lj))
            inds = (inds - outpi) - innb
            for col, (i, j) in inobj:
                gi[i + li, j + lj] = col
            for col, (i, j) in outp:
                go[i, j] = col
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading: the multicolour template is a STAMP. Wherever only its indicator
    part appears (at scale 1..5), the template is stamped there at that scale,
    which completes the body.  Scale 1 is stamped by copying the template's own
    body (CopyI, masked to the body cells) and pasting it.  At a larger scale the
    first occurrence is built by painting its upscaled body; that finished body is
    then copied (CopyO, masked to the cells just painted) and pasted onto every
    further occurrence of the same scale.  Scales are processed smallest first.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    def finish():
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    if examples:
        bgc = Counter(int(v) for ex in examples for v in np.asarray(ex[0]).flatten()).most_common(1)[0][0]
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]; seen[r, c] = True; cells = []
                while stack:
                    y, x = stack.pop(); cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] != bgc:
                                seen[ny, nx] = True; stack.append((ny, nx))
                comps.append(cells)
    if not comps:
        return finish()

    ti = max(range(len(comps)), key=lambda k: len({int(I[r, c]) for r, c in comps[k]}))
    tmpl = comps[ti]
    other_counts = Counter(int(I[r, c]) for k, cs in enumerate(comps) if k != ti for (r, c) in cs)
    tmpl_counts = Counter(int(I[r, c]) for r, c in tmpl)
    if other_counts:
        indic = other_counts.most_common(1)[0][0]
    else:
        indic = min(tmpl_counts, key=lambda col: tmpl_counts[col])
    body_cols = [col for col in tmpl_counts if col != indic]
    if not body_cols:
        return finish()
    mainc = max(body_cols, key=lambda col: tmpl_counts[col])

    tr0 = min(r for r, c in tmpl); tc0 = min(c for r, c in tmpl)
    norm = [(r - tr0, c - tc0, int(I[r, c])) for r, c in tmpl]
    ind_cells = [(a, b) for a, b, col in norm if col == indic]
    body_cells = [(a, b) for a, b, col in norm if col == mainc]
    all_cells = [(a, b) for a, b, col in norm]
    if not ind_cells or not body_cells:
        return finish()
    oh = max(a for a, b in all_cells) + 1
    ow = max(b for a, b in all_cells) + 1

    H, W = 3 * h, 3 * w
    P = np.full((H, W), bgc, dtype=int)
    P[h:2 * h, w:2 * w] = I
    comp_of = {}
    for k, cs in enumerate(comps):
        for (r, c) in cs:
            comp_of[(r + h, c + w)] = k
    comp_size = [len(cs) for cs in comps]
    anchor = min(ind_cells)
    ind_positions = [(r + h, c + w) for r in range(h) for c in range(w) if I[r, c] == indic]

    def fits(i, j, fac):
        return 0 <= i and 0 <= j and i + oh * fac <= H and j + ow * fac <= W

    def matches(i, j, fac):
        for (a, b) in ind_cells:
            for da in range(fac):
                for db in range(fac):
                    if P[i + a * fac + da, j + b * fac + db] != indic:
                        return False
        for (a, b) in body_cells:
            for da in range(fac):
                for db in range(fac):
                    if P[i + a * fac + da, j + b * fac + db] != bgc:
                        return False
        return True

    by_fac = {}
    seen_occ = set()
    for fac in range(1, 6):
        need = len(ind_cells) * fac * fac
        for (pr, pc) in ind_positions:
            i = pr - anchor[0] * fac; j = pc - anchor[1] * fac
            if (fac, i, j) in seen_occ:
                continue
            seen_occ.add((fac, i, j))
            if not fits(i, j, fac) or not matches(i, j, fac):
                continue
            hit = set()
            for (a, b) in all_cells:
                for da in range(fac):
                    for db in range(fac):
                        k = comp_of.get((i + a * fac + da, j + b * fac + db))
                        if k is not None:
                            hit.add(k)
            if sum(comp_size[k] for k in hit) != need:
                continue
            by_fac.setdefault(fac, []).append((i - h, j - w))

    def body_at(i, j, fac):
        return [(i + a * fac + da, j + b * fac + db) for (a, b) in body_cells
                for da in range(fac) for db in range(fac)]

    def inside(cells):
        return all(0 <= r < h and 0 <= c < w for r, c in cells)

    def paint(cells):
        cells = [(r, c) for r, c in cells if 0 <= r < h and 0 <= c < w]
        if cells:
            ops.append(int(mainc)); sels.append(sel_of(cells))

    for fac in sorted(by_fac):
        occs = sorted(by_fac[fac])
        full = [o for o in occs if inside(body_at(o[0], o[1], fac))]
        part = [o for o in occs if o not in full]
        if mainc == 0 or not full:
            # colour 0 cannot travel through the clipboard: paint directly
            for o in occs:
                paint(body_at(o[0], o[1], fac))
            continue
        if fac == 1:
            # stamp source = the template's own body, copied from the input
            src = [(tr0 + a, tc0 + b) for (a, b) in body_cells]
            targets = full
        else:
            # build the first full occurrence of this scale, then reuse it as the stamp
            first = full[0]
            src = body_at(first[0], first[1], fac)
            paint(src)
            targets = full[1:]
        if targets:
            ops.append(28 if fac == 1 else 29); sels.append(sel_of(src))
            br = min(r for r, c in src); bc = min(c for r, c in src)
            sr, sc = (tr0, tc0) if fac == 1 else first
            dr0, dc0 = br - sr, bc - sc  # body-bbox offset inside the occurrence frame
            for (i, j) in targets:
                ops.append(30); sels.append([i + dr0, j + dc0, 0, 0])
        for o in part:
            paint(body_at(o[0], o[1], fac))

    return finish()


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
                        f"num_examples+1 ({num_examples + 1}) for task 447fd412"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 447fd412"
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
                                f"for task 447fd412"
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
                    f"Failed to build a complete episode for task 447fd412 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"447fd412-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
