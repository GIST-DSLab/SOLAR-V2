"""
ARC Task: f8c80d96 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque

from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------- colors
def sample_colors(num_examples=None) -> dict:
    """bgc = canvas colour, linc = colour the nested boxes are drawn with.
    5 is hardcoded by the task (output background) so it is excluded."""
    cols = [c for c in range(10) if c != 5]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


# ----------------------------------------------------------------------------- generator
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, linc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    ow = randint(1, 3 if h > 10 else 2)
    oh = randint(1, 3 if w > 10 else 2)
    loci = randint(-oh + 1, h - 1)
    locj = randint(-ow + 1, w - 1)
    obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = canvas(bgc, (h, w))
    go = canvas(5, (h, w))
    ulci, ulcj = decrement(ulcorner(obj))
    lrci, lrcj = increment(lrcorner(obj))
    hoffs = randint(2, 4 if h > 12 else 3)
    woffs = randint(2, 4 if w > 12 else 3)
    lns = []
    for k in range(max(h, w) // min(hoffs, woffs) + 1):
        lnx = box(frozenset({(ulci - hoffs * k, ulcj - woffs * k),
                             (lrci + hoffs * k, lrcj + woffs * k)}))
        lns.append(lnx)
    inds = asindices(gi)
    lns = sfilter(lns, lambda ln: len(ln & inds) > 0)
    nlns = len(lns)
    nmissing = unifint(diff_lb, diff_ub, (0, max(0, nlns - 2)))
    npresent = nlns - nmissing
    for k in range(npresent):
        gi = fill(gi, linc, lns[k])
    for ln in lns:
        go = fill(go, linc, ln)
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------- derivation
def _box_cells(t, l, b, r, h, w):
    """axis-aligned rectangle OUTLINE (t,l)-(b,r), clipped to the h x w grid."""
    cells = set()
    for c in range(l, r + 1):
        for rr in (t, b):
            if 0 <= rr < h and 0 <= c < w:
                cells.add((rr, c))
    for rr in range(t, b + 1):
        for cc in (l, r):
            if 0 <= rr < h and 0 <= cc < w:
                cells.add((rr, cc))
    return cells


def _build_rings(t, l, b, r, ho, wo, h, w):
    """the whole box family: ring k = ring0 grown by (ho, wo) per step,
    until a ring no longer touches the grid (it encloses it completely)."""
    rings = []
    k = 0
    limit = max(h, w) + 4
    while k <= limit:
        tt, ll, bb, rr = t - ho * k, l - wo * k, b + ho * k, r + wo * k
        cells = _box_cells(tt, ll, bb, rr, h, w)
        if cells:
            rings.append(sorted(cells))
        elif tt < 0 and ll < 0 and bb > h - 1 and rr > w - 1:
            break
        k += 1
    return rings


def _components(mask, h, w):
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                comps.append(cells)
    return comps


def _fit(I, linc, bgc):
    """Measure the box family from I alone:
       innermost box = outbox of the small bgc rectangle it encloses,
       period (ho, wo) = step to the next box out.
       Returns (all_rings, n_present)."""
    h, w = I.shape
    S = {(r, c) for r in range(h) for c in range(w) if I[r, c] == linc}
    if not S:
        return None
    comps = _components(I == bgc, h, w)
    def _bb(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        return min(rs), min(cs), max(rs), max(cs)
    comps.sort(key=lambda cc: (lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1))(_bb(cc)))
    for comp in comps[:10]:
        r0, c0, r1, c1 = _bb(comp)
        # the enclosed rectangle may run off the grid edge -> allow extension there
        tops = [r0 - 2, r0 - 1, r0] if r0 == 0 else [r0]
        bots = [r1, r1 + 1, r1 + 2] if r1 == h - 1 else [r1]
        lefts = [c0 - 2, c0 - 1, c0] if c0 == 0 else [c0]
        rights = [c1, c1 + 1, c1 + 2] if c1 == w - 1 else [c1]
        for a in tops:
            for bo in bots:
                if not 1 <= bo - a + 1 <= 3:
                    continue
                for cl in lefts:
                    for cr in rights:
                        if not 1 <= cr - cl + 1 <= 3:
                            continue
                        t, l, b, r = a - 1, cl - 1, bo + 1, cr + 1
                        ring0 = _box_cells(t, l, b, r, h, w)
                        if not ring0 or not ring0 <= S:
                            continue
                        for ho in (2, 3, 4):
                            for wo in (2, 3, 4):
                                ring1 = _box_cells(t - ho, l - wo, b + ho, r + wo, h, w)
                                if not ring1 or not ring1 <= S:
                                    continue
                                rings = _build_rings(t, l, b, r, ho, wo, h, w)
                                acc = set()
                                for k in range(len(rings) + 1):
                                    if acc == S:
                                        return rings, k
                                    if k < len(rings):
                                        acc |= set(rings[k])
    return None


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    colors = sorted(set(I.flatten().tolist()),
                    key=lambda c: int((I == c).sum()))
    fit = None
    linc = bgc = None
    for cand_linc in colors:
        for cand_bgc in colors:
            if cand_bgc == cand_linc:
                continue
            fit = _fit(I, cand_linc, cand_bgc)
            if fit is not None:
                linc, bgc = cand_linc, cand_bgc
                break
        if fit is not None:
            break

    if fit is None:
        # safety net: never expected to trigger
        for c in sorted(set(O.flatten().tolist())):
            cells = [(r, cc) for r in range(h) for cc in range(w)
                     if O[r, cc] == c and I[r, cc] != c]
            if cells:
                ops.append(int(c))
                sels.append(sel_of(cells))
        ops.append(34)
        sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
        return ops, sels

    rings, npresent = fit

    # 1) the canvas turns to 5 everywhere that is background
    bg_cells = [(r, c) for r in range(h) for c in range(w) if I[r, c] == bgc]
    if bg_cells:
        ops.append(5)
        sels.append(sel_of(bg_cells))

    # 2) continue the box family outward: one stamp per missing ring,
    #    innermost missing ring first
    for k in range(npresent, len(rings)):
        ops.append(int(linc))
        sels.append(sel_of(rings[k]))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task f8c80d96"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f8c80d96"
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
                                f"for task f8c80d96"
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
                    f"Failed to build a complete episode for task f8c80d96 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f8c80d96-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
