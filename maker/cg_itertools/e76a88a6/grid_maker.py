"""
ARC Task: e76a88a6 (RE-ARC) — LLM-generated grid_maker
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
from dsl import *


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    dmyc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "dmyc": dmyc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dmyc=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if dmyc is None:
        dmyc = choice(remove(bgc, cols))
    while True:
        h = unifint(diff_lb, diff_ub, (8, max(8, max_h)))
        w = unifint(diff_lb, diff_ub, (8, max(8, max_w)))
        objh = unifint(diff_lb, diff_ub, (2, 5))
        objw = unifint(diff_lb, diff_ub, (2, 5))
        bounds = asindices(canvas(0, (objh, objw)))
        shp = {choice(totuple(bounds))}
        nc = unifint(diff_lb, diff_ub, (2, len(bounds) - 2))
        for j in range(nc):
            ij = choice(totuple((bounds - shp) & mapply(dneighbors, shp)))
            shp.add(ij)
        shp = normalize(shp)
        remcols = remove(dmyc, remove(bgc, cols))
        oh, ow = shape(shp)
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        shpp = shift(shp, (loci, locj))
        numco = unifint(diff_lb, diff_ub, (2, 8))
        colll = sample(remcols, numco)
        shppc = frozenset({(choice(colll), ij) for ij in shpp})
        while numcolors(shppc) == 1:
            shppc = frozenset({(choice(colll), ij) for ij in shpp})
        shppcn = normalize(shppc)
        gi = canvas(bgc, (h, w))
        gi = paint(gi, shppc)
        go = tuple(e for e in gi)
        ub = ((h * w) / (oh * ow)) // 2
        ub = max(1, ub)
        numlocs = unifint(diff_lb, diff_ub, (1, ub))
        cnt = 0
        fails = 0
        maxfails = 5 * numlocs
        idns = (asindices(gi) - shpp) - mapply(dneighbors, shpp)
        idns = sfilter(idns, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        while cnt < numlocs and fails < maxfails:
            if len(idns) == 0:
                break
            loc = choice(totuple(idns))
            plcd = shift(shppcn, loc)
            plcdi = toindices(plcd)
            if plcdi.issubset(idns):
                go = paint(go, plcd)
                gi = fill(gi, dmyc, plcdi)
                cnt += 1
                idns = (idns - plcdi) - mapply(dneighbors, plcdi)
            else:
                fails += 1
        if cnt >= 1:
            return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # ---- silhouette colour: read from the demonstrations ------------------
    # (the cells that change in every example are exactly the silhouette cells)
    dmyc = None
    if examples:
        cnt = Counter()
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int); eo = np.asarray(eo, dtype=int)
            if ei.shape != eo.shape:
                continue
            for v in ei[ei != eo].tolist():
                cnt[v] += 1
        if cnt:
            dmyc = cnt.most_common(1)[0][0]
    if dmyc is None or not (I == dmyc).any():
        # fallback: the most frequent colour after the background that is not
        # part of the multicoloured template -> pick most common non-majority colour
        c = Counter(I.flatten().tolist()).most_common()
        dmyc = c[1][0] if len(c) > 1 else c[0][0]

    # background: majority among the non-silhouette colours (template is tiny)
    rest = Counter(v for v in I.flatten().tolist() if v != dmyc)
    if not rest:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    bgc = rest.most_common(1)[0][0]

    # ---- template: every cell that is neither background nor silhouette ----
    tmpl = sorted((r, c) for r in range(h) for c in range(w)
                  if I[r, c] != bgc and I[r, c] != dmyc)
    if not tmpl:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    ar, ac = tmpl[0]                       # anchor = first template cell (raster order)
    pattern = [(r - ar, c - ac, int(I[r, c])) for r, c in tmpl]

    # ---- silhouettes: 4-connected components of the silhouette colour ------
    seen = np.zeros((h, w), dtype=bool)
    sils = []
    for r0 in range(h):
        for c0 in range(w):
            if I[r0, c0] == dmyc and not seen[r0, c0]:
                stack = [(r0, c0)]; seen[r0, c0] = True; cells = []
                while stack:
                    rr, cc = stack.pop(); cells.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] == dmyc:
                            seen[nr, nc] = True; stack.append((nr, nc))
                sils.append(sorted(cells))
    sils.sort(key=lambda cs: cs[0])

    # ---- paint each silhouette with the template, anchored at its first cell.
    # Selections are the silhouette's own cells, split by the template colour
    # that lands on them (no clipboard, so template colour 0 is handled too).
    for cs in sils:
        sr, sc = cs[0]
        cellset = set(cs)
        buckets = {}
        for dr, dc, col in pattern:
            rr, cc = sr + dr, sc + dc
            if (rr, cc) in cellset and col != dmyc:
                buckets.setdefault(col, []).append((rr, cc))
        for col in sorted(buckets):
            ops.append(int(col)); sels.append(sel_of(sorted(buckets[col])))

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
                        f"num_examples+1 ({num_examples + 1}) for task e76a88a6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e76a88a6"
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
                                f"for task e76a88a6"
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
                    f"Failed to build a complete episode for task e76a88a6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e76a88a6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
