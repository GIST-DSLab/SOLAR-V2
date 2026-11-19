"""
ARC Task: 321b1fc6 (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(cols)                       # canvas background
    rem = [c for c in cols if c != bgc]
    dmyc = random.choice(rem)                       # colour the blank copies are drawn in
    rem2 = [c for c in rem if c != dmyc]
    numco = random.randint(2, min(8, len(rem2)))
    colll = random.sample(rem2, numco)              # palette of the multicoloured template
    return {"bgc": bgc, "dmyc": dmyc, "colll": list(colll)}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, dmyc=None, colll=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(totuple(cols))
    if dmyc is None:
        dmyc = choice(totuple(remove(bgc, cols)))
    if colll is None:
        rem = remove(dmyc, remove(bgc, cols))
        colll = sample(rem, unifint(diff_lb, diff_ub, (2, 8)))
    colll = list(colll)

    hmax = max(8, min(30, int(max_h)))
    wmax = max(8, min(30, int(max_w)))
    h = unifint(diff_lb, diff_ub, (8, hmax))
    w = unifint(diff_lb, diff_ub, (8, wmax))
    objh = unifint(diff_lb, diff_ub, (2, 5))
    objw = unifint(diff_lb, diff_ub, (2, 5))
    bounds = asindices(canvas(0, (objh, objw)))
    shp = {choice(totuple(bounds))}
    nc = unifint(diff_lb, diff_ub, (2, len(bounds) - 2))
    for j in range(nc):
        ij = choice(totuple((bounds - shp) & mapply(dneighbors, shp)))
        shp.add(ij)
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    shpp = shift(shp, (loci, locj))
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
    go = fill(go, bgc, shpp)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    def comps_of(bg):
        seen = [[False] * w for _ in range(h)]
        out = []
        for r in range(h):
            for c in range(w):
                if int(I[r][c]) != bg and not seen[r][c]:
                    st = [(r, c)]
                    seen[r][c] = True
                    cells = []
                    while st:
                        rr, cc = st.pop()
                        cells.append((rr, cc))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and (not seen[nr][nc]) and int(I[nr][nc]) != bg:
                                seen[nr][nc] = True
                                st.append((nr, nc))
                    out.append(sorted(cells))
        return out

    def norm(cells):
        mr = min(r for r, _ in cells)
        mc = min(c for _, c in cells)
        return frozenset((r - mr, c - mc) for r, c in cells)

    def ncols(cells):
        return len({int(I[r][c]) for r, c in cells})

    # --- identify background, the multicoloured template, and the blank copies -------------
    bgc = None
    tmpl = None
    copies = []
    for cand, _ in Counter(I.flatten().tolist()).most_common():
        cs = comps_of(cand)
        multi = [k for k in cs if ncols(k) > 1]
        if len(multi) != 1:
            continue
        t = multi[0]
        rest = [k for k in cs if k is not t]
        tn = norm(t)
        if any(ncols(k) != 1 or norm(k) != tn for k in rest):
            continue
        if len({int(I[k[0][0]][k[0][1]]) for k in rest}) > 1:
            continue
        bgc, tmpl, copies = cand, t, rest
        break
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
        cs = comps_of(bgc)
        tmpl = max(cs, key=ncols)
        tn = norm(tmpl)
        copies = [k for k in cs if k is not tmpl and norm(k) == tn]

    tr = min(r for r, _ in tmpl)
    tc = min(c for _, c in tmpl)
    th = max(r for r, _ in tmpl) - tr + 1
    tw = max(c for _, c in tmpl) - tc + 1
    rel = {(r - tr, c - tc): int(I[r][c]) for r, c in tmpl}
    holes = [(i, j) for i in range(th) for j in range(tw) if (i, j) not in rel]
    zero_rel = sorted(k for k, v in rel.items() if v == 0)
    # the stamp's bounding rectangle may only be copied wholesale if its gaps are pure background
    clip_clean = all(int(I[tr + i][tc + j]) == bgc for i, j in holes)

    G = I.copy()
    ops, sels = [], []
    copied = False

    # --- stamp the coloured template onto every blank copy, one copy at a time -------------
    for cell_list in sorted(copies, key=lambda k: (min(r for r, _ in k), min(c for _, c in k))):
        cr = min(r for r, _ in cell_list)
        cc = min(c for _, c in cell_list)
        # a bbox paste is only safe when nothing foreign sits in the stamp's gaps here
        safe = clip_clean and all(int(G[cr + i][cc + j]) == bgc for i, j in holes)
        if safe:
            if not copied:
                # bbox selection is intentional: we copy the whole stamp rectangle from the input
                ops.append(28)
                sels.append([tr, tc, th - 1, tw - 1])
                copied = True
            ops.append(30)
            sels.append([cr, cc, 0, 0])
            for (i, j), v in rel.items():
                if v != 0:
                    G[cr + i][cc + j] = v
            for i, j in holes:
                hv = int(I[tr + i][tc + j])
                if hv != 0:
                    G[cr + i][cc + j] = hv
            if zero_rel:
                # Paste never writes 0 cells - draw the stamp's black pixels explicitly
                tgt = [(cr + i, cc + j) for i, j in zero_rel if int(G[cr + i][cc + j]) != 0]
                if tgt:
                    ops.append(0)
                    sels.append(sel_of(sorted(tgt)))
                    for r, c in tgt:
                        G[r][c] = 0
        else:
            for v in sorted(set(rel.values())):
                cells = [(cr + i, cc + j) for (i, j), vv in rel.items() if vv == v
                         and int(G[cr + i][cc + j]) != v]
                if cells:
                    ops.append(int(v))
                    sels.append(sel_of(sorted(cells)))
                    for r, c in cells:
                        G[r][c] = v

    # --- the original template is wiped out ------------------------------------------------
    tcells = [(r, c) for r, c in sorted(tmpl) if int(G[r][c]) != bgc]
    if tcells:
        ops.append(int(bgc))
        sels.append(sel_of(tcells))
        for r, c in tcells:
            G[r][c] = bgc

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
                        f"num_examples+1 ({num_examples + 1}) for task 321b1fc6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 321b1fc6"
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
                                f"for task 321b1fc6"
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
                    f"Failed to build a complete episode for task 321b1fc6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"321b1fc6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
