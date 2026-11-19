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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def _dn(cells):
    out = set()
    for r, c in cells:
        out |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
    return out


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    dmyc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "dmyc": dmyc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dmyc=None, **kw) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if dmyc is None or dmyc == bgc:
        dmyc = random.choice([c for c in cols if c != bgc])
    h = _unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = _unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    objh = _unifint(diff_lb, diff_ub, (2, 5))
    objw = _unifint(diff_lb, diff_ub, (2, 5))
    bounds = {(i, j) for i in range(objh) for j in range(objw)}
    shp = {random.choice(sorted(bounds))}
    nc = _unifint(diff_lb, diff_ub, (2, len(bounds) - 2))
    for _ in range(nc):
        cand = sorted((bounds - shp) & _dn(shp))
        shp.add(random.choice(cand))
    mr = min(r for r, _ in shp); mc = min(c for _, c in shp)
    shp = {(r - mr, c - mc) for r, c in shp}
    remcols = [c for c in cols if c not in (bgc, dmyc)]
    oh = max(r for r, _ in shp) + 1
    ow = max(c for _, c in shp) + 1
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    shpp = {(r + loci, c + locj) for r, c in shp}
    numco = _unifint(diff_lb, diff_ub, (2, 8))
    colll = random.sample(remcols, numco)
    while True:
        shppc = {ij: random.choice(colll) for ij in shpp}
        if len(set(shppc.values())) > 1:
            break
    shppcn = {(r - loci, c - locj): v for (r, c), v in shppc.items()}
    gi = np.full((h, w), bgc, dtype=int)
    for (r, c), v in shppc.items():
        gi[r, c] = v
    go = gi.copy()
    ub = max(1, int(((h * w) / (oh * ow)) // 2))
    numlocs = _unifint(diff_lb, diff_ub, (1, ub))
    cnt = 0
    fails = 0
    maxfails = 5 * numlocs
    allidx = {(i, j) for i in range(h) for j in range(w)}
    idns = (allidx - shpp) - _dn(shpp)
    idns = {(i, j) for i, j in idns if i <= h - oh and j <= w - ow}
    while cnt < numlocs and fails < maxfails:
        if not idns:
            break
        loc = random.choice(sorted(idns))
        plcd = {(r + loc[0], c + loc[1]): v for (r, c), v in shppcn.items()}
        plcdi = set(plcd)
        if plcdi <= idns:
            for (r, c), v in plcd.items():
                go[r, c] = v
                gi[r, c] = dmyc
            cnt += 1
            idns = (idns - plcdi) - _dn(plcdi)
        else:
            fails += 1
    if cnt == 0:
        raise ValueError("no copies placed")
    for r, c in shpp:
        go[r, c] = bgc
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    def comps_of(bg):
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for r in range(h):
            for c in range(w):
                if I[r, c] != bg and not seen[r, c]:
                    st = [(r, c)]
                    seen[r, c] = True
                    cells = []
                    while st:
                        rr, cc = st.pop()
                        cells.append((rr, cc))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] != bg:
                                seen[nr, nc] = True
                                st.append((nr, nc))
                    out.append(sorted(cells))
        return out

    def norm(cells):
        mr = min(r for r, _ in cells)
        mc = min(c for _, c in cells)
        return frozenset((r - mr, c - mc) for r, c in cells)

    def ncols(cells):
        return len({int(I[r, c]) for r, c in cells})

    # background / multicoloured template / single-colour blank copies, read from I
    bgc, tmpl, copies = None, None, []
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
        if len({int(I[k[0][0], k[0][1]]) for k in rest}) > 1:
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
    rel = {(r - tr, c - tc): int(I[r, c]) for r, c in tmpl}
    holes = [(i, j) for i in range(th) for j in range(tw) if (i, j) not in rel]
    has_zero = any(v == 0 for v in rel.values())
    dmyc = int(I[copies[0][0][0], copies[0][0][1]]) if copies else None
    # the template's bounding rectangle can be stamped wholesale only if its gaps are background
    clip_clean = all(int(I[tr + i, tc + j]) == bgc for i, j in holes)

    G = I.copy()
    ops, sels = [], []
    copied = False

    def apply_paste(cr, cc):
        for i in range(th):
            for j in range(tw):
                v = int(I[tr + i, tc + j])
                if v != 0:
                    G[cr + i, cc + j] = v

    for cells in sorted(copies, key=lambda k: (min(r for r, _ in k), min(c for _, c in k))):
        cr = min(r for r, _ in cells)
        cc = min(c for _, c in cells)
        # with a non-zero background the clip's gaps overwrite, so they must land on background
        safe = clip_clean and (bgc == 0 or all(int(G[cr + i, cc + j]) == bgc for i, j in holes))
        if safe:
            if not copied:
                # whole template rectangle: its gaps are pure background
                ops.append(28); sels.append([tr, tc, th - 1, tw - 1])
                copied = True
            ops.append(30); sels.append([cr, cc, 0, 0])   # stamp onto this blank copy
            apply_paste(cr, cc)
        else:
            # gaps would hit a neighbour: paint this copy colour by colour
            for v in sorted(set(rel.values())):
                tgt = [(cr + i, cc + j) for (i, j), vv in rel.items() if vv == v
                       and int(G[cr + i, cc + j]) != v]
                if tgt:
                    ops.append(v); sels.append(sel_of(tgt))
                    for r, c in tgt:
                        G[r, c] = v

    # Paste never writes black: every cell still in the blank colour is a black stamp pixel
    if has_zero and dmyc is not None:
        left = [(r, c) for r in range(h) for c in range(w) if int(G[r, c]) == dmyc]
        if left:
            ops.append(0); sels.append(sel_of(left))
            for r, c in left:
                G[r, c] = 0

    # wipe the original template
    tcells = [(r, c) for r, c in tmpl if int(G[r, c]) != bgc]
    if tcells:
        ops.append(int(bgc)); sels.append(sel_of(tcells))
        for r, c in tcells:
            G[r, c] = bgc

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
