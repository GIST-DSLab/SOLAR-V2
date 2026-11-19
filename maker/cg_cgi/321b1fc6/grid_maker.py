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

# Reading compared with the code-golf entry: it also treats the multicoloured template
# as ONE unit. It shifts that unit onto every blank copy, lined up on the copy's
# first/top-left cell, and wipes the template's original spot. The maker reads the
# task the same way, so the operations below follow that reading: stamp the template
# copy by copy, then cover the source.


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    dmyc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "dmyc": dmyc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dmyc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round(lb * (b - a))
        hi = a + round(ub * (b - a))
        return random.randint(min(lo, hi), max(lo, hi))

    def dnb(c):
        r, cc = c
        return {(r - 1, cc), (r + 1, cc), (r, cc - 1), (r, cc + 1)}

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if dmyc is None:
        dmyc = random.choice([c for c in cols if c != bgc])
    h = unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    objh = unifint(diff_lb, diff_ub, (2, 5))
    objw = unifint(diff_lb, diff_ub, (2, 5))
    bounds = {(i, j) for i in range(objh) for j in range(objw)}
    shp = {random.choice(sorted(bounds))}
    nc = unifint(diff_lb, diff_ub, (2, len(bounds) - 2))
    for _ in range(nc):
        cand = set()
        for c in shp:
            cand |= dnb(c)
        cand = (bounds - shp) & cand
        shp.add(random.choice(sorted(cand)))
    mr = min(i for i, _ in shp); mc = min(j for _, j in shp)
    shp = {(i - mr, j - mc) for i, j in shp}
    remcols = [c for c in cols if c not in (bgc, dmyc)]
    oh = max(i for i, _ in shp) + 1
    ow = max(j for _, j in shp) + 1
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    shpp = {(i + loci, j + locj) for i, j in shp}
    numco = unifint(diff_lb, diff_ub, (2, 8))
    colll = random.sample(remcols, numco)
    while True:
        shppc = {ij: random.choice(colll) for ij in shpp}
        if len(set(shppc.values())) > 1:
            break
    shppcn = {(i - loci, j - locj): v for (i, j), v in shppc.items()}
    gi = [[bgc] * w for _ in range(h)]
    for (i, j), v in shppc.items():
        gi[i][j] = v
    go = [row[:] for row in gi]
    ub = max(1, int(((h * w) / (oh * ow)) // 2))
    numlocs = unifint(diff_lb, diff_ub, (1, ub))
    cnt = 0
    fails = 0
    maxfails = 5 * numlocs
    allidx = {(i, j) for i in range(h) for j in range(w)}
    nb = set()
    for c in shpp:
        nb |= dnb(c)
    idns = (allidx - shpp) - nb
    idns = {(i, j) for i, j in idns if i <= h - oh and j <= w - ow}
    while cnt < numlocs and fails < maxfails:
        if not idns:
            break
        loc = random.choice(sorted(idns))
        plcd = {(i + loc[0], j + loc[1]): v for (i, j), v in shppcn.items()}
        if set(plcd).issubset(idns):
            for (i, j), v in plcd.items():
                go[i][j] = v
                gi[i][j] = dmyc
            cnt += 1
            nb = set()
            for c in plcd:
                nb |= dnb(c)
            idns = (idns - set(plcd)) - nb
        else:
            fails += 1
    if cnt == 0:
        raise ValueError("no copies placed")
    for i, j in shpp:
        go[i][j] = bgc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    def comps_of(bg):
        seen = np.zeros((h, w), bool)
        out = []
        for r in range(h):
            for c in range(w):
                if I[r, c] != bg and not seen[r, c]:
                    st = [(r, c)]; seen[r, c] = True; cells = []
                    while st:
                        a, b = st.pop(); cells.append((a, b))
                        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            na, nb = a + da, b + db
                            if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] != bg:
                                seen[na, nb] = True; st.append((na, nb))
                    out.append(sorted(cells))
        return out

    def norm(cells):
        mr = min(r for r, _ in cells); mc = min(c for _, c in cells)
        return frozenset((r - mr, c - mc) for r, c in cells)

    def ncols(cells):
        return len({int(I[r, c]) for r, c in cells})

    # background: the colour the demonstration inputs are painted on (shared across the episode)
    bg_votes = Counter()
    if examples:
        for ei, _ in examples:
            ei = np.asarray(ei, dtype=int)
            bg_votes[Counter(ei.flatten().tolist()).most_common(1)[0][0]] += 1
    order = [c for c, _ in bg_votes.most_common()] + \
            [c for c, _ in Counter(I.flatten().tolist()).most_common()]
    bgc = tmpl = None
    copies = []
    for cand in dict.fromkeys(order):
        cs = comps_of(cand)
        multi = [k for k in cs if ncols(k) > 1]
        if len(multi) != 1:
            continue
        t = multi[0]
        rest = [k for k in cs if k is not t]
        tn = norm(t)
        if any(ncols(k) != 1 or norm(k) != tn for k in rest):
            continue
        bgc, tmpl, copies = cand, t, rest
        break
    if bgc is None:
        bgc = order[0]
        cs = comps_of(bgc)
        tmpl = max(cs, key=ncols)
        tn = norm(tmpl)
        copies = [k for k in cs if k is not tmpl and norm(k) == tn]

    tr = min(r for r, _ in tmpl); tc = min(c for _, c in tmpl)
    th = max(r for r, _ in tmpl) - tr + 1
    tw = max(c for _, c in tmpl) - tc + 1
    rel = {(r - tr, c - tc): int(I[r, c]) for r, c in tmpl}
    holes = [(i, j) for i in range(th) for j in range(tw) if (i, j) not in rel]
    zero_rel = sorted(k for k, v in rel.items() if v == 0)
    clip_clean = all(I[tr + i, tc + j] == bgc for i, j in holes)

    G = I.copy()
    ops, sels = [], []
    copied = False
    # every blank copy receives the coloured template, aligned on its own top-left corner
    for cl in sorted(copies, key=lambda k: (min(r for r, _ in k), min(c for _, c in k))):
        cr = min(r for r, _ in cl); cc = min(c for _, c in cl)
        safe = clip_clean and all(G[cr + i, cc + j] == bgc for i, j in holes)
        if safe:
            if not copied:
                # bbox intentional: the whole template rectangle (its gaps are pure background)
                ops.append(28); sels.append([tr, tc, th - 1, tw - 1])
                copied = True
            ops.append(30); sels.append([cr, cc, 0, 0])
            for i in range(th):
                for j in range(tw):
                    v = int(I[tr + i, tc + j])
                    if v != 0:
                        G[cr + i, cc + j] = v
            if zero_rel:
                # Paste cannot write colour 0: draw the template's 0-cells on this copy
                tgt = [(cr + i, cc + j) for i, j in zero_rel if G[cr + i, cc + j] != 0]
                if tgt:
                    ops.append(0); sels.append(sel_of(sorted(tgt)))
                    for r, c in tgt:
                        G[r, c] = 0
        else:
            # a foreign cell sits in the stamp's gaps here: paint the template colour by colour
            for v in sorted(set(rel.values())):
                cells = [(cr + i, cc + j) for (i, j), vv in rel.items()
                         if vv == v and G[cr + i, cc + j] != v]
                if cells:
                    ops.append(v); sels.append(sel_of(sorted(cells)))
                    for r, c in cells:
                        G[r, c] = v

    # the original coloured template is removed
    tcells = [(r, c) for r, c in tmpl if G[r, c] != bgc]
    if tcells:
        ops.append(int(bgc)); sels.append(sel_of(sorted(tcells)))
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
