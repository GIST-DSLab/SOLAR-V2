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
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    dmyc = random.choice(rem)
    pool = [c for c in rem if c != dmyc]
    return {"bgc": bgc, "dmyc": dmyc, "pool": pool}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dmyc=None, pool=None, **kw) -> dict:
    def unifint(lo, hi):
        if hi <= lo:
            return lo
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, lo + int(round((hi - lo) * d))))

    def dn(ij):
        i, j = ij
        return {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}

    if bgc is None:
        bgc = random.randrange(10)
    if dmyc is None:
        dmyc = random.choice([c for c in range(10) if c != bgc])
    if pool is None:
        pool = [c for c in range(10) if c not in (bgc, dmyc)]
    h = unifint(8, max(8, max_h))
    w = unifint(8, max(8, max_w))
    objh = unifint(2, 5)
    objw = unifint(2, 5)
    bounds = {(i, j) for i in range(objh) for j in range(objw)}
    shp = {random.choice(sorted(bounds))}
    nc = unifint(2, len(bounds) - 2)
    for _ in range(nc):
        cand = set()
        for x in shp:
            cand |= dn(x)
        cand = (bounds - shp) & cand
        shp.add(random.choice(sorted(cand)))
    mi = min(i for i, _ in shp); mj = min(j for _, j in shp)
    shp = {(i - mi, j - mj) for i, j in shp}
    oh = max(i for i, _ in shp) + 1
    ow = max(j for _, j in shp) + 1
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    shpp = {(i + loci, j + locj) for i, j in shp}
    numco = unifint(2, min(8, len(pool)))
    colll = random.sample(pool, numco)
    while True:
        cmap = {ij: random.choice(colll) for ij in shp}
        if len(set(cmap.values())) > 1:
            break
    gi = [[bgc] * w for _ in range(h)]
    for (i, j), v in cmap.items():
        gi[i + loci][j + locj] = v
    go = [row[:] for row in gi]
    ub = max(1, int(((h * w) / (oh * ow)) // 2))
    numlocs = unifint(1, ub)
    cnt = 0
    fails = 0
    maxfails = 5 * numlocs
    nb = set()
    for x in shpp:
        nb |= dn(x)
    idns = {(i, j) for i in range(h) for j in range(w)} - shpp - nb
    idns = {(i, j) for i, j in idns if i <= h - oh and j <= w - ow}
    while cnt < numlocs and fails < maxfails:
        if not idns:
            break
        li, lj = random.choice(sorted(idns))
        plcdi = {(i + li, j + lj) for i, j in shp}
        if plcdi <= idns:
            for (i, j), v in cmap.items():
                go[i + li][j + lj] = v
                gi[i + li][j + lj] = dmyc
            cnt += 1
            nb = set()
            for x in plcdi:
                nb |= dn(x)
            idns = idns - plcdi - nb
        else:
            fails += 1
    for i, j in shpp:
        go[i][j] = bgc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    def comps_of(G, bg):
        H, W = G.shape
        seen = [[False] * W for _ in range(H)]
        out = []
        for r in range(H):
            for c in range(W):
                if int(G[r][c]) != bg and not seen[r][c]:
                    st = [(r, c)]
                    seen[r][c] = True
                    cells = []
                    while st:
                        rr, cc = st.pop()
                        cells.append((rr, cc))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W and not seen[nr][nc] and int(G[nr][nc]) != bg:
                                seen[nr][nc] = True
                                st.append((nr, nc))
                    out.append(sorted(cells))
        return out

    def norm(cells):
        mr = min(r for r, _ in cells)
        mc = min(c for _, c in cells)
        return frozenset((r - mr, c - mc) for r, c in cells)

    def parse(G, bg):
        cs = comps_of(G, bg)
        multi = [k for k in cs if len({int(G[r][c]) for r, c in k}) > 1]
        if len(multi) != 1:
            return None
        t = multi[0]
        rest = [k for k in cs if k is not t]
        tn = norm(t)
        if any(len({int(G[r][c]) for r, c in k}) != 1 or norm(k) != tn for k in rest):
            return None
        if len({int(G[k[0][0]][k[0][1]]) for k in rest}) > 1:
            return None
        return t, rest

    # background: the colour the demonstrations are drawn on (shared across the episode)
    cands = [c for c, _ in Counter(I.flatten().tolist()).most_common()]
    if examples:
        good = []
        for c in cands:
            if all(parse(np.asarray(ei, dtype=int), c) is not None for ei, _ in examples):
                good.append(c)
        if good:
            cands = good + [c for c in cands if c not in good]
    bgc, tmpl, copies = None, None, []
    for c in cands:
        p = parse(I, c)
        if p is not None:
            bgc = c
            tmpl, copies = p
            break
    if bgc is None:
        bgc = cands[0]
        cs = comps_of(I, bgc)
        tmpl = max(cs, key=lambda k: len({int(I[r][c]) for r, c in k}))
        copies = [k for k in cs if k is not tmpl and norm(k) == norm(tmpl)]

    tr = min(r for r, _ in tmpl)
    tc = min(c for _, c in tmpl)
    th = max(r for r, _ in tmpl) - tr + 1
    tw = max(c for _, c in tmpl) - tc + 1
    rel = {(r - tr, c - tc): int(I[r][c]) for r, c in tmpl}
    holes = [(i, j) for i in range(th) for j in range(tw) if (i, j) not in rel]
    zero_rel = sorted(k for k, v in rel.items() if v == 0)

    # CopyI reads the input, so the template's bbox gaps must be pure background there
    clip_clean = all(int(I[tr + i][tc + j]) == bgc for i, j in holes)

    G = I.copy()
    ops, sels = [], []

    # 1. lift the coloured template off the canvas (it does not survive in place)
    ops.append(int(bgc)); sels.append(sel_of(sorted(tmpl)))
    for r, c in tmpl:
        G[r][c] = bgc

    # 2. stamp the template onto every blank copy, anchored at the copy's corner
    copied = False
    for k in sorted(copies, key=lambda k: (min(r for r, _ in k), min(c for _, c in k))):
        cr = min(r for r, _ in k)
        cc = min(c for _, c in k)
        # the template's bbox gaps hold bgc in I; Paste writes them only when bgc != 0
        safe = clip_clean and (bgc == 0 or all(int(G[cr + i][cc + j]) == bgc for i, j in holes))
        if safe:
            if not copied:
                # full-rectangle CopyI of the template's bbox from the untouched input
                ops.append(28); sels.append([tr, tc, th - 1, tw - 1])
                copied = True
            ops.append(30); sels.append([cr, cc, 0, 0])
            for (i, j), v in rel.items():
                if v != 0:
                    G[cr + i][cc + j] = v
            if bgc != 0:
                for i, j in holes:
                    G[cr + i][cc + j] = bgc
            # Paste never writes 0 — draw the template's black pixels explicitly
            tgt = [(cr + i, cc + j) for i, j in zero_rel if int(G[cr + i][cc + j]) != 0]
            if tgt:
                ops.append(0); sels.append(sel_of(sorted(tgt)))
                for r, c in tgt:
                    G[r][c] = 0
        else:
            # paste unsafe here: paint each template colour onto the copy's own cells
            for v in sorted(set(rel.values())):
                cells = [(cr + i, cc + j) for (i, j), vv in rel.items()
                         if vv == v and int(G[cr + i][cc + j]) != v]
                if cells:
                    ops.append(int(v)); sels.append(sel_of(sorted(cells)))
                    for r, c in cells:
                        G[r][c] = v

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
