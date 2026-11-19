"""
ARC Task: 234bbc79 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    import random
    cols = list(range(10))
    bgc = random.choice(cols)
    dotc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc) -> dict:
    import random
    cols = interval(0, 10, 1)
    hub = max(5, min(30, max_h))
    wub = max(6, min(20, max_w))
    fwub = max(6, min(30, max_w))
    while True:
        h = unifint(diff_lb, diff_ub, (5, hub))
        w = unifint(diff_lb, diff_ub, (6, wub))
        # body colours must stay non-zero: ARCLE object ops ignore 0-valued cells
        remcols = [c for c in difference(cols, (bgc, dotc)) if c != 0]
        if len(remcols) == 0:
            continue
        go = canvas(bgc, (h, w))
        ncols = unifint(diff_lb, diff_ub, (1, min(8, len(remcols))))
        ccols = random.sample(remcols, ncols)

        # ---- grow the snake (self avoiding, rightwards) ----
        spi = random.randint(0, h - 1)
        snek = [(spi, 0)]
        go = fill(go, dotc, {(spi, 0)})
        while True:
            previ, prevj = snek[-1]
            if prevj == w - 1:
                if random.choice((True, False, False)):
                    break
            options = []
            if previ < h - 1 and go[previ + 1][prevj] == bgc:
                options.append((previ + 1, prevj))
            if previ > 0 and go[previ - 1][prevj] == bgc:
                options.append((previ - 1, prevj))
            if prevj < w - 1:
                options.append((previ, prevj + 1))
            if len(options) == 0:
                break
            loc = random.choice(options)
            snek.append(loc)
            go = fill(go, dotc, {loc})

        # ---- cut the snake into segments at rightward steps ----
        objs = []
        cobj = []
        for idx, cel in enumerate(snek):
            if len(cobj) > 2 and width(frozenset(cobj)) > 1 and snek[idx - 1] == add(cel, (0, -1)):
                objs.append(cobj)
                cobj = [cel]
            else:
                cobj.append(cel)
        if len(objs) == 0:
            continue
        objs[-1] = objs[-1] + cobj
        nobjs = len(objs)
        if nobjs < 2:
            continue
        ntokeep = unifint(diff_lb, diff_ub, (2, nobjs))
        ntorem = nobjs - ntokeep
        for k in range(ntorem):
            if len(objs) < 3:
                break
            idx = random.randint(0, len(objs) - 2)
            objs = objs[:idx] + [objs[idx] + objs[idx + 1]] + objs[idx + 2:]
        if len(objs) < 2:
            continue

        inobjs = []
        for idx, obj in enumerate(objs):
            col = random.choice(ccols)
            go = fill(go, col, set(obj))
            centerpart = recolor(col, set(obj[1:-1]))
            leftpart = {(dotc if idx > 0 else col, obj[0])}
            rightpart = {(dotc if idx < len(objs) - 1 else col, obj[-1])}
            inobj = centerpart | leftpart | rightpart
            inobjs.append(inobj)

        # ---- scatter the pieces horizontally ----
        spacings = [1 for idx in range(len(inobjs) - 1)]
        fullw = unifint(diff_lb, diff_ub, (w, max(w, fwub)))
        for k in range(fullw - w - len(inobjs) - 1):
            idx = random.randint(0, len(spacings) - 1)
            spacings[idx] += 1
        widths = [width(o) for o in inobjs]
        required = sum(widths) + sum(spacings)
        if required > fwub:
            continue
        fullw = max(fullw, required)
        lspacings = [0] + spacings
        gi = canvas(bgc, (h, fullw))
        ofs = 0
        ok = True
        for i, (lsp, obj) in enumerate(zip(lspacings, inobjs)):
            obj = set(obj)
            if height(obj) > h:
                ok = False
                break
            if i == 0:
                ulc = ulcorner(obj)
            else:
                ulci = random.randint(0, h - height(obj))
                ulcj = ofs + lsp
                ulc = (ulci, ulcj)
            ofs += width(obj) + lsp
            plcd = shift(normalize(obj), ulc)
            gi = paint(gi, plcd)
        if not ok:
            continue
        break

    ins = size(merge(fgpartition(gi)))
    while True:
        go2 = dmirror(dmirror(go)[:-1])
        if len(go2[0]) == 0:
            break
        if size(sfilter(asobject(go2), lambda cij: cij[0] != bgc)) < ins:
            break
        else:
            go = go2
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # ---------- structure discovery (only from I) ----------
    def comps_of(bg):
        seen = [[False] * wi for _ in range(hi)]
        out = []
        for r in range(hi):
            for c in range(wi):
                if I[r][c] != bg and not seen[r][c]:
                    st = [(r, c)]
                    seen[r][c] = True
                    cur = []
                    while st:
                        y, x = st.pop()
                        cur.append((y, x))
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < hi and 0 <= nx < wi and not seen[ny][nx] and I[ny][nx] != bg:
                                seen[ny][nx] = True
                                st.append((ny, nx))
                    out.append(cur)
        return out

    def analyze(bg):
        comps = comps_of(bg)
        if len(comps) < 2:
            return None
        pal = sorted(set(int(v) for v in np.unique(I)) - {bg})
        cands = []
        for col in pal:
            good = True
            for comp in comps:
                k = sum(1 for (r, c) in comp if int(I[r][c]) == col)
                if k < 1 or k > 2:
                    good = False
                    break
            if good:
                cands.append(col)
        if not cands:
            return None
        cands.sort(key=lambda col: (int((I == col).sum()), col))
        dotc = cands[0]
        pieces = []
        for comp in comps:
            body = set(int(I[r][c]) for (r, c) in comp if int(I[r][c]) != dotc)
            if len(body) != 1:
                return None
            dots = sorted([(r, c) for (r, c) in comp if int(I[r][c]) == dotc],
                          key=lambda p: (p[1], p[0]))
            if len(dots) == 0:
                return None
            pieces.append({'cells': sorted(comp), 'body': body.pop(), 'dots': dots})
        pieces.sort(key=lambda p: min(c for (r, c) in p['cells']))
        rngs = [(min(c for r, c in p['cells']), max(c for r, c in p['cells'])) for p in pieces]
        for a, b in zip(rngs, rngs[1:]):
            if a[1] >= b[0]:
                return None
        if rngs[0][0] != 0:
            return None
        return pieces

    bgc = None
    pieces = None
    for cand, _cnt in Counter(I.flatten().tolist()).most_common():
        res = analyze(int(cand))
        if res is not None:
            bgc = int(cand)
            pieces = res
            break
    if pieces is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
        return [34], [[0, 0, hi - 1, wi - 1]]

    # ---------- plan the chain (pure python, measured from I) ----------
    # piece 0 stays where it is; every following piece is translated so that its
    # LEFT dot lands one cell to the right of the chain's current RIGHT dot.
    plan = []
    p0 = pieces[0]
    chain_dot = p0['dots'][-1]                      # right end marker of the head piece
    maxcol = max(c for r, c in p0['cells'])
    plan.append({'piece': p0, 'dr': 0, 'dc': 0, 'final': list(p0['cells'])})
    for p in pieces[1:]:
        left = p['dots'][0]
        dr = chain_dot[0] - left[0]
        dc = chain_dot[1] + 1 - left[1]
        final = [(r + dr, c + dc) for (r, c) in p['cells']]
        plan.append({'piece': p, 'dr': dr, 'dc': dc, 'final': final})
        maxcol = max(maxcol, max(c for r, c in final))
        rd = p['dots'][-1] if len(p['dots']) > 1 else p['dots'][0]
        chain_dot = (rd[0] + dr, rd[1] + dc)
    W = maxcol + 1                                   # width of the reassembled snake

    # ---------- emit ----------
    ops, sels = [], []
    for entry in plan:
        p = entry['piece']
        # the connector markers belong to their own piece body
        if p['dots']:
            ops.append(int(p['body']))
            sels.append(sel_of(p['dots']))
        dr, dc = entry['dr'], entry['dc']
        steps = []
        if dr:
            steps += [(21 if dr > 0 else 20, (1 if dr > 0 else -1, 0))] * abs(dr)
        if dc:
            steps += [(22 if dc > 0 else 23, (0, 1 if dc > 0 else -1))] * abs(dc)
        if steps:
            cur = list(p['cells'])
            first = True
            for opc, (sr, sc) in steps:
                ops.append(opc)
                sels.append(sel_of(cur) if first else sel_of([]))
                first = False
                cur = [(r + sr, c + sc) for (r, c) in cur]
            if bgc != 0:
                hole = sorted(x for x in (set(p['cells']) - set(cur)) if x[1] < W)
                if hole:
                    ops.append(int(bgc))
                    sels.append(sel_of(hole))

    # keep only the columns the reassembled snake occupies
    ops.append(33)
    sels.append([0, 0, hi - 1, W - 1])
    ops.append(34)
    sels.append([0, 0, hi - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 234bbc79"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 234bbc79"
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
                                f"for task 234bbc79"
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
                    f"Failed to build a complete episode for task 234bbc79 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"234bbc79-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
