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
import random
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # bgc = canvas colour, dotc = the connector-marker colour (fixed role for whole episode).
    # Segment body colours carry no rule information (only their position/pattern matters),
    # so they stay random per instance.  All foreground colours are kept non-zero so that
    # ARCLE object ops (Move) never treat a real object cell as "empty".
    bgc = random.choice(list(range(10)))
    dotc = random.choice([c for c in range(1, 10) if c != bgc])
    return {"bgc": bgc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc) -> dict:
    hub = max(5, min(30, max_h))
    wub = max(6, min(20, max_w - 4))
    while True:
        h = unifint(diff_lb, diff_ub, (5, hub))
        w = unifint(diff_lb, diff_ub, (6, wub))
        remcols = [c for c in range(1, 10) if c != bgc and c != dotc]
        if len(remcols) == 0:
            continue
        go = canvas(bgc, (h, w))
        ncols = unifint(diff_lb, diff_ub, (1, min(8, len(remcols))))
        ccols = sample(remcols, ncols)
        spi = randint(0, h - 1)
        snek = [(spi, 0)]
        go = fill(go, dotc, {(spi, 0)})
        while True:
            previ, prevj = snek[-1]
            if prevj == w - 1:
                if choice((True, False, False)):
                    break
            options = []
            if previ < h - 1:
                if go[previ + 1][prevj] == bgc:
                    options.append((previ + 1, prevj))
            if previ > 0:
                if go[previ - 1][prevj] == bgc:
                    options.append((previ - 1, prevj))
            if prevj < w - 1:
                options.append((previ, prevj + 1))
            if len(options) == 0:
                break
            loc = choice(options)
            snek.append(loc)
            go = fill(go, dotc, {loc})
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
        # keep enough room so that every segment (plus >=1 blank column between them)
        # really fits inside the input canvas
        maxkeep = max_w - 1 - w
        if maxkeep < 2:
            continue
        ntokeep = unifint(diff_lb, diff_ub, (2, min(nobjs, maxkeep)))
        ntorem = nobjs - ntokeep
        for k in range(ntorem):
            idx = randint(0, len(objs) - 2)
            objs = objs[:idx] + [objs[idx] + objs[idx + 1]] + objs[idx + 2:]
        inobjs = []
        for idx, obj in enumerate(objs):
            col = choice(ccols)
            go = fill(go, col, set(obj))
            centerpart = recolor(col, set(obj[1:-1]))
            leftpart = {(dotc if idx > 0 else col, obj[0])}
            rightpart = {(dotc if idx < len(objs) - 1 else col, obj[-1])}
            inobj = centerpart | leftpart | rightpart
            inobjs.append(inobj)
        spacings = [1 for idx in range(len(inobjs) - 1)]
        fullw = unifint(diff_lb, diff_ub, (w + len(inobjs) + 1, max_w))
        for k in range(fullw - w - len(inobjs) - 1):
            idx = randint(0, len(spacings) - 1)
            spacings[idx] += 1
        lspacings = [0] + spacings
        gi = canvas(bgc, (h, fullw))
        ofs = 0
        ok = True
        for i, (lsp, obj) in enumerate(zip(lspacings, inobjs)):
            obj = set(obj)
            if i == 0:
                ulc = ulcorner(obj)
            else:
                if h - height(obj) < 0:
                    ok = False
                    break
                ulci = randint(0, h - height(obj))
                ulcj = ofs + lsp
                ulc = (ulci, ulcj)
            ofs += width(obj) + lsp
            plcd = shift(normalize(obj), ulc)
            gi = paint(gi, plcd)
        if not ok:
            continue
        if ofs > fullw:
            continue
        # background must stay the dominant colour so it is detectable from the input alone
        if colorcount(gi, bgc) * 2 <= h * fullw:
            continue
        return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- connected components (4-conn) of non-background cells = the snake segments ---
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r0 in range(hi):
        for c0 in range(wi):
            if I[r0, c0] != bgc and not seen[r0, c0]:
                stack = [(r0, c0)]
                seen[r0, c0] = True
                cur = []
                while stack:
                    r, c = stack.pop()
                    cur.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] != bgc:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(sorted(cur))
    comps.sort(key=lambda cl: min(c for _, c in cl))
    n = len(comps)

    def build(dotc, strict):
        if n < 2:
            return None
        infos = []
        for cid, comp in enumerate(comps):
            dots = [p for p in comp if I[p[0], p[1]] == dotc]
            body = [p for p in comp if I[p[0], p[1]] != dotc]
            if not body:
                return None
            bcols = set(int(I[r, c]) for r, c in body)
            if len(bcols) != 1:
                return None
            need = 1 if (cid == 0 or cid == n - 1) else 2
            if len(dots) != need:
                return None
            entry = min(dots, key=lambda p: p[1]) if cid > 0 else None
            exit_ = max(dots, key=lambda p: p[1]) if cid < n - 1 else None
            if entry is not None and exit_ is not None and entry[1] >= exit_[1]:
                return None
            # the entry marker is the segment's leftmost cell, the exit marker its rightmost
            if entry is not None and entry[1] != min(c for _, c in comp):
                return None
            if exit_ is not None and exit_[1] != max(c for _, c in comp):
                return None
            infos.append({'cells': comp, 'dots': dots, 'col': bcols.pop(),
                          'entry': entry, 'exit': exit_})
        # chain: segment i's entry marker lands one column right of segment i-1's exit marker
        offs = [(0, 0)]
        for i in range(1, n):
            pr, pc = infos[i - 1]['exit']
            odr, odc = offs[i - 1]
            tr, tc = pr + odr, pc + odc + 1
            er, ec = infos[i]['entry']
            offs.append((tr - er, tc - ec))
        W = 0
        for i, inf in enumerate(infos):
            dr, dc = offs[i]
            for r, c in inf['cells']:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < hi and 0 <= nc < wi):
                    return None
                if nc + 1 > W:
                    W = nc + 1
        if W >= wi:
            return None
        if strict:
            pred = np.full((hi, W), bgc, dtype=int)
            for i, inf in enumerate(infos):
                dr, dc = offs[i]
                dset = set(inf['dots'])
                for r, c in inf['cells']:
                    pred[r + dr, c + dc] = inf['col'] if (r, c) in dset else int(I[r, c])
            if pred.shape != O.shape or not np.array_equal(pred, O):
                return None
        return infos, offs, W

    # marker colour: present in the input, gone from the output (every marker gets welded)
    pal_i = set(int(v) for v in np.unique(I)) - {bgc}
    pal_o = set(int(v) for v in np.unique(O))
    counts = Counter(I.flatten().tolist())
    cands = sorted(pal_i - pal_o, key=lambda c: counts[c])
    cands += sorted(pal_i & pal_o, key=lambda c: counts[c])

    res = None
    for c in cands:
        res = build(c, True)
        if res is not None:
            break
    if res is None:
        for c in cands:
            res = build(c, False)
            if res is not None:
                break
    if res is None:
        return [34], [[0, 0, ho - 1, wo - 1]]

    infos, offs, W = res

    ops, sels = [], []

    # 1. slide every segment (left to right) onto the tail of the assembled snake
    for i in range(1, n):
        cells = infos[i]['cells']
        dr, dc = offs[i]
        steps = []
        if dr != 0:
            steps += [20 if dr < 0 else 21] * abs(dr)
        if dc != 0:
            steps += [23 if dc < 0 else 22] * abs(dc)
        if not steps:
            continue
        for k, op in enumerate(steps):
            ops.append(op)
            # first step grabs the object; the rest carry an empty selection so ARCLE
            # keeps the same object grabbed and restores everything it glides over
            sels.append(sel_of(cells) if k == 0 else sel_of([]))
        dst = set((r + dr, c + dc) for r, c in cells)
        hole = sorted(set(cells) - dst)
        if bgc != 0 and hole:
            ops.append(int(bgc))
            sels.append(sel_of(hole))

    # 2. weld the joints: each segment's marker cells take that segment's own colour
    for i in range(n):
        dr, dc = offs[i]
        pts = sorted((r + dr, c + dc) for r, c in infos[i]['dots'])
        ops.append(int(infos[i]['col']))
        sels.append(sel_of(pts))

    # 3. crop the canvas down to the assembled snake's width
    #    (bbox = the exact full rectangle rows 0..hi-1, cols 0..W-1)
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
