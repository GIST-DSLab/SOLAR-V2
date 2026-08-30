"""
ARC Task: 7e0986d6 (RE-ARC) — LLM-generated grid_maker
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
    # The rule is purely structural (component size + bounding box), so only the
    # colour ROLES need to stay stable across the episode: the background, the
    # palette the rectangles are drawn from, and the palette the speckles use.
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    random.shuffle(rem)
    nsq = random.randint(1, 5)
    sqcols = rem[:nsq]          # rectangle colours
    noisecols = rem[nsq:]       # speckle colours (always disjoint from sqcols)
    return {"bgc": bgc, "sqcols": sqcols, "noisecols": noisecols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqcols, noisecols) -> dict:
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))

    sqcols = list(sqcols)
    noisecols = list(noisecols)
    nsqcols = unifint(diff_lb, diff_ub, (1, len(sqcols)))
    sqc = sample(tuple(sqcols), nsqcols)
    nnoisecols = unifint(diff_lb, diff_ub, (1, len(noisecols)))
    noisec = sample(tuple(noisecols), nnoisecols)

    numsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    succ = 0
    tr = 0
    maxtr = 5 * numsq
    go = canvas(bgc, (h, w))
    inds = asindices(go)
    while tr < maxtr and succ < numsq:
        tr += 1
        oh = randint(2, 7)
        ow = randint(2, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if sq.issubset(inds):
            succ += 1
            inds = (inds - sq) - outbox(sq)
            col = choice(list(sqc))
            go = fill(go, col, sq)

    # ---- helpers used to guarantee the instance is actually derivable ----
    def comps_of(g):
        hh = len(g)
        ww = len(g[0])
        seen = [[False] * ww for _ in range(hh)]
        out = []
        for r in range(hh):
            for c in range(ww):
                if seen[r][c]:
                    continue
                col = g[r][c]
                seen[r][c] = True
                st = [(r, c)]
                cells = []
                while st:
                    a, b = st.pop()
                    cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < hh and 0 <= nb < ww and not seen[na][nb] and g[na][nb] == col:
                            seen[na][nb] = True
                            st.append((na, nb))
                out.append((col, cells))
        return out

    def rule(g):
        cs = comps_of(g)
        big = max(cs, key=lambda t: len(t[1]))
        if big[0] != bgc:
            return None  # background must be recoverable as the largest region
        res = [list(row) for row in g]
        for col, cells in cs:
            if col != bgc and len(cells) < 3:
                for (a, b) in cells:
                    res[a][b] = bgc
        for col, cells in cs:
            if col == bgc or len(cells) < 3:
                continue
            r0 = min(a for a, b in cells)
            r1 = max(a for a, b in cells)
            c0 = min(b for a, b in cells)
            c1 = max(b for a, b in cells)
            for a in range(r0, r1 + 1):
                for b in range(c0, c1 + 1):
                    res[a][b] = col
        return tuple(tuple(row) for row in res)

    namt = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 9)))
    gi = tuple(e for e in go)
    for attempt in range(30):
        amt = max(1, namt // (1 + attempt))
        gg = tuple(e for e in go)
        cands = asindices(gg)
        for k in range(amt):
            if len(cands) == 0:
                break
            loc = choice(totuple(cands))
            col = gg[loc[0]][loc[1]]
            torem = neighbors(loc) & ofcolor(gg, col)
            cands = cands - torem
            noisecol = choice(list(noisec))
            gg = fill(gg, noisecol, {loc})
        if rule(gg) == go:
            gi = gg
            break

    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # ---- everything below is measured from I only ----
    # 4-connected, single-coloured components of the INPUT
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            st = [(r, c)]
            cells = []
            while st:
                a, b = st.pop()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] == col:
                        seen[na, nb] = True
                        st.append((na, nb))
            comps.append((col, cells))

    # background = colour of the single large surrounding region
    counts = Counter(I.flatten().tolist())
    bgc = max(comps, key=lambda t: (len(t[1]), counts[t[0]]))[0]

    # speckles = tiny components (size < 3); blocks = everything else
    speckles = [(col, cells) for col, cells in comps if col != bgc and len(cells) < 3]
    blocks = [(col, cells) for col, cells in comps if col != bgc and len(cells) >= 3]

    ops, sels = [], []
    cur = I.copy()

    # 1) wipe the stray speckles away, one op per speckle colour
    by_color = {}
    for col, cells in speckles:
        by_color.setdefault(col, []).extend(cells)
    for col in sorted(by_color):
        cells = sorted(by_color[col])
        ops.append(int(bgc))
        sels.append(sel_of(cells))
        for (a, b) in cells:
            cur[a, b] = bgc

    # 2) restore every remaining block to its full bounding rectangle
    #    (selection is exactly that whole rectangle, background included)
    for col, cells in sorted(blocks, key=lambda t: (min(a for a, b in t[1]), min(b for a, b in t[1]))):
        r0 = min(a for a, b in cells)
        r1 = max(a for a, b in cells)
        c0 = min(b for a, b in cells)
        c1 = max(b for a, b in cells)
        rect = [(a, b) for a in range(r0, r1 + 1) for b in range(c0, c1 + 1)]
        if any(cur[a, b] != col for a, b in rect):
            ops.append(int(col))
            sels.append(sel_of(rect))
            for (a, b) in rect:
                cur[a, b] = col

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
                        f"num_examples+1 ({num_examples + 1}) for task 7e0986d6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7e0986d6"
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
                                f"for task 7e0986d6"
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
                    f"Failed to build a complete episode for task 7e0986d6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7e0986d6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
