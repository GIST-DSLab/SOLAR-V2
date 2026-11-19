"""
ARC Task: 90f3ed37 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, fgc=2, **kwargs) -> dict:
    def shift(p, d):
        return {(i + d[0], j + d[1]) for i, j in p}

    def rect(hh, ww):
        return [(i, j) for i in range(hh) for j in range(ww)]

    while True:
        h = _unifint(diff_lb, diff_ub, (8, max(8, max_h)))
        w = _unifint(diff_lb, diff_ub, (8, max(8, max_w)))
        pathh = _unifint(diff_lb, diff_ub, (1, max(1, h // 4)))
        pathh = _unifint(diff_lb, diff_ub, (pathh, max(1, h // 4)))
        Lpatper = _unifint(diff_lb, diff_ub, (1, w // 7))
        Rpatper = _unifint(diff_lb, diff_ub, (1, w // 7))
        hh = random.randint(1, pathh)
        Linds = rect(hh, Lpatper)
        Rinds = rect(hh, Rpatper)
        lpatsd = _unifint(diff_lb, diff_ub, (0, (hh * Lpatper) // 2))
        rpatsd = _unifint(diff_lb, diff_ub, (0, (hh * Rpatper) // 2))
        lpats = random.choice((lpatsd, hh * Lpatper - lpatsd))
        rpats = random.choice((rpatsd, hh * Rpatper - rpatsd))
        lpats = min(max(Lpatper, lpats), hh * Lpatper)
        rpats = min(max(Rpatper, rpats), hh * Rpatper)
        lpat = set(random.sample(Linds, lpats))
        rpat = set(random.sample(Rinds, rpats))
        midpatw = random.randint(0, w - 2 * Lpatper - 2 * Rpatper)
        if midpatw == 0 or Lpatper == hh == 1:
            midpat = set()
            midpatw = 0
        else:
            mi = rect(hh, midpatw)
            midpat = set(random.sample(mi, random.randint(midpatw, hh * midpatw)))
        if shift(midpat, (0, 2 * Lpatper - midpatw)).issubset(lpat):
            midpat = set()
            midpatw = 0
        loci = random.randint(0, h - pathh)
        lplac = shift(lpat, (loci, 0)) | shift(lpat, (loci, Lpatper))
        mplac = shift(midpat, (loci, 2 * Lpatper))
        rplac = shift(rpat, (loci, 2 * Lpatper + midpatw)) | shift(rpat, (loci, 2 * Lpatper + midpatw + Rpatper))
        sp = 2 * Lpatper + midpatw + Rpatper
        for k in range(w // Lpatper + 1):
            lplac |= shift(lpat, (loci, -k * Lpatper))
        for k in range(w // Rpatper + 1):
            rplac |= shift(rpat, (loci, sp + k * Rpatper))
        pat = lplac | mplac | rplac
        patn = shift(pat, (-loci, 0))
        gi = [[bgc] * w for _ in range(h)]
        go = [[bgc] * w for _ in range(h)]

        def fill(g, col, cells):
            for i, j in cells:
                if 0 <= i < h and 0 <= j < w:
                    g[i][j] = col

        fill(gi, fgc, pat)
        fill(go, fgc, pat)
        options = set(range(0, h - pathh + 1)) - set(range(loci - pathh - 1, loci + 2 * pathh))
        nplacements = _unifint(diff_lb, diff_ub, (1, max(1, len(options) // pathh)))
        for _ in range(nplacements):
            if len(options) == 0:
                break
            locii = random.choice(sorted(options))
            options -= set(range(locii - pathh - 1, locii + 2 * pathh))
            hoffs = random.randint(0, max(Rpatper, w - sp - 2))
            cutoffopts = list(range(2 * Lpatper + midpatw, 2 * Lpatper + midpatw + hoffs + 1))[::-1]
            idx = _unifint(diff_lb, diff_ub, (0, len(cutoffopts) - 1))
            cutoff = cutoffopts[idx]
            patnc = {(i, j) for i, j in patn if j <= cutoff}
            fill(go, 1, shift(patn, (locii, hoffs)))
            fill(gi, fgc, shift(patnc, (locii, hoffs)))
            fill(go, fgc, shift(patnc, (locii, hoffs)))
        if any(1 in row for row in go):
            break
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def _align(P, group, h, w):
    """Find (dr, dc) such that `group` is the prototype path P shifted by (dr, dc)
    and cut off at its rightmost column.  Returns (matched, dr, dc) or None."""
    gset = set(group)
    gmax = max(c for _, c in group)
    grows = sorted({r for r, _ in group})
    prows = sorted({r for r, _ in P})
    best = None
    for dr in sorted({gr - pr for gr in grows for pr in prows}):
        if not all(0 <= r + dr < h for r in prows):
            continue
        for dc in range(0, w):
            if dc > gmax:
                break
            sh = {(r + dr, c + dc) for r, c in P if c + dc <= gmax}
            part = {(r, c) for r, c in gset if c >= dc}
            if sh == part and part:
                cand = (len(part), dr, dc)
                if best is None or cand[0] > best[0]:
                    best = cand
                break  # smallest dc for this dr is the tightest alignment
    return best


def _period(P, w):
    """Smallest horizontal period of the prototype's right-hand (repeating) end."""
    cols = {}
    for r, c in P:
        cols.setdefault(c, set()).add(r)
    for p in range(1, max(2, w // 2 + 1)):
        if all(cols.get(c, set()) == cols.get(c + p, set()) for c in range(w - 2 * p, w - p)):
            return p
    return 1


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # background = canvas colour; it is the same on every input of the episode
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            cnt.update(np.asarray(ex[0], dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    fg = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    if fg:
        row_cells = {}
        for r, c in fg:
            row_cells.setdefault(r, []).append(c)
        rows_with = sorted(row_cells)

        # paths = horizontal bands of rows; a path may have internal blank rows,
        # distinct paths are separated by a wider gap -> try every gap threshold
        groupings, seen = [], set()
        for t in range(0, h + 1):
            groups, cur = [], [rows_with[0]]
            for r in rows_with[1:]:
                if r - cur[-1] - 1 <= t:
                    cur.append(r)
                else:
                    groups.append(cur)
                    cur = [r]
            groups.append(cur)
            if len(groups) < 2:
                continue
            key = tuple(tuple(g) for g in groups)
            if key not in seen:
                seen.add(key)
                groupings.append(groups)

        best = None
        for groups in groupings:
            gcells = [sorted((r, c) for r in g for c in row_cells[r]) for g in groups]
            for pi in range(len(groups)):
                P = set(gcells[pi])          # candidate complete path (prototype)
                score, n_bad, plan = len(P), 0, []
                for ti in range(len(groups)):
                    if ti == pi:
                        continue
                    res = _align(P, gcells[ti], h, w)
                    if res is None:
                        n_bad += 1
                        continue
                    matched, dr, dc = res
                    score += matched
                    gmax = max(c for _, c in gcells[ti])
                    plan.append((groups[ti][0], dr, dc, gmax))
                has_cont = any(
                    any(gmax < c + dc < w for _, c in P) for _, dr, dc, gmax in plan)
                cand = (-n_bad, 1 if has_cont else 0, score)
                if best is None or cand > best[0]:
                    best = (cand, P, plan)

        if best is not None:
            _, P, plan = best
            p = _period(P, w)
            # extend each cut-off path to the right, one period of its repeating
            # end at a time (the stamp is the prototype path shifted onto it)
            for _top, dr, dc, gmax in sorted(plan):
                stamp = [(r + dr, c + dc) for r, c in P if gmax < c + dc < w]
                c0 = gmax + 1
                while c0 < w:
                    block = [(r, c) for r, c in stamp
                             if c0 <= c < c0 + p and I[r, c] == bgc]
                    if block:
                        ops.append(1)            # Color1 on one period of the extension
                        sels.append(sel_of(block))
                    c0 += p

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 90f3ed37"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 90f3ed37"
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
                                f"for task 90f3ed37"
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
                    f"Failed to build a complete episode for task 90f3ed37 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"90f3ed37-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
