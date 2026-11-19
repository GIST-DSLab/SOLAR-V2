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
from dsl import *
from utils import *
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    while True:
        h = unifint(diff_lb, diff_ub, (8, max_h))
        w = unifint(diff_lb, diff_ub, (8, max_w))
        pathh = unifint(diff_lb, diff_ub, (1, max(1, h//4)))
        pathh = unifint(diff_lb, diff_ub, (pathh, max(1, h//4)))
        Lpatper = unifint(diff_lb, diff_ub, (1, w//7))
        Rpatper = unifint(diff_lb, diff_ub, (1, w//7))
        hh = randint(1, pathh)
        Linds = asindices(canvas(-1, (hh, Lpatper)))
        Rinds = asindices(canvas(-1, (hh, Rpatper)))
        lpatsd = unifint(diff_lb, diff_ub, (0, (hh * Lpatper) // 2))
        rpatsd = unifint(diff_lb, diff_ub, (0, (hh * Rpatper) // 2))
        lpats = choice((lpatsd, hh * Lpatper - lpatsd))
        rpats = choice((rpatsd, hh * Rpatper - rpatsd))
        lpats = min(max(Lpatper, lpats), hh * Lpatper)
        rpats = min(max(Rpatper, rpats), hh * Rpatper)
        lpat = set(sample(totuple(Linds), lpats))
        rpat = set(sample(totuple(Rinds), rpats))
        midpatw = randint(0, w-2*Lpatper-2*Rpatper)
        if midpatw == 0 or Lpatper == hh == 1:
            midpat = set()
            midpatw = 0
        else:
            midpat = set(sample(totuple(asindices(canvas(-1, (hh, midpatw)))), randint(midpatw, (hh * midpatw))))
        if shift(midpat, (0, 2*Lpatper-midpatw)).issubset(lpat):
            midpat = set()
            midpatw = 0
        loci = randint(0, h - pathh)
        lplac = shift(lpat, (loci, 0)) | shift(lpat, (loci, Lpatper))
        mplac = shift(midpat, (loci, 2*Lpatper))
        rplac = shift(rpat, (loci, 2*Lpatper+midpatw)) | shift(rpat, (loci, 2*Lpatper+midpatw+Rpatper))
        sp = 2*Lpatper+midpatw+Rpatper
        for k in range(w//Lpatper+1):
            lplac |= shift(lpat, (loci, -k*Lpatper))
        for k in range(w//Rpatper+1):
            rplac |= shift(rpat, (loci, sp+k*Rpatper))
        pat = lplac | mplac | rplac
        patn = shift(pat, (-loci, 0))
        gi = canvas(bgc, (h, w))
        gi = fill(gi, fgc, pat)
        options = interval(0, h - pathh + 1, 1)
        options = difference(options, interval(loci-pathh-1, loci+2*pathh, 1))
        nplacements = unifint(diff_lb, diff_ub, (1, max(1, len(options) // pathh)))
        go = tuple(e for e in gi)
        for k in range(nplacements):
            if len(options) == 0:
                break
            locii = choice(options)
            options = difference(options, interval(locii-pathh-1, locii+2*pathh, 1))
            hoffs = randint(0, max(Rpatper, w-sp-2))
            cutoffopts = interval(2*Lpatper+midpatw, 2*Lpatper+midpatw+hoffs+1, 1)
            cutoffopts = cutoffopts[::-1]
            idx = unifint(diff_lb, diff_ub, (0, len(cutoffopts) - 1))
            cutoff = cutoffopts[idx]
            patnc = sfilter(patn, lambda ij: ij[1] <= cutoff)
            go = fill(go, 1, shift(patn, (locii, hoffs)))
            gi = fill(gi, fgc, shift(patnc, (locii, hoffs)))
            go = fill(go, fgc, shift(patnc, (locii, hoffs)))
        if 1 in palette(go):
            break
    return {'input': gi, 'output': go}


def _bg_of(I, examples=None):
    # background = the colour of the blank rows separating the paths
    G = np.asarray(I, dtype=int)
    cnt = Counter()
    for row in G:
        if len(set(row.tolist())) == 1:
            cnt[int(row[0])] += 1
    if cnt:
        return cnt.most_common(1)[0][0]
    return Counter(G.flatten().tolist()).most_common(1)[0][0]


def _fit(P, T, h, w):
    """Best placement of the full path P over the truncated path T.
    The full path's left part is periodic (period p), so columns of T left of
    the shift must repeat that period.  Returns the continuation cells (the
    shifted full path right of T's last column) or None."""
    prows = [r for r, _ in P]
    trows = [r for r, _ in T]
    Tset = set(T)
    tmax = max(c for _, c in T)
    pcol = {}
    for r, c in P:
        pcol.setdefault(c, set()).add(r)
    periods = [p for p in range(1, max(1, w // 7) + 1)
               if all(pcol.get(c, set()) == pcol.get(c + p, set()) for c in range(p))]
    best = None
    for dr in range(max(trows) - max(prows), min(trows) - min(prows) + 1):
        if min(prows) + dr < 0 or max(prows) + dr >= h:
            continue
        tcol = {}
        for r, c in T:
            tcol.setdefault(c, set()).add(r - dr)
        for dc in range(0, w):
            S = {(r + dr, c + dc) for r, c in P if 0 <= c + dc < w}
            Tv = {(r, c) for r, c in Tset if c >= dc}
            Sv = {(r, c) for r, c in S if c <= tmax}
            if Tv != Sv or not Tv:
                continue
            if dc > 0 and not any(
                    all(tcol.get(c, set()) == pcol.get((c - dc) % p, set()) for c in range(dc))
                    for p in periods):
                continue
            key = (len(Tv), -dc)
            if best is None or key > best[0]:
                cont = sorted((r, c) for r, c in S if c > tmax)
                best = (key, cont)
    return None if best is None else best[1]


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    bgc = _bg_of(I, examples)
    fg = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    if fg:
        row_cells = {}
        for r, c in fg:
            row_cells.setdefault(r, []).append(c)
        rows_with = sorted(row_cells)
        # Paths are horizontal bands; a band may have internal blank rows,
        # so try every gap threshold for splitting rows into bands.
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
            key = tuple(map(tuple, groups))
            if key not in seen:
                seen.add(key)
                groupings.append(groups)
        best = None
        for groups in groupings:
            gcells = [[(r, c) for r in g for c in row_cells[r]] for g in groups]
            # the full path = the band that explains every other band as a
            # shifted, right-truncated copy of itself
            for pi in range(len(groups)):
                P = gcells[pi]
                n_bad, plan = 0, []
                for ti in range(len(groups)):
                    if ti == pi:
                        continue
                    cont = _fit(P, gcells[ti], h, w)
                    if cont is None:
                        n_bad += 1
                    elif cont:
                        plan.append((groups[ti][0], cont))
                cand = (-n_bad, len(groups), len(P), 1 if plan else 0)
                if best is None or cand > best[0]:
                    best = (cand, plan)
        if best is not None:
            # each truncated copy of the path is continued to the right edge in colour 1
            for _top, cont in sorted(best[1]):
                cells = [(r, c) for r, c in cont if I[r, c] == bgc]
                if cells:
                    ops.append(1)
                    sels.append(sel_of(cells))
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
