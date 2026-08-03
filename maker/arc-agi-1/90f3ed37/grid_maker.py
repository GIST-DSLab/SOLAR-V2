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
from collections import Counter

import numpy as np

from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------- colors
def sample_colors(num_examples=None) -> dict:
    # generator: cols = interval(0,10,1) minus 1  ->  bgc, fgc = sample(cols, 2)
    cols = [c for c in range(10) if c != 1]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


# ----------------------------------------------------------------------------- generator
def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    while True:
        h = unifint(diff_lb, diff_ub, (8, max(8, max_h)))
        w = unifint(diff_lb, diff_ub, (8, max(8, max_w)))
        pathh = unifint(diff_lb, diff_ub, (1, max(1, h // 4)))
        pathh = unifint(diff_lb, diff_ub, (pathh, max(1, h // 4)))
        Lpatper = unifint(diff_lb, diff_ub, (1, max(1, w // 7)))
        Rpatper = unifint(diff_lb, diff_ub, (1, max(1, w // 7)))
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
        midpatw = randint(0, max(0, w - 2 * Lpatper - 2 * Rpatper))
        if midpatw == 0 or Lpatper == hh == 1:
            midpat = set()
            midpatw = 0
        else:
            midpat = set(sample(totuple(asindices(canvas(-1, (hh, midpatw)))),
                                randint(midpatw, (hh * midpatw))))
        if shift(midpat, (0, 2 * Lpatper - midpatw)).issubset(lpat):
            midpat = set()
            midpatw = 0
        loci = randint(0, h - pathh)
        lplac = shift(lpat, (loci, 0)) | shift(lpat, (loci, Lpatper))
        mplac = shift(midpat, (loci, 2 * Lpatper))
        rplac = shift(rpat, (loci, 2 * Lpatper + midpatw)) | \
            shift(rpat, (loci, 2 * Lpatper + midpatw + Rpatper))
        sp = 2 * Lpatper + midpatw + Rpatper
        for k in range(w // Lpatper + 1):
            lplac |= shift(lpat, (loci, -k * Lpatper))
        for k in range(w // Rpatper + 1):
            rplac |= shift(rpat, (loci, sp + k * Rpatper))
        pat = lplac | mplac | rplac
        patn = shift(pat, (-loci, 0))
        gi = canvas(bgc, (h, w))
        gi = fill(gi, fgc, pat)
        options = interval(0, h - pathh + 1, 1)
        options = difference(options, interval(loci - pathh - 1, loci + 2 * pathh, 1))
        nplacements = unifint(diff_lb, diff_ub, (1, max(1, len(options) // pathh)))
        go = tuple(e for e in gi)
        for k in range(nplacements):
            if len(options) == 0:
                break
            locii = choice(options)
            options = difference(options, interval(locii - pathh - 1, locii + 2 * pathh, 1))
            hoffs = randint(0, max(Rpatper, w - sp - 2))
            cutoffopts = interval(2 * Lpatper + midpatw, 2 * Lpatper + midpatw + hoffs + 1, 1)
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


# ----------------------------------------------------------------------------- ops
def _align_path(P, prows, tcells, h, w):
    """Align the prototype path P (cells relative to its own top row, absolute cols)
    onto a truncated path `tcells`.

    A truncated path is the prototype translated by (dr, dc), dc >= 0, and cut off at
    some column.  Because the prototype's own left flank is clipped by the grid edge,
    only columns >= dc are comparable: there the shifted prototype must reproduce the
    truncated path EXACTLY (no missing cell, no extra cell) up to the path's rightmost
    column.  Everything the shifted prototype puts further right is the continuation.
    Returns (matched_count, dc, continuation_cells) or None."""
    tset = set(tcells)
    maxc = max(c for _, c in tset)
    t_top = min(r for r, _ in tset)
    best = None
    for pr in prows:                      # target's top row may be any prototype row
        dr = t_top - pr
        for dc in range(0, w):
            pred = set()
            for (r, c) in P:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    pred.add((rr, cc))
            pred_seen = {(r, c) for (r, c) in pred if c <= maxc}
            t_seen = {(r, c) for (r, c) in tset if c >= dc}
            if pred_seen and pred_seen == t_seen:
                cont = sorted((r, c) for (r, c) in pred if c > maxc)
                cand = (len(pred_seen), -dc, cont)
                if best is None or cand[:2] > best[:2]:
                    best = cand
    return best


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape

    ops, sels = [], []

    # background = canvas colour the generator paints before drawing paths
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    fg = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    if fg:
        row_cells = {}
        for r, c in fg:
            row_cells.setdefault(r, []).append(c)
        rows_with = sorted(row_cells)

        # --- group rows into paths -------------------------------------------------
        # A path may contain internal empty rows; different paths are separated by a
        # larger vertical gap.  The gap threshold is unknown, so try every threshold
        # and keep the reading of the grid that explains the most foreground cells as
        # one prototype path plus translated copies of it.
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

        best_plan = None   # (n_invalid, has_cont, score, plan)
        for groups in groupings:
            gcells = [sorted((r, c) for r in g for c in row_cells[r]) for g in groups]
            for pi in range(len(groups)):
                top = groups[pi][0]
                P = {(r - top, c) for (r, c) in gcells[pi]}
                prows = sorted({r for r, _ in P})
                score = len(P)
                n_bad = 0
                plan = []
                for ti in range(len(groups)):
                    if ti == pi:
                        continue
                    res = _align_path(P, prows, gcells[ti], h, w)
                    if res is None:
                        n_bad += 1
                        continue
                    matched, _negdc, cont = res
                    score += matched
                    if cont:
                        plan.append((groups[ti][0], cont))
                total_cont = sum(len(c) for _, c in plan)
                cand = (-n_bad, 1 if total_cont else 0, score)
                if best_plan is None or cand > best_plan[0]:
                    best_plan = (cand, plan)

        # --- paint each truncated path's continuation ------------------------------
        if best_plan is not None:
            for _top, cont in sorted(best_plan[1]):
                cells = [(r, c) for (r, c) in cont if I[r, c] == bgc]
                if cells:
                    ops.append(1)                 # Color1
                    sels.append(sel_of(cells))

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
