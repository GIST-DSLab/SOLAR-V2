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
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(b, max(a, round(a + (b - a) * d)))


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def _solve(I, bgc):
    """Return list of (group_top, fill_cells): for every truncated path, the
    cells of the prototype path (shifted onto it) that are still background."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    rows = [r for r in range(h) if any(I[r, c] != bgc for c in range(w))]
    if not rows:
        return []
    # paths = maximal runs of rows holding foreground
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if r == cur[-1] + 1:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r]
    groups.append(cur)
    if len(groups) < 2:
        return []
    gcells = [{(r, c) for r in g for c in range(w) if I[r, c] != bgc} for g in groups]

    def match(P, G):
        ptop = min(r for r, _ in P); pbot = max(r for r, _ in P)
        gtop = min(r for r, _ in G); gbot = max(r for r, _ in G)
        rmost = max(c for _, c in G)
        best = None
        for dr in range(gtop - pbot, gbot - ptop + 1):
            if ptop + dr < 0 or pbot + dr >= h:
                continue
            for dc in range(0, w):
                S = {(r + dr, c + dc) for r, c in P if c + dc < w}
                inter = len(S & G)
                if inter == 0:
                    continue
                fill = [(r, c) for r, c in S if I[r, c] == bgc]
                # the shifted prototype must agree with the path up to its cut-off
                if any(c <= rmost for _, c in fill):
                    continue
                key = (inter, -dc)
                if best is None or key > best[0]:
                    best = (key, sorted(fill))
        return best

    best_plan = None
    for pi in range(len(groups)):
        P = gcells[pi]
        plan, total, ok = [], 0, True
        for ti in range(len(groups)):
            if ti == pi:
                continue
            m = match(P, gcells[ti])
            if m is None:
                ok = False
                break
            total += m[0][0]
            if m[1]:
                plan.append((groups[ti][0], m[1]))
        if not ok:
            continue
        key = (1 if plan else 0, total)
        if best_plan is None or key > best_plan[0]:
            best_plan = (key, plan)
    return sorted(best_plan[1]) if best_plan else []


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, fgc=None) -> dict:
    if bgc is None or fgc is None:
        cols = [c for c in range(10) if c != 1]
        bgc, fgc = random.sample(cols, 2)
    max_h = max(8, min(30, max_h)); max_w = max(8, min(30, max_w))
    for _ in range(500):
        h = _unifint(diff_lb, diff_ub, (8, max_h))
        w = _unifint(diff_lb, diff_ub, (8, max_w))
        pathh = _unifint(diff_lb, diff_ub, (1, max(1, h // 4)))
        pathh = _unifint(diff_lb, diff_ub, (pathh, max(1, h // 4)))
        Lpatper = _unifint(diff_lb, diff_ub, (1, w // 7))
        Rpatper = _unifint(diff_lb, diff_ub, (1, w // 7))
        hh = random.randint(1, pathh)
        Linds = [(i, j) for i in range(hh) for j in range(Lpatper)]
        Rinds = [(i, j) for i in range(hh) for j in range(Rpatper)]
        lpatsd = _unifint(diff_lb, diff_ub, (0, (hh * Lpatper) // 2))
        rpatsd = _unifint(diff_lb, diff_ub, (0, (hh * Rpatper) // 2))
        lpats = random.choice((lpatsd, hh * Lpatper - lpatsd))
        rpats = random.choice((rpatsd, hh * Rpatper - rpatsd))
        lpats = min(max(Lpatper, lpats), hh * Lpatper)
        rpats = min(max(Rpatper, rpats), hh * Rpatper)
        lpat = set(random.sample(Linds, lpats))
        rpat = set(random.sample(Rinds, rpats))
        midpatw = random.randint(0, max(0, w - 2 * Lpatper - 2 * Rpatper))
        if midpatw == 0 or Lpatper == hh == 1:
            midpat = set(); midpatw = 0
        else:
            mi = [(i, j) for i in range(hh) for j in range(midpatw)]
            midpat = set(random.sample(mi, random.randint(midpatw, hh * midpatw)))
        sh = lambda p, d: {(i + d[0], j + d[1]) for i, j in p}
        if sh(midpat, (0, 2 * Lpatper - midpatw)).issubset(lpat):
            midpat = set(); midpatw = 0
        loci = random.randint(0, h - pathh)
        lplac = sh(lpat, (loci, 0)) | sh(lpat, (loci, Lpatper))
        mplac = sh(midpat, (loci, 2 * Lpatper))
        rplac = sh(rpat, (loci, 2 * Lpatper + midpatw)) | sh(rpat, (loci, 2 * Lpatper + midpatw + Rpatper))
        sp = 2 * Lpatper + midpatw + Rpatper
        for k in range(w // Lpatper + 1):
            lplac |= sh(lpat, (loci, -k * Lpatper))
        for k in range(w // Rpatper + 1):
            rplac |= sh(rpat, (loci, sp + k * Rpatper))
        pat = lplac | mplac | rplac
        patn = sh(pat, (-loci, 0))
        gi = np.full((h, w), bgc, dtype=int)

        def fill(g, col, cells):
            for i, j in cells:
                if 0 <= i < h and 0 <= j < w:
                    g[i, j] = col
        fill(gi, fgc, pat)
        rm = lambda opts, lo, hi: [o for o in opts if not (lo <= o < hi)]
        options = rm(list(range(0, h - pathh + 1)), loci - pathh - 1, loci + 2 * pathh)
        nplacements = _unifint(diff_lb, diff_ub, (1, max(1, len(options) // pathh)))
        go = gi.copy()
        for k in range(nplacements):
            if len(options) == 0:
                break
            locii = random.choice(options)
            options = rm(options, locii - pathh - 1, locii + 2 * pathh)
            hoffs = random.randint(0, max(Rpatper, w - sp - 2))
            cutoffopts = list(range(2 * Lpatper + midpatw, 2 * Lpatper + midpatw + hoffs + 1))[::-1]
            idx = _unifint(diff_lb, diff_ub, (0, len(cutoffopts) - 1))
            cutoff = cutoffopts[idx]
            patnc = {(i, j) for i, j in patn if j <= cutoff}
            fill(go, 1, sh(patn, (locii, hoffs)))
            fill(gi, fgc, sh(patnc, (locii, hoffs)))
            fill(go, fgc, sh(patnc, (locii, hoffs)))
        if 1 not in go:
            continue
        # keep only instances whose rule is recoverable from the input alone
        pred = gi.copy()
        for _, cells in _solve(gi, bgc):
            for r, c in cells:
                pred[r, c] = 1
        if not np.array_equal(pred, go):
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    # background: the canvas colour, read from this input and the demonstration inputs
    cnt = Counter(I.flatten().tolist())
    for ex in (examples or []):
        cnt.update(np.asarray(ex[0], dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    ops, sels = [], []
    # one stroke per truncated path: paint the continuation of the prototype path
    # (prototype shifted onto the truncated path, beyond its cut-off) with colour 1
    for _top, cells in _solve(I, bgc):
        ops.append(1)
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
