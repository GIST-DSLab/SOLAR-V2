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

# Reading of the task (shared by the maker and the code-golf entry, so the op
# structure does not change): one row-band holds a complete periodic "path".
# Every other band is a copy of it, shifted right and cut off at some column.
# Each cut-off copy is continued with the prototype's remaining cells in colour 1.
# Each continuation is one Color1 op on "the cells the shifted prototype lands on
# past the cut-off". That set is named from I alone; O is never read.


def _unifint(lb, ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(lb, ub)
    return max(a, min(b, int(round(a + (b - a) * d))))


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, fgc=2, **kw) -> dict:
    H = max(8, min(30, max_h))
    W = max(8, min(30, max_w))
    for _ in range(500):
        h = _unifint(diff_lb, diff_ub, (8, H))
        w = _unifint(diff_lb, diff_ub, (8, W))
        pathh = _unifint(diff_lb, diff_ub, (1, max(1, h // 4)))
        pathh = _unifint(diff_lb, diff_ub, (pathh, max(1, h // 4)))
        Lp = _unifint(diff_lb, diff_ub, (1, w // 7))
        Rp = _unifint(diff_lb, diff_ub, (1, w // 7))
        hh = random.randint(1, pathh)
        Linds = [(i, j) for i in range(hh) for j in range(Lp)]
        Rinds = [(i, j) for i in range(hh) for j in range(Rp)]
        lsd = _unifint(diff_lb, diff_ub, (0, (hh * Lp) // 2))
        rsd = _unifint(diff_lb, diff_ub, (0, (hh * Rp) // 2))
        lpats = random.choice((lsd, hh * Lp - lsd))
        rpats = random.choice((rsd, hh * Rp - rsd))
        lpats = min(max(Lp, lpats), hh * Lp)
        rpats = min(max(Rp, rpats), hh * Rp)
        lpat = set(random.sample(Linds, lpats))
        rpat = set(random.sample(Rinds, rpats))
        midw = random.randint(0, w - 2 * Lp - 2 * Rp)
        if midw == 0 or Lp == hh == 1:
            mid = set()
            midw = 0
        else:
            mi = [(i, j) for i in range(hh) for j in range(midw)]
            mid = set(random.sample(mi, random.randint(midw, hh * midw)))
        sh = lambda S, d: {(i + d[0], j + d[1]) for i, j in S}
        if sh(mid, (0, 2 * Lp - midw)).issubset(lpat):
            mid = set()
            midw = 0
        loci = random.randint(0, h - pathh)
        lpl = sh(lpat, (loci, 0)) | sh(lpat, (loci, Lp))
        mpl = sh(mid, (loci, 2 * Lp))
        rpl = sh(rpat, (loci, 2 * Lp + midw)) | sh(rpat, (loci, 2 * Lp + midw + Rp))
        sp = 2 * Lp + midw + Rp
        for k in range(w // Lp + 1):
            lpl |= sh(lpat, (loci, -k * Lp))
        for k in range(w // Rp + 1):
            rpl |= sh(rpat, (loci, sp + k * Rp))
        pat = lpl | mpl | rpl
        patn = sh(pat, (-loci, 0))
        gi = [[bgc] * w for _ in range(h)]

        def fill(g, c, S):
            for i, j in S:
                if 0 <= i < h and 0 <= j < w:
                    g[i][j] = c

        fill(gi, fgc, pat)
        bad = set(range(loci - pathh - 1, loci + 2 * pathh))
        options = [o for o in range(0, h - pathh + 1) if o not in bad]
        npl = _unifint(diff_lb, diff_ub, (1, max(1, len(options) // pathh)))
        go = [row[:] for row in gi]
        for k in range(npl):
            if not options:
                break
            lo = random.choice(options)
            bad = set(range(lo - pathh - 1, lo + 2 * pathh))
            options = [o for o in options if o not in bad]
            hoffs = random.randint(0, max(Rp, w - sp - 2))
            cuts = list(range(2 * Lp + midw, 2 * Lp + midw + hoffs + 1))[::-1]
            idx = _unifint(diff_lb, diff_ub, (0, len(cuts) - 1))
            cutoff = cuts[idx]
            patnc = {(i, j) for i, j in patn if j <= cutoff}
            fill(go, 1, sh(patn, (lo, hoffs)))
            fill(gi, fgc, sh(patnc, (lo, hoffs)))
            fill(go, fgc, sh(patnc, (lo, hoffs)))
        if any(1 in r for r in go):
            return {"input": gi, "output": go}
    raise RuntimeError("generation failed")


def _bands(rows_with, t):
    groups, cur = [], [rows_with[0]]
    for r in rows_with[1:]:
        if r - cur[-1] - 1 <= t:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r]
    groups.append(cur)
    return groups


def _fit(proto, band, h, w):
    """A cut-off band is the prototype shifted right by dc >= 0 (and vertically
    by dr), cut after its last column.  Returns (matched cells, continuation)."""
    bset = set(band)
    cutoff = max(c for _, c in band)
    ptop = min(r for r, _ in proto); btop = min(r for r, _ in band)
    pbot = max(r for r, _ in proto); bbot = max(r for r, _ in band)
    best = None
    for dr in sorted({btop - ptop, bbot - pbot}):
        for dc in range(0, w):
            S = {(r + dr, c + dc) for r, c in proto}
            if any(not (0 <= r < h) for r, _ in S):
                continue
            if not {r for r, _ in bset} <= {r for r, _ in S}:
                continue
            # left of dc the shifted prototype is unseen; compare the rest
            known_b = {(r, c) for r, c in bset if c >= dc}
            pref = {(r, c) for r, c in S if c <= cutoff}
            if known_b != pref or not pref:
                continue
            cont = sorted((r, c) for r, c in S if cutoff < c < w)
            key = (len(pref), -dc)
            if best is None or key > best[0]:
                best = (key, cont)
    return None if best is None else (best[0][0], best[1])


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # background: the canvas colour shared by every demonstration input
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ei, _eo in examples:
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    fg = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    ops, sels = [], []
    if fg:
        rows_with = sorted({r for r, _ in fg})
        chosen = None
        fallback = None
        for t in range(0, h):
            groups = _bands(rows_with, t)
            if len(groups) < 2:
                break
            gcells = [[(r, c) for (r, c) in fg if r in set(g)] for g in groups]
            best = None
            for pi in range(len(groups)):
                plan, nfit, tmatch = [], 0, 0
                for k in range(len(groups)):
                    if k == pi:
                        continue
                    res = _fit(gcells[pi], gcells[k], h, w)
                    if res is None:
                        continue
                    nfit += 1
                    tmatch += res[0]
                    plan.append((groups[k][0], res[1]))
                # prototype: explains every path, reaches furthest right,
                # and explains the most visible cells
                key = (nfit, max(c for _, c in gcells[pi]), tmatch)
                if best is None or key > best[0]:
                    best = (key, plan)
            ok = best[0][0] == len(groups) - 1
            # paths are at most h//4 tall and stacked at least one path-height
            # apart; keep the coarsest grouping that respects this and in which
            # every path is a cut-off copy of the prototype
            span = max(g[-1] - g[0] + 1 for g in groups)
            gap = min(groups[i + 1][0] - groups[i][-1] - 1 for i in range(len(groups) - 1))
            if ok and gap >= span and span <= max(1, h // 4):
                chosen = best[1]
            if ok and fallback is None:
                fallback = best[1]
        if chosen is None:
            chosen = fallback
        for _top, cont in sorted(chosen or []):
            # continue each cut-off path with the prototype's remaining cells, in colour 1
            cells = [(r, c) for r, c in cont if 0 <= r < h and I[r, c] == bgc]
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
