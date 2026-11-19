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

# Reading (golf entry and maker agree, so the op structure is unchanged): ONE
# row-band carries the complete periodic path; every other band is that same
# path translated (vertically + rightward) and cut off at some column.  The golf
# entry fixes one prototype band and slides it over each other band keeping the
# best-overlapping slide; the maker searches the prototype among all bands.  Both
# end in the same action: continue each cut-off band with the slid prototype's
# cells past its cut, in colour 1 -> one Color1 per cut-off band, whose selection
# is "the cells the slid prototype lands on right of that band's last cell".


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
    sh = lambda S, d: {(i + d[0], j + d[1]) for i, j in S}
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
            mid, midw = set(), 0
        else:
            mi = [(i, j) for i in range(hh) for j in range(midw)]
            mid = set(random.sample(mi, random.randint(midw, hh * midw)))
        if sh(mid, (0, 2 * Lp - midw)).issubset(lpat):
            mid, midw = set(), 0
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
            cutoff = cuts[_unifint(diff_lb, diff_ub, (0, len(cuts) - 1))]
            patnc = {(i, j) for i, j in patn if j <= cutoff}
            fill(go, 1, sh(patn, (lo, hoffs)))
            fill(gi, fgc, sh(patnc, (lo, hoffs)))
            fill(go, fgc, sh(patnc, (lo, hoffs)))
        if any(1 in r for r in go):
            return {"input": gi, "output": go}
    raise RuntimeError("generation failed")


def _group_rows(rows, t):
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if r - cur[-1] - 1 <= t:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r]
    groups.append(cur)
    return groups


def _slide(proto, band, h, w):
    """Slide the prototype band over a cut-off band (vertical align + rightward
    shift dc).  Columns left of dc hold the path's unseen periodic tail, so only
    columns >= dc are compared; up to the band's last cell they must agree exactly.
    Returns (overlap, continuation cells) for the best slide, or None."""
    bset = set(band)
    last = max(c for _, c in band)
    brows = {r for r, _ in band}
    ptop, pbot = min(r for r, _ in proto), max(r for r, _ in proto)
    btop, bbot = min(brows), max(brows)
    best = None
    for dr in sorted({btop - ptop, bbot - pbot}):
        if ptop + dr < 0 or pbot + dr >= h:
            continue
        for dc in range(0, w):
            S = {(r + dr, c + dc) for r, c in proto}
            if not brows <= {r for r, _ in S}:
                continue
            seen = {(r, c) for r, c in bset if c >= dc}
            pref = {(r, c) for r, c in S if c <= last}
            if not pref or seen != pref:
                continue
            key = (len(pref), -dc)
            if best is None or key > best[0]:
                best = (key, sorted((r, c) for r, c in S if last < c < w))
    return None if best is None else (best[0][0], best[1])


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # background = the canvas colour every demonstration input is drawn on
    cnt = Counter(I.flatten().tolist())
    for ei, _eo in (examples or []):
        cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    fg = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    ops, sels = [], []
    plan = None
    if fg:
        rows = sorted({r for r, _ in fg})
        fallback = None
        # a path may contain empty rows; paths are separated by >= one path-height
        for t in range(0, h):
            groups = _group_rows(rows, t)
            if len(groups) < 2:
                break
            gsets = [set(g) for g in groups]
            cells = [[(r, c) for r, c in fg if r in s] for s in gsets]
            best = None
            for pi in range(len(groups)):
                cand, nfit, ov = [], 0, 0
                for k in range(len(groups)):
                    if k == pi:
                        continue
                    res = _slide(cells[pi], cells[k], h, w)
                    if res is None:
                        continue
                    nfit += 1
                    ov += res[0]
                    cand.append((groups[k][0], res[1]))
                # prototype: explains every other band, reaches furthest right
                key = (nfit, max(c for _, c in cells[pi]), ov)
                if best is None or key > best[0]:
                    best = (key, cand)
            if best[0][0] != len(groups) - 1:
                continue
            span = max(g[-1] - g[0] + 1 for g in groups)
            gap = min(groups[i + 1][0] - groups[i][-1] - 1 for i in range(len(groups) - 1))
            if fallback is None:
                fallback = best[1]
            if gap >= span and span <= max(1, h // 4):
                plan = best[1]
        if plan is None:
            plan = fallback
    for _top, cont in sorted(plan or []):
        # continue this cut-off path with the slid prototype's remaining cells
        sel = [(r, c) for r, c in cont if I[r, c] == bgc]
        if sel:
            ops.append(1)
            sels.append(sel_of(sel))
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
