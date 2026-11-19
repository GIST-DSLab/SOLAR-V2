"""
ARC Task: 39e1d7f9 (RE-ARC) — LLM-generated grid_maker
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
    bgc, linc, dotc = random.sample(range(10), 3)
    return {"bgc": bgc, "linc": linc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, dotc) -> dict:
    def unif(a, b):
        b = max(a, b)
        lo = a + int(round(diff_lb * (b - a)))
        hi = a + int(round(diff_ub * (b - a)))
        return random.randint(min(lo, hi), max(lo, hi))

    D4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    hub = max(5, min(10, (max_h + 1) // 2))
    wub = max(5, min(10, (max_w + 1) // 2))
    h = unif(5, hub)
    w = unif(5, wub)
    remcols = [c for c in range(10) if c not in (bgc, linc, dotc)]
    gi = [[bgc] * w for _ in range(h)]
    loci = random.randint(1, h - 2)
    locj = random.randint(1, w - 2)
    if h == 5:
        loci = random.choice((1, h - 2))
    if w == 5:
        locj = random.choice((1, w - 2))
    npix = unif(1, 8)
    ncols = unif(1, 7)
    ccols = random.sample(remcols, ncols)
    cands = {(loci + a, locj + b) for a in (-1, 0, 1) for b in (-1, 0, 1)} - {(loci, locj)}
    pixs = {(loci, locj)}
    for _ in range(npix):
        grow = {(r + a, c + b) for (r, c) in pixs for a, b in D4}
        opts = sorted((grow & cands) - pixs)
        pixs.add(random.choice(opts))
    pixs = sorted(pixs - {(loci, locj)})
    obj = [(random.choice(ccols), ij) for ij in pixs]
    gi[loci][locj] = dotc
    for col, (r, c) in obj:
        gi[r][c] = col
    go = [row[:] for row in gi]
    noccs = unif(1, max(1, (h * w) // (2 * len(pixs) + 1)))
    succ = tr = 0
    maxtr = 6 * noccs
    nb = {(loci + a, locj + b) for a in (-1, 0, 1) for b in (-1, 0, 1)}
    excl = {(r + a, c + b) for (r, c) in nb for a, b in D4} | nb
    inds = {(r, c) for r in range(h) for c in range(w) if gi[r][c] == bgc} - excl
    objn = [(col, (r - loci, c - locj)) for col, (r, c) in obj]
    failed = set()
    while (tr < maxtr and succ < noccs) or succ == 0:
        lc = sorted(inds - failed)
        if not lc:
            break
        tr += 1
        a, b = random.choice(lc)
        plcd = [(col, (r + a, c + b)) for col, (r, c) in objn]
        pi = {ij for _, ij in plcd}
        if pi.issubset(inds):
            inds = inds - (pi | {(a, b)})
            succ += 1
            gi[a][b] = dotc
            go[a][b] = dotc
            for col, (r, c) in plcd:
                go[r][c] = col
        else:
            failed.add((a, b))
    if succ == 0:
        raise ValueError("no marker placed")
    hfac = unif(1, max(1, (max_h - h + 1) // h))
    wfac = unif(1, max(1, (max_w - w + 1) // w))
    fh, fw = hfac * h + h - 1, wfac * w + w - 1

    def up(g):
        out = [[linc] * fw for _ in range(fh)]
        for a in range(h):
            for b in range(w):
                for i in range(hfac):
                    for j in range(wfac):
                        out[a * (hfac + 1) + i][b * (wfac + 1) + j] = g[a][b]
        return out

    return {"input": up(gi), "output": up(go)}


def derive_operations(I, O=None, examples=None):
    """Compress the block grid, take the multi-colour object as a stamp anchored on its
    marker block, and paint that object (its own blocks only, not the background
    rectangle around it) onto every lone marker block. Everything is measured from I."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    full = [0, 0, hi - 1, wi - 1]

    sep_rows = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    sep_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    if not sep_rows or not sep_cols:
        return [34], [full]
    hfac, wfac = sep_rows[0], sep_cols[0]
    hs, ws = hfac + 1, wfac + 1
    h, w = (hi + 1) // hs, (wi + 1) // ws
    g = [[int(I[a * hs, b * ws]) for b in range(w)] for a in range(h)]
    bgc = Counter(v for row in g for v in row).most_common(1)[0][0]

    seen = set()
    comps = []
    for a in range(h):
        for b in range(w):
            if g[a][b] != bgc and (a, b) not in seen:
                st = [(a, b)]; seen.add((a, b)); cur = []
                while st:
                    r, c = st.pop(); cur.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        n = (r + dr, c + dc)
                        if 0 <= n[0] < h and 0 <= n[1] < w and n not in seen and g[n[0]][n[1]] != bgc:
                            seen.add(n); st.append(n)
                comps.append(cur)
    if len(comps) < 2:
        return [34], [full]
    tmpl = max(comps, key=lambda cm: (len({g[r][c] for r, c in cm}), len(cm)))
    others = [cell for cm in comps if cm is not tmpl for cell in cm]
    dotc = Counter(g[r][c] for r, c in others).most_common(1)[0][0]
    anchor = [cell for cell in tmpl if g[cell[0]][cell[1]] == dotc]
    if not anchor:
        return [34], [full]
    ar, ac = min(anchor)
    # every lone marker block individually (adjacent markers are still separate anchors)
    markers = sorted(cell for cell in others if g[cell[0]][cell[1]] == dotc)

    def block(a, b):
        return [(a * hs + i, b * ws + j) for i in range(hfac) for j in range(wfac)]

    r0 = min(r for r, _ in tmpl)
    c0 = min(c for _, c in tmpl)
    rel = [((r - ar, c - ac), g[r][c]) for (r, c) in tmpl]
    zero_rel = [d for d, col in rel if col == 0 and col != dotc]
    has_paint = any(col != 0 and col != dotc for _, col in rel)

    if has_paint:
        # copy exactly the template object's blocks (marker block included as anchor)
        cells = [p for (r, c) in tmpl for p in block(r, c)]
        ops.append(28); sels.append(sel_of(cells))
    for (mr, mc) in markers:
        if has_paint:
            # paste so the template's marker block lands on this marker block
            pr, pc = (mr + (r0 - ar)) * hs, (mc + (c0 - ac)) * ws
            ops.append(30); sels.append([pr, pc, 0, 0])
        if zero_rel:
            # colour-0 blocks of the object are invisible to Copy/Paste: paint them here
            zc = [p for (dr, dc) in zero_rel for p in block(mr + dr, mc + dc)]
            ops.append(0); sels.append(sel_of(zc))
    ops.append(34); sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task 39e1d7f9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 39e1d7f9"
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
                                f"for task 39e1d7f9"
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
                    f"Failed to build a complete episode for task 39e1d7f9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"39e1d7f9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
