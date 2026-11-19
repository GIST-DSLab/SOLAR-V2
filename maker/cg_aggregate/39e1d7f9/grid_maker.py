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
from random import randint, choice, sample
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of
from dsl import *
from utils import unifint


def sample_colors(num_examples=None) -> dict:
    bgc, linc, dotc = random.sample(range(10), 3)
    return {"bgc": bgc, "linc": linc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, linc=None, dotc=None, **kw) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or linc is None or dotc is None:
        bgc, linc, dotc = sample(cols, 3)
    hmax = max(5, min(10, (max_h + 1) // 2))
    wmax = max(5, min(10, (max_w + 1) // 2))
    h = unifint(diff_lb, diff_ub, (5, hmax))
    w = unifint(diff_lb, diff_ub, (5, wmax))
    remcols = difference(cols, (bgc, linc, dotc))
    gi = canvas(bgc, (h, w))
    loci = randint(1, h - 2)
    locj = randint(1, w - 2)
    if h == 5:
        loci = choice((1, h - 2))
    if w == 5:
        locj = choice((1, w - 2))
    npix = unifint(diff_lb, diff_ub, (1, 8))
    ncols = unifint(diff_lb, diff_ub, (1, 7))
    ccols = sample(remcols, ncols)
    candsss = neighbors((loci, locj))
    pixs = {(loci, locj)}
    for k in range(npix):
        pixs.add(choice(totuple((mapply(dneighbors, pixs) & candsss) - pixs)))
    pixs = totuple(remove((loci, locj), pixs))
    obj = {(choice(ccols), ij) for ij in pixs}
    gi = fill(gi, dotc, {(loci, locj)})
    gi = paint(gi, obj)
    go = tuple(e for e in gi)
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // (2 * len(pixs) + 1)))
    succ = 0
    tr = 0
    maxtr = 6 * noccs
    inds = ofcolor(gi, bgc) - mapply(dneighbors, neighbors((loci, locj)))
    objn = shift(obj, (-loci, -locj))
    triedandfailed = set()
    while (tr < maxtr and succ < noccs) or succ == 0:
        lopcands = totuple(inds - triedandfailed)
        if len(lopcands) == 0:
            break
        tr += 1
        loci, locj = choice(lopcands)
        plcd = shift(objn, (loci, locj))
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            inds = inds - (plcdi | {(loci, locj)})
            succ += 1
            gi = fill(gi, dotc, {(loci, locj)})
            go = fill(go, dotc, {(loci, locj)})
            go = paint(go, plcd)
        else:
            triedandfailed.add((loci, locj))
    hfac = unifint(diff_lb, diff_ub, (1, max(1, (max_h - h + 1) // h)))
    wfac = unifint(diff_lb, diff_ub, (1, max(1, (max_w - w + 1) // w)))
    fullh = hfac * h + h - 1
    fullw = wfac * w + w - 1
    gi2 = canvas(linc, (fullh, fullw))
    go2 = canvas(linc, (fullh, fullw))
    bd = asindices(canvas(-1, (hfac, wfac)))
    for a in range(h):
        for b in range(w):
            gi2 = fill(gi2, gi[a][b], shift(bd, (a * (hfac + 1), b * (wfac + 1))))
    for a in range(h):
        for b in range(w):
            go2 = fill(go2, go[a][b], shift(bd, (a * (hfac + 1), b * (wfac + 1))))
    return {'input': gi2, 'output': go2}


def derive_operations(I, O=None, examples=None):
    """Golf reading: the template is the 3x3 block-neighbourhood around the marker
    that sits inside the multi-colour shape. Every other marker block gets that same
    neighbourhood stamped around it, block by block (one Color op per template colour
    per stamp). Background blocks of the neighbourhood are never pasted, so stamps
    never clobber each other and no repair pass is needed. Everything is read from I."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    submit = [0, 0, hi - 1, wi - 1]
    ops, sels = [], []

    # separator colour: the only colour that fills a whole pixel column
    # (a background column always crosses separator rows)
    sep_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    if not sep_cols:
        return [34], [submit]
    linc = int(I[0, sep_cols[0]])
    sep_rows = [r for r in range(hi) if all(int(v) == linc for v in I[r])]
    if not sep_rows:
        return [34], [submit]
    hfac, wfac = sep_rows[0], sep_cols[0]
    hs, ws = hfac + 1, wfac + 1
    h, w = (hi + 1) // hs, (wi + 1) // ws

    g = [[int(I[a * hs, b * ws]) for b in range(w)] for a in range(h)]
    bgc = Counter(v for row in g for v in row).most_common(1)[0][0]

    # components of non-background blocks (4-connected)
    seen = [[False] * w for _ in range(h)]
    comps = []
    for a in range(h):
        for b in range(w):
            if g[a][b] != bgc and not seen[a][b]:
                st = [(a, b)]; seen[a][b] = True; cur = []
                while st:
                    r, c = st.pop(); cur.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and g[nr][nc] != bgc:
                            seen[nr][nc] = True; st.append((nr, nc))
                comps.append(cur)
    if len(comps) < 2:
        return [34], [submit]
    ti = max(range(len(comps)), key=lambda k: (len({g[r][c] for r, c in comps[k]}), len(comps[k])))
    template = comps[ti]
    others = [cell for k, cm in enumerate(comps) if k != ti for cell in cm]
    dotc = Counter(g[r][c] for r, c in others).most_common(1)[0][0]
    centers = [(r, c) for r, c in template if g[r][c] == dotc]
    if not centers:
        return [34], [submit]
    cr, cc = centers[0]

    # template = the coloured blocks of the 3x3 block-neighbourhood of the centre marker
    tmpl = {}
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr, dc) == (0, 0):
                continue
            r, c = cr + dr, cc + dc
            if 0 <= r < h and 0 <= c < w and g[r][c] != bgc:
                tmpl[(dr, dc)] = g[r][c]

    markers = sorted((r, c) for r, c in others if g[r][c] == dotc)

    def block_pixels(a, b):
        return [(a * hs + i, b * ws + j) for i in range(hfac) for j in range(wfac)]

    cur = [row[:] for row in g]
    for (mr, mc) in markers:                       # one stamp at a time
        by_col = {}
        for (dr, dc), col in sorted(tmpl.items(),
                                    key=lambda t: (abs(t[0][0]) + abs(t[0][1]), t[0])):
            r, c = mr + dr, mc + dc
            if 0 <= r < h and 0 <= c < w and cur[r][c] != col:
                by_col.setdefault(col, []).append((r, c))
        for col, blocks in by_col.items():         # one Color op per template colour
            cells = [p for (r, c) in blocks for p in block_pixels(r, c)]
            ops.append(col); sels.append(sel_of(cells))
            for r, c in blocks:
                cur[r][c] = col
    ops.append(34); sels.append(submit)
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
