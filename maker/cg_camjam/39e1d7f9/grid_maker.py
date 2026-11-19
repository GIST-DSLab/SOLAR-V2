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
from dsl import *
from utils import unifint


def sample_colors(num_examples=None) -> dict:
    bgc, linc, dotc = random.sample(range(10), 3)
    return {"bgc": bgc, "linc": linc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, dotc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or linc is None or dotc is None:
        bgc, linc, dotc = sample(cols, 3)
    h = unifint(diff_lb, diff_ub, (5, max(5, min(10, (max_h + 1) // 2))))
    w = unifint(diff_lb, diff_ub, (5, max(5, min(10, (max_w + 1) // 2))))
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
    """Reading: the template is the 3x3 block-neighbourhood centred on its marker block.
    That whole window (marker, its 8 neighbour blocks and the separators between them)
    is copied from I and pasted centred on every lone marker block (window trimmed at grid
    edges); blocks where a window's background covered a stamp (or a zero-coloured
    template block that Paste cannot write) are then fixed block by block."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    submit_sel = [0, 0, hi - 1, wi - 1]

    # block lattice: separator rows/cols are the fully constant ones
    sep_rows = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    sep_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    if not sep_rows or not sep_cols:
        return [34], [submit_sel]
    hfac, wfac = sep_rows[0], sep_cols[0]
    hs, ws = hfac + 1, wfac + 1
    h, w = (hi + 1) // hs, (wi + 1) // ws
    g = [[int(I[a * hs, b * ws]) for b in range(w)] for a in range(h)]
    bgc = Counter([v for row in g for v in row]).most_common(1)[0][0]

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
        return [34], [submit_sel]
    ti = max(range(len(comps)), key=lambda k: (len({g[r][c] for r, c in comps[k]}), len(comps[k])))
    template = comps[ti]
    others = [cm for k, cm in enumerate(comps) if k != ti]
    dotc = Counter([g[r][c] for cm in others for r, c in cm]).most_common(1)[0][0]
    centres = [(r, c) for r, c in template if g[r][c] == dotc]
    if not centres:
        return [34], [submit_sel]
    ca, cb = centres[0]
    markers = sorted((r, c) for cm in others for r, c in cm if g[r][c] == dotc)
    rel = {(r - ca, c - cb): g[r][c] for r, c in template if (r, c) != (ca, cb)}

    def blk(a, b):
        return [(a * hs + i, b * ws + j) for i in range(hfac) for j in range(wfac)]

    sim = I.copy()
    want = I.copy()
    clip_key = None
    for (ma, mb) in markers:
        # window block offsets -1..1 that stay on the grid around this marker
        da0, da1 = max(-1, -ma), min(1, h - 1 - ma)
        db0, db1 = max(-1, -mb), min(1, w - 1 - mb)
        sr, sc = (ca + da0) * hs, (cb + db0) * ws
        bh = (da1 - da0 + 1) * hs - 1
        bw = (db1 - db0 + 1) * ws - 1
        pr, pc = (ma + da0) * hs, (mb + db0) * ws
        clip = I[sr:sr + bh, sc:sc + bw]
        new = sim.copy()
        m = clip != 0
        new[pr:pr + bh, pc:pc + bw][m] = clip[m]
        for (dr, dc), col in rel.items():
            a, b = ma + dr, mb + dc
            if 0 <= a < h and 0 <= b < w:
                for (r, c) in blk(a, b):
                    want[r, c] = col
        if not np.array_equal(new, sim):
            key = (sr, sc, bh, bw)
            if key != clip_key:
                # full rectangle on purpose: the template's 3x3 block window of I
                ops.append(28); sels.append([sr, sc, bh - 1, bw - 1])
                clip_key = key
            ops.append(30); sels.append([pr, pc, 0, 0])  # paste origin = window corner at marker
            sim = new

    # repair: blocks of the template or any stamp that the pastes left differing from the rule
    targets = [(ca + dr, cb + dc) for (dr, dc) in rel]
    for (ma, mb) in markers:
        targets += [(ma + dr, mb + dc) for (dr, dc) in rel]
    done = set()
    for (a, b) in targets:
        if (a, b) in done or not (0 <= a < h and 0 <= b < w):
            continue
        done.add((a, b))
        cells = blk(a, b)
        col = int(want[cells[0]])
        if any(sim[r, c] != col for r, c in cells):
            ops.append(col); sels.append(sel_of(cells))
            for r, c in cells:
                sim[r, c] = col

    ops.append(34); sels.append(submit_sel)
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
