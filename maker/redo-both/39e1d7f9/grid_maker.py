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


def sample_colors(num_examples=None) -> dict:
    # bgc is fixed to 0 (ARC background convention): the block-grid is drawn on it and
    # every stamped object colour is drawn from the remaining palette.
    bgc = 0
    others = [c for c in range(1, 10)]
    linc, dotc = random.sample(others, 2)          # separator-line colour, marker colour
    pool = [c for c in others if c not in (linc, dotc)]
    random.shuffle(pool)                            # episode-consistent object palette
    return {"bgc": bgc, "linc": linc, "dotc": dotc, "ccols_pool": pool}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, dotc, ccols_pool) -> dict:
    def _uf(a, b):
        if b < a:
            b = a
        vals = list(range(a, b + 1))
        lo = int(diff_lb * (len(vals) - 1))
        hi = int(diff_ub * (len(vals) - 1)) + 1
        sub = vals[lo:hi]
        return random.choice(sub if sub else vals)

    D4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    hub = max(5, min(10, (max_h + 1) // 2))
    wub = max(5, min(10, (max_w + 1) // 2))

    g = go = None
    h = w = 0
    for _attempt in range(300):
        h = _uf(5, hub)
        w = _uf(5, wub)
        g = [[bgc] * w for _ in range(h)]

        loci = random.randint(1, h - 2)
        locj = random.randint(1, w - 2)
        if h == 5:
            loci = random.choice([1, h - 2])
        if w == 5:
            locj = random.choice([1, w - 2])

        npix = _uf(1, 8)
        ncols = _uf(1, min(7, len(ccols_pool)))
        ccols = list(ccols_pool)[:ncols]

        ring = {(loci + dr, locj + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        ring.discard((loci, locj))
        pixs = {(loci, locj)}
        for _ in range(npix):
            grow = set()
            for (r, c) in pixs:
                for dr, dc in D4:
                    grow.add((r + dr, c + dc))
            opts = sorted((grow & ring) - pixs)
            if not opts:
                break
            pixs.add(random.choice(opts))
        pixs = sorted(pixs - {(loci, locj)})
        if not pixs:
            continue

        obj = {(r, c): random.choice(ccols) for (r, c) in pixs}
        g[loci][locj] = dotc
        for (r, c), col in obj.items():
            g[r][c] = col
        go = [row[:] for row in g]

        # normalized stamp (marker at (0,0)) and its bounding box
        objn = {(r - loci, c - locj): col for (r, c), col in obj.items()}
        cells = list(objn.keys()) + [(0, 0)]
        r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
        bbox_rel = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

        # cells forbidden for further markers: around the template (as in the original)
        ex = set()
        for (rr, cc) in ring:
            for dr, dc in D4:
                ex.add((rr + dr, cc + dc))
        ex |= {(loci + r, locj + c) for (r, c) in bbox_rel}
        for (rr, cc) in list(ex):
            for dr, dc in D4:
                ex.add((rr + dr, cc + dc))

        inds = {(r, c) for r in range(h) for c in range(w) if g[r][c] == bgc} - ex

        noccs = _uf(1, max(1, (h * w) // (2 * len(pixs) + 1)))
        succ = 0
        tr = 0
        maxtr = 6 * noccs
        tried = set()
        while (tr < maxtr and succ < noccs) or (succ == 0 and tr < 60):
            cand = sorted(inds - tried)
            if not cand:
                break
            tr += 1
            a, b = random.choice(cand)
            bb = [(a + r, b + c) for (r, c) in bbox_rel]
            if all(p in inds for p in bb):
                rem = set(bb)
                for (r, c) in bb:
                    for dr, dc in D4:
                        rem.add((r + dr, c + dc))
                inds = inds - rem
                succ += 1
                g[a][b] = dotc
                go[a][b] = dotc
                for (rr, cc), col in objn.items():
                    go[a + rr][b + cc] = col
            else:
                tried.add((a, b))
        if succ > 0:
            break

    hfac = _uf(1, max(1, (max_h - h + 1) // h))
    wfac = _uf(1, max(1, (max_w - w + 1) // w))
    fullh = hfac * h + h - 1
    fullw = wfac * w + w - 1

    def up(src):
        out = [[linc] * fullw for _ in range(fullh)]
        for a in range(h):
            for b in range(w):
                col = src[a][b]
                for i in range(hfac):
                    for j in range(wfac):
                        out[a * (hfac + 1) + i][b * (wfac + 1) + j] = col
        return tuple(tuple(row) for row in out)

    return {"input": up(g), "output": up(go)}


def derive_operations(I, O):
    """Replicate the multi-colour template block-shape onto every lone marker block.

    Everything (block grid geometry, background, marker colour, which block-region is
    the template, where it is anchored, where the copies go) is measured from I only.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    submit_sel = [0, 0, hi - 1, wi - 1]

    # --- block-grid geometry: the separator lines are the constant rows/columns ---
    sep_rows = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    sep_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    if not sep_rows or not sep_cols:
        return [34], [submit_sel]
    hfac = sep_rows[0]                 # block height in pixels
    wfac = sep_cols[0]                 # block width in pixels
    hs, ws = hfac + 1, wfac + 1        # block pitch (block + separator)
    h = (hi + 1) // hs
    w = (wi + 1) // ws
    if h < 1 or w < 1:
        return [34], [submit_sel]

    # logical (compressed) grid: one value per block
    g = [[int(I[a * hs, b * ws]) for b in range(w)] for a in range(h)]
    bgc = Counter([g[a][b] for a in range(h) for b in range(w)]).most_common(1)[0][0]

    # --- connected block-components of non-background blocks ---
    seen = [[False] * w for _ in range(h)]
    comps = []
    for a in range(h):
        for b in range(w):
            if g[a][b] != bgc and not seen[a][b]:
                stack = [(a, b)]
                seen[a][b] = True
                cur = []
                while stack:
                    r, c = stack.pop()
                    cur.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and g[nr][nc] != bgc:
                            seen[nr][nc] = True
                            stack.append((nr, nc))
                comps.append(cur)
    if not comps:
        return [34], [submit_sel]

    # the template is the component using the most distinct colours
    ti = max(range(len(comps)), key=lambda k: (len({g[r][c] for r, c in comps[k]}), len(comps[k])))
    template = comps[ti]
    others = [comps[k] for k in range(len(comps)) if k != ti]
    if not others:
        return [34], [submit_sel]

    # marker colour = the colour of the lone components
    dot_counts = Counter([g[r][c] for cm in others for (r, c) in cm])
    dotc = dot_counts.most_common(1)[0][0]

    dot_cells = [(r, c) for (r, c) in template if g[r][c] == dotc]
    if not dot_cells:
        return [34], [submit_sel]
    dot_r = min(r for r, _ in dot_cells)
    dot_c = min(c for _, c in dot_cells)

    r0 = min(r for r, _ in template); r1 = max(r for r, _ in template)
    c0 = min(c for _, c in template); c1 = max(c for _, c in template)
    th, tw = r1 - r0 + 1, c1 - c0 + 1
    off_r, off_c = dot_r - r0, dot_c - c0
    rel = {(r - r0, c - c0): g[r][c] for (r, c) in template}

    # destinations: each lone marker, aligned so the template's marker lands on it
    dsts = []
    for cm in others:
        ur = min(r for r, _ in cm)
        uc = min(c for _, c in cm)
        d = (ur - off_r, uc - off_c)
        if d != (r0, c0) and d not in dsts:
            dsts.append(d)
    dsts.sort()
    if not dsts:
        return [34], [submit_sel]

    src_pr, src_pc = r0 * hs, c0 * ws
    box_h = th * hs - 1                 # pixel height of the template block-region
    box_w = tw * ws - 1

    # 1) copy the template block-region from the input (full rectangle: blocks + separators)
    ops.append(28); sels.append([src_pr, src_pc, box_h - 1, box_w - 1])
    # 2) stamp it on every marker
    for (dr_, dc_) in dsts:
        ops.append(30); sels.append([dr_ * hs, dc_ * ws, 0, 0])   # paste origin

    # --- repair, measured from I: where two stamp rectangles overlap, the later paste can
    # cover an earlier stamp's block with its own background block. Simulate the pastes and
    # the intended stamping (both from I) and restore the covered blocks.
    sim = I.copy()
    clip = I[src_pr:src_pr + box_h, src_pc:src_pc + box_w].copy()
    for (dr_, dc_) in dsts:
        pr, pc = dr_ * hs, dc_ * ws
        for i in range(box_h):
            for j in range(box_w):
                v = int(clip[i, j])
                if v != 0 and 0 <= pr + i < hi and 0 <= pc + j < wi:
                    sim[pr + i, pc + j] = v

    want = I.copy()
    for (dr_, dc_) in dsts:
        for (rr, cc), col in rel.items():
            ar, ac = dr_ + rr, dc_ + cc
            if 0 <= ar < h and 0 <= ac < w:
                want[ar * hs:ar * hs + hfac, ac * ws:ac * ws + wfac] = col

    for (dr_, dc_) in dsts:                      # one stamp region at a time
        for rr in range(th):
            for cc in range(tw):
                ar, ac = dr_ + rr, dc_ + cc
                if not (0 <= ar < h and 0 <= ac < w):
                    continue
                pr, pc = ar * hs, ac * ws
                blk_now = sim[pr:pr + hfac, pc:pc + wfac]
                blk_want = want[pr:pr + hfac, pc:pc + wfac]
                if not np.array_equal(blk_now, blk_want):
                    col = int(blk_want[0, 0])
                    ops.append(col); sels.append([pr, pc, hfac - 1, wfac - 1])  # one block
                    sim[pr:pr + hfac, pc:pc + wfac] = col

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
