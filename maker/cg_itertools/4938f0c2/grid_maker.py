"""
ARC Task: 4938f0c2 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    cc = random.choice(rem)
    rem = [c for c in rem if c != cc]
    objc = random.choice(rem)
    return {"bgc": bgc, "cc": cc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, cc, objc) -> dict:
    from random import randint, sample, choice

    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, round(a + (b - a) * d)))

    H = min(31, max_h + 1)
    W = min(31, max_w + 1)
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
    oh = unifint(diff_lb, diff_ub, (2, (h - 3) // 2))
    ow = unifint(diff_lb, diff_ub, (2, (w - 3) // 2))
    sg = [[bgc] * ow for _ in range(oh)]
    locc = (oh - 1, ow - 1)
    sg[locc[0]][locc[1]] = cc
    reminds = [(i, j) for i in range(oh) for j in range(ow) if (i, j) != locc]

    def shp(cs):
        rs = [r for r, _ in cs]; cs_ = [c for _, c in cs]
        return (max(rs) - min(rs) + 1, max(cs_) - min(cs_) + 1)

    ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
    cells = sample(reminds, ncells)
    while ncells == 4 and shp(cells) == (2, 2):
        ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
        cells = sample(reminds, ncells)
    for (i, j) in cells:
        sg[i][j] = objc
    sg = np.array(sg)
    G1 = sg; G2 = sg[:, ::-1]; G3 = sg[::-1, :]; G4 = sg[::-1, ::-1]
    vbar = np.full((oh, 1), bgc)
    hbar = np.full((1, ow), bgc)
    cp = np.full((1, 1), cc)
    topg = np.hstack([G1, vbar, G2])
    botg = np.hstack([G3, vbar, G4])
    ggm = np.hstack([hbar, cp, hbar])
    GG = np.vstack([topg, ggm, botg])
    loci = randint(0, h - 2 * oh - 1)
    locj = randint(0, w - 2 * ow - 1)
    go = np.full((h, w), bgc)
    go[loci:loci + GG.shape[0], locj:locj + GG.shape[1]] = GG
    gi = np.full((h, w), bgc)
    gi[loci:loci + oh, locj:locj + ow] = sg
    gi[go == cc] = cc
    k = choice((0, 1, 2, 3))
    gi = np.rot90(gi, k); go = np.rot90(go, k)
    rr, cc_ = np.where(gi == cc)
    ci = int(rr.min() + (rr.max() - rr.min()) // 2)
    cj = int(cc_.min() + (cc_.max() - cc_.min()) // 2)
    gi = np.delete(np.delete(gi, ci, 0), cj, 1)
    go = np.delete(np.delete(go, ci, 0), cj, 1)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def bg_of(g):
        g = np.asarray(g, dtype=int)
        return int(Counter(g.flatten().tolist()).most_common(1)[0][0])

    def markers(g):
        g = np.asarray(g, dtype=int)
        b = bg_of(g)
        bycol = {}
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                v = int(g[r, c])
                if v != b:
                    bycol.setdefault(v, []).append((r, c))
        out = {}
        for col, cs in bycol.items():
            if len(cs) == 4:
                rr = sorted({r for r, _ in cs}); cl = sorted({c for _, c in cs})
                if len(rr) == 2 and len(cl) == 2 and rr[1] == rr[0] + 1 and cl[1] == cl[0] + 1:
                    out[col] = (rr[0], cl[0])
        return out

    full = [0, 0, hi - 1, wi - 1]
    bgc = bg_of(I)
    cand = markers(I)
    if not cand:
        return [34], [full]
    if len(cand) == 1:
        mcol = next(iter(cand))
    else:
        votes = Counter()
        for ex in (examples or []):
            ei = ex.get("input") if isinstance(ex, dict) else ex[0]
            if ei is None:
                continue
            for k in markers(ei):
                votes[k] += 1
        pool = [c for c in cand if votes.get(c, 0) > 0]
        mcol = max(pool, key=lambda c: votes[c]) if pool else sorted(cand)[0]
    br, bc = cand[mcol]

    obj = [(r, c) for r in range(hi) for c in range(wi)
           if int(I[r, c]) != bgc and int(I[r, c]) != mcol]
    if not obj:
        return [34], [full]
    objc = int(I[obj[0][0], obj[0][1]])
    rmin = min(r for r, _ in obj); rmax = max(r for r, _ in obj)
    cmin = min(c for _, c in obj); cmax = max(c for _, c in obj)

    ops, sels = [], []

    if objc == 0:
        # 0 cannot travel through the clipboard: paint each reflected copy instead
        ar, ac = 2 * br + 1, 2 * bc + 1
        for image in ([(r, ac - c) for r, c in obj], [(ar - r, c) for r, c in obj],
                      [(ar - r, ac - c) for r, c in obj]):
            cells = sorted({(r, c) for r, c in image if 0 <= r < hi and 0 <= c < wi})
            if cells:
                ops.append(0); sels.append(sel_of(cells))
        ops.append(34); sels.append(full)
        return ops, sels

    # quadrant holding the picture: from the object's far edge up to the marker
    # (the marker's 2x2 is the four quadrants' shared inner corners)
    if rmax <= br:
        q_r0, q_r1 = rmin, br
    else:
        q_r0, q_r1 = br + 1, rmax
    if cmax <= bc:
        q_c0, q_c1 = cmin, bc
    else:
        q_c0, q_c1 = bc + 1, cmax
    qh, qw = q_r1 - q_r0 + 1, q_c1 - q_c0 + 1
    d_r0 = 2 * br + 1 - q_r1      # mirrored rows across the marker's horizontal axis
    d_c0 = 2 * bc + 1 - q_c1      # mirrored cols across the marker's vertical axis

    G = I.copy()
    mk = {(br + i, bc + j) for i in (0, 1) for j in (0, 1)}

    def stray(r0, c0, hh, ww):
        # with bgc == 0 the transparent paste leaves the marker cell that already sat in the
        # destination rectangle; the flip carries it off the marker -> put it back to background
        cells = [(r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww)
                 if (r, c) not in mk and int(G[r, c]) == mcol]
        if cells:
            ops.append(bgc); sels.append(sel_of(cells))
            for r, c in cells:
                G[r, c] = bgc
        # with marker colour 0 the clipboard cannot carry it: restore the marker cells in place
        lost = [(r, c) for (r, c) in sorted(mk)
                if r0 <= r < r0 + hh and c0 <= c < c0 + ww and int(G[r, c]) != mcol]
        if lost:
            ops.append(mcol); sels.append(sel_of(lost))
            for r, c in lost:
                G[r, c] = mcol

    # stage 1: the picture's half-strip -> mirror it to the other side of the marker rows
    ops.append(28); sels.append([q_r0, q_c0, qh - 1, qw - 1])   # full quadrant rectangle
    ops.append(30); sels.append([d_r0, q_c0, 0, 0])
    blk = I[q_r0:q_r1 + 1, q_c0:q_c1 + 1]
    G[d_r0:d_r0 + qh, q_c0:q_c0 + qw] = np.where(blk != 0, blk, G[d_r0:d_r0 + qh, q_c0:q_c0 + qw])
    dst = G[d_r0:d_r0 + qh, q_c0:q_c0 + qw]
    if not np.array_equal(dst, dst[::-1, :]):
        ops.append(27); sels.append([d_r0, q_c0, qh - 1, qw - 1])  # flip the pasted rectangle
        G[d_r0:d_r0 + qh, q_c0:q_c0 + qw] = dst[::-1, :].copy()
    stray(d_r0, q_c0, qh, qw)

    # stage 2: the whole column strip (both copies + marker rows) -> mirror across marker cols
    s_r0, s_r1 = min(q_r0, d_r0), max(q_r1, d_r0 + qh - 1)
    sh = s_r1 - s_r0 + 1
    ops.append(29); sels.append([s_r0, q_c0, sh - 1, qw - 1])   # full strip rectangle
    ops.append(30); sels.append([s_r0, d_c0, 0, 0])
    blk = G[s_r0:s_r1 + 1, q_c0:q_c1 + 1].copy()
    G[s_r0:s_r1 + 1, d_c0:d_c0 + qw] = np.where(blk != 0, blk, G[s_r0:s_r1 + 1, d_c0:d_c0 + qw])
    dst = G[s_r0:s_r1 + 1, d_c0:d_c0 + qw]
    if not np.array_equal(dst, dst[:, ::-1]):
        ops.append(26); sels.append([s_r0, d_c0, sh - 1, qw - 1])  # flip the pasted strip
        G[s_r0:s_r1 + 1, d_c0:d_c0 + qw] = dst[:, ::-1].copy()
    stray(s_r0, d_c0, sh, qw)

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
                        f"num_examples+1 ({num_examples + 1}) for task 4938f0c2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4938f0c2"
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
                                f"for task 4938f0c2"
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
                    f"Failed to build a complete episode for task 4938f0c2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4938f0c2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
