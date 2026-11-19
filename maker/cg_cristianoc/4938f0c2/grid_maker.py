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
from collections import Counter

import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    cc = random.choice(rem)
    rem = [c for c in rem if c != cc]
    objc = random.choice(rem)
    return {"bgc": bgc, "cc": cc, "objc": objc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, cc=1, objc=2, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        d = random.uniform(lb, ub)
        return min(b, max(a, a + int(round(d * (b - a)))))

    H = min(31, max_h + 1)
    W = min(31, max_w + 1)
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
    oh = unifint(diff_lb, diff_ub, (2, max(2, (h - 3) // 2)))
    ow = unifint(diff_lb, diff_ub, (2, max(2, (w - 3) // 2)))

    sg = [[bgc] * ow for _ in range(oh)]
    locc = (oh - 1, ow - 1)
    sg[locc[0]][locc[1]] = cc
    reminds = [(i, j) for i in range(oh) for j in range(ow) if (i, j) != locc]

    def shape_of(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
    ncells = min(ncells, len(reminds))
    cells = random.sample(reminds, ncells)
    while ncells == 4 and shape_of(cells) == (2, 2):
        ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
        ncells = min(ncells, len(reminds))
        cells = random.sample(reminds, ncells)
    for (i, j) in cells:
        sg[i][j] = objc

    sg = np.array(sg, dtype=int)
    G1 = sg
    G2 = np.fliplr(sg)
    G3 = np.flipud(sg)
    G4 = np.flipud(np.fliplr(sg))
    vbar = np.full((oh, 1), bgc, dtype=int)
    hbar = np.full((1, ow), bgc, dtype=int)
    cp = np.full((1, 1), cc, dtype=int)
    topg = np.hstack([G1, vbar, G2])
    botg = np.hstack([G3, vbar, G4])
    ggm = np.hstack([hbar, cp, hbar])
    GG = np.vstack([topg, ggm, botg])

    loci = random.randint(0, h - 2 * oh - 1)
    locj = random.randint(0, w - 2 * ow - 1)
    go = np.full((h, w), bgc, dtype=int)
    go[loci:loci + GG.shape[0], locj:locj + GG.shape[1]] = GG
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + oh, locj:locj + ow] = sg
    gi[go == cc] = cc

    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)

    pts = np.argwhere(gi == cc)
    rs = pts[:, 0]
    cs = pts[:, 1]
    ccpi = int(rs.min() + (rs.max() - rs.min() + 1) // 2)
    ccpj = int(cs.min() + (cs.max() - cs.min() + 1) // 2)
    gi = np.delete(np.delete(gi, ccpi, axis=0), ccpj, axis=1)
    go = np.delete(np.delete(go, ccpi, axis=0), ccpj, axis=1)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def bg_of(g):
        g = np.asarray(g, dtype=int)
        return int(Counter(g.flatten().tolist()).most_common(1)[0][0])

    def markers(g):
        """colours whose whole cell set is a solid 2x2 block -> {colour: (top, left)}"""
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
                rr = sorted({r for r, _ in cs})
                cc_ = sorted({c for _, c in cs})
                if len(rr) == 2 and len(cc_) == 2 and rr[1] == rr[0] + 1 and cc_[1] == cc_[0] + 1:
                    out[col] = (rr[0], cc_[0])
        return out

    bgc = bg_of(I)
    cand = markers(I)
    if not cand:
        return [34], [[0, 0, hi - 1, wi - 1]]
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
    ar2 = 2 * br + 1   # doubled horizontal mirror axis (between marker rows)
    ac2 = 2 * bc + 1   # doubled vertical mirror axis (between marker cols)

    obj = [(r, c) for r in range(hi) for c in range(wi)
           if int(I[r, c]) != bgc and int(I[r, c]) != mcol]
    if not obj:
        return [34], [[0, 0, hi - 1, wi - 1]]
    objc = int(I[obj[0][0], obj[0][1]])

    # Golf reading: the unit is ONE source pixel and its orbit under the two
    # mirror axes through the marker. For each object pixel, paint its three
    # reflected images (left-right, up-down, both) together in one Color op.
    # The orbit of an input pixel is fully determined by I and the marker.
    # Sweep outward from the marker, nearest pixel first.
    cy, cx = ar2 / 2.0, ac2 / 2.0
    obj_sorted = sorted(obj, key=lambda p: (abs(p[0] - cy) + abs(p[1] - cx), p))

    cur = I.copy()
    ops, sels = [], []
    for (r, c) in obj_sorted:
        orbit = [(r, ac2 - c), (ar2 - r, c), (ar2 - r, ac2 - c)]
        cells = sorted({(y, x) for (y, x) in orbit
                        if 0 <= y < hi and 0 <= x < wi and int(cur[y, x]) != objc})
        if not cells:
            continue
        ops.append(objc)
        sels.append(sel_of(cells))
        for (y, x) in cells:
            cur[y, x] = objc

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])   # full-grid rectangle: submit
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
