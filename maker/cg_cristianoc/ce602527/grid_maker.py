"""
ARC Task: ce602527 (RE-ARC) — LLM-generated grid_maker
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
    bgc, c1, c2, c3 = random.sample(range(10), 4)
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2, c3) -> dict:
    def unifint(lo, hi):
        if hi < lo:
            hi = lo
        d = random.uniform(diff_lb, diff_ub)
        return lo + int(round(d * (hi - lo)))

    def dn(r, c):
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    def shape(p):
        rs = [r for r, _ in p]; cs = [c for _, c in p]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    def normalize(p):
        r0 = min(r for r, _ in p); c0 = min(c for _, c in p)
        return {(r - r0, c - c0) for r, c in p}

    def shift(p, d):
        return {(r + d[0], c + d[1]) for r, c in p}

    h = unifint(12, max(12, min(30, max_h)))
    w = unifint(12, max(12, min(30, max_w)))
    while True:
        objs = []
        for _ in range(2):
            oh = unifint(3, h // 3 - 1)
            ow = unifint(3, w // 3 - 1)
            bounds = {(i, j) for i in range(oh) for j in range(ow)}
            numcd = unifint(0, max(0, (oh * ow) // 2 - 4))
            numc = random.choice((numcd, oh * ow - numcd))
            numc = min(max(3, numc), oh * ow - 3)
            obj = {random.choice(sorted(bounds))}
            while len(obj) < numc or shape(obj) != (oh, ow):
                nb = set()
                for r, c in obj:
                    nb |= dn(r, c)
                obj.add(random.choice(sorted((bounds - obj) & nb)))
            objs.append(normalize(obj))
        a, b = objs
        maxinh = min(shape(a)[0], shape(b)[0]) // 2 + 1
        maxinw = min(shape(a)[1], shape(b)[1]) // 2 + 1
        ca = {(r, c) for r, c in a if r < maxinh and c < maxinw}
        cb = {(r, c) for r, c in b if r < maxinh and c < maxinw}
        if ca != cb:
            break
    trgo = random.choice(objs)
    trgo2 = {(2 * r + dr, 2 * c + dc) for r, c in trgo for dr in (0, 1) for dc in (0, 1)}
    th = shape(trgo2)[0]
    staysinh = unifint(maxinh * 2, shape(trgo)[0] * 2)
    nout = th - staysinh
    loci = h - th + nout
    locj = random.randint(0, w - maxinw * 2)
    gi = np.full((h, w), bgc, dtype=int)
    for r, c in shift(trgo2, (loci, locj)):
        if 0 <= r < h and 0 <= c < w:
            gi[r, c] = c3
    (lcol, lobj), (rcol, robj) = random.sample([(c1, a), (c2, b)], 2)
    cands = {(i, j) for i in range(1, h - 1) for j in range(1, w - 1) if gi[i, j] == bgc}
    lca = {ij for ij in cands if ij[1] < w // 3 * 2}
    lcands = [ij for ij in sorted(lca) if shift(lobj, ij) <= lca]
    if not lcands:
        raise ValueError("no placement")
    for _ in range(2000):
        lloc = random.choice(lcands)
        rloc = random.choice(lcands)
        lp = shift(lobj, lloc); rp = shift(robj, rloc)
        if lp <= cands and rp <= cands and not (lp & rp):
            break
    else:
        raise ValueError("no placement")
    for r, c in lp:
        gi[r, c] = lcol
    for r, c in rp:
        gi[r, c] = rcol
    th0, tw0 = shape(trgo)
    go = np.full((th0, tw0), bgc, dtype=int)
    for r, c in trgo:
        go[r, c] = c1 if trgo == a else c2
    mf = random.choice([
        lambda g: g, lambda g: np.rot90(g, 3), lambda g: np.rot90(g, 2),
        lambda g: np.rot90(g, 1), lambda g: np.rot90(g, 2).T,
        lambda g: g.T, lambda g: np.flipud(g), lambda g: np.fliplr(g)])
    gi, go = mf(gi), mf(go)
    # reject instances where the visible piece of the big object fits both
    # small objects' 2x upscales (the rule would be ambiguous)
    vis = {(r, c) for r, c in shift(trgo2, (loci, locj)) if 0 <= r < h and 0 <= c < w}
    vr0 = min(r for r, _ in vis); vc0 = min(c for _, c in vis)
    vh, vw = shape(vis)
    vm = np.zeros((vh, vw), dtype=int)
    for r, c in vis:
        vm[r - vr0, c - vc0] = 1
    nmatch = 0
    for o in objs:
        oh_, ow_ = shape(o)
        om = np.zeros((oh_, ow_), dtype=int)
        for r, c in o:
            om[r, c] = 1
        up = np.kron(om, np.ones((2, 2), dtype=int))
        if any(np.array_equal(up[r:r + vh, c:c + vw], vm)
               for r in range(up.shape[0] - vh + 1) for c in range(up.shape[1] - vw + 1)):
            nmatch += 1
    if nmatch != 1:
        raise ValueError("ambiguous target")
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    colors = [int(c) for c in np.unique(I)]

    # background: the only colour spanning the whole grid (corner-to-corner)
    bgc = None
    for c in colors:
        rs, cs = np.where(I == c)
        if rs.min() == 0 and rs.max() == hi - 1 and cs.min() == 0 and cs.max() == wi - 1:
            bgc = c
            break
    if bgc is None:
        bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])
    fgs = [c for c in colors if c != bgc]

    def bbox(col):
        rs, cs = np.where(I == col)
        return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())

    def mask(col):
        r0, c0, r1, c1 = bbox(col)
        return (I[r0:r1 + 1, c0:c1 + 1] == col).astype(int)

    # the big (2x upscaled, partly off-grid) object is the one touching the border
    big, smalls = None, []
    for c in fgs:
        r0, c0, r1, c1 = bbox(c)
        if big is None and (r0 == 0 or c0 == 0 or r1 == hi - 1 or c1 == wi - 1):
            big = c
        else:
            smalls.append(c)
    if big is None:
        big, smalls = fgs[0], fgs[1:]
    big_m = mask(big)
    bh, bw = big_m.shape

    # the target is the small object whose 2x upscale contains the visible big piece
    target = smalls[-1]
    for c in smalls:
        up = np.kron(mask(c), np.ones((2, 2), dtype=int))
        uh, uw = up.shape
        if any(np.array_equal(up[r:r + bh, cc:cc + bw], big_m)
               for r in range(uh - bh + 1) for cc in range(uw - bw + 1)):
            target = c
            break

    r0, c0, r1, c1 = bbox(target)
    ho, wo = r1 - r0 + 1, c1 - c0 + 1
    ops, sels = [], []
    # crop canvas to the target object's bounding rectangle (full rectangle intended)
    ops.append(33); sels.append([r0, c0, ho - 1, wo - 1])
    # each foreign object intruding into that rectangle is blanked as one unit
    # (its cells inside the target bbox, named from I; a 0-coloured intruder was
    #  dropped to 0 by the transparent crop and still needs bgc)
    for c in fgs:
        if c == target:
            continue
        cells = [(r - r0, cc - c0) for r in range(r0, r1 + 1) for cc in range(c0, c1 + 1)
                 if I[r, cc] == c]
        if cells:
            ops.append(bgc); sels.append(sel_of(cells))
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task ce602527"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ce602527"
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
                                f"for task ce602527"
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
                    f"Failed to build a complete episode for task ce602527 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ce602527-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
