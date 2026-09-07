"""
ARC Task: 3345333e (RE-ARC) — LLM-generated grid_maker
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


VARIANTS = [{"axis": "v"}, {"axis": "h"}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # object color must be non-zero: the restoration copies the intact half of the
    # shape with CopyI/Paste, and 0 is "nothing" for the clipboard.
    objc = random.choice([c for c in cols if c != bgc and c != 0])
    occcol = random.choice([c for c in cols if c not in (bgc, objc)])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "occcol": occcol, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, objc=None, occcol=None, axis=None) -> dict:
    if axis is None:
        axis = choice(('v', 'h'))
    lim = min(max_h, max_w)                 # rot90-family transforms may transpose
    lo = min(10, lim)
    h = unifint(diff_lb, diff_ub, (lo, lim))
    w = unifint(diff_lb, diff_ub, (lo, lim))
    oh = unifint(diff_lb, diff_ub, (4, h - 2))
    ow = unifint(diff_lb, diff_ub, (4, (w - 2) // 2))
    nc = unifint(diff_lb, diff_ub, (min(oh, ow), (oh * ow) // 3 * 2))
    shp = {(0, 0)}
    bounds = asindices(canvas(-1, (oh, ow)))
    for j in range(nc):
        ij = choice(totuple((bounds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    while height(shp) < 3 or width(shp) < 3:
        ij = choice(totuple((bounds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    vmshp = vmirror(shp)
    if choice((True, False)):
        vmshp = sfilter(vmshp, lambda ij: ij[1] != width(shp) - 1)
    shp = normalize(combine(shp, shift(vmshp, (0, -width(vmshp)))))
    oh, ow = shape(shp)
    loci = randint(1, h - oh - 1)
    locj = randint(1, w - ow - 1)
    loc = (loci, locj)
    shp = shift(shp, loc)
    c = canvas(bgc, (h, w))
    go = fill(c, objc, shp)
    boxh = unifint(diff_lb, diff_ub, (2, oh - 1))
    boxw = unifint(diff_lb, diff_ub, (2, ow // 2))
    ulci = randint(loci - 1, loci + oh - boxh + 1)
    ulcj = randint(locj + ow // 2 + 1, locj + ow - boxw + 1)
    bx = backdrop(frozenset({(ulci, ulcj), (ulci + boxh - 1, ulcj + boxw - 1)}))
    gi = fill(go, occcol, bx)
    # shape is mirror-symmetric about a vertical axis here; transposing transforms
    # turn that into a horizontal-axis symmetry. Pick the transform set matching `axis`.
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    transposing = (dmirror, cmirror, rot90, rot270)
    nmfs = choice((1, 2))
    while True:
        fns = sample(mfs, nmfs)
        par = sum(1 for fn in fns if fn in transposing) % 2
        if (par == 1) == (axis == 'h'):
            break
    for fn in fns:
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # background = border color
    border = I[0].tolist() + I[-1].tolist() + I[:, 0].tolist() + I[:, -1].tolist()
    bgc = Counter(border).most_common(1)[0][0]

    # two remaining colors: the occluder is the one filling its bbox solidly
    cols = [c for c in np.unique(I).tolist() if c != bgc]
    info = {}
    for c in cols:
        rs, cs = np.where(I == c)
        a, b, d, e = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
        solid = (b - a + 1) * (e - d + 1) == len(rs)
        info[c] = (solid, len(rs), (a, d, b, e))
    solids = [c for c in cols if info[c][0]]
    occ_c = solids[0] if len(solids) == 1 else min(cols, key=lambda c: info[c][1])
    obj_c = [c for c in cols if c != occ_c][0]
    r0, c0, r1, c1 = info[occ_c][2]

    V = set(zip(*[a.tolist() for a in np.where(I == obj_c)]))   # visible shape cells
    B = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}  # occluded area
    allowed = V | B

    # The shape is mirror-symmetric; find the axis (row-mirror r->K-r or col-mirror
    # c->K-c) whose reflection of the visible shape stays inside shape+occluded area,
    # never straddles the occluder, and overlaps the visible shape the most.
    best = None
    for kind in ('h', 'v'):
        span = hi if kind == 'h' else wi
        for K in range(0, 2 * span - 1):
            if kind == 'h':
                if K - r1 < 0 or K - r0 > hi - 1:
                    continue
                if not (K - r1 > r1 or K - r0 < r0):
                    continue
                mir = {(K - r, c) for (r, c) in V}
            else:
                if K - c1 < 0 or K - c0 > wi - 1:
                    continue
                if not (K - c1 > c1 or K - c0 < c0):
                    continue
                mir = {(r, K - c) for (r, c) in V}
            if not mir <= allowed:
                continue
            score = len(mir & V)
            if best is None or score > best[0]:
                best = (score, kind, K)

    ops, sels = [], []
    dh, dw = r1 - r0, c1 - c0

    if best is None:
        ops.append(int(bgc)); sels.append([r0, c0, dh, dw])
    else:
        _, kind, K = best
        # Paste is transparent to 0: when bgc==0 the occluder must be cleared first.
        if bgc == 0:
            ops.append(0); sels.append([r0, c0, dh, dw])
        if kind == 'h':
            ops.append(28); sels.append([K - r1, c0, dh, dw])   # intact mirror half
            ops.append(30); sels.append([r0, c0, 0, 0])         # onto occluded area
            ops.append(27); sels.append([r0, c0, dh, dw])       # mirror it up<->down
        else:
            ops.append(28); sels.append([r0, K - c1, dh, dw])
            ops.append(30); sels.append([r0, c0, 0, 0])
            ops.append(26); sels.append([r0, c0, dh, dw])       # mirror it left<->right

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 3345333e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3345333e"
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
                                f"for task 3345333e"
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
                    f"Failed to build a complete episode for task 3345333e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3345333e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
