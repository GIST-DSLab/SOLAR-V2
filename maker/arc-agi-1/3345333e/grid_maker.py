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
import numpy as np
from collections import Counter

from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# The four structural cases this task can produce:
#   the shape is mirror-symmetric about ONE axis, and the occluding rectangle
#   always sits strictly on ONE side of that axis.
#     identity -> vertical axis, occluder on the right
#     vmirror  -> vertical axis, occluder on the left
#     dmirror  -> horizontal axis, occluder on the bottom
#     cmirror  -> horizontal axis, occluder on the top
# ----------------------------------------------------------------------------
VARIANTS = [
    {"tf": "identity"},
    {"tf": "vmirror"},
    {"tf": "dmirror"},
    {"tf": "cmirror"},
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    # object colour must be non-zero: the reflection is carried out with
    # CopyI/Paste, and 0 is "nothing" to the clipboard.
    objc = random.choice([c for c in cols if c != 0])
    bgc = random.choice([c for c in cols if c != objc])
    occcol = random.choice([c for c in cols if c not in (objc, bgc)])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "occcol": occcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, occcol, tf=None, **kwargs) -> dict:
    if tf is None:
        tf = choice(("identity", "vmirror", "dmirror", "cmirror"))
    tfmap = {
        "identity": identity, "dmirror": dmirror, "cmirror": cmirror,
        "vmirror": vmirror, "hmirror": hmirror,
        "rot90": rot90, "rot180": rot180, "rot270": rot270,
    }
    fn = tfmap[tf]
    swaps = tf in ("dmirror", "cmirror", "rot90", "rot270")

    hcap = max(10, min(30, max_w if swaps else max_h))
    wcap = max(10, min(30, max_h if swaps else max_w))

    h = unifint(diff_lb, diff_ub, (10, hcap))
    w = unifint(diff_lb, diff_ub, (10, wcap))
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
    shp = shift(shp, (loci, locj))

    c = canvas(bgc, (h, w))
    go = fill(c, objc, shp)

    bx = None
    for _attempt in range(64):
        boxh = unifint(diff_lb, diff_ub, (2, oh - 1))
        boxw = unifint(diff_lb, diff_ub, (2, ow // 2))
        ulci = randint(loci - 1, loci + oh - boxh + 1)
        ulcj = randint(locj + ow // 2 + 1, locj + ow - boxw + 1)
        cand = backdrop(frozenset({(ulci, ulcj), (ulci + boxh - 1, ulcj + boxw - 1)}))
        bx = cand
        if len(cand & shp) > 0:      # the occluder must actually hide part of the shape
            break

    gi = fill(go, occcol, bx)

    gi = fn(gi)
    go = fn(go)
    return {"input": gi, "output": go}


# ----------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    def cells_of(sel):
        if isinstance(sel, dict):
            return [(int(r), int(c)) for r, c in sel.get("cells", [])]
        r, c, dh, dw = sel
        return [(rr, cc) for rr in range(r, r + dh + 1) for cc in range(c, c + dw + 1)]

    def sim(ops, sels):
        g = I.copy()
        clip = None
        states = []
        for op, sel in zip(ops, sels):
            if op == 34:
                states.append(g.copy())
                continue
            if op < 10:
                for (r, c) in cells_of(sel):
                    if 0 <= r < hi and 0 <= c < wi:
                        g[r, c] = op
            elif op == 28:
                r, c, dh, dw = sel
                clip = I[r:r + dh + 1, c:c + dw + 1].copy()
            elif op == 30:
                r, c = sel[0], sel[1]
                if clip is not None:
                    ch, cw = clip.shape
                    for i in range(ch):
                        for j in range(cw):
                            if clip[i, j] != 0 and 0 <= r + i < hi and 0 <= c + j < wi:
                                g[r + i, c + j] = clip[i, j]
            elif op in (26, 27):
                r, c, dh, dw = sel
                hh, ww = dh + 1, dw + 1
                region = g[r:r + hh, c:c + ww].copy()
                obj = np.where(region != 0, region, 0)
                g[r:r + hh, c:c + ww] = np.where(region != 0, 0, region)
                f = np.fliplr(obj) if op == 26 else np.flipud(obj)
                tgt = g[r:r + hh, c:c + ww]
                g[r:r + hh, c:c + ww] = np.where(f != 0, f, tgt)
            states.append(g.copy())
        return g, states

    # --- colours -------------------------------------------------------------
    ring = I[0, :].tolist() + I[-1, :].tolist() + I[:, 0].tolist() + I[:, -1].tolist()
    bgc = Counter(ring).most_common(1)[0][0]
    ocols = [int(c) for c in np.unique(O).tolist() if c != bgc]
    objc = ocols[0] if ocols else int(bgc)
    icols = [int(c) for c in np.unique(I).tolist() if c != bgc and c != objc]
    occc = icols[0] if icols else None

    ops, sels = [], []

    # --- 1. remove the occluding rectangle ----------------------------------
    cur = I.copy()
    if occc is not None:
        occ_cells = [(int(r), int(c)) for r, c in np.argwhere(I == occc)]
        if occ_cells:
            ops.append(int(bgc))
            sels.append(sel_of(occ_cells))
            cur[I == occc] = bgc

    diff = [(int(r), int(c)) for r, c in np.argwhere(cur != O)]
    if not diff:
        ops.append(34)
        sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
        return ops, sels

    # --- 2. reflect the intact half onto the damaged half -------------------
    mask = (O == objc)
    rr, cc = np.where(mask)
    r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
    sub = mask[r0:r1 + 1, c0:c1 + 1]
    Hh, Ww = r1 - r0 + 1, c1 - c0 + 1

    plans = []
    if Ww >= 2 and np.array_equal(sub, sub[:, ::-1]):
        m = Ww // 2
        left = (c0, c0 + m - 1)
        right = (c1 - m + 1, c1)
        dc = set(c for _, c in diff)
        if all(left[0] <= c <= left[1] for c in dc):
            plans.append(("V", (r0, left[0]), (r0, right[0]), Hh, m))
        elif all(right[0] <= c <= right[1] for c in dc):
            plans.append(("V", (r0, right[0]), (r0, left[0]), Hh, m))
    if Hh >= 2 and np.array_equal(sub, sub[::-1, :]):
        m = Hh // 2
        top = (r0, r0 + m - 1)
        bot = (r1 - m + 1, r1)
        dr = set(r for r, _ in diff)
        if all(top[0] <= r <= top[1] for r in dr):
            plans.append(("H", (top[0], c0), (bot[0], c0), m, Ww))
        elif all(bot[0] <= r <= bot[1] for r in dr):
            plans.append(("H", (bot[0], c0), (top[0], c0), m, Ww))

    def build(plan):
        axis, (dr0, dc0), (sr0, sc0), rh, rw = plan
        o2, s2 = list(ops), list(sels)
        # lay a clean bgc base over the damaged half (only cells that still hold
        # something else) so the in-place flip has nothing pre-existing to catch
        clear = [(r, c) for r in range(dr0, dr0 + rh) for c in range(dc0, dc0 + rw)
                 if cur[r, c] != bgc]
        if clear:
            o2.append(int(bgc))
            s2.append(sel_of(clear))
        # CopyI / Paste / Flip selections are FULL RECTANGLES (background included
        # on purpose): a half of the shape's bounding box is exactly a rectangle.
        o2.append(28)
        s2.append([sr0, sc0, rh - 1, rw - 1])          # grab the intact half from the input
        o2.append(30)
        s2.append([dr0, dc0, 0, 0])                    # drop it onto the damaged half
        o2.append(26 if axis == "V" else 27)           # and mirror it in place
        s2.append([dr0, dc0, rh - 1, rw - 1])
        return o2, s2

    chosen = None
    for plan in plans:
        o2, s2 = build(plan)
        g, _ = sim(o2, s2)
        if np.array_equal(g, O):
            chosen = (o2, s2)
            break

    if chosen is None:
        # fallback: no reflection matched — paint what the rule leaves to paint
        o2, s2 = list(ops), list(sels)
        need = {}
        for (r, c) in diff:
            need.setdefault(int(O[r, c]), []).append((r, c))
        for col, cl in need.items():
            o2.append(int(col))
            s2.append(sel_of(cl))
        chosen = (o2, s2)

    o2, s2 = chosen

    # --- prune ops that leave the grid untouched (CopyI works on the clipboard)
    g = I.copy()
    keep_ops, keep_sels = [], []
    prev = I.copy()
    for idx in range(len(o2)):
        gg, _ = sim(o2[:idx + 1], s2[:idx + 1])
        if o2[idx] == 28 or not np.array_equal(gg, prev):
            keep_ops.append(o2[idx])
            keep_sels.append(s2[idx])
        prev = gg
    gfin, _ = sim(keep_ops, keep_sels)
    if np.array_equal(gfin, O):
        o2, s2 = keep_ops, keep_sels

    o2.append(34)
    s2.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
    return o2, s2


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
