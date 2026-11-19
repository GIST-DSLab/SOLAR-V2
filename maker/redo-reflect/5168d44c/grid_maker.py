"""
ARC Task: 5168d44c (RE-ARC) — LLM-generated grid_maker
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

# the dot-chain direction is a discrete structural variant -> plan it per episode
DIRECS = [{"direc": "down"}, {"direc": "right"}, {"direc": "unity"}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    dotcol = random.choice(rem)
    rem = [c for c in rem if c != dotcol]
    boxcol = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(DIRECS):
        examples = [dict(v) for v in DIRECS]
        examples += [dict(random.choice(DIRECS)) for _ in range(n_ex - len(DIRECS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(DIRECS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "dotcol": dotcol, "boxcol": boxcol, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotcol, boxcol, direc=None) -> dict:
    if direc is None:
        direc = random.choice([v["direc"] for v in DIRECS])
    hub = max(7, min(int(max_h), 30))
    wub = max(7, min(int(max_w), 30))
    h = _unifint(diff_lb, diff_ub, (7, hub))
    w = _unifint(diff_lb, diff_ub, (7, wub))
    doth = _unifint(diff_lb, diff_ub, (1, h // 3))
    dotw = _unifint(diff_lb, diff_ub, (1, w // 3))
    borderh = _unifint(diff_lb, diff_ub, (1, h // 4))
    borderw = _unifint(diff_lb, diff_ub, (1, w // 4))
    dvr, dvc = {"down": (1, 0), "right": (0, 1), "unity": (1, 1)}[direc]
    hi_i = h - doth - 1 if direc == "right" else h - doth - borderh - 1
    hi_j = w - dotw - 1 if direc == "down" else w - dotw - borderw - 1
    loci = random.randint(0, max(0, hi_i))
    locj = random.randint(0, max(0, hi_j))
    offr = dvr * (doth + borderh)
    offc = dvc * (dotw + borderw)

    gi = np.full((h, w), bgc, dtype=int)
    for k in range(-15, 16):                      # periodic chain of identical dots
        rr, cc = loci + k * offr, locj + k * offc
        for a in range(rr, rr + doth):
            for b in range(cc, cc + dotw):
                if 0 <= a < h and 0 <= b < w:
                    gi[a, b] = dotcol
    box = []                                      # ring around the starter dot
    for a in range(loci - borderh, loci + doth + borderh):
        for b in range(locj - borderw, locj + dotw + borderw):
            if loci <= a < loci + doth and locj <= b < locj + dotw:
                continue
            box.append((a, b))
    go = gi.copy()
    for (a, b) in box:                            # output: ring around the NEXT dot
        if 0 <= a + offr < h and 0 <= b + offc < w:
            go[a + offr, b + offc] = boxcol
    for (a, b) in box:                            # input: ring around the starter dot
        if 0 <= a < h and 0 <= b < w:
            gi[a, b] = boxcol
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    """The box jumps to the next dot of the chain. Equivalently: every cell takes the
    value of its mirror partner across the mid-line/mid-point between the box's dot and
    the next dot (the dot chain is invariant under that reflection, the box is not).
    So the route performs that reflection with FlipV/FlipH on the largest on-grid region
    that is symmetric about it, and then paints only what the reflection genuinely cannot
    reach: box cells whose mirror partner lay off the grid, and old-box cells whose
    mirror partner lay off the grid (so they were never carried away)."""
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    present = sorted(set(I.flatten().tolist()))

    # ---- read the structure off I: the box color is the one whose cells fill their
    #      bounding box except for one solid rectangular hole (the starter dot).
    parse = None
    for boxcol in present:
        rs, cs = np.nonzero(I == boxcol)
        if len(rs) == 0:
            continue
        r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
        sub = I[r0:r1 + 1, c0:c1 + 1]
        hole = (sub != boxcol)
        if not hole.any():
            continue
        hr, hc = np.nonzero(hole)
        hh = int(hr.max() - hr.min() + 1)
        hw = int(hc.max() - hc.min() + 1)
        if int(hole.sum()) != hh * hw:
            continue
        vals = set(sub[hole].tolist())
        if len(vals) != 1:
            continue
        dotcol = int(vals.pop())
        di, dj = r0 + int(hr.min()), c0 + int(hc.min())
        dh, dw = hh, hw
        # one of the two border sides is always unclipped, so max() recovers its true size
        bh = max(di - r0, r1 - (di + dh - 1))
        bw = max(dj - c0, c1 - (dj + dw - 1))
        if bh < 1 or bw < 1:
            continue
        dotcells = set((int(a), int(b)) for a, b in zip(*np.nonzero(I == dotcol)))
        for (offr, offc) in ((dh + bh, 0), (0, dw + bw), (dh + bh, dw + bw)):
            pred = set()
            for k in range(-40, 41):
                rr, cc = di + k * offr, dj + k * offc
                for a in range(rr, rr + dh):
                    for b in range(cc, cc + dw):
                        if 0 <= a < H and 0 <= b < W:
                            pred.add((a, b))
            if pred == dotcells:                  # this spacing explains every dot
                parse = (boxcol, dotcol, di, dj, dh, dw, bh, bw, offr, offc)
                break
        if parse is not None:
            break

    boxcol, dotcol, di, dj, dh, dw, bh, bw, offr, offc = parse
    others = [c for c in present if c not in (boxcol, dotcol)]
    if len(others) == 1:
        bgc = others[0]
    else:
        cnt = Counter(I.flatten().tolist())
        for c in (boxcol, dotcol):
            cnt.pop(c, None)
        bgc = cnt.most_common(1)[0][0]

    rt, rb = di - bh, di + dh + bh - 1             # true (unclipped) ring bbox
    cl, cr = dj - bw, dj + dw + bw - 1

    def ring_at(shr, shc):
        out = set()
        for a in range(rt + shr, rb + shr + 1):
            for b in range(cl + shc, cr + shc + 1):
                if di + shr <= a < di + shr + dh and dj + shc <= b < dj + shc + dw:
                    continue
                if 0 <= a < H and 0 <= b < W:
                    out.add((a, b))
        return out

    src = ring_at(0, 0)
    dst = ring_at(offr, offc)

    ops, sels = [], []
    allcells = src | dst
    rows = [r for r, _ in allcells]
    cols = [c for _, c in allcells]

    def sym_span(lo_need, hi_need, S, N):
        """widest on-grid interval [a, S-a] centred on the reflection axis"""
        a = min(lo_need, S - hi_need)
        a = max(a, 0, S - (N - 1))
        return a, S - a

    if offr > 0:
        ra, rbb = sym_span(min(rows), max(rows), rt + rb + offr, H)
    else:
        ra, rbb = min(rows), max(rows)
    if offc > 0:
        ca, cbb = sym_span(min(cols), max(cols), cl + cr + offc, W)
    else:
        ca, cbb = min(cols), max(cols)

    g = I.copy()
    # bbox selection: the reflection acts on this WHOLE rectangle, background included
    rect = [ra, ca, rbb - ra, cbb - ca]
    if offr > 0:
        ops.append(27); sels.append(list(rect))    # FlipV: reflect across the horizontal mid-line
        g[ra:rbb + 1, ca:cbb + 1] = np.flipud(g[ra:rbb + 1, ca:cbb + 1])
    if offc > 0:
        ops.append(26); sels.append(list(rect))    # FlipH: reflect across the vertical mid-line
        g[ra:rbb + 1, ca:cbb + 1] = np.fliplr(g[ra:rbb + 1, ca:cbb + 1])

    def components(cells):
        cells = set(cells)
        out = []
        while cells:
            seed = min(cells)
            comp, stack = {seed}, [seed]
            cells.discard(seed)
            while stack:
                r, c = stack.pop()
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (nr, nc) in cells:
                        cells.discard((nr, nc))
                        comp.add((nr, nc))
                        stack.append((nr, nc))
            out.append(sorted(comp))
        return out

    # finish the box arms the reflection could not fetch (their partner was off-grid)
    for comp in components([p for p in sorted(dst) if g[p] != boxcol]):
        ops.append(int(boxcol)); sels.append(sel_of(comp))
        for p in comp:
            g[p] = boxcol
    # clear the old-box remains the reflection could not carry away
    for comp in components([p for p in sorted(src - dst) if g[p] != bgc]):
        ops.append(int(bgc)); sels.append(sel_of(comp))
        for p in comp:
            g[p] = bgc

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 5168d44c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5168d44c"
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
                                f"for task 5168d44c"
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
                    f"Failed to build a complete episode for task 5168d44c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5168d44c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
