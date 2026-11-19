"""
ARC Task: 3e980e27 (RE-ARC) — LLM-generated grid_maker
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
import random, math
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of

# The code-golf entry reads the task the same way the maker does: there are two
# templates, each keyed by one 2 cell or one 3 cell, and every other lone 2 or 3
# is a marker that gets that template stamped on it. The 2-template is stamped
# mirrored left-right. Both use the same unit (one template per key colour), the
# same order (one template at a time, one marker at a time) and the same fixed
# quantity (the key cell lands on the marker). So the operations are the
# maker's, with two fixes:
#   - a left-right symmetric 2-template gets no flip, because the flip would
#     not change the grid;
#   - a landing rectangle that holds anything besides its own marker sends that
#     template to the per-stamp paint route.


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 3)]
    bgc, rcol, gcol = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    # The generator sometimes wipes out every 2-object (or every 3-object).
    # That case stays possible, but at least two examples show both stamp rules.
    # The test case is drawn from the cases the examples showed.
    drops = [None] * n_ex
    for i in range(2, n_ex):
        if random.random() < 0.3:
            drops[i] = random.choice((2, 3))
    random.shuffle(drops)
    plan = [{"drop": d} for d in drops]
    plan.append({"drop": random.choice(drops)})
    return {"bgc": bgc, "rcol": rcol, "gcol": gcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, rcol=None, gcol=None,
             drop="unset") -> dict:
    cols = difference(interval(0, 10, 1), (2, 3))
    if bgc is None or rcol is None or gcol is None:
        bgc, rcol, gcol = sample(cols, 3)
    if drop == "unset":
        drop = choice((2, 3)) if unifint(diff_lb, diff_ub, (1, 100)) < 30 else None
    h = unifint(diff_lb, diff_ub, (11, max(11, max_h)))
    w = unifint(diff_lb, diff_ub, (11, max(11, max_w)))
    objs = []
    for (fixc, remc) in ((2, rcol), (3, gcol)):
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (2, 5))
        bounds = asindices(canvas(-1, (oh, ow)))
        obj = {choice(totuple(bounds))}
        ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(2, ncells), oh * ow)
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        obj = normalize(obj)
        fixp = choice(totuple(obj))
        rem = remove(fixp, obj)
        obj = {(fixc, fixp)} | recolor(remc, rem)
        objs.append(obj)
    robj, gobj = objs
    obj1, obj2 = sample(objs, 2)
    loci1 = randint(0, h - height(obj1) - height(obj2) - 1)
    locj1 = randint(0, w - width(obj1))
    loci2 = randint(loci1 + height(obj1) + 1, h - height(obj2))
    locj2 = randint(0, w - width(obj2))
    gi = canvas(bgc, (h, w))
    obj1p = shift(obj1, (loci1, locj1))
    obj2p = shift(obj2, (loci2, locj2))
    gi = paint(gi, obj1p)
    gi = paint(gi, obj2p)
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // int(1.5 * (len(robj) + len(gobj)))))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    robj = vmirror(robj)
    inds = ofcolor(gi, bgc) - (mapply(neighbors, toindices(obj1p)) | mapply(neighbors, toindices(obj2p)))
    go = tuple(e for e in gi)
    objopts = [robj, gobj]
    while tr < maxtr and succ < noccs:
        tr += 1
        obj = choice(objopts)
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            succ += 1
            inds = (inds - plcdi) - mapply(neighbors, plcdi)
            gi = paint(gi, sfilter(plcd, lambda cij: cij[0] in (2, 3)))
            go = paint(go, plcd)
    if drop in (2, 3):
        c = drop
        giobjs = objects(gi, F, T, T)
        goobjs = objects(go, F, T, T)
        gi = fill(gi, bgc, mfilter(giobjs, lambda o: c in palette(o)))
        go = fill(go, bgc, mfilter(goobjs, lambda o: c in palette(o)))
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Rule: the grid holds two templates. One is a shape keyed by a single 2 cell,
    the other a shape keyed by a single 3 cell. Every other lone 2 or 3 is a marker,
    and the matching template is stamped onto it with its key cell on the marker.
    The 2-template is stamped mirrored left-right.
    For each template: copy it out of the input once. Then, for each marker, paste
    it so the key lands on the marker. For a mirrored stamp, paste the plain copy
    and flip that landing rectangle."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    def sim(grid, ops, sels):
        g = grid.copy()
        clip = None
        for op, s in zip(ops, sels):
            r, c, h, w = s
            if op <= 9:
                g[r:r + h + 1, c:c + w + 1] = op
            elif op == 26:
                g[r:r + h + 1, c:c + w + 1] = np.fliplr(g[r:r + h + 1, c:c + w + 1])
            elif op == 28:
                clip = I[r:r + h + 1, c:c + w + 1].copy()
            elif op == 30:
                ch, cw = clip.shape
                for i in range(ch):
                    for j in range(cw):
                        if clip[i, j] != 0 and r + i < hi and c + j < wi:
                            g[r + i, c + j] = int(clip[i, j])
        return g

    # 8-connected non-background components of I
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            stack, comp = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] != bgc:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comps.append(comp)

    templates = {}
    for comp in comps:
        vals = set(int(I[y, x]) for y, x in comp)
        if len(vals) < 2:
            continue
        for fc in (2, 3):
            if fc in vals:
                templates[fc] = comp
    tcells = set()
    for comp in templates.values():
        tcells |= set(comp)
    markers = {2: [], 3: []}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v in (2, 3) and (r, c) not in tcells:
                markers[v].append((r, c))

    ops, sels = [], []
    cur = I.copy()
    for fc in (2, 3):
        comp = templates.get(fc)
        if comp is None or not markers[fc]:
            continue
        rs = [y for y, x in comp]
        cs = [x for y, x in comp]
        tr, tc = min(rs), min(cs)
        th, tw = max(rs) - tr + 1, max(cs) - tc + 1
        ar, ac = [(y, x) for y, x in comp if int(I[y, x]) == fc][0]
        lr, lc = ar - tr, ac - tc
        body = [(y - ar, x - ac) for y, x in comp if int(I[y, x]) != fc]
        bodycol = int(I[body[0][0] + ar, body[0][1] + ac])
        box = I[tr:tr + th, tc:tc + tw]
        mask = np.zeros((th, tw), dtype=bool)
        for (y, x) in comp:
            mask[y - tr, x - tc] = True
        # the 2-template is mirrored; a left-right symmetric template needs no flip
        mirror = (fc == 2) and not np.array_equal(np.where(mask, box, -1),
                                                  np.fliplr(np.where(mask, box, -1)))

        def stamp_cells(m):
            mr, mc = m
            return [(mr + dy, mc - dx if mirror else mc + dx) for dy, dx in body]

        expect = cur.copy()   # what the stamps look like by the rule (built from I only)
        for m in markers[fc]:
            for (y, x) in stamp_cells(m):
                expect[y, x] = bodycol

        # copy the template's full bbox rectangle out of the input, once
        c_ops, c_sels = [28], [[tr, tc, th - 1, tw - 1]]
        for (mr, mc) in markers[fc]:
            r0 = mr - lr
            c0 = mc - (tw - 1 - lc) if mirror else mc - lc
            if mirror and bgc == 0:
                # clear the landing rectangle (it holds the marker) so the flip moves only the stamp
                c_ops.append(0)
                c_sels.append([r0, c0, th - 1, tw - 1])
            c_ops.append(30)
            c_sels.append([r0, c0, 0, 0])
            if mirror:
                # mirror the newly stamped rectangle in place
                c_ops.append(26)
                c_sels.append([r0, c0, th - 1, tw - 1])
        # each landing rectangle must hold nothing but background and its own marker
        clean = True
        for (mr, mc) in markers[fc]:
            r0 = mr - lr
            c0 = mc - (tw - 1 - lc) if mirror else mc - lc
            sub = cur[r0:r0 + th, c0:c0 + tw]
            if int((sub != bgc).sum()) != 1:
                clean = False
        got = sim(cur, c_ops, c_sels)
        if clean and bodycol != 0 and np.array_equal(got, expect):
            ops += c_ops
            sels += c_sels
            cur = got
            continue

        # Copy/Paste cannot carry a body colour of 0, and it must not run when a
        # landing rectangle holds other content. In those cases paint each stamp's
        # body instead, one stamp at a time, one row run at a time.
        p_ops, p_sels = [], []
        for m in markers[fc]:
            rows = {}
            for (y, x) in stamp_cells(m):
                rows.setdefault(y, []).append(x)
            for y in sorted(rows):
                xs = sorted(rows[y])
                start = prev = xs[0]
                for x in xs[1:] + [None]:
                    if x is not None and x == prev + 1:
                        prev = x
                        continue
                    p_ops.append(bodycol)
                    p_sels.append(sel_of([(y, k) for k in range(start, prev + 1)]))
                    if x is not None:
                        start = prev = x
        ops += p_ops
        sels += p_sels
        cur = expect

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3e980e27"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3e980e27"
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
                                f"for task 3e980e27"
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
                    f"Failed to build a complete episode for task 3e980e27 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3e980e27-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
