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
import random
from collections import Counter

import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 3)]
    bgc, rcol, gcol = random.sample(cols, 3)
    return {"bgc": bgc, "rcol": rcol, "gcol": gcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, rcol=None, gcol=None, **kwargs) -> dict:
    cols = difference(interval(0, 10, 1), (2, 3))
    if bgc is None or rcol is None or gcol is None:
        bgc, rcol, gcol = sample(cols, 3)
    h = unifint(diff_lb, diff_ub, (11, max(11, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (11, max(11, min(30, max_w))))
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
    if unifint(diff_lb, diff_ub, (1, 100)) < 30:
        c = choice((2, 3))
        giobjs = objects(gi, F, T, T)
        goobjs = objects(go, F, T, T)
        gi = fill(gi, bgc, mfilter(giobjs, lambda o: c in palette(o)))
        go = fill(go, bgc, mfilter(goobjs, lambda o: c in palette(o)))
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Rule: two templates sit in the grid, each a multi-coloured shape keyed by one
    2 cell or one 3 cell.  Every lone 2 / 3 is a marker: stamp its template there,
    anchored on the key cell; the 2-template is stamped left-right mirrored.

    Reading vs. the code-golf entry: it is the SAME reading (template = the non-isolated
    component holding the key colour, markers = isolated cells of that colour, offsets
    taken relative to the key, x negated for key 2).  Nothing is carved differently, so
    the route stays template-by-template: copy the template out of the input once, then
    stamp it on every marker (flipping the fresh stamp in place for the 2-template).
    Where a copy would carry or hit foreign content (or the body colour is 0, which
    Copy/Paste cannot carry), the stamp's body is painted on exactly the cells it lands on.
    Everything is computed from I and the grid built so far; O is used only for nothing."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

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

    templates, markers = {}, {2: [], 3: []}
    for comp in comps:
        vals = set(int(I[y, x]) for y, x in comp)
        if len(comp) == 1:
            v = int(I[comp[0]])
            if v in (2, 3):
                markers[v].append(comp[0])
            continue
        for fc in (2, 3):
            if fc in vals:
                templates[fc] = comp

    cur = I.copy()
    ops, sels = [], []
    for fc in (2, 3):
        comp = templates.get(fc)
        if comp is None or not markers[fc]:
            continue
        mirror = (fc == 2)
        rs = [y for y, x in comp]
        cs = [x for y, x in comp]
        tr, tc = min(rs), min(cs)
        th, tw = max(rs) - tr + 1, max(cs) - tc + 1
        ar, ac = [(y, x) for y, x in comp if int(I[y, x]) == fc][0]
        lr, lc = ar - tr, ac - tc
        comp_set = set(comp)
        bodycol = int(I[[p for p in comp if p != (ar, ac)][0]])
        # clipboard = the template's bounding box in I: it must hold only template + background,
        # and the body colour must survive Copy (non-zero)
        clip = I[tr:tr + th, tc:tc + tw]
        clip_clean = all(I[y, x] == bgc for y in range(tr, tr + th) for x in range(tc, tc + tw)
                         if (y, x) not in comp_set)
        shape_cells = [(y - tr, x - tc) for y, x in comp]           # unmirrored, box-relative
        sym = np.array_equal(clip, clip[:, ::-1])
        do_flip = mirror and not sym
        copied = False

        for (mr, mc) in markers[fc]:
            r0 = mr - lr
            c0 = mc - (tw - 1 - lc) if mirror else mc - lc
            final_cells = [(r0 + dy, c0 + (tw - 1 - dx if mirror else dx)) for dy, dx in shape_cells]
            paste_cells = [(r0 + dy, c0 + dx) for dy, dx in shape_cells]
            box = [(y, x) for y in range(r0, r0 + th) for x in range(c0, c0 + tw)]

            def free(cells):
                return all(0 <= y < hi and 0 <= x < wi and (cur[y, x] == bgc or (y, x) == (mr, mc))
                           for y, x in cells)

            use_copy = bodycol != 0 and clip_clean and free(final_cells) and free(paste_cells)
            if use_copy and bgc != 0:
                use_copy = free(box)        # opaque paste writes the whole box
            if use_copy:
                if not copied:
                    ops.append(28); sels.append([tr, tc, th - 1, tw - 1])   # CopyI the template box
                    copied = True
                ops.append(30); sels.append([r0, c0, 0, 0])               # stamp it on the marker
                for dy in range(th):
                    for dx in range(tw):
                        if clip[dy, dx] != 0:
                            cur[r0 + dy, c0 + dx] = clip[dy, dx]
                if do_flip:
                    if bgc != 0:
                        # full rectangle intended: the whole landing box (template + bg) is the stamp
                        ops.append(26); sels.append([r0, c0, th - 1, tw - 1])
                        cur[r0:r0 + th, c0:c0 + tw] = cur[r0:r0 + th, c0:c0 + tw][:, ::-1].copy()
                    else:
                        # flip exactly the cells the paste landed on
                        ops.append(26); sels.append(sel_of(paste_cells))
                        vals = {(y, x): cur[y, x] for y, x in paste_cells}
                        for y, x in paste_cells:
                            cur[y, x] = 0
                        for y, x in paste_cells:
                            cur[y, 2 * c0 + tw - 1 - x] = vals[(y, x)]
            else:
                body = [p for p, (dy, dx) in zip(final_cells, shape_cells)
                        if (dy + tr, dx + tc) != (ar, ac)]
                ops.append(bodycol); sels.append(sel_of(body))            # paint the stamp's body
                for y, x in body:
                    cur[y, x] = bodycol

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
