"""
ARC Task: 72322fa7 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # rule is colour-agnostic (template completion by shape) -> only bgc must be shared
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = interval(0, 10, 1)
    hlo = min(10, max_h)
    wlo = min(10, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, max_h))
    w = unifint(diff_lb, diff_ub, (wlo, max_w))
    remcols = remove(bgc, cols)
    nobjs = unifint(diff_lb, diff_ub, (1, 4))
    nobjs = min(nobjs, len(remcols) // 2)
    nobjs = max(1, nobjs)
    ccols = sample(remcols, 2 * nobjs)
    cpairs = list(zip(ccols[:nobjs], ccols[nobjs:]))
    objs = []
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    for ca, cb in cpairs:
        oh = unifint(diff_lb, diff_ub, (1, 4))
        ow = unifint(diff_lb, diff_ub, (2 if oh == 1 else 1, 4))
        if choice((True, False)):
            oh, ow = ow, oh
        bounds = asindices(canvas(-1, (oh, ow)))
        obj = {choice(totuple(bounds))}
        ncells = randint(2, oh * ow)
        for k in range(ncells - 1):
            rem = (bounds - obj) & mapply(neighbors, obj)
            if len(rem) == 0:
                break
            obj.add(choice(totuple(rem)))
        objn = normalize(obj)
        objt = totuple(objn)
        if len(objt) < 2:
            continue
        apart = sample(objt, randint(1, len(objt) - 1))
        bpart = difference(objt, apart)
        obj = recolor(ca, set(apart)) | recolor(cb, set(bpart))
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: shift(objn, ij).issubset(inds))
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        gi = paint(gi, plcd)
        plcdi = toindices(plcd)
        inds = (inds - plcdi) - mapply(neighbors, plcdi)
        objs.append(obj)
    if len(objs) == 0:
        return generate(diff_lb, diff_ub, max_h, max_w, bgc)
    avgs = sum([len(o) for o in objs]) / len(objs)
    ub = max(1, int((h * w) // (avgs * 2)))
    noccs = unifint(diff_lb, diff_ub, (1, ub))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    go = tuple(e for e in gi)
    while tr < maxtr and succ < noccs:
        tr += 1
        obj = choice(objs)
        pal = list(palette(obj))
        if len(pal) != 2:
            continue
        ca, cb = pal
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
            go = paint(go, plcd)
            col = choice((ca, cb))
            gi = paint(gi, sfilter(plcd, lambda cij: cij[0] == col))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (measured from I only):
      * find the 2-colour 'key' objects in I (diagonally connected non-bg components
        whose palette has exactly 2 colours)
      * split each key into its two single-colour parts A and B
      * every place in I where part A occurs on its own is an incomplete copy of the key
        -> stamp part B at the matching offset (and symmetrically for part B)
    O is never inspected to decide what/where to paint.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # background = colour the generator fills the canvas with (sparse grids -> majority)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- connected components of non-background cells (diagonal connectivity) ---
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] != bgc:
                                seen[ny, nx] = True
                                q.append((ny, nx))
                comps.append(cells)

    # --- keys = components made of exactly two colours ---
    keys = [cl for cl in comps if len({int(I[y, x]) for y, x in cl}) == 2]

    ops, sels = [], []
    grid = I.copy()

    for cells in keys:
        kcols = sorted({int(I[y, x]) for y, x in cells})
        for ca in kcols:
            part = [(y, x) for y, x in cells if int(I[y, x]) == ca]
            partner = [(y, x) for y, x in cells if int(I[y, x]) != ca]
            if not part or not partner:
                continue
            cb = int(I[partner[0][0], partner[0][1]])
            # normalise both parts relative to the ulcorner of the searched part
            ar = min(y for y, _ in part)
            ac = min(x for _, x in part)
            prel = [(y - ar, x - ac) for y, x in part]
            brel = [(y - ar, x - ac) for y, x in partner]
            ph = max(y for y, _ in prel)
            pw = max(x for _, x in prel)
            # slide the single-colour part over I: every exact match is an occurrence
            for i in range(h - ph):
                for j in range(w - pw):
                    if all(I[i + y, j + x] == ca for y, x in prel):
                        tgt = [(i + y, j + x) for y, x in brel]
                        tgt = [(y, x) for y, x in tgt if 0 <= y < h and 0 <= x < w]
                        need = [(y, x) for y, x in tgt if grid[y, x] != cb]
                        if need:
                            # one op per occurrence: stamp the missing partner part
                            ops.append(int(cb))
                            sels.append(sel_of(need))
                            for y, x in need:
                                grid[y, x] = cb

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 72322fa7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 72322fa7"
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
                                f"for task 72322fa7"
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
                    f"Failed to build a complete episode for task 72322fa7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"72322fa7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
