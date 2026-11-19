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
from dsl import *
from utils import unifint
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Rule depends only on the presence/pattern of 2-colour keys, not on which colours;
    # fix the background for the episode.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    if bgc is None:
        bgc = choice(cols)
    remcols = remove(bgc, cols)
    nobjs = unifint(diff_lb, diff_ub, (1, 4))
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
            obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        objn = normalize(obj)
        objt = totuple(objn)
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
    avgs = sum([len(o) for o in objs]) / len(objs)
    ub = max(1, (h * w) // (avgs * 2))
    noccs = unifint(diff_lb, diff_ub, (1, ub))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    go = tuple(e for e in gi)
    while tr < maxtr and succ < noccs:
        tr += 1
        obj = choice(objs)
        ca, cb = list(palette(obj))
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


def derive_operations(I, O, examples=None):
    """
    Reading (taken from the code-golf solution): each two-colour object is a KEY.
    Every place where one of its single-colour parts occurs alone is an incomplete
    copy of that key, and the WHOLE key is stamped there. Stamping the whole key is
    written as CopyI(key's own cells) -> Paste at the copy's origin (the key is the
    unit, not only its missing colour). All matches of each key's majority part are
    stamped first, then those of the minority part. Everything is measured from I;
    O only sets the Submit extent.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # background = generator's canvas colour; confirmed against the demonstrations
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    if examples:
        votes = Counter()
        for ei, _ in examples:
            ei = np.asarray(ei, dtype=int)
            votes[Counter(ei.flatten().tolist()).most_common(1)[0][0]] += 1
        cand = votes.most_common(1)[0][0]
        if (I == cand).any():
            bgc = cand

    # 8-connected non-background components
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; cells = []
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] != bgc:
                                seen[ny, nx] = True; q.append((ny, nx))
                comps.append(cells)

    keys = [cl for cl in comps if len({int(I[y, x]) for y, x in cl}) == 2]

    # plan stamps from I: (key index, top-left origin of the whole key at that copy)
    major_stamps, minor_stamps = [], []
    for ki, cells in enumerate(keys):
        cnt = Counter(int(I[y, x]) for y, x in cells)
        mc = cnt.most_common(1)[0][0]
        kr = min(y for y, _ in cells); kc = min(x for _, x in cells)
        for part_is_major in (True, False):
            part = [(y, x) for y, x in cells if (int(I[y, x]) == mc) == part_is_major]
            pr = min(y for y, _ in part); pc = min(x for _, x in part)
            prel = [(y - pr, x - pc) for y, x in part]
            ph = max(y for y, _ in prel); pw = max(x for _, x in prel)
            col = int(I[part[0]])
            for i in range(h - ph):
                for j in range(w - pw):
                    if all(I[i + y, j + x] == col for y, x in prel):
                        origin = (i + kr - pr, j + kc - pc)
                        (major_stamps if part_is_major else minor_stamps).append((ki, origin))

    ops, sels = [], []
    grid = I.copy()
    clip_key = None
    for ki, (orr, occ) in major_stamps + minor_stamps:
        cells = keys[ki]
        kr = min(y for y, _ in cells); kc = min(x for _, x in cells)
        rel = [(y - kr, x - kc, int(I[y, x])) for y, x in cells]
        tgt = [(orr + a, occ + b, v) for a, b, v in rel]
        inside = [(y, x, v) for y, x, v in tgt if 0 <= y < h and 0 <= x < w]
        need = [(y, x, v) for y, x, v in inside if grid[y, x] != v]
        if not need:
            continue  # already a complete copy (e.g. the key itself) -> stamping would be a no-op
        paste_ok = (orr >= 0 and occ >= 0 and len(inside) == len(tgt)
                    and all(v != 0 for _, _, v in rel))
        if paste_ok:
            if clip_key != ki:
                ops.append(28); sels.append(sel_of(cells))     # CopyI: the key object's own cells
                clip_key = ki
            ops.append(30); sels.append(sel_of([(orr, occ)]))  # Paste: stamp whole key at this copy
            for y, x, v in inside:
                grid[y, x] = v
        else:
            # key contains colour 0 (Paste cannot write 0) or the stamp runs off the grid:
            # stamp the key colour by colour at this copy with Color ops
            for v in sorted({v for _, _, v in need}):
                cs = [(y, x) for y, x, vv in need if vv == v]
                ops.append(v); sels.append(sel_of(cs))
                for y, x in cs:
                    grid[y, x] = v

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
