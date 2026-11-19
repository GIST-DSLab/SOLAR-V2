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


# ----------------------------------------------------------------------------
# 1. colours: the rule ("complete the half-drawn copies of each two-colour key")
#    is colour-agnostic, so only the background has to be pinned for the episode.
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


# ----------------------------------------------------------------------------
# 2. generator (RE-ARC 72322fa7, 30 -> max_h/max_w, bgc injected)
# ----------------------------------------------------------------------------
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    remcols = remove(bgc, cols)
    nobjs = unifint(diff_lb, diff_ub, (1, 4))
    nobjs = min(nobjs, len(remcols) // 2)
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
    if len(objs) == 0:
        raise ValueError("no key objects placed")
    avgs = sum([len(o) for o in objs]) / len(objs)
    ub = max(1, (h * w) // (avgs * 2))
    noccs = unifint(diff_lb, diff_ub, (1, int(ub)))
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
    if succ == 0:
        raise ValueError("no partial occurrences placed")
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. derive_operations
#
# Rule (read off I alone):
#   * the "keys" are the diagonally-connected non-background components whose
#     palette holds exactly two colours;
#   * a key splits into two single-colour halves A and B;
#   * wherever half A appears on its own in I, that is a half-drawn copy of the
#     key -> the other half B belongs there, at the key's own relative offset
#     (and symmetrically for B).  O is never consulted.
#
# The route through that rule is chosen from the input itself - the anchor key's
# position, the background colour, the grid size, the number of keys - so the
# same grid always gives the same route while different grids are worked in
# different orders, from different corners, and with different stamping habits
# (one copy at a time / all missing cells of a shape at once / copy the key and
# stamp it).  No operation is added or kept that the answer can be reached
# without.
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # ---- diagonally connected non-background components -------------------
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

    keys = [cl for cl in comps if len({int(I[y, x]) for y, x in cl}) == 2]

    ops, sels = [], []
    if not keys:
        ops.append(34)
        sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    key_info = []
    for cells in keys:
        r0 = min(y for y, _ in cells)
        c0 = min(x for _, x in cells)
        r1 = max(y for y, _ in cells)
        c1 = max(x for _, x in cells)
        key_info.append({'cells': cells, 'ul': (r0, c0), 'lr': (r1, c1), 'size': len(cells)})

    # ---- route choices, all read out of the input -------------------------
    # anchor = the biggest key (ties: the topmost-leftmost one) - the one a
    # person's eye lands on first.
    anchor = max(range(len(key_info)),
                 key=lambda k: (key_info[k]['size'], -key_info[k]['ul'][0], -key_info[k]['ul'][1]))
    ar, ac = key_info[anchor]['ul']
    nkeys = len(key_info)

    key_mode = (ar + ac) % 3            # which key to deal with first
    sweep_mode = (bgc + ac) % 4         # which direction to sweep its copies
    part_desc = ((ar + w) % 2 == 1)     # which half of the pair to chase first
    stamp_mode = (bgc + h + nkeys) % 3  # one-at-a-time / all-at-once / copy the key

    idxs = list(range(nkeys))
    if key_mode == 0:                   # reading order
        idxs.sort(key=lambda k: key_info[k]['ul'])
    elif key_mode == 1:                 # from the bottom-right back up
        idxs.sort(key=lambda k: key_info[k]['ul'], reverse=True)
    else:                               # anchor first, then outward from it
        idxs.sort(key=lambda k: (max(abs(key_info[k]['ul'][0] - ar),
                                     abs(key_info[k]['ul'][1] - ac)), key_info[k]['ul']))

    def sweep(locs, ref_r, ref_c):
        if sweep_mode == 0:
            return sorted(locs)                                   # top-down, left-right
        if sweep_mode == 1:
            return sorted(locs, reverse=True)                     # bottom-up, right-left
        if sweep_mode == 2:
            return sorted(locs, key=lambda t: (t[1], t[0]))       # column by column
        return sorted(locs, key=lambda t: (max(abs(t[0] - ref_r), abs(t[1] - ref_c)), t))

    grid = I.copy()

    def paste_ok(dr, dc, template):
        """The key stamped whole at (dr, dc) may only land on background or on
        cells that already agree with it, and must change at least one cell."""
        th, tw = template.shape
        if dr < 0 or dc < 0 or dr + th > h or dc + tw > w:
            return False
        changed = False
        for y in range(th):
            for x in range(tw):
                v = int(template[y, x])
                if v == 0:                      # 0 is never copied nor pasted
                    continue
                g = int(grid[dr + y, dc + x])
                if g != v:
                    if g != bgc:
                        return False
                    changed = True
        return changed

    for k in idxs:
        info = key_info[k]
        cells = info['cells']
        kcols = sorted({int(I[y, x]) for y, x in cells})
        if part_desc:
            kcols = kcols[::-1]
        kr0, kc0 = info['ul']
        kr1, kc1 = info['lr']
        template = I[kr0:kr1 + 1, kc0:kc1 + 1]
        kh, kw = template.shape
        # stamping the key wholesale needs every cell of it to survive the
        # clipboard (0 is transparent to Copy/Paste).
        use_copy = (stamp_mode == 2) and (0 not in kcols) and (bgc != 0 or True)
        if 0 in kcols:
            use_copy = False
        clip_ready = False

        for ca in kcols:
            part = [(y, x) for y, x in cells if int(I[y, x]) == ca]
            partner = [(y, x) for y, x in cells if int(I[y, x]) != ca]
            if not part or not partner:
                continue
            cb = int(I[partner[0][0], partner[0][1]])
            pr = min(y for y, _ in part)
            pc = min(x for _, x in part)
            prel = [(y - pr, x - pc) for y, x in part]
            brel = [(y - pr, x - pc) for y, x in partner]
            ph = max(y for y, _ in prel)
            pw = max(x for _, x in prel)

            locs = []
            for i in range(h - ph):
                for j in range(w - pw):
                    if all(I[i + y, j + x] == ca for y, x in prel):
                        locs.append((i, j))
            locs = sweep(locs, pr, pc)

            merged = []
            for (i, j) in locs:
                tgt = [(i + y, j + x) for y, x in brel]
                tgt = [(y, x) for y, x in tgt if 0 <= y < h and 0 <= x < w]
                need = [(y, x) for y, x in tgt if int(grid[y, x]) != cb]
                if not need:
                    continue                      # already a complete copy
                dr, dc = i + (kr0 - pr), j + (kc0 - pc)
                if use_copy and paste_ok(dr, dc, template):
                    if not clip_ready:
                        # whole key rectangle, background included - that is
                        # exactly the stamp being picked up.
                        ops.append(28)
                        sels.append([kr0, kc0, kh - 1, kw - 1])
                        clip_ready = True
                    ops.append(30)
                    sels.append([dr, dc, 0, 0])
                    for y in range(kh):
                        for x in range(kw):
                            v = int(template[y, x])
                            if v != 0:
                                grid[dr + y, dc + x] = v
                elif stamp_mode == 1:
                    merged.extend(need)           # all of this half at once
                    for y, x in need:
                        grid[y, x] = cb
                else:
                    ops.append(cb)                # this copy, finished now
                    sels.append(sel_of(need))
                    for y, x in need:
                        grid[y, x] = cb
            if merged:
                ops.append(cb)
                sels.append(sel_of(merged))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
