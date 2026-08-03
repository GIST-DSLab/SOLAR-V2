"""
ARC Task: f25fbde4 (RE-ARC) — LLM-generated grid_maker
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
    # both roles get a visible color: the drawing is duplicated with Copy/Paste
    # and ARCLE reads color 0 as "nothing there"
    cols = list(range(1, 10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    h = unifint(diff_lb, diff_ub, (2, max_h))
    w = unifint(diff_lb, diff_ub, (2, max_w))
    # output is the drawing's box at double size, so keep that box small enough
    max_oh = max(1, min(15, h - 1, max_h // 2))
    max_ow = max(1, min(15, w - 1, max_w // 2))
    if max_oh * max_ow >= 2:
        ncd = unifint(diff_lb, diff_ub, (1, max(1, (max_oh * max_ow) // 2)))
        nc = min(max(1, ncd), max_oh * max_ow - 1)
    else:
        nc = 0
    c = canvas(bgc, (h, w))
    bounds = asindices(canvas(-1, (max_oh, max_ow)))
    ch = choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc):
        candidates = totuple((bounds - shp) & mapply(neighbors, shp))
        if not candidates:
            break
        shp.add(choice(candidates))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    plcd = shift(shp, (loci, locj))
    gi = fill(c, fgc, plcd)
    go = upscale(compress(gi), 2)
    return {'input': gi, 'output': go}


def _replay(I, ops, sels, ho, wo):
    """Faithful ARCLE replay of the ops used here (0, 28, 29, 30, 33)."""
    H, W = max(I.shape[0], ho), max(I.shape[1], wo)
    inp = np.zeros((H, W), dtype=int)
    inp[:I.shape[0], :I.shape[1]] = I
    idim = [I.shape[0], I.shape[1]]
    grid = inp.copy()
    gdim = [I.shape[0], I.shape[1]]
    clip = np.zeros((H, W), dtype=int)
    cdim = [0, 0]
    for op, (r, c, h, w) in zip(ops, sels):
        x0, y0, x1, y1 = r, c, r + h, c + w
        if op == 0:
            grid[x0:x1 + 1, y0:y1 + 1] = 0
        elif op in (28, 29):
            src, sdim = (inp, idim) if op == 28 else (grid, gdim)
            if x1 >= sdim[0] or y1 > sdim[1]:
                continue
            clip[:] = 0
            cdim[:] = [x1 - x0 + 1, y1 - y0 + 1]
            patch = src[x0:x1 + 1, y0:y1 + 1]
            np.copyto(clip[:cdim[0], :cdim[1]], patch, where=(patch != 0))
        elif op == 30:
            if cdim[0] == 0 or cdim[1] == 0 or x0 >= H or y0 >= W:
                continue
            ex, ey = min(x0 + cdim[0], H), min(y0 + cdim[1], W)
            patch = clip[:cdim[0], :cdim[1]][:ex - x0, :ey - y0]
            np.copyto(grid[x0:ex, y0:ey], patch, where=(patch > 0))
        elif op == 33:
            gh, gw = x1 - x0 + 1, y1 - y0 + 1
            reg = grid[x0:x1 + 1, y0:y1 + 1]
            patch = np.zeros((gh, gw), dtype=int)
            np.copyto(patch, reg, where=(reg != 0))
            grid[:] = 0
            grid[:gh, :gw] = patch
            gdim[:] = [gh, gw]
    return grid[:gdim[0], :gdim[1]]


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background = color of a plain row; the drawing never spans a whole row
    bgc = None
    for r in range(hi):
        if int(I[r].min()) == int(I[r].max()):
            bgc = int(I[r, 0])
            break
    if bgc is None:
        bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    mask = I != bgc
    rr = np.where(mask.any(axis=1))[0]
    cc = np.where(mask.any(axis=0))[0]
    rmin, rmax = int(rr[0]), int(rr[-1])
    cmin, cmax = int(cc[0]), int(cc[-1])
    bh, bw = rmax - rmin + 1, cmax - cmin + 1

    setup, body = [], []
    if (rmin, cmin) != (0, 0):
        # drop the empty margin: keep the drawing's bounding box
        setup.append((33, [rmin, cmin, bh - 1, bw - 1]))
        g = I[rmin:rmax + 1, cmin:cmax + 1].copy()
    else:
        g = I.copy()                    # the drawing already starts in the corner
    if g.shape != (2 * bh, 2 * bw):
        # open the canvas to twice the drawing's size
        setup.append((33, [0, 0, 2 * bh - 1, 2 * bw - 1]))
        gg = np.zeros((2 * bh, 2 * bw), dtype=int)
        a = min(g.shape[0], 2 * bh)
        b = min(g.shape[1], 2 * bw)
        gg[:a, :b] = g[:a, :b]
        g = gg
    clip = [None]

    def stretch(lines, span, vertical):
        # each line of the drawing is stamped onto the two lines that replace it
        for i in lines:
            # rows are read off the input, columns off the row-doubled grid
            src = (g[:span, i] if vertical else I[rmin + i, cmin:cmin + span]).copy()
            for t in (2 * i, 2 * i + 1):
                if t == i:
                    continue                        # this line already sits there
                cur = g[:span, t] if vertical else g[t, :span]
                if np.array_equal(cur, src):
                    continue                        # already identical
                stamp = bool(np.any((src != 0) & (cur != src)))
                # a stamp is see-through, so blank what it cannot cover itself
                gap = (src == 0) & (cur != 0)
                k = 0
                while k < span:
                    if gap[k]:
                        e = k
                        while e + 1 < span and gap[e + 1]:
                            e += 1
                        body.append((0, [k, t, e - k, 0] if vertical else [t, k, 0, e - k]))
                        cur[k:e + 1] = 0
                        k = e + 1
                    else:
                        k += 1
                if not stamp:
                    continue
                key = (vertical, tuple(src.tolist()))
                if clip[0] != key:                  # clipboard may still hold it
                    if vertical:
                        body.append((29, [0, i, span - 1, 0]))
                    else:
                        body.append((28, [rmin + i, cmin, 0, span - 1]))
                    clip[0] = key
                body.append((30, [0, t, 0, 0] if vertical else [t, 0, 0, 0]))
                np.copyto(cur, src, where=(src != 0))

    # double every row, then double every column of the result
    stretch(range(bh), bw, False)
    # rightmost column first, so the columns still to be read stay untouched
    stretch(range(bw - 1, -1, -1), 2 * bh, True)

    steps = setup + body

    def hoist():
        # a grab changes nothing on the grid: keep each next to the stamp it feeds
        j = 0
        while j < len(steps):
            if steps[j][0] in (28, 29):
                k = j
                while k > 0 and steps[k - 1][0] not in (28, 29, 30):
                    a, b = steps[j][1], steps[k - 1][1]
                    apart = (a[0] + a[2] < b[0] or b[0] + b[2] < a[0]
                             or a[1] + a[3] < b[1] or b[1] + b[3] < a[1])
                    # a row grab reads the input, which no step ever touches
                    if steps[j][0] == 28 or (steps[k - 1][0] == 0 and apart):
                        k -= 1
                    else:
                        break
                if k != j:
                    steps.insert(k, steps.pop(j))
            j += 1

    def prune():
        # a see-through stamp can leave an earlier step with nothing left to do
        dropped = False
        for k in range(len(steps) - 1, -1, -1):
            trial = steps[:k] + steps[k + 1:]
            if np.array_equal(_replay(I, [o for o, _ in trial],
                                      [s for _, s in trial], ho, wo), O):
                steps.pop(k)
                dropped = True
        return dropped

    while True:
        hoist()
        if not prune():
            break
    assert np.array_equal(
        _replay(I, [o for o, _ in steps], [s for _, s in steps], ho, wo), O), "mismatch"
    return [o for o, _ in steps] + [34], [s for _, s in steps] + [[0, 0, ho - 1, wo - 1]]


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
                        f"num_examples+1 ({num_examples + 1}) for task f25fbde4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f25fbde4"
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
                                f"for task f25fbde4"
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
                    f"Failed to build a complete episode for task f25fbde4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f25fbde4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
