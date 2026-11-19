"""
ARC Task: d8c310e9 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of  # noqa: F401  (all selections here are full rectangles)


def sample_colors(num_examples=None) -> dict:
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    cols = interval(0, 10, 1)
    mfs = (identity, rot90, rot180, rot270)
    fn = choice(mfs)
    if fn in (rot90, rot270):
        hmax, wmax = max_w, max_h
    else:
        hmax, wmax = max_h, max_w
    h = unifint(diff_lb, diff_ub, (3, max(3, hmax)))
    w = unifint(diff_lb, diff_ub, (10, max(10, wmax)))
    p = unifint(diff_lb, diff_ub, (2, (w - 1) // 3))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numc)
    obj = set()
    for j in range(p):
        numcells = unifint(diff_lb, diff_ub, (1, h - 1))
        for ii in range(h - 1, h - numcells - 1, -1):
            obj.add((choice(ccols), (ii, j)))
    gi = canvas(bgc, (h, w))
    minobj = obj | shift(obj, (0, p))
    addonw = randint(0, p)
    addon = sfilter(obj, lambda cij: cij[1][1] < addonw)
    fullobj = minobj | addon
    leftshift = randint(0, addonw)
    fullobj = shift(fullobj, (0, -leftshift))
    gi = paint(gi, fullobj)
    go = tuple(e for e in gi)
    for j in range(w // (2 * p) + 2):
        go = paint(go, shift(fullobj, (0, j * 2 * p)))
    gi = fn(gi)
    go = fn(go)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """Continue the striped block across the grid by copy-and-paste doubling.

    Frame, background, march direction and period are all read from I (O is never
    looked at).  Each round copies the whole stretch filled so far (a rectangle) and
    pastes it a whole number of periods further on, so the filled stretch roughly
    doubles per Paste.
    """
    A = np.asarray(I, dtype=int)
    H, W = A.shape
    full = [0, 0, H - 1, W - 1]

    # frame: the quarter turn that leaves the top row and right column pure background
    axis = {0: (0, 1), 3: (-1, 0), 2: (0, -1), 1: (1, 0)}
    kk = None
    for k in (0, 3, 2, 1):
        C = np.rot90(A, k)
        if len(set(C[0].tolist())) == 1 and len(set(C[:, -1].tolist())) == 1:
            kk = k
            break
    if kk is None:
        return [34], [full]
    C = np.rot90(A, kk)
    bgc = int(C[0, -1])
    dr, dc = axis[kk]
    cells = np.argwhere(C != bgc)
    if len(cells) == 0:
        return [34], [full]

    # period along the march axis (canonical frame: columns): the shortest slide
    # that maps every column of the block onto an identical column
    minc, maxc = int(cells[:, 1].min()), int(cells[:, 1].max())
    per = maxc - minc + 1
    for cand in range(1, maxc - minc + 1):
        if all((C[:, c] == C[:, c - cand]).all() for c in range(minc + cand, maxc + 1)):
            per = cand
            break

    # block bbox in the grid as given
    oc = np.argwhere(A != bgc)
    r0, r1 = int(oc[:, 0].min()), int(oc[:, 0].max())
    c0, c1 = int(oc[:, 1].min()), int(oc[:, 1].max())
    if dc != 0:
        a, b, L, d = c0, c1, W, dc
    else:
        a, b, L, d = r0, r1, H, dr

    def rect(lo, hi):  # stretch [lo, hi] along the axis, full block width across it
        if dc != 0:
            return [r0, lo, r1 - r0, hi - lo]
        return [lo, c0, hi - lo, c1 - c0]

    ops, sels = [], []
    g = A.copy()  # the working grid as it stands, to skip stamps that change nothing

    def cells_of(rc):
        r, c, hh, ww = rc
        return slice(r, r + hh + 1), slice(c, c + ww + 1)

    # the clipboard skips colour 0: if the block has black cells on a non-black
    # background, lay the strip still to be filled down black first
    blockvals = A[r0:r1 + 1, c0:c1 + 1]
    if d > 0:
        strip = (b + 1, L - 1) if b + 1 <= L - 1 else None
    else:
        strip = (0, a - 1) if a >= 1 else None
    if strip is None:
        return [34], [full]
    if bgc != 0 and (blockvals == 0).any():
        sr = rect(*strip)
        ops.append(0); sels.append(sr)  # full rectangle of the strip still to fill
        g[cells_of(sr)] = 0

    lo, hi = a, b  # filled stretch
    while (d > 0 and hi < L - 1) or (d < 0 and lo > 0):
        n = hi - lo + 1
        s = per * (n // per)
        if d > 0:
            src = (lo, min(hi, L - 1 - s))
            dst = lo + s
            newlo, newhi = lo, min(L - 1, hi + s)
        else:
            t = lo - s
            src = (lo, hi) if t >= 0 else (lo - t, hi)
            dst = max(t, 0)
            newlo, newhi = max(0, t), hi
        srect = rect(*src)
        clip = g[cells_of(srect)].copy()
        tr = rect(dst, dst)
        ch, cw = clip.shape
        after = g.copy()
        reg = after[tr[0]:tr[0] + ch, tr[1]:tr[1] + cw]
        reg[clip != 0] = clip[clip != 0]
        if not np.array_equal(after, g):
            copy_op = 28 if (src[0] >= a and src[1] <= b) else 29
            ops.append(copy_op); sels.append(srect)  # full rectangle: filled stretch
            ops.append(30); sels.append([tr[0], tr[1], 0, 0])
            g = after
        lo, hi = newlo, newhi

    ops.append(34); sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
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
                                f"for task d8c310e9"
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
                    f"Failed to build a complete episode for task d8c310e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d8c310e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
