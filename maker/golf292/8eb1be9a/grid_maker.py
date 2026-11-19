"""
ARC Task: 8eb1be9a (RE-ARC) — LLM-generated grid_maker
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
from dsl import *
from utils import *
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    h = unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = unifint(diff_lb, diff_ub, (4, max(4, max_w)))
    oh = unifint(diff_lb, diff_ub, (2, h // 3))
    ow = unifint(diff_lb, diff_ub, (2, w))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncells = unifint(diff_lb, diff_ub, (2, (oh * ow) // 3 * 2))
    obj = normalize(frozenset(sample(totuple(bounds), ncells)))
    oh, ow = shape(obj)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, ncols)
    obj = frozenset({(choice(ccols), ij) for ij in obj})
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    obj = shift(obj, (loci, locj))
    c = canvas(bgc, (h, w))
    gi = paint(c, obj)
    go = paint(c, obj)
    for k in range(h // oh + 1):
        go = paint(go, shift(obj, (-oh * k, 0)))
        go = paint(go, shift(obj, (oh * k, 0)))
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # background: the object covers at most ~2/9 of the canvas, and every
    # demonstration input shares the same background -> majority colour.
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ei, _ in examples:
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    cells = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    if not cells:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
    oh = r1 - r0 + 1
    ow = c1 - c0 + 1

    # landing spots of the object's 0-coloured cells (fixed by one Color0 at the end)
    Z = np.zeros((h, w), bool)
    if bgc != 0:
        for k in range(-(h // oh) - 1, h // oh + 2):
            for (r, c) in cells:
                if I[r, c] == 0 and 0 <= r + k * oh < h:
                    Z[r + k * oh, c] = True

    # Python-side mirror of the working grid / clipboard (only to skip no-op ops)
    G = I.copy()
    clip = {"arr": None, "src": None, "len": 0}
    pending = []   # a copy op is emitted only together with a paste that changes the grid
    prev = {}

    def copy(kind, src, ln):
        if not pending:
            prev.clear(); prev.update(clip)
        base = I if kind == 28 else G
        clip["arr"] = base[src:src + ln, c0:c1 + 1].copy()
        clip["src"] = src
        clip["len"] = ln
        pending[:] = [(kind, [src, c0, ln - 1, ow - 1])]   # band rows src..src+ln-1 of the object columns

    def apply(arr, t):
        newG = G.copy()
        for i in range(arr.shape[0]):
            rr = t + i
            if rr >= h:
                break
            for j in range(arr.shape[1]):
                if arr[i, j] != 0:
                    newG[rr, c0 + j] = arr[i, j]
        return newG

    def paste(t):
        newG = apply(clip["arr"], t)
        if np.array_equal(newG[~Z], G[~Z]):
            return                       # nothing visible beyond what the final Color0 settles
        if pending and pending[0][0] == 29 and prev.get("arr") is not None:
            oldG = apply(prev["arr"], t)
            if np.array_equal(oldG[~Z], newG[~Z]):
                clip.update(prev); pending.clear(); newG = oldG   # old clipboard already does it
        for k, s in pending:
            ops.append(k); sels.append(s)
        pending.clear()
        ops.append(30); sels.append([t, c0, 0, 0])   # paste the band block at row t
        G[:, :] = newG

    # ---- downward: repeat the object band, doubling the copied block ----
    copy(28, r0, oh)            # the object's band (CopyI of its bbox)
    F = oh                      # rows r0 .. r0+F-1 already hold whole periods
    while r0 + F < h:
        R = h - (r0 + F)
        if R > clip["len"] and F > clip["len"]:
            copy(29, r0, F)     # grab everything filled so far
        paste(r0 + F)
        F += clip["len"]

    # ---- upward: fill rows 0 .. s-1 from the periodic part below ----
    s = r0
    while s > 0:
        t_reuse = None
        if clip["arr"] is not None:
            Lc = clip["len"]
            if clip["src"] >= s:
                for t in range(0, s):
                    if (t - clip["src"]) % oh == 0 and t + Lc >= s:
                        t_reuse = t
                        break
        t_new = None
        for t in range(0, s):
            ln = s - t
            src = s + ((t - s) % oh)
            if src + ln - 1 <= h - 1:
                t_new = (t, src, ln)
                break
        if t_reuse is not None and (t_new is None or t_reuse <= t_new[0]):
            paste(t_reuse)
            s = t_reuse
        else:
            t, src, ln = t_new
            copy(29, src, ln)
            paste(t)
            s = t

    # ---- object cells coloured 0 cannot be carried by Paste: stamp them ----
    if bgc != 0:
        zc = [(r, c) for (r, c) in cells if I[r, c] == 0]
        z = []
        for k in range(-(h // oh) - 1, h // oh + 2):
            for (r, c) in zc:
                rr = r + k * oh
                if 0 <= rr < h and G[rr, c] != 0:
                    z.append((rr, c))
        if z:
            ops.append(0); sels.append(sel_of(z))   # every landing spot of the object's 0-cells
            for rr, c in z:
                G[rr, c] = 0

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8eb1be9a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8eb1be9a"
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
                                f"for task 8eb1be9a"
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
                    f"Failed to build a complete episode for task 8eb1be9a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8eb1be9a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
