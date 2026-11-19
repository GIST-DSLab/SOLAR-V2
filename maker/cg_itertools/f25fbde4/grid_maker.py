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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if fgc is None:
        fgc = choice(remove(bgc, cols))
    h = unifint(diff_lb, diff_ub, (2, max_h))
    w = unifint(diff_lb, diff_ub, (2, max_w))
    ncd = unifint(diff_lb, diff_ub, (1, max(1, (min(15, h - 1) * min(15, w - 1)) // 2)))
    nc = min(max(1, ncd), (h - 1) * (w - 1) - 1)
    c = canvas(bgc, (h, w))
    bounds = asindices(canvas(-1, (min(15, h - 1), min(15, w - 1))))
    ch = choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc):
        cand = (bounds - shp) & mapply(neighbors, shp)
        if len(cand) == 0:
            break
        shp.add(choice(totuple(cand)))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    plcd = shift(shp, (loci, locj))
    gi = fill(c, fgc, plcd)
    go = compress(gi)
    go = upscale(go, 2)
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    # Reading: one axis at a time.  Pass 1 keeps only the columns that touch
    # the shape and doubles every kept column; pass 2 does the same with rows.
    # Lines are doubled far end first so no source line is overwritten before
    # it is copied.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    if examples:
        cnt = Counter()
        for ex_i, _ in examples:
            cnt.update(np.asarray(ex_i, dtype=int).flatten().tolist())
        cnt.update(I.flatten().tolist())
        bgc = cnt.most_common(1)[0][0]

    rows = [r for r in range(hi) if any(I[r, c] != bgc for c in range(wi))]
    cols = [c for c in range(wi) if any(I[r, c] != bgc for r in range(hi))]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    ops, sels = [], []

    def double_lines(G, n, length, axis, live=None):
        """Double each of the n lines (columns if axis==1, rows if axis==0)."""
        clip = None
        for j in range(n - 1, -1, -1):
            if axis == 1:
                src = [int(G[i, j]) for i in range(length)]
            else:
                src = [int(G[j, i]) for i in range(length)]
            for t in (2 * j, 2 * j + 1):
                cur = [int(G[i, t]) if axis == 1 else int(G[t, i]) for i in range(length)]
                # only the cells that survive the next canvas reshape matter
                idx = live if live is not None else range(length)
                if all(cur[i] == src[i] for i in idx):
                    continue
                # Paste never writes 0.  When this target line still holds
                # something where the source has 0 (0 is a real colour here),
                # wipe the whole target line segment to 0 first, then paste.
                if any(src[i] == 0 and cur[i] != 0 for i in idx):
                    lo, hi_ = min(idx), max(idx)
                    if axis == 1:
                        ops.append(0); sels.append([lo, t, hi_ - lo, 0])
                        G[lo:hi_ + 1, t] = 0
                    else:
                        ops.append(0); sels.append([t, lo, 0, hi_ - lo])
                        G[t, lo:hi_ + 1] = 0
                    cur = [int(G[i, t]) if axis == 1 else int(G[t, i]) for i in range(length)]
                if any(src[i] != 0 and cur[i] != src[i] for i in idx):
                    key = (axis, tuple(src))
                    if clip != key:
                        # CopyO the whole source line (full rectangle)
                        if axis == 1:
                            ops.append(29); sels.append([0, j, length - 1, 0])
                        else:
                            ops.append(29); sels.append([j, 0, 0, length - 1])
                        clip = key
                    if axis == 1:
                        ops.append(30); sels.append([0, t, 0, 0])
                    else:
                        ops.append(30); sels.append([t, 0, 0, 0])
                    for i in range(length):
                        if src[i] != 0:
                            if axis == 1:
                                G[i, t] = src[i]
                            else:
                                G[t, i] = src[i]

    def crop(G, r, c, hh, ww):
        # ARCLE CropGrid: transparent copy of the selected rectangle to (0,0)
        patch = np.zeros((hh, ww), dtype=int)
        sub = G[r:r + hh, c:c + ww]
        patch[:sub.shape[0], :sub.shape[1]] = sub
        G[:, :] = 0
        G[:hh, :ww] = patch

    G = np.zeros((30, 30), dtype=int)
    G[:hi, :wi] = I

    # ---- pass 1: columns -------------------------------------------------
    # keep only the columns the shape occupies (all rows; full rectangle)
    if c0 > 0:
        ops.append(33); sels.append([0, c0, hi - 1, w - 1])
        crop(G, 0, c0, hi, w)
    # double each column: column j -> columns 2j, 2j+1.  Copies that land to
    # the right of the canvas are taken in by the next canvas reshape.
    double_lines(G, w, hi, axis=1, live=range(r0, r1 + 1))

    # ---- pass 2: rows ----------------------------------------------------
    W2 = 2 * w
    if r0 > 0:
        # keep only the rows the shape occupies, over the full doubled width
        ops.append(33); sels.append([r0, 0, h - 1, W2 - 1])
        crop(G, r0, 0, h, W2)
    # set the canvas to the final 2h x 2w size (full rectangle)
    ops.append(33); sels.append([0, 0, 2 * h - 1, W2 - 1])
    crop(G, 0, 0, 2 * h, W2)
    # double each row: row i -> rows 2i, 2i+1
    double_lines(G, h, W2, axis=0)

    ops.append(34); sels.append([0, 0, 2 * h - 1, W2 - 1])
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
