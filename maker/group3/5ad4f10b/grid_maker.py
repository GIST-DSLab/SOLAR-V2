"""
ARC Task: 5ad4f10b (RE-ARC) — LLM-generated grid_maker
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
    bgc, noisec, objc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, objc) -> dict:
    from dsl import (apply, lbind, shift, asindices, canvas, totuple, sfilter, normalize,
                     shape, fill, asobject, upscale, replace, paint, ofcolor)
    from utils import unifint
    from random import choice, sample, randint
    nbh = {(0, 0), (1, 0), (0, 1), (1, 1)}
    nbhs = apply(lbind(shift, nbh), {(0, 0), (-1, 0), (0, -1), (-1, -1)})
    lim_h = max(2, min(6, (max_h - 2) // 2))
    lim_w = max(2, min(6, (max_w - 2) // 2))
    oh = unifint(diff_lb, diff_ub, (2, lim_h))
    ow = unifint(diff_lb, diff_ub, (2, lim_w))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(1, ncells), oh * ow - 1)
    obj = set(sample(totuple(bounds), ncells))
    while len(sfilter(obj, lambda ij: sum([len(obj & shift(nbh, ij)) < 4 for nbh in nbhs]) > 0)) == 0:
        ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
        ncells = choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow - 1)
        obj = set(sample(totuple(bounds), ncells))
    obj = normalize(obj)
    if len(obj) == shape(obj)[0] * shape(obj)[1]:
        # a solid rectangle has no unique scale factor -> ambiguous; resample
        return generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, objc)
    oh, ow = shape(obj)
    go = canvas(bgc, (oh, ow))
    go = fill(go, noisec, obj)
    fub = min(28 // oh, 28 // ow, (max_h - 2) // oh, (max_w - 2) // ow)
    fac = unifint(diff_lb, diff_ub, (2, max(2, fub)))
    gobj = asobject(upscale(replace(go, noisec, objc), fac))
    oh, ow = shape(gobj)
    h = unifint(diff_lb, diff_ub, (oh + 2, max(oh + 2, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (ow + 2, max(ow + 2, min(30, max_w))))
    loci = randint(1, h - oh - 1)
    locj = randint(1, w - ow - 1)
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shift(gobj, (loci, locj)))
    cands = ofcolor(gi, bgc)
    namt = unifint(diff_lb, diff_ub, (2, max(2, len(cands) // 4)))
    noise = sample(totuple(cands), namt)
    gi = fill(gi, noisec, noise)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background = dominant colour of the border ring (the object never touches it)
    border = I[0, :].tolist() + I[-1, :].tolist() + I[:, 0].tolist() + I[:, -1].tolist()
    bgc = Counter(border).most_common(1)[0][0]
    others = [int(c) for c in np.unique(I).tolist() if int(c) != bgc]

    # object colour = the colour built of solid blocks (no isolated cells), noise is scattered
    def block_score(c):
        cs = set(map(tuple, np.argwhere(I == c).tolist()))
        iso = sum(1 for (r, cc) in cs
                  if not any((r + dr, cc + dc) in cs for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))))
        return (1 if iso == 0 else 0, len(cs))

    objc = max(others, key=block_score)
    rest = [c for c in others if c != objc]
    if rest:
        noisec = rest[0]
    else:  # read the noise colour from the demonstrations (episode-fixed colours)
        noisec = None
        for ei, _eo in (examples or []):
            ei = np.asarray(ei, dtype=int)
            cand = [int(v) for v in np.unique(ei).tolist() if int(v) not in (bgc, objc)]
            if cand:
                noisec = cand[0]
                break
        if noisec is None:
            noisec = (bgc + 1) % 10 if (bgc + 1) % 10 != objc else (bgc + 2) % 10

    # object bbox and upscale factor
    cells = np.argwhere(I == objc)
    r0, c0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    r1, c1 = int(cells[:, 0].max()), int(cells[:, 1].max())
    H, W = r1 - r0 + 1, c1 - c0 + 1
    M = (I[r0:r1 + 1, c0:c1 + 1] == objc)
    fac = 1
    for k in range(min(H, W), 0, -1):
        if H % k or W % k:
            continue
        small = M.reshape(H // k, k, W // k, k).any(axis=(1, 3))
        if np.array_equal(small.repeat(k, 0).repeat(k, 1), M):
            fac = k
            break
    ph, pw = H // fac, W // fac

    G = I.copy()
    ops, sels = [], []

    def emit(op, sel, cells_):
        ops.append(op); sels.append(sel)
        if op <= 9:
            for (r, c) in cells_:
                G[r, c] = op

    # 1. noise specks lying inside the object's frame are not part of the picture -> background
    inside_noise = [(int(r), int(c)) for r, c in np.argwhere(I == noisec)
                    if r0 <= r <= r1 and c0 <= c <= c1]
    if inside_noise:
        emit(int(bgc), sel_of(inside_noise), inside_noise)

    # 2. colour swap: the whole big object takes the noise colour (all cells of objc)
    objcells = [(int(r), int(c)) for r, c in cells]
    emit(int(noisec), sel_of(objcells), objcells)

    zero_role = 0 in (bgc, noisec)

    def strip_copy(src_cells, dst_cells, src_sel, dst_origin, dst_sel):
        src_vals = [G[r, c] for r, c in src_cells]
        dst_vals = [G[r, c] for r, c in dst_cells]
        if src_vals == dst_vals:
            return
        if zero_role and any(sv == 0 and dv != 0 for sv, dv in zip(src_vals, dst_vals)):
            # paste is transparent for 0: clear the whole destination strip (a full row/column
            # segment rectangle) first
            emit(0, dst_sel, dst_cells)
        if [G[r, c] for r, c in dst_cells] == src_vals:
            return
        ops.append(29); sels.append(src_sel)      # CopyO the sampled strip (full rectangle strip)
        ops.append(30); sels.append(dst_origin)   # Paste it onto the compressed position
        for (sr, sc), (dr, dc) in zip(src_cells, dst_cells):
            if G[sr, sc] != 0:
                G[dr, dc] = G[sr, sc]

    # 3. squeeze rows: block-row i (its first row r0+i*fac) is copied into row r0+i
    for i in range(1, ph):
        sr, dr = r0 + i * fac, r0 + i
        src = [(sr, c) for c in range(c0, c1 + 1)]
        dst = [(dr, c) for c in range(c0, c1 + 1)]
        strip_copy(src, dst, [sr, c0, 0, W - 1], [dr, c0, 0, 0], [dr, c0, 0, W - 1])

    # 4. squeeze columns of the now ph-tall band: column c0+j*fac is copied into column c0+j
    for j in range(1, pw):
        sc, dc = c0 + j * fac, c0 + j
        src = [(r, sc) for r in range(r0, r0 + ph)]
        dst = [(r, dc) for r in range(r0, r0 + ph)]
        strip_copy(src, dst, [r0, sc, ph - 1, 0], [r0, dc, 0, 0], [r0, dc, ph - 1, 0])

    # 5. crop to the shrunken picture (full rectangle)
    ops.append(33); sels.append([r0, c0, ph - 1, pw - 1])
    ops.append(34); sels.append([0, 0, ph - 1, pw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 5ad4f10b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5ad4f10b"
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
                                f"for task 5ad4f10b"
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
                    f"Failed to build a complete episode for task 5ad4f10b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5ad4f10b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
