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

# unifint / choice / sample / DSL helpers come from the harness's
# `from utils import *` and `from dsl import *` (unifint lives in utils, not dsl).


def sample_colors(num_examples=None) -> dict:
    bgc, noisec, objc = random.sample(list(range(10)), 3)
    return {"bgc": bgc, "noisec": noisec, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, objc=None) -> dict:
    if bgc is None or noisec is None or objc is None:
        bgc, noisec, objc = random.sample(list(range(10)), 3)
    nbh = {(0, 0), (1, 0), (0, 1), (1, 1)}
    nbhs = apply(lbind(shift, nbh), {(0, 0), (-1, 0), (0, -1), (-1, -1)})
    oh = unifint(diff_lb, diff_ub, (2, max(2, min(6, (max_h - 2) // 2))))
    ow = unifint(diff_lb, diff_ub, (2, max(2, min(6, (max_w - 2) // 2))))
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
    oh, ow = shape(obj)
    go = canvas(bgc, (oh, ow))
    go = fill(go, noisec, obj)
    # the downscale factor must be recoverable: the output itself must not be an upscale
    for k in range(2, max(oh, ow) + 1):
        if oh % k == 0 and ow % k == 0 and upscale(downscale(go, k), k) == go:
            return generate(diff_lb, diff_ub, max_h, max_w, bgc=bgc, noisec=noisec, objc=objc)
    if oh == ow and len(obj) == oh * ow:
        return generate(diff_lb, diff_ub, max_h, max_w, bgc=bgc, noisec=noisec, objc=objc)
    fmax = max(2, min((max_h - 2) // oh, (max_w - 2) // ow))
    fac = unifint(diff_lb, diff_ub, (2, fmax))
    gobj = asobject(upscale(replace(go, noisec, objc), fac))
    oh2, ow2 = shape(gobj)
    h = unifint(diff_lb, diff_ub, (oh2 + 2, max(oh2 + 2, max_h)))
    w = unifint(diff_lb, diff_ub, (ow2 + 2, max(ow2 + 2, max_w)))
    loci = randint(1, h - oh2 - 1)
    locj = randint(1, w - ow2 - 1)
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shift(gobj, (loci, locj)))
    cands = ofcolor(gi, bgc)
    namt = unifint(diff_lb, diff_ub, (2, max(2, len(cands) // 4)))
    noise = sample(totuple(cands), namt)
    gi = fill(gi, noisec, noise)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared with the code-golf entry, different from the maker's
    # "shrink in place, then crop"): crop to the big object, swap its colour
    # for the noise colour and clear the noise to background, then downscale
    # by taking the first row/column of each block, and crop to the small grid.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: dominant colour of the border ring (the upscaled object never touches it)
    border = I[0, :].tolist() + I[-1, :].tolist() + I[:, 0].tolist() + I[:, -1].tolist()
    bgc = Counter(border).most_common(1)[0][0]
    others = [int(c) for c in np.unique(I).tolist() if int(c) != bgc]

    # object colour: the colour with no isolated cell (it is made of fac x fac blocks)
    def block_score(c):
        cs = set(map(tuple, np.argwhere(I == c).tolist()))
        iso = sum(1 for (r, cc) in cs
                  if not any((r + dr, cc + dc) in cs for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))))
        return (1 if iso == 0 else 0, len(cs))
    objc = max(others, key=block_score)
    rest = [c for c in others if c != objc]
    if rest:
        noisec = rest[0]
    else:  # colours are fixed per episode: read the noise colour from the demonstrations
        noisec = None
        for ei, eo in (examples or []):
            eo = np.asarray(eo)
            cand = [int(v) for v in np.unique(eo).tolist() if int(v) != bgc]
            if cand:
                noisec = cand[0]
                break
        if noisec is None:
            noisec = bgc

    ops, sels = [], []

    # 1. crop the canvas to the object's bounding box (noise outside it is discarded)
    cells = np.argwhere(I == objc)
    r0, c0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    r1, c1 = int(cells[:, 0].max()), int(cells[:, 1].max())
    H, W = r1 - r0 + 1, c1 - c0 + 1
    ops.append(33); sels.append([r0, c0, H - 1, W - 1])   # full rectangle: the object's bbox
    G = I[r0:r1 + 1, c0:c1 + 1].copy()   # crop onto a cleared canvas keeps every value

    # 2. recolour by colour class: noise inside the box -> background,
    #    then the object colour -> noise colour
    nz = [tuple(x) for x in np.argwhere(G == noisec).tolist()]
    if nz and noisec != bgc:
        ops.append(int(bgc)); sels.append(sel_of(nz))
        G[G == noisec] = bgc
    oc = [tuple(x) for x in np.argwhere(G == objc).tolist()]
    ops.append(int(noisec)); sels.append(sel_of(oc))
    G[G == objc] = noisec

    # scale factor: largest k for which the box is a k-upscale of a smaller grid
    fac = 1
    for k in range(max(H, W), 1, -1):
        if H % k or W % k:
            continue
        small = G[::k, ::k]
        if np.array_equal(small.repeat(k, 0).repeat(k, 1), G):
            fac = k
            break
    ph, pw = H // fac, W // fac

    def copy_line(src, dst, is_row, length):
        # copy line `src` of the current grid onto line `dst`; 0 cells don't paste,
        # so first clear the dst cells that the paste cannot overwrite
        if is_row:
            s, d = G[src, :length].copy(), G[dst, :length]
        else:
            s, d = G[:length, src].copy(), G[:length, dst]
        if np.array_equal(s, d):
            return
        stuck = [k for k in range(length) if s[k] == 0 and d[k] != 0]
        if stuck:
            ops.append(0)
            sels.append(sel_of([(dst, k) if is_row else (k, dst) for k in stuck]))
        if np.any((s != 0) & (s != d)):
            ops.append(29)   # CopyO the whole source line (a full 1-wide rectangle)
            sels.append([src, 0, 0, length - 1] if is_row else [0, src, length - 1, 0])
            ops.append(30)
            sels.append([dst, 0, 0, 0] if is_row else [0, dst, 0, 0])
        if is_row:
            G[dst, :length] = s
        else:
            G[:length, dst] = s

    # 3. downscale rows: row i takes the first row of block i
    for i in range(1, ph):
        copy_line(i * fac, i, True, W)
    # 4. downscale columns over the rows that survive
    for j in range(1, pw):
        copy_line(j * fac, j, False, ph)

    # 5. crop to the downscaled grid
    ops.append(33); sels.append([0, 0, ph - 1, pw - 1])
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
