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
from collections import Counter, deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, noisec, objc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, objc=None) -> dict:
    from random import choice, randint, sample

    def unifint(lb, ub, rng):
        a, b = rng
        if a >= b:
            return a
        d = random.uniform(lb, ub)
        return int(round(a + d * (b - a)))

    if bgc is None or noisec is None or objc is None:
        bgc, noisec, objc = random.sample(range(10), 3)

    nbh = {(0, 0), (1, 0), (0, 1), (1, 1)}
    nbhs = [{(a + i, b + j) for a, b in nbh} for i, j in {(0, 0), (-1, 0), (0, -1), (-1, -1)}]

    def ok(obj):
        for (i, j) in obj:
            cnt = 0
            for nb in nbhs:
                if len(obj & {(a + i, b + j) for a, b in nb}) < 4:
                    cnt += 1
            if cnt > 0:
                return True
        return False

    oh = unifint(diff_lb, diff_ub, (2, 6))
    ow = unifint(diff_lb, diff_ub, (2, 6))
    bounds = [(i, j) for i in range(oh) for j in range(ow)]
    ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(1, ncells), oh * ow - 1)
    obj = set(sample(bounds, ncells))
    while not ok(obj):
        ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
        ncells = choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow)
        obj = set(sample(bounds, ncells))
    mi = min(i for i, j in obj)
    mj = min(j for i, j in obj)
    obj = {(i - mi, j - mj) for i, j in obj}
    oh = max(i for i, j in obj) + 1
    ow = max(j for i, j in obj) + 1
    go = np.full((oh, ow), bgc, dtype=int)
    for i, j in obj:
        go[i, j] = noisec
    fmax = min((max_h - 2) // oh, (max_w - 2) // ow)
    if fmax < 2:
        raise ValueError("grid too small")
    fac = unifint(diff_lb, diff_ub, (2, fmax))
    small = np.where(go == noisec, objc, bgc)
    big = small.repeat(fac, 0).repeat(fac, 1)
    bh, bw = big.shape
    h = unifint(diff_lb, diff_ub, (bh + 2, max_h))
    w = unifint(diff_lb, diff_ub, (bw + 2, max_w))
    loci = randint(1, h - bh - 1)
    locj = randint(1, w - bw - 1)
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + bh, locj:locj + bw] = big
    cands = [tuple(x) for x in np.argwhere(gi == bgc).tolist()]
    namt = unifint(diff_lb, diff_ub, (2, max(1, len(cands) // 4)))
    namt = min(namt, len(cands))
    for (i, j) in sample(cands, namt):
        gi[i, j] = noisec
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background = dominant colour of the border ring (object never touches it)
    border = I[0, :].tolist() + I[-1, :].tolist() + I[:, 0].tolist() + I[:, -1].tolist()
    bgc = Counter(border).most_common(1)[0][0]
    others = [int(c) for c in np.unique(I).tolist() if int(c) != bgc]

    # object colour = the colour with no isolated cells (solid upscaled blocks)
    def block_score(c):
        cs = set(map(tuple, np.argwhere(I == c).tolist()))
        iso = sum(1 for (r, cc) in cs
                  if not any((r + dr, cc + dc) in cs
                             for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))))
        return (1 if iso == 0 else 0, len(cs))

    objc = max(others, key=block_score)
    rest = [c for c in others if c != objc]
    if rest:
        noisec = rest[0]
    else:
        noisec = None
        for ei, eo in (examples or []):
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            eb = Counter(ei[0, :].tolist() + ei[-1, :].tolist()
                         + ei[:, 0].tolist() + ei[:, -1].tolist()).most_common(1)[0][0]
            if eb == bgc:
                vals = [int(v) for v in np.unique(eo).tolist() if int(v) != bgc]
                if vals:
                    noisec = vals[0]
                    break
        if noisec is None:
            noisec = [c for c in range(10) if c not in (bgc, objc)][0]

    # object's bbox and its upscale factor (largest k with exact block structure)
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
    P = M[::fac, ::fac]          # the object at scale 1
    ph, pw = P.shape

    ops, sels = [], []
    # 1. crop canvas to the object's downscaled footprint at its top-left corner
    #    (full rectangle intended)
    ops.append(33); sels.append([r0, c0, ph - 1, pw - 1])
    # 2. lay the background over the whole new canvas (full rectangle intended)
    win = I[r0:r0 + ph, c0:c0 + pw]
    if bool(np.any(win != bgc)):
        ops.append(int(bgc)); sels.append([0, 0, ph - 1, pw - 1])
    # 3. draw the downscaled object (one cell per fac×fac block of the input object)
    #    in the noise colour
    obj_cells = [(int(r), int(c)) for r, c in np.argwhere(P).tolist()]
    ops.append(int(noisec)); sels.append(sel_of(obj_cells))
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
