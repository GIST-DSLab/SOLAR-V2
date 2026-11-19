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
from collections import Counter, deque

import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, noisec, objc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, objc=None) -> dict:
    if bgc is None or noisec is None or objc is None:
        bgc, noisec, objc = random.sample(range(10), 3)

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    nbh = [(0, 0), (1, 0), (0, 1), (1, 1)]
    offs = [(0, 0), (-1, 0), (0, -1), (-1, -1)]
    nbhs = [[(a + da, b + db) for a, b in nbh] for da, db in offs]

    def bad(obj):
        for (i, j) in obj:
            for nb in nbhs:
                cnt = sum(1 for (a, b) in nb if (i + a, j + b) in obj)
                if cnt < 4:
                    return False
        return True

    oh = unifint(diff_lb, diff_ub, (2, 6))
    ow = unifint(diff_lb, diff_ub, (2, 6))
    bounds = [(i, j) for i in range(oh) for j in range(ow)]
    ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
    ncells = random.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(1, ncells), oh * ow - 1)
    obj = set(random.sample(bounds, ncells))
    while bad(obj):
        ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
        ncells = random.choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow - 1)
        obj = set(random.sample(bounds, ncells))
    mi = min(i for i, j in obj)
    mj = min(j for i, j in obj)
    obj = {(i - mi, j - mj) for i, j in obj}
    oh = max(i for i, j in obj) + 1
    ow = max(j for i, j in obj) + 1
    go = [[bgc] * ow for _ in range(oh)]
    for i, j in obj:
        go[i][j] = noisec

    lim = min(max_h, max_w, 30) - 2
    fmax = max(2, min(lim // oh, lim // ow))
    fac = unifint(diff_lb, diff_ub, (2, fmax))
    gh, gw = oh * fac, ow * fac
    h = unifint(diff_lb, diff_ub, (gh + 2, max(gh + 2, max_h)))
    w = unifint(diff_lb, diff_ub, (gw + 2, max(gw + 2, max_w)))
    loci = random.randint(1, h - gh - 1)
    locj = random.randint(1, w - gw - 1)
    gi = [[bgc] * w for _ in range(h)]
    for i in range(gh):
        for j in range(gw):
            if go[i // fac][j // fac] == noisec:
                gi[loci + i][locj + j] = objc
    cands = [(i, j) for i in range(h) for j in range(w) if gi[i][j] == bgc]
    namt = unifint(diff_lb, diff_ub, (2, max(2, len(cands) // 4)))
    namt = min(namt, len(cands))
    for i, j in random.sample(cands, namt):
        gi[i][j] = noisec
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background = dominant colour of the border ring
    border = I[0, :].tolist() + I[-1, :].tolist() + I[:, 0].tolist() + I[:, -1].tolist()
    bgc = Counter(border).most_common(1)[0][0]
    others = [int(c) for c in np.unique(I).tolist() if int(c) != bgc]

    # object colour = the colour with no isolated cells (noise is scattered pixels)
    def block_score(c):
        cs = set(map(tuple, np.argwhere(I == c).tolist()))
        iso = sum(1 for (r, cc) in cs
                  if not any((r + dr, cc + dc) in cs for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))))
        return (1 if iso == 0 else 0, len(cs))

    objc = max(others, key=block_score)
    rest = [c for c in others if c != objc]
    noisec = rest[0]

    # subgrid of the object (golf/verifier reading: keep only rows/cols holding the object)
    cells = np.argwhere(I == objc)
    r0, c0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    r1, c1 = int(cells[:, 0].max()), int(cells[:, 1].max())
    H, W = r1 - r0 + 1, c1 - c0 + 1
    M = (I[r0:r1 + 1, c0:c1 + 1] == objc)

    # scale factor: largest k with upscale(downscale(M,k),k) == M
    fac = 1
    for k in range(min(H, W), 0, -1):
        if H % k or W % k:
            continue
        small = M[::k, ::k]
        if np.array_equal(small.repeat(k, 0).repeat(k, 1), M):
            fac = k
            break
    ph, pw = H // fac, W // fac

    G = I.copy()
    ops, sels = [], []
    win = [(r0 + i, c0 + j) for i in range(ph) for j in range(pw)]
    # sample value of each block (its top-left cell) decides each collapsed cell
    tgt_obj = {(r0 + i, c0 + j): bool(M[i * fac, j * fac]) for i in range(ph) for j in range(pw)}

    # step 1: inside the subgrid, noise pixels become background (switch part 1)
    noise_in = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if I[r, c] == noisec]
    if any(not tgt_obj[p] and I[p] == noisec for p in win if p in set(noise_in)):
        ops.append(int(bgc)); sels.append(sel_of(noise_in))
        for p in noise_in:
            G[p] = bgc

    # step 2: the object takes the noise colour (switch part 2)
    obj_cells = [tuple(x) for x in cells.tolist()]
    if any(tgt_obj[p] and G[p] == objc for p in win):
        ops.append(int(noisec)); sels.append(sel_of(obj_cells))
        for p in obj_cells:
            G[p] = noisec
    else:
        # object colour never recoloured: collapsed object cells must be painted directly
        pass

    # step 3: downscale — collapse each fac x fac block onto one cell at the subgrid's corner
    to_obj = [p for p in win if tgt_obj[p] and G[p] != noisec]
    if to_obj:
        ops.append(int(noisec)); sels.append(sel_of(to_obj))
        for p in to_obj:
            G[p] = noisec
    to_bg = [p for p in win if not tgt_obj[p] and G[p] != bgc]
    if to_bg:
        ops.append(int(bgc)); sels.append(sel_of(to_bg))
        for p in to_bg:
            G[p] = bgc

    # step 4: keep only the collapsed object (full rectangle crop)
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
