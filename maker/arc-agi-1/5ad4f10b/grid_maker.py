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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, noisec, objc = random.sample(cols, 3)
    return {"bgc": bgc, "noisec": noisec, "objc": objc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, noisec: int, objc: int) -> dict:
    nbh = {(0, 0), (1, 0), (0, 1), (1, 1)}
    nbhs = apply(lbind(shift, nbh), {(0, 0), (-1, 0), (0, -1), (-1, -1)})

    oh_ub = min(6, max(2, (max_h - 2) // 2))
    ow_ub = min(6, max(2, (max_w - 2) // 2))
    oh = unifint(diff_lb, diff_ub, (2, oh_ub))
    ow = unifint(diff_lb, diff_ub, (2, ow_ub))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(1, ncells), oh * ow - 1)
    obj = set(sample(totuple(bounds), ncells))
    while len(sfilter(obj, lambda ij: sum([len(obj & shift(nb, ij)) < 4 for nb in nbhs]) > 0)) == 0:
        ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
        ncells = choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow)
        obj = set(sample(totuple(bounds), ncells))
    obj = normalize(obj)
    oh, ow = shape(obj)

    go = canvas(bgc, (oh, ow))
    go = fill(go, noisec, obj)

    fac_ub = max(2, min((max_h - 2) // oh, (max_w - 2) // ow))
    fac = unifint(diff_lb, diff_ub, (2, fac_ub))
    gobj = asobject(upscale(replace(go, noisec, objc), fac))
    uh, uw = shape(gobj)

    h = unifint(diff_lb, diff_ub, (uh + 2, max_h))
    w = unifint(diff_lb, diff_ub, (uw + 2, max_w))
    loci = randint(1, h - uh - 1)
    locj = randint(1, w - uw - 1)
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shift(gobj, (loci, locj)))
    cands = ofcolor(gi, bgc)
    namt = unifint(diff_lb, diff_ub, (2, max(1, len(cands) // 4)))
    namt = min(namt, len(cands))
    noise = sample(totuple(cands), namt)
    gi = fill(gi, noisec, noise)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # --- background = dominant colour of the border ring (object never touches it) ---
    border = I[0, :].tolist() + I[-1, :].tolist() + I[:, 0].tolist() + I[:, -1].tolist()
    bgc = Counter(border).most_common(1)[0][0]

    others = [int(c) for c in np.unique(I).tolist() if int(c) != bgc]

    # --- object colour = the one made of solid blocks (no isolated cell); noise has isolated cells ---
    def block_score(c):
        cs = set(map(tuple, np.argwhere(I == c).tolist()))
        iso = sum(
            1 for (r, cc) in cs
            if not any((r + dr, cc + dc) in cs for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        )
        return (1 if iso == 0 else 0, len(cs))

    objc = max(others, key=block_score)
    rest = [c for c in others if c != objc]
    if rest:
        noisec = rest[0]
    else:
        noisec = int([v for v in np.unique(O).tolist() if int(v) != bgc][0])

    # --- upscaled object bbox and its scale factor ---
    cells = np.argwhere(I == objc)
    r0, c0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    r1, c1 = int(cells[:, 0].max()), int(cells[:, 1].max())
    H, W = r1 - r0 + 1, c1 - c0 + 1
    M = (I[r0:r1 + 1, c0:c1 + 1] == objc)

    fac = 1
    for k in range(min(H, W), 0, -1):
        if H % k or W % k:
            continue
        blocks = M.reshape(H // k, k, W // k, k)
        if blocks.all(axis=(1, 3)).astype(bool).repeat(k, 0).repeat(k, 1).shape == M.shape:
            pass
        small = blocks.any(axis=(1, 3))
        if np.array_equal(small.repeat(k, 0).repeat(k, 1), M):
            fac = k
            break

    P = M[::fac, ::fac]          # the object at scale 1 (ho x wo)
    ph, pw = P.shape

    ops, sels = [], []

    # --- the object shrinks in place onto its own top-left corner ---
    win = I[r0:r0 + ph, c0:c0 + pw]
    if bool(np.any(win != bgc)):
        ops.append(int(bgc))
        sels.append([r0, c0, ph - 1, pw - 1])

    # --- repaint each connected piece of the shrunk object, piece by piece ---
    filled = set(map(tuple, np.argwhere(P).tolist()))
    seen = set()
    comps = []
    for cell in sorted(filled):
        if cell in seen:
            continue
        comp, q = set(), deque([cell])
        seen.add(cell)
        while q:
            r, c = q.popleft()
            comp.add((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (r + dr, c + dc)
                if nb in filled and nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        comps.append(comp)

    for comp in comps:
        rem = set(comp)
        while rem:
            r, c = min(rem)
            w = 1
            while (r, c + w) in rem:
                w += 1
            h = 1
            while all((r + h, c + k) in rem for k in range(w)):
                h += 1
            for i in range(h):
                for k in range(w):
                    rem.discard((r + i, c + k))
            ops.append(int(noisec))
            sels.append([r0 + r, c0 + c, h - 1, w - 1])

    # --- keep only the shrunk object ---
    ops.append(33)
    sels.append([r0, c0, ph - 1, pw - 1])
    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
