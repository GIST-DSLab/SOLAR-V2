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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, noisec, objc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, objc=None) -> dict:
    if bgc is None or noisec is None or objc is None:
        bgc, noisec, objc = random.sample(range(10), 3)
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
    while len(sfilter(obj, lambda ij: sum([len(obj & shift(nb, ij)) < 4 for nb in nbhs]) > 0)) == 0:
        ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
        ncells = choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow - 1)
        obj = set(sample(totuple(bounds), ncells))
    obj = normalize(obj)
    oh, ow = shape(obj)
    go = canvas(bgc, (oh, ow))
    go = fill(go, noisec, obj)
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


def derive_operations(I, O=None, examples=None):
    import numpy as np
    from collections import Counter
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background = most common colour on the border (as in the verifier)
    border = list(I[0, :]) + list(I[-1, :]) + list(I[:, 0]) + list(I[:, -1])
    bgc = Counter([int(v) for v in border]).most_common(1)[0][0]
    others = sorted(set(int(v) for v in I.flatten()) - {bgc})

    def isolated_count(col):
        n = 0
        for r in range(hi):
            for c in range(wi):
                if I[r, c] != col:
                    continue
                ok = False
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < hi and 0 <= cc < wi and I[rr, cc] == col:
                        ok = True
                        break
                if not ok:
                    n += 1
        return n

    # object colour: the upscaled one (no isolated cells); the other one is noise
    objc = min(others, key=isolated_count)
    noisec = [c for c in others if c != objc][0] if len(others) > 1 else objc

    rows, cols = np.where(I == objc)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    pat = (I[r0:r1 + 1, c0:c1 + 1] == objc).astype(int)
    ph, pw = pat.shape
    best = 1
    for f in range(1, max(ph, pw)):
        if ph % f or pw % f:
            continue
        d = pat[::f, ::f]
        if np.array_equal(np.kron(d, np.ones((f, f), dtype=int)), pat):
            best = f
    small = pat[::best, ::best]
    oh, ow = small.shape

    ops, sels = [], []
    # Crop canvas to the downscaled size, anchored at the object's top-left (full rectangle)
    ops.append(33); sels.append([int(r0), int(c0), oh - 1, ow - 1])
    cur = I[r0:r0 + oh, c0:c0 + ow].copy()
    cur[cur == 0] = 0
    # Lay a background base over the whole new canvas (full rectangle) if needed
    if np.any(cur != bgc):
        ops.append(bgc); sels.append([0, 0, oh - 1, ow - 1])
        cur[:, :] = bgc
    # Paint the downscaled object cells with the noise colour
    cells = [(r, c) for r in range(oh) for c in range(ow) if small[r, c]]
    if cells and any(cur[r, c] != noisec for r, c in cells):
        ops.append(noisec); sels.append(sel_of(cells))
    ops.append(34); sels.append([0, 0, oh - 1, ow - 1])
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
