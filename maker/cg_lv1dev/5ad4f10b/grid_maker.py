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
    bgc, noisec, objc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "objc": objc}


def _unifint(lb, ub, rng):
    a, b = rng
    if b <= a:
        return a
    lo = a + int(lb * (b - a))
    hi = a + int(round(ub * (b - a)))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _is_upscale(M, k):
    H, W = M.shape
    if H % k or W % k:
        return False
    small = M[::k, ::k]
    return np.array_equal(np.repeat(np.repeat(small, k, 0), k, 1), M)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, objc=None) -> dict:
    if bgc is None or noisec is None or objc is None:
        bgc, noisec, objc = random.sample(range(10), 3)
    for _ in range(500):
        oh = _unifint(diff_lb, diff_ub, (2, 6))
        ow = _unifint(diff_lb, diff_ub, (2, 6))
        n = oh * ow
        ncd = _unifint(diff_lb, diff_ub, (1, n // 2))
        nc = random.choice((ncd, n - ncd))
        nc = min(max(1, nc), n - 1)
        bounds = [(i, j) for i in range(oh) for j in range(ow)]
        obj = set(random.sample(bounds, nc))
        mi = min(i for i, _ in obj); mj = min(j for _, j in obj)
        obj = {(i - mi, j - mj) for i, j in obj}
        oh = max(i for i, _ in obj) + 1
        ow = max(j for _, j in obj) + 1
        P = np.zeros((oh, ow), dtype=bool)
        for i, j in obj:
            P[i, j] = True
        if P.all():
            continue
        # reject patterns that are themselves upscaled (scale factor would be ambiguous)
        if any(_is_upscale(P, k) for k in range(2, min(oh, ow) + 1)):
            continue
        fmax = min((max_h - 2) // oh, (max_w - 2) // ow)
        if fmax < 2:
            continue
        fac = _unifint(diff_lb, diff_ub, (2, fmax))
        gh, gw = oh * fac, ow * fac
        h = _unifint(diff_lb, diff_ub, (gh + 2, max_h))
        w = _unifint(diff_lb, diff_ub, (gw + 2, max_w))
        li = random.randint(1, h - gh - 1)
        lj = random.randint(1, w - gw - 1)
        gi = np.full((h, w), bgc, dtype=int)
        big = np.repeat(np.repeat(P, fac, 0), fac, 1)
        gi[li:li + gh, lj:lj + gw][big] = objc
        cands = [tuple(x) for x in np.argwhere(gi == bgc).tolist()]
        namt = _unifint(diff_lb, diff_ub, (2, max(2, len(cands) // 4)))
        for (r, c) in random.sample(cands, min(namt, len(cands))):
            gi[r, c] = noisec
        go = np.where(P, noisec, bgc)
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O, examples=None):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)

    # The code-golf entry reads the task the same way the maker does: the big object
    # is a stride-`fac` upscale of the answer. Take one cell from each block, swap the
    # object's colour for the noise colour, and that is the output. Painting the
    # shrunk shape onto the object's own top-left corner and then cropping does
    # exactly this sampling, so the maker's grouping of the ops is kept. The one
    # change is that the colour roles now come from the demonstrations
    # (the object colour never appears in an output; the noise colour does)
    # rather than being guessed, or read from O.

    border = I[0, :].tolist() + I[-1, :].tolist() + I[:, 0].tolist() + I[:, -1].tolist()
    bgc = Counter(border).most_common(1)[0][0]
    others = [int(c) for c in np.unique(I).tolist() if int(c) != bgc]

    objc = noisec = None
    if examples:
        out_cols = set()
        for ei, eo in examples:
            out_cols |= set(np.asarray(eo).flatten().tolist())
        cand_obj = [c for c in others if c not in out_cols]
        cand_noise = [c for c in others if c in out_cols]
        if len(cand_obj) == 1:
            objc = cand_obj[0]
        if len(cand_noise) == 1:
            noisec = cand_noise[0]

    if objc is None:
        def block_score(c):
            cs = set(map(tuple, np.argwhere(I == c).tolist()))
            iso = sum(1 for (r, cc) in cs
                      if not any((r + dr, cc + dc) in cs for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))))
            return (1 if iso == 0 else 0, len(cs))
        objc = max(others, key=block_score)
    if noisec is None:
        rest = [c for c in others if c != objc]
        noisec = rest[0] if rest else bgc

    # upscaled object bbox and its scale factor (largest k that M is a k-upscale of)
    cells = np.argwhere(I == objc)
    r0, c0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    r1, c1 = int(cells[:, 0].max()), int(cells[:, 1].max())
    M = (I[r0:r1 + 1, c0:c1 + 1] == objc)
    fac = 1
    for k in range(min(M.shape), 0, -1):
        if _is_upscale(M, k):
            fac = k
            break
    P = M[::fac, ::fac]          # one sample per block
    ph, pw = P.shape

    ops, sels = [], []
    # base: clear the ph x pw window at the object's top-left corner to background
    # (full rectangle -- it becomes the whole output canvas)
    win = I[r0:r0 + ph, c0:c0 + pw]
    if bool(np.any(win != bgc)):
        ops.append(int(bgc)); sels.append([r0, c0, ph - 1, pw - 1])

    # draw the sampled shape in the noise colour, one connected piece at a time
    filled = set(map(tuple, np.argwhere(P).tolist()))
    seen = set()
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
                    seen.add(nb); q.append(nb)
        if noisec == bgc:
            continue
        ops.append(int(noisec))
        sels.append(sel_of(sorted((r0 + r, c0 + c) for r, c in comp)))

    # keep only the shrunk object (full-rectangle crop)
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
