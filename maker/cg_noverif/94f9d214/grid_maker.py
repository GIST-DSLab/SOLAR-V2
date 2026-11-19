"""
ARC Task: 94f9d214 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque
from maker.sel_helpers import sel_of

# Reading of the task: this agrees with both the maker and the golf entry.
# The input is two equal halves, A and B, drawn on a shared background.
# Each output cell depends only on the two cells at the same position in A and B:
# it is 2 when both of them are background, and background otherwise.
# The golf entry pairs the halves row by row and cell by cell in the same way,
# so the carving is the same. Steps:
#   1. Paint 2 on each connected region of cells that are background in both halves.
#   2. Flood A's own foreground blobs back to the background colour.
#   3. Crop the canvas down to half A.
# Every selection is computed from I alone.

MIRROR_VARIANTS = [{"mirror": False}, {"mirror": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    acol = random.choice(rem)
    rem = [c for c in rem if c != acol]
    bcol = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(MIRROR_VARIANTS):
        ex = [dict(v) for v in MIRROR_VARIANTS]
        ex += [dict(random.choice(MIRROR_VARIANTS)) for _ in range(n_ex - len(MIRROR_VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(MIRROR_VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "instance_plan": plan}


def _unifint(lb, ub, rng):
    a, b = rng
    lo = a + int(round((b - a) * lb))
    hi = a + int(round((b - a) * ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if lo > hi:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, acol=None, bcol=None,
             mirror=None, **kw) -> dict:
    if bgc is None:
        d = sample_colors()
        bgc, acol, bcol = d["bgc"], d["acol"], d["bcol"]
    if mirror is None:
        mirror = random.choice([False, True])
    if mirror:
        # input becomes (2w, h)
        h = _unifint(diff_lb, diff_ub, (2, max(2, min(30, max_w))))
        w = _unifint(diff_lb, diff_ub, (2, max(2, min(14, max_h // 2))))
    else:
        h = _unifint(diff_lb, diff_ub, (2, max(2, min(30, max_h))))
        w = _unifint(diff_lb, diff_ub, (2, max(2, min(14, max_w // 2))))
    inds = [(r, c) for r in range(h) for c in range(w)]
    numadev = _unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = _unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = random.choice((numadev, h * w - numadev))
    numb = random.choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = random.sample(inds, numa)
    bset = random.sample(inds, numb)
    A = np.full((h, w), bgc, dtype=int)
    B = np.full((h, w), bgc, dtype=int)
    for r, c in aset:
        A[r, c] = acol
    for r, c in bset:
        B[r, c] = bcol
    gi = np.concatenate([A, B], axis=1)
    go = np.full((h, w), bgc, dtype=int)
    res = (set(inds) - set(aset)) & (set(inds) - set(bset))
    for r, c in res:
        go[r, c] = 2
    if mirror:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Candidate splits into two equal halves (A is the top or left half).
    cands = []
    if wi % 2 == 0:
        cands.append((I[:, :wi // 2], I[:, wi // 2:]))
    if hi % 2 == 0:
        cands.append((I[:hi // 2, :], I[hi // 2:, :]))
    A = B = bgc = None
    for a, b in cands:
        pa = set(a.flatten().tolist())
        pb = set(b.flatten().tolist())
        shared = pa & pb
        if len(pa) == 2 and len(pb) == 2 and len(shared) == 1:
            A, B, bgc = a, b, shared.pop()
            break
    if A is None:
        A, B = cands[0]
        shared = set(A.flatten().tolist()) & set(B.flatten().tolist())
        bgc = sorted(shared)[0] if shared else int(np.bincount(A.flatten()).argmax())
    ho, wo = A.shape

    def components(cellset):
        rem = set(cellset)
        comps = []
        while rem:
            seed = min(rem)
            rem.discard(seed)
            comp = [seed]
            q = deque([seed])
            while q:
                r, c = q.popleft()
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (nr, nc) in rem:
                        rem.discard((nr, nc))
                        comp.append((nr, nc))
                        q.append((nr, nc))
            comps.append(comp)
        comps.sort(key=lambda cm: (-len(cm), min(cm)))
        return comps

    ops, sels = [], []

    # 1. Each connected region where both halves are background becomes 2.
    both_bg = {(r, c) for r in range(ho) for c in range(wo)
               if A[r, c] == bgc and B[r, c] == bgc}
    for comp in components(both_bg):
        ops.append(2)
        sels.append(sel_of(sorted(comp)))

    # 2. Erase A's own foreground: flood each blob back to the background colour.
    a_cells = {(r, c) for r in range(ho) for c in range(wo) if A[r, c] != bgc}
    for comp in components(a_cells):
        seed = min(comp)
        ops.append(10 + int(bgc))
        sels.append([seed[0], seed[1], 0, 0])

    # 3. Crop the canvas to half A (a full rectangle).
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task 94f9d214"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 94f9d214"
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
                                f"for task 94f9d214"
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
                    f"Failed to build a complete episode for task 94f9d214 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"94f9d214-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
