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
from maker.sel_helpers import sel_of

VARIANTS = [{"mirror": False}, {"mirror": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    acol = random.choice(rem)
    rem = [c for c in rem if c != acol]
    bcol = random.choice(rem)
    n = num_examples if num_examples else 3
    if n >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS] + [dict(random.choice(VARIANTS)) for _ in range(n - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, acol=1, bcol=3, mirror=None, **kw) -> dict:
    if mirror is None:
        mirror = random.choice([True, False])

    def unifint(lb, ub):
        if ub < lb:
            ub = lb
        return lb + int(round(random.uniform(diff_lb, diff_ub) * (ub - lb)))

    while True:
        if mirror:
            hub, wub = min(30, max_w), min(14, max_h // 2)
        else:
            hub, wub = min(30, max_h), min(14, max_w // 2)
        h = unifint(2, max(2, hub))
        w = unifint(2, max(2, wub))
        inds = [(i, j) for i in range(h) for j in range(w)]
        numadev = unifint(0, (h * w) // 2)
        numbdev = unifint(0, (h * w) // 2)
        numa = random.choice((numadev, h * w - numadev))
        numb = random.choice((numadev, h * w - numbdev))
        numa = min(max(1, numa), h * w - 1)
        numb = min(max(1, numb), h * w - 1)
        aset = random.sample(inds, numa)
        bset = random.sample(inds, numb)
        A = np.full((h, w), bgc, dtype=int)
        B = A.copy()
        go = np.full((h, w), bgc, dtype=int)
        for p in aset:
            A[p] = acol
        for p in bset:
            B[p] = bcol
        go[(A == bgc) & (B == bgc)] = 2
        gi = np.hstack([A, B])
        if mirror:
            gi = gi.T
            go = go.T
        # reject grids where the other split also looks like two 2-colour halves (ambiguous)
        H, W = gi.shape

        def ok(a, b):
            pa, pb = set(a.flatten().tolist()), set(b.flatten().tolist())
            return len(pa) == 2 and len(pb) == 2 and len(pa & pb) == 1

        alt = ok(gi[:, :W // 2], gi[:, W // 2:]) if mirror else ok(gi[:H // 2], gi[H // 2:])
        altpossible = (W % 2 == 0) if mirror else (H % 2 == 0)
        if not (altpossible and alt):
            break
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # candidate splits (top/bottom first, like the verifier)
    cands = []
    if hi % 2 == 0:
        cands.append((hi // 2, wi, hi // 2, 0))
    if wi % 2 == 0:
        cands.append((hi, wi // 2, 0, wi // 2))
    pick = None
    for (ho, wo, dr, dc) in cands:
        A = I[:ho, :wo]
        B = I[dr:dr + ho, dc:dc + wo]
        pa, pb = set(A.flatten().tolist()), set(B.flatten().tolist())
        if len(pa) == 2 and len(pb) == 2 and len(pa & pb) == 1:
            pick = (ho, wo, dr, dc, A, B, (pa & pb).pop())
            break
    if pick is None:
        ho, wo, dr, dc = cands[0]
        A = I[:ho, :wo]
        B = I[dr:dr + ho, dc:dc + wo]
        sh = set(A.flatten().tolist()) & set(B.flatten().tolist())
        pick = (ho, wo, dr, dc, A, B, sorted(sh)[0] if sh else int(np.bincount(A.flatten()).argmax()))
    ho, wo, dr, dc, A, B, bgc = pick
    acol = (set(A.flatten().tolist()) - {bgc}).pop()
    bcol = (set(B.flatten().tolist()) - {bgc}).pop()

    ops, sels = [], []
    # 1. lay the base: the whole A half (full rectangle) becomes 2
    ops.append(2); sels.append([0, 0, ho - 1, wo - 1])
    # 2. A's own colour layer knocks its cells back to background
    ops.append(bgc)
    sels.append(sel_of([(r, c) for r in range(ho) for c in range(wo) if A[r, c] == acol]))
    if bcol != 0:
        # 3. isolate B's colour layer: blank B's background so a copy carries only bcol
        if bgc != 0:
            ops.append(0)
            sels.append(sel_of([(r + dr, c + dc) for r in range(ho) for c in range(wo) if B[r, c] == bgc]))
        # 4. copy the B half (full rectangle) from the working grid and lay it over A
        ops.append(29); sels.append([dr, dc, ho - 1, wo - 1])
        ops.append(30); sels.append([0, 0, 0, 0])
        # 5. the pasted bcol layer, now visible in A, knocks out to background
        ops.append(bgc)
        sels.append(sel_of([(r, c) for r in range(ho) for c in range(wo) if B[r, c] == bcol]))
    else:
        # bcol==0 cannot be carried by a copy: knock out the cells B's pattern lands on,
        # skipping cells already at bgc from A's knockout
        land = [(r, c) for r in range(ho) for c in range(wo) if B[r, c] == bcol and A[r, c] != acol]
        if land:
            ops.append(bgc); sels.append(sel_of(land))
    # 6. B has served its purpose: crop to the A half
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
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
