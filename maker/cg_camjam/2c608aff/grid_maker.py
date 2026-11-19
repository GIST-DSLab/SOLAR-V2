"""
ARC Task: 2c608aff (RE-ARC) — LLM-generated grid_maker
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
from dsl import *
from utils import unifint
from random import randint, sample
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    ccol = random.choice([c for c in cols if c != bgc])
    dcol = random.choice([c for c in cols if c not in (bgc, ccol)])
    return {"bgc": bgc, "ccol": ccol, "dcol": dcol}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, ccol=1, dcol=2) -> dict:
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    boxh = unifint(diff_lb, diff_ub, (2, h // 2))
    boxw = unifint(diff_lb, diff_ub, (2, w // 2))
    loci = randint(0, h - boxh)
    locj = randint(0, w - boxw)
    bd = backdrop(frozenset({(loci, locj), (loci + boxh - 1, locj + boxw - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, ccol, bd)
    reminds = totuple(asindices(gi) - backdrop(outbox(bd)))
    noiseb = max(1, len(reminds) // 4)
    nnoise = unifint(diff_lb, diff_ub, (0, noiseb))
    noise = sample(reminds, nnoise)
    gi = fill(gi, dcol, noise)
    go = tuple(e for e in gi)
    hs = interval(loci, loci + boxh, 1)
    ws = interval(locj, locj + boxw, 1)
    for ij in noise:
        a, b = ij
        if a in hs:
            go = fill(go, dcol, connect(ij, (a, locj)))
        elif b in ws:
            go = fill(go, dcol, connect(ij, (loci, b)))
    go = fill(go, ccol, bd)
    return {'input': gi, 'output': go}


def _solid_box(G, col):
    rs, cs = np.where(G == col)
    if len(rs) == 0:
        return None
    r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
    bh, bw = r1 - r0 + 1, c1 - c0 + 1
    if bh >= 2 and bw >= 2 and len(rs) == bh * bw:
        return int(r0), int(c0), int(bh), int(bw)
    return None


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # --- colour roles: read them from the demonstrations when available
    ccol = dcol = None
    if examples:
        exs = [(np.asarray(a, dtype=int), np.asarray(b, dtype=int)) for a, b in examples]
        cands = None
        for a, _ in exs:
            s = {c for c in set(a.flatten().tolist()) if _solid_box(a, c) is not None}
            cands = s if cands is None else cands & s
        if cands and len(cands) == 1:
            ccol = next(iter(cands))
        added = Counter()
        for a, b in exs:
            m = a != b
            added.update(b[m].tolist())
        if added:
            dcol = added.most_common(1)[0][0]
    if ccol is None or _solid_box(I, ccol) is None:
        best = None
        for col in set(I.flatten().tolist()):
            bx = _solid_box(I, col)
            if bx and (best is None or bx[2] * bx[3] > best[0]):
                best = (bx[2] * bx[3], col)
        if best is None:
            return [34], [[0, 0, hi - 1, wi - 1]]
        ccol = best[1]
    if dcol is None:
        rest = [c for c in set(I.flatten().tolist()) if c != ccol]
        if len(rest) < 2:
            return [34], [[0, 0, hi - 1, wi - 1]]
        rest.sort(key=lambda c: int((I == c).sum()))
        dcol = rest[0]

    r0, c0, bh, bw = _solid_box(I, ccol)
    r1, c1 = r0 + bh - 1, c0 + bw - 1

    # Reading: the box is the target; each of its four sides is handled as a
    # unit. On one side, every lane (row / column) facing that side that holds
    # noise is a ray shot from the farthest noise cell straight into the box
    # wall; the noise cells in that lane merge into one stroke.
    def ray(cells):
        # cells: lane coordinates ordered from the far end toward the box wall
        k = 0
        while k < len(cells) and I[cells[k]] == dcol:
            k += 1          # already-noise prefix needs no paint
        rest = cells[k:]
        if rest:
            ops.append(int(dcol)); sels.append(sel_of(rest))

    # left side: rows of the box, cells to the left of the wall
    for r in range(r0, r1 + 1):
        hits = [c for c in range(c0) if I[r, c] == dcol]
        if hits:
            ray([(r, c) for c in range(min(hits), c0)])
    # right side
    for r in range(r0, r1 + 1):
        hits = [c for c in range(c1 + 1, wi) if I[r, c] == dcol]
        if hits:
            ray([(r, c) for c in range(max(hits), c1, -1)])
    # top side
    for c in range(c0, c1 + 1):
        hits = [r for r in range(r0) if I[r, c] == dcol]
        if hits:
            ray([(r, c) for r in range(min(hits), r0)])
    # bottom side
    for c in range(c0, c1 + 1):
        hits = [r for r in range(r1 + 1, hi) if I[r, c] == dcol]
        if hits:
            ray([(r, c) for r in range(max(hits), r1, -1)])

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 2c608aff"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2c608aff"
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
                                f"for task 2c608aff"
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
                    f"Failed to build a complete episode for task 2c608aff "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2c608aff-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
