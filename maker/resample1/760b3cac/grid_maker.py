"""
ARC Task: 760b3cac (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    # object colour kept non-zero so CopyI/Paste carry every object cell
    objc = random.choice([c for c in cols if c not in (bgc, 0)])
    indc = random.choice([c for c in cols if c not in (bgc, objc)])
    return {"bgc": bgc, "objc": objc, "indc": indc}


def _is_indicator(cells):
    cells = set(cells)
    if len(cells) != 5:
        return False
    rs = [r for r, c in cells]; cs = [c for r, c in cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    if r1 - r0 != 2 or c1 - c0 != 2:
        return False
    box = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
           if r in (r0, r1) or c in (c0, c1)}
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    return len(cells & box) == 4 and len(cells & corners) == 1


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, indc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, round(a + (b - a) * d)))
    if bgc is None:
        bgc, objc, indc = random.sample(range(10), 3)
    objL = {(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)}
    objR = {(r, 2 - c) for r, c in objL}
    hmax = min(30, max_h, max_w)
    wmax = (min(30, max_w, max_h) - 1) // 2
    for _ in range(200):
        h = unifint(diff_lb, diff_ub, (5, hmax))
        w = unifint(diff_lb, diff_ub, (3, max(3, wmax)))
        w = 2 * w + 1
        if w > min(max_h, max_w) or h > min(max_h, max_w):
            continue
        objh = unifint(diff_lb, diff_ub, (1, h - 3))
        objw = unifint(diff_lb, diff_ub, (1, w // 6))
        objw = 2 * objw + 1
        gi = np.full((h, w), bgc, dtype=int)
        if random.choice((True, False)):
            obj, sgn = objL, -1
        else:
            obj, sgn = objR, 1
        for r, c in obj:
            gi[h - 3 + r, w // 2 - 1 + c] = indc
        inds = {(i, j) for i in range(objh) for j in range(objw)}
        sp = random.choice(sorted(inds))
        objx = {sp}

        def nb():
            s = set()
            for (i, j) in objx:
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di or dj:
                            s.add((i + di, j + dj))
            return sorted((inds - objx) & s)
        numcd = unifint(diff_lb, diff_ub, (0, (objh * objw) // 2))
        numc = random.choice((numcd, objh * objw - numcd))
        numc = min(max(1, numc), objh * objw)
        for _k in range(numc - 1):
            objx.add(random.choice(nb()))
        while max(j for i, j in objx) - min(j for i, j in objx) + 1 != objw:
            objx.add(random.choice(nb()))
        mi = min(i for i, j in objx); mj = min(j for i, j in objx)
        objx = {(i - mi, j - mj) for i, j in objx}
        if _is_indicator(objx):
            continue
        oh = max(i for i, j in objx) + 1
        ow = max(j for i, j in objx) + 1
        loci = random.randint(0, h - 3 - oh)
        locj = w // 2 - ow // 2
        plcd = {(i + loci, j + locj) for i, j in objx}
        for i, j in plcd:
            gi[i, j] = objc
        go = gi.copy()
        if locj + ow * sgn < 0 or locj + ow - 1 + ow * sgn >= w:
            continue
        for i, j in plcd:
            jj = 2 * locj + ow - 1 - j + ow * sgn
            go[i, jj] = objc
        k = random.choice((0, 1, 2, 3))
        gi = np.rot90(gi, -k); go = np.rot90(go, -k)
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("failed")


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    groups = {}
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc:
                groups.setdefault(int(I[r, c]), []).append((r, c))
    ind = None
    for col, cells in groups.items():
        if _is_indicator(cells):
            ind = col
            break
    objcol = [c for c in groups if c != ind][0]
    ic = groups[ind]
    rs = [r for r, c in ic]; cs = [c for r, c in ic]
    down = sum(1 for r, c in ic if r == max(rs)) == 2
    up = sum(1 for r, c in ic if r == min(rs)) == 2
    right = sum(1 for r, c in ic if c == max(cs)) == 2
    left = sum(1 for r, c in ic if c == min(cs)) == 2
    dr = int(down) - int(up)
    dc = int(right) - int(left)
    oc = groups[objcol]
    r0 = min(r for r, c in oc); r1 = max(r for r, c in oc)
    c0 = min(c for r, c in oc); c1 = max(c for r, c in oc)
    oh, ow = r1 - r0 + 1, c1 - c0 + 1
    tr, tc = r0 + dr * oh, c0 + dc * ow
    ops, sels = [], []
    # copy the object's full bounding rectangle from the input
    ops.append(28); sels.append([r0, c0, oh - 1, ow - 1])
    # paste it right next to the object, on the side the indicator points to
    ops.append(30); sels.append([tr, tc, 0, 0])
    # mirror the pasted copy in place (full destination rectangle, bg included)
    block = I[r0:r1 + 1, c0:c1 + 1]
    vertical = (dr != 0)
    flipped = block[::-1, :] if vertical else block[:, ::-1]
    if not np.array_equal(flipped, block):
        ops.append(27 if vertical else 26); sels.append([tr, tc, oh - 1, ow - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 760b3cac"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 760b3cac"
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
                                f"for task 760b3cac"
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
                    f"Failed to build a complete episode for task 760b3cac "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"760b3cac-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
