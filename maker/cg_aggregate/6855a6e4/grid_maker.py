"""
ARC Task: 6855a6e4 (RE-ARC) — LLM-generated grid_maker
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

from maker.sel_helpers import sel_of  # noqa: F401  (all selections here are full rectangles)


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = a + int(round((b - a) * diff_lb))
    hi = a + int(round((b - a) * diff_ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    return random.randint(min(lo, hi), max(lo, hi))


ORIENTATIONS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    bgc, objc, boxc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENTATIONS):
        examples = [dict(v) for v in ORIENTATIONS]
        examples += [dict(random.choice(ORIENTATIONS)) for _ in range(n_ex - len(ORIENTATIONS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(ORIENTATIONS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "boxc": boxc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, objc=1, boxc=2,
             transpose=None, **_ignored) -> dict:
    # Frame = two parallel rails, each with two inward stubs. Each pattern outside
    # a rail is MIRRORED ACROSS THAT RAIL into the frame (line-by-line reflection
    # about the rail row/column). This is the reading the examples follow in every
    # case where they agree with one another.
    if transpose is None:
        transpose = random.choice([True, False])
    if transpose and (max_w < 10 or max_h < 4):
        transpose = False
    if not transpose and (max_h < 10 or max_w < 4):
        transpose = True
    HL, WL = (max_w, max_h) if transpose else (max_h, max_w)
    HL, WL = min(HL, 30), min(WL, 30)

    h = _unifint(diff_lb, diff_ub, (10, HL))
    w = _unifint(diff_lb, diff_ub, (4, WL))
    fullh = _unifint(diff_lb, diff_ub, (10, h))
    fullw = _unifint(diff_lb, diff_ub, (3, w))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    loci = random.randint(0, h - fullh)
    locj = random.randint(0, w - fullw)
    objh = (fullh // 2 - 3) // 2
    cands = [(i, j) for i in range(objh) for j in range(1, fullw - 1)]
    tot = len(cands)
    for k in range(2):
        mr = (lambda i: i) if k == 0 else (lambda i, fh=fullh: fh - 1 - i)
        frame = [(objh + 1, j) for j in range(fullw)] + [(objh + 2, 0), (objh + 2, fullw - 1)]
        for i, j in frame:
            gi[loci + mr(i)][locj + j] = boxc
            go[loci + mr(i)][locj + j] = boxc
        nd = _unifint(diff_lb, diff_ub, (0, tot // 2))
        n = min(max(1, random.choice((nd, tot - nd))), tot)
        for i, j in random.sample(cands, n):
            gi[loci + mr(i)][locj + j] = objc
            # reflection about the rail row objh+1
            go[loci + mr(2 * objh + 2 - i)][locj + j] = objc
    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    import numpy as np
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []

    # --- the frame: two full rails + four inward stubs, all on its bbox border ---
    frame = None
    for c in sorted(set(I.flatten().tolist())):
        pts = np.argwhere(I == c)
        r0, r1 = int(pts[:, 0].min()), int(pts[:, 0].max())
        c0, c1 = int(pts[:, 1].min()), int(pts[:, 1].max())
        hh, ww, n = r1 - r0 + 1, c1 - c0 + 1, len(pts)
        if hh >= 4 and ww >= 3 and n == 2 * ww + 4 \
                and (I[r0, c0:c1 + 1] == c).all() and (I[r1, c0:c1 + 1] == c).all() \
                and all(I[a, b] == c for a, b in ((r0 + 1, c0), (r0 + 1, c1),
                                                  (r1 - 1, c0), (r1 - 1, c1))):
            frame = (c, True, r0, r1, c0, c1)
            break
        if ww >= 4 and hh >= 3 and n == 2 * hh + 4 \
                and (I[r0:r1 + 1, c0] == c).all() and (I[r0:r1 + 1, c1] == c).all() \
                and all(I[a, b] == c for a, b in ((r0, c0 + 1), (r1, c0 + 1),
                                                  (r0, c1 - 1), (r1, c1 - 1))):
            frame = (c, False, r0, r1, c0, c1)
            break
    boxc, horizontal, r0, r1, c0, c1 = frame
    # the cell just inside a stub is always background
    bgc = int(I[r0 + 1, c0 + 1])
    obj = np.argwhere((I != boxc) & (I != bgc))

    # Each pattern is reflected ACROSS ITS OWN RAIL: select the full rectangle
    # spanning the gap between the stubs, symmetric about the rail line and just
    # reaching the pattern's farthest line, and flip it whole. The rail maps onto
    # itself, the outside pattern lands mirrored inside the frame, and the empty
    # inside lands outside. (Intended cells are exactly that full rectangle.)
    if horizontal:
        top = [int(r) for r, _ in obj if r < r0]
        bot = [int(r) for r, _ in obj if r > r1]
        if top:
            k = r0 - min(top)
            ops.append(27); sels.append([r0 - k, c0 + 1, 2 * k, c1 - c0 - 2])
        if bot:
            k = max(bot) - r1
            ops.append(27); sels.append([r1 - k, c0 + 1, 2 * k, c1 - c0 - 2])
    else:
        lef = [int(c) for _, c in obj if c < c0]
        rig = [int(c) for _, c in obj if c > c1]
        if lef:
            k = c0 - min(lef)
            ops.append(26); sels.append([r0 + 1, c0 - k, r1 - r0 - 2, 2 * k])
        if rig:
            k = max(rig) - c1
            ops.append(26); sels.append([r0 + 1, c1 - k, r1 - r0 - 2, 2 * k])

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6855a6e4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6855a6e4"
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
                                f"for task 6855a6e4"
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
                    f"Failed to build a complete episode for task 6855a6e4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6855a6e4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
