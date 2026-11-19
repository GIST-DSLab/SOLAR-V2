"""
ARC Task: 9ecd008a (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    v = int(round(a + (b - a) * random.uniform(diff_lb, diff_ub)))
    return max(a, min(b, v))


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level colour role; the rule (restore the
    # zeroed hole from the grid's mirror symmetry) does not depend on fg colours.
    bgc = random.choice(range(1, 10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(range(1, 10))
    hmax = max(3, min(15, max_h // 2, max_w // 2))
    h = _unifint(diff_lb, diff_ub, (3, hmax))
    w = h
    remcols = [c for c in range(1, 10) if c != bgc]
    numcols = _unifint(diff_lb, diff_ub, (1, 8))
    remcols = random.sample(remcols, numcols)
    g = np.full((h, w), bgc, dtype=int)
    nc = _unifint(diff_lb, diff_ub, (1, h * w))
    bx = {(i, j) for i in range(h) for j in range(w)}
    start = random.choice(sorted(bx))
    objcells = {start}
    g[start] = random.choice(remcols)
    for _ in range(nc - 1):
        dns = set()
        for (i, j) in objcells:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        dns.add((i + di, j + dj))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        g[ch] = random.choice(remcols)
        objcells.add(ch)
        bx = bx - {ch}
    # make the quadrant diagonal-symmetric (upper triangle taken from transpose)
    gt = g.T.copy()
    for i in range(h):
        for j in range(w):
            if j >= i:
                g[i, j] = gt[i, j]
    g = np.hstack([g, g[:, ::-1]])
    g = np.vstack([g, g[::-1, :]])
    locidev = _unifint(diff_lb, diff_ub, (1, 2 * h))
    locjdev = _unifint(diff_lb, diff_ub, (1, w))
    loci = 2 * h - locidev
    locj = w - locjdev
    loci2 = _unifint(diff_lb, diff_ub, (loci, 2 * h - 1))
    locj2 = _unifint(diff_lb, diff_ub, (locj, w - 1))
    go = g[loci:loci2 + 1, locj:locj2 + 1].copy()
    gi = g.copy()
    gi[loci:loci2 + 1, locj:locj2 + 1] = 0
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # the hole = the zeroed rectangle
    zs = np.argwhere(I == 0)
    r0, c0 = zs.min(axis=0)
    r1, c1 = zs.max(axis=0)
    hh, ww = int(r1 - r0 + 1), int(c1 - c0 + 1)

    # mirror-image source regions of the hole
    vsrc = (int(r0), int(wi - 1 - c1))   # left<->right mirror
    hsrc = (int(hi - 1 - r1), int(c0))   # up<->down mirror

    def region(rc):
        r, c = rc
        return I[r:r + hh, c:c + ww]

    # pick the mirror whose source region is intact (fewest zeros)
    if int((region(vsrc) == 0).sum()) <= int((region(hsrc) == 0).sum()):
        src, flip_op = vsrc, 26
        raw = region(vsrc)
        flipped = raw[:, ::-1]
    else:
        src, flip_op = hsrc, 27
        raw = region(hsrc)
        flipped = raw[::-1, :]

    ops, sels = [], []
    # copy the mirror-image rectangle from the input (whole rectangle intended)
    ops.append(28); sels.append([src[0], src[1], hh - 1, ww - 1])
    # paste it into the hole
    ops.append(30); sels.append([int(r0), int(c0), 0, 0])
    # mirror the pasted rectangle in place (whole hole rectangle intended)
    if not np.array_equal(raw, flipped):
        ops.append(flip_op); sels.append([int(r0), int(c0), hh - 1, ww - 1])
    # crop the canvas to the restored hole
    ops.append(33); sels.append([int(r0), int(c0), hh - 1, ww - 1])
    ops.append(34); sels.append([0, 0, hh - 1, ww - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9ecd008a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9ecd008a"
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
                                f"for task 9ecd008a"
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
                    f"Failed to build a complete episode for task 9ecd008a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9ecd008a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
