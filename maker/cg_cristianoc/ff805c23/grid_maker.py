"""
ARC Task: ff805c23 (RE-ARC) — LLM-generated grid_maker
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

# How this differs from the maker: the maker flips the WHOLE mosaic and then crops
# where the hole was. The golf entry reads the task another way. The hole is a gap in
# a picture that is symmetric both ways, so it gets REPAIRED IN PLACE from its intact
# mirror counterpart, and then the repaired rectangle is cut out. The operations below
# follow that reading. Copy the intact counterpart block from the input, paste it onto
# the hole, mirror it in place, and crop to the hole rectangle. Both readings use the
# same crop rectangle (the hole), but the repair is a different unit of work.


def sample_colors(num_examples=None) -> dict:
    # Only the background is sampled as an episode-level colour. The rule (fill the
    # 0-hole from its mirror) does not depend on which foreground colours are used.
    return {"bgc": random.randint(1, 9)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    def unifint(lo, hi):
        a = lo + (hi - lo) * diff_lb
        b = lo + (hi - lo) * diff_ub
        return max(lo, min(hi, random.randint(int(round(a)), int(round(b)))))

    if bgc is None:
        bgc = random.randint(1, 9)
    hmax = max(3, min(15, max_h // 2, max_w // 2))
    h = unifint(3, hmax)
    w = h
    remcols = [c for c in range(1, 10) if c != bgc]
    numcols = unifint(1, 8)
    remcols = random.sample(remcols, numcols)
    g = np.full((h, w), bgc, dtype=int)
    nc = unifint(1, h * w)
    bx = {(i, j) for i in range(h) for j in range(w)}
    start = random.choice(sorted(bx))
    obj = {start: random.choice(remcols)}
    for _ in range(nc - 1):
        dns = set()
        for (i, j) in obj:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        dns.add((i + di, j + dj))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        obj[ch] = random.choice(remcols)
        bx.discard(ch)
    for (i, j), v in obj.items():
        g[i, j] = v
    for i in range(h):
        for j in range(i, w):
            g[i, j] = g[j, i]
    g = np.hstack([g, np.fliplr(g)])
    g = np.vstack([g, np.flipud(g)])
    locidev = unifint(1, 2 * h)
    locjdev = unifint(1, w)
    loci = 2 * h - locidev
    locj = w - locjdev
    loci2 = unifint(loci, 2 * h - 1)
    locj2 = unifint(locj, w - 1)
    go = g[loci:loci2 + 1, locj:locj2 + 1].copy()
    gi = g.copy()
    gi[loci:loci2 + 1, locj:locj2 + 1] = 0
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, -k)
    go = np.rot90(go, -k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape

    # The hole is the solid block of 0s. Mosaic colours are always 1-9.
    hole = np.argwhere(I == 0)
    r0, c0 = hole.min(0).tolist()
    r1, c1 = hole.max(0).tolist()
    hh, ww = r1 - r0 + 1, c1 - c0 + 1

    def disjoint(a0, a1, b0, b1):
        return a1 < r0 or a0 > r1 or b1 < c0 or b0 > c1

    # The mosaic is mirror-symmetric both up/down and left/right. The hole always lies
    # within one half, so at least one counterpart block is intact.
    # Prefer the up/down counterpart, as the golf entry does.
    mr0, mr1 = H - 1 - r1, H - 1 - r0
    if disjoint(mr0, mr1, c0, c1) and not (I[mr0:mr1 + 1, c0:c1 + 1] == 0).any():
        src = (mr0, c0)
        flip = 27                                   # FlipV (up/down)
        block = I[mr0:mr1 + 1, c0:c1 + 1]
    else:
        mc0 = W - 1 - c1
        src = (r0, mc0)
        flip = 26                                   # FlipH (left/right)
        block = I[r0:r1 + 1, mc0:mc0 + ww]

    ops, sels = [], []
    # 1. Copy the intact mirror counterpart from the input. The selection is that full
    #    rectangle, and every cell in it is non-zero.
    ops.append(28); sels.append([src[0], src[1], hh - 1, ww - 1])
    # 2. Paste it onto the hole. Every 0 in the hole gets overwritten.
    ops.append(30); sels.append([r0, c0, 0, 0])
    # 3. Mirror the patched rectangle in place so it matches its true orientation.
    #    The selection is exactly the full hole rectangle. Skip this step when the block
    #    is self-symmetric, because the flip would then change nothing.
    flipped = np.flipud(block) if flip == 27 else np.fliplr(block)
    if not np.array_equal(flipped, block):
        ops.append(flip); sels.append([r0, c0, hh - 1, ww - 1])
    # 4. Cut out the repaired hole rectangle. The selection is exactly that rectangle.
    ops.append(33); sels.append([r0, c0, hh - 1, ww - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task ff805c23"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ff805c23"
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
                                f"for task ff805c23"
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
                    f"Failed to build a complete episode for task ff805c23 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ff805c23-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
