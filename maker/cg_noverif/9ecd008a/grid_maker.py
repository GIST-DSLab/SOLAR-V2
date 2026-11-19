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


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level color role; the object palette is
    # free per instance (the rule reads the picture's own symmetry, not its colors).
    bgc = random.choice(range(1, 10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(range(1, 10))
    hmax = max(3, min(15, max_h // 2, max_w // 2))
    h = random.randint(3, hmax)
    w = h
    remcols = [c for c in range(1, 10) if c != bgc]
    numcols = random.randint(1, 8)
    remcols = random.sample(remcols, numcols)

    g = [[bgc] * w for _ in range(h)]
    nc = random.randint(1, h * w)
    allcells = {(i, j) for i in range(h) for j in range(w)}
    start = random.choice(sorted(allcells))
    obj = {start: random.choice(remcols)}
    bx = set(allcells)
    for _ in range(nc - 1):
        dns = set()
        for (i, j) in obj:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    dns.add((i + di, j + dj))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        obj[ch] = random.choice(remcols)
        bx.discard(ch)
    for (i, j), c in obj.items():
        g[i][j] = c
    # make the tile symmetric about its main diagonal
    gT = [[g[j][i] for j in range(h)] for i in range(w)]
    for i in range(h):
        for j in range(w):
            if j >= i:
                g[i][j] = gT[i][j]
    g = [row + row[::-1] for row in g]
    g = g + g[::-1]
    gi = np.array(g, dtype=int)

    loci = 2 * h - random.randint(1, 2 * h)
    locj = w - random.randint(1, w)
    loci2 = random.randint(loci, 2 * h - 1)
    locj2 = random.randint(locj, w - 1)
    go = gi[loci:loci2 + 1, locj:locj2 + 1].copy()
    gi[loci:loci2 + 1, locj:locj2 + 1] = 0

    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # The hole: the rectangle of 0s punched out of a mirror-symmetric picture.
    zr, zc = np.where(I == 0)
    r0, c0 = int(zr.min()), int(zc.min())
    hh = int(zr.max()) - r0 + 1
    ww = int(zc.max()) - c0 + 1

    # Reading: each row through the hole holds its missing segment at the mirrored
    # columns on the other side of the picture, just written backwards.
    # Take that mirrored segment block (crop it out), then reverse it.
    mc0 = wi - c0 - ww                       # left column of the left-right mirror image of the hole
    lr_block = I[r0:r0 + hh, mc0:mc0 + ww]
    if mc0 >= 0 and not (lr_block == 0).any():
        src = [r0, mc0, hh - 1, ww - 1]
        flip_op = 26                         # FlipH: reverse each row segment
    else:
        # hole spans the left/right divide -> its intact twin lies across the up/down divide
        mr0 = hi - r0 - hh
        src = [mr0, c0, hh - 1, ww - 1]
        flip_op = 27                         # FlipV: reverse each column segment

    ops, sels = [], []
    # Crop to the hole's mirror-image rectangle (a full rectangle, all non-zero content)
    ops.append(33); sels.append(src)
    # Reverse it in place: whole remaining canvas, full rectangle intended
    full = [0, 0, hh - 1, ww - 1]
    flipped = np.fliplr(I[src[0]:src[0] + hh, src[1]:src[1] + ww]) if flip_op == 26 \
        else np.flipud(I[src[0]:src[0] + hh, src[1]:src[1] + ww])
    if not np.array_equal(flipped, I[src[0]:src[0] + hh, src[1]:src[1] + ww]):
        ops.append(flip_op); sels.append(full)
    ops.append(34); sels.append(full)
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
