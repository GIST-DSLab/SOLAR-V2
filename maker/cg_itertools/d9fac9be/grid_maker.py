"""
ARC Task: d9fac9be (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # The generator samples three distinct colours: background, the "noise"
    # colour (which also fills the ring centre) and the ring colour.
    bgc, noisec, ringc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "ringc": ringc}


def _ring_occurrences(g, col):
    # Top-left corners (r, c) of every 3x3 window whose 8 border cells all equal col.
    h, w = g.shape
    occ = []
    for r in range(h - 2):
        for c in range(w - 2):
            win = g[r:r + 3, c:c + 3]
            ok = True
            for dr in range(3):
                for dc in range(3):
                    if dr == 1 and dc == 1:
                        continue
                    if win[dr, dc] != col:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                occ.append((r, c))
    return occ


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, noisec=None, ringc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if bgc is None or noisec is None or ringc is None:
        bgc, noisec, ringc = random.sample(range(10), 3)

    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    h = max(6, min(h, max_h))
    w = max(6, min(w, max_w))

    gi = np.full((h, w), bgc, dtype=int)
    nnoise1 = unifint(diff_lb, diff_ub, (1, (h * w) // 3 - 1))
    nnoise2 = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 3 - 9)))
    inds = [(r, c) for r in range(h) for c in range(w)]
    nnoise1 = max(1, min(nnoise1, len(inds)))
    noise1 = random.sample(inds, nnoise1)
    s1 = set(noise1)
    rest = [p for p in inds if p not in s1]
    nnoise2 = max(1, min(nnoise2, len(rest)))
    noise2 = random.sample(rest, nnoise2)
    for (r, c) in noise1:
        gi[r, c] = noisec
    for (r, c) in noise2:
        gi[r, c] = ringc

    ring = [(dr, dc) for dr in range(3) for dc in range(3) if not (dr == 1 and dc == 1)]
    # Break accidental full rings of either foreground colour (as the generator does).
    for (r0, c0) in _ring_occurrences(gi, noisec):
        dr, dc = random.choice(ring)
        gi[r0 + dr, c0 + dc] = random.choice((bgc, ringc))
    for (r0, c0) in _ring_occurrences(gi, ringc):
        dr, dc = random.choice(ring)
        gi[r0 + dr, c0 + dc] = random.choice((bgc, noisec))

    loci = random.randint(0, h - 3)
    locj = random.randint(0, w - 3)
    for dr, dc in ring:
        gi[loci + dr, locj + dc] = ringc
    gi[loci + 1, locj + 1] = noisec

    go = np.full((1, 1), noisec, dtype=int)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Rule, read from I alone: exactly one 3x3 window has a uniform
    # non-background 8-cell ring around a centre of the other foreground colour.
    # The answer is that centre's colour.
    #
    # The golf solution reads the task as "output a colour" and picks it by
    # how often each colour appears. That counting trick happens to work on
    # the samples, but the generator does not guarantee it: either noise
    # colour can be the more common one. Treating the answer as a single
    # colour does match the maker, though. The cell that holds that colour,
    # and that can be found without looking at O, is the ring centre. Both
    # readings give the same 1x1 answer, so the operations stay the maker's:
    # crop straight to the ring centre.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # The generator paints the background first and keeps each foreground
    # colour under a third of the grid, so the background is the majority colour.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    cr, cc = 0, 0
    found = False
    for r in range(1, hi - 1):
        for c in range(1, wi - 1):
            ctr = int(I[r, c])
            if ctr == bgc:
                continue
            ring = set()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    ring.add(int(I[r + dr, c + dc]))
            if len(ring) != 1:
                continue
            rc = ring.pop()
            if rc == bgc or rc == ctr:
                continue
            cr, cc, found = r, c, True
            break
        if found:
            break

    ops, sels = [], []
    # CropGrid onto the single ring-centre cell (a 1x1 rectangle): that pixel
    # becomes the whole grid.
    ops.append(33)
    sels.append([cr, cc, 0, 0])
    ops.append(34)
    sels.append([0, 0, 0, 0])
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
                        f"num_examples+1 ({num_examples + 1}) for task d9fac9be"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d9fac9be"
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
                                f"for task d9fac9be"
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
                    f"Failed to build a complete episode for task d9fac9be "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d9fac9be-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
