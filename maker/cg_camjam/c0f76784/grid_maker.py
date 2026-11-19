"""
ARC Task: c0f76784 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque

import numpy as np

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Colours 6, 7, 8 are reserved for the hole fills, so they are never used as
    # the background or as frame colours.
    cols = [c for c in range(10) if c not in (6, 7, 8)]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(1, len(remcols))
    ccols = random.sample(remcols, numcols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    cols = [c for c in range(10) if c not in (6, 7, 8)]
    if bgc is None:
        bgc = random.choice(cols)
    if not ccols:
        rem = [c for c in cols if c != bgc]
        ccols = random.sample(rem, random.randint(1, len(rem)))

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
    indss = {(i, j) for i in range(h) for j in range(w)}
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if not indss:
            break
        oh = random.choice((3, 4, 5))
        ow = oh
        subs = [ij for ij in indss if ij[0] < h - oh and ij[1] < w - ow]
        if not subs:
            tr += 1
            continue
        loci, locj = random.choice(subs)
        bd = {(loci + a, locj + b) for a in range(oh) for b in range(ow)}
        col = random.choice(ccols)
        if bd.issubset(indss):
            for (a, b) in bd:
                gi[a][b] = col
                go[a][b] = col
            ccc = oh + 3
            for a in range(loci + 1, loci + oh - 1):
                for b in range(locj + 1, locj + ow - 1):
                    gi[a][b] = bgc
                    go[a][b] = ccc
            outbox = set()
            for a in range(loci - 1, loci + oh + 1):
                for b in range(locj - 1, locj + ow + 1):
                    if (a, b) not in bd:
                        outbox.add((a, b))
            indss = (indss - bd) - outbox
            succ += 1
        tr += 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the canvas colour the generator paints first (the dominant colour).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Reading taken from the golf solution: a box's hole is read ROW BY ROW. Each row
    # of the hole is a run of background lying between two cells of the frame, and the
    # fill colour is 5 + the run's length (1 -> 6, 2 -> 7, 3 -> 8). Because the hole is
    # square, every row of one hole has the same length, so each box gets one colour.
    # Each Color op covers one interior row: a run of bgc cells closed in on the left
    # and right by the frame and cut off from the outer background, which can be seen
    # in I alone.
    seen = np.zeros((hi, wi), dtype=bool)
    holes = []
    for sr in range(hi):
        for sc in range(wi):
            if I[sr, sc] != bgc or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells = []
            touches_border = False
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                if y in (0, hi - 1) or x in (0, wi - 1):
                    touches_border = True
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == bgc:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if touches_border:
                continue
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
            k = r1 - r0 + 1
            if k != c1 - c0 + 1 or k * k != len(cells) or k not in (1, 2, 3):
                continue
            # The surrounding ring must be a single frame colour.
            ring = set()
            for a in range(r0 - 1, r1 + 2):
                for b in range(c0 - 1, c1 + 2):
                    if not (r0 <= a <= r1 and c0 <= b <= c1):
                        ring.add(int(I[a, b]))
            if len(ring) != 1 or bgc in ring:
                continue
            holes.append((r0, c0, k))

    ops, sels = [], []
    for (r0, c0, k) in holes:              # finish one box before starting the next
        colour = 5 + k                     # run length between the frame walls -> colour
        for r in range(r0, r0 + k):        # one interior row at a time, top to bottom
            ops.append(colour)
            sels.append(sel_of([(r, c) for c in range(c0, c0 + k)]))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c0f76784"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c0f76784"
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
                                f"for task c0f76784"
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
                    f"Failed to build a complete episode for task c0f76784 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c0f76784-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
