"""
ARC Task: 7f4411dc (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, fgc = random.sample(range(10), 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        d = random.uniform(lb, ub)
        return min(max(a, round(a + (b - a) * d)), b)

    if bgc is None or fgc is None:
        bgc, fgc = random.sample(range(10), 2)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 15)))
    maxtr = 4 * nsq
    tr = 0
    succ = 0
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    while tr < maxtr and succ < nsq:
        oh = random.randint(2, 6)
        ow = random.randint(2, 6)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if len(cands) == 0:
            break
        loci, locj = random.choice(sorted(cands))
        obj = {(loci + a, locj + b) for a in range(oh) for b in range(ow)}
        if obj.issubset(inds):
            for (a, b) in obj:
                go[a][b] = fgc
            succ += 1
            outbox = set()
            for a in range(loci - 1, loci + oh + 1):
                for b in range(locj - 1, locj + ow + 1):
                    if (a, b) not in obj:
                        outbox.add((a, b))
            inds = (inds - obj) - outbox
        tr += 1
    inds = {(i, j) for i in range(h) for j in range(w) if go[i][j] == bgc}
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, len(inds) // 2 - 1)))
    gi = [row[:] for row in go]
    for _ in range(nnoise):
        if not inds:
            break
        loc = random.choice(sorted(inds))
        i, j = loc
        inds = inds - {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
        gi[i][j] = fgc
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    # Rule: the grid holds solid fgc rectangles (sides 2..6) plus single-pixel fgc
    # noise.  A fgc cell survives iff it lies in some solid 2x2 fgc block; every other
    # fgc cell is an isolated noise pixel and is repainted to the background colour.
    # (A neighbour-count reading "keep if >=2 same neighbours" breaks on noise that
    # sits in the 1-cell gap between two rectangles, so the 2x2-block test is used.)
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    def blocks(color):
        m = (I == color)
        cov = np.zeros_like(m)
        for r in range(hi - 1):
            for c in range(wi - 1):
                if m[r, c] and m[r + 1, c] and m[r, c + 1] and m[r + 1, c + 1]:
                    cov[r:r + 2, c:c + 2] = True
        noise = [(r, c) for r in range(hi) for c in range(wi) if m[r, c] and not cov[r, c]]
        return cov, noise

    def is_foreground(cov, noise):
        seen = np.zeros_like(cov)
        for r in range(hi):
            for c in range(wi):
                if not cov[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and cov[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                rs = [p[0] for p in cells]
                cs = [p[1] for p in cells]
                bh = max(rs) - min(rs) + 1
                bw = max(cs) - min(cs) + 1
                if bh * bw != len(cells) or not (2 <= bh <= 6) or not (2 <= bw <= 6):
                    return False
        nset = set(noise)
        for (r, c) in noise:
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (r + dy, c + dx) in nset:
                    return False
        return True

    colors = sorted(set(I.flatten().tolist()))
    bgc = fgc = None

    # Background convention read from the demonstrations: the colour that
    # dominates every example output (the cleaned canvas).
    if examples:
        cnt = Counter()
        for ex in examples:
            eo = np.asarray(ex[1], dtype=int)
            cnt.update(eo.flatten().tolist())
        if cnt:
            cand_bg = cnt.most_common(1)[0][0]
            if cand_bg in colors:
                others = [c for c in colors if c != cand_bg]
                if len(others) == 1:
                    bgc, fgc = cand_bg, others[0]

    if bgc is None and len(colors) == 2:
        cands = []
        for col in colors:
            cov, noise = blocks(col)
            if is_foreground(cov, noise):
                cands.append((len(noise), col))
        if cands:
            cands.sort()
            fgc = cands[-1][1]
            bgc = colors[0] if colors[1] == fgc else colors[1]

    if bgc is not None and fgc is not None:
        _, noise = blocks(fgc)
        # each noise pixel is its own isolated object: erase it to background
        for (r, c) in noise:
            ops.append(int(bgc))
            sels.append(sel_of([(int(r), int(c))]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 7f4411dc"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7f4411dc"
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
                                f"for task 7f4411dc"
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
                    f"Failed to build a complete episode for task 7f4411dc "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7f4411dc-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
