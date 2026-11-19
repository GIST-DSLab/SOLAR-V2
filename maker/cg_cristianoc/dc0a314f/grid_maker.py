"""
ARC Task: dc0a314f (RE-ARC) — LLM-generated grid_maker
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
    # 3 is hardcoded as the occluder colour; only the background is a free role.
    cols = [c for c in range(10) if c != 3]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def _unifint(lb, ub, rng):
    a, b = rng
    if b <= a:
        return a
    d = random.uniform(lb, ub)
    return max(a, min(b, a + int(round((b - a) * d))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    if bgc is None:
        bgc = random.choice(cols)
    hmax = max(3, min(15, min(max_h, max_w) // 2))
    h = _unifint(diff_lb, diff_ub, (3, hmax))
    w = h
    remcols = [c for c in cols if c != bgc]
    numcols = _unifint(diff_lb, diff_ub, (1, 8))
    remcols = random.sample(remcols, min(numcols, len(remcols)))
    canv = [[bgc] * w for _ in range(h)]
    nc = _unifint(diff_lb, diff_ub, (1, h * w))
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
        bx = bx - {ch}
    for (i, j), v in obj.items():
        canv[i][j] = v
    # diagonal symmetry: cells with j >= i take value from the transpose
    g = [row[:] for row in canv]
    for i in range(h):
        for j in range(w):
            if j >= i:
                g[i][j] = canv[j][i]
    g = [row + row[::-1] for row in g]
    g = g + g[::-1]
    H, W = 2 * h, 2 * w
    locidev = _unifint(diff_lb, diff_ub, (1, 2 * h))
    locjdev = _unifint(diff_lb, diff_ub, (1, w))
    loci = 2 * h - locidev
    locj = w - locjdev
    loci2 = _unifint(diff_lb, diff_ub, (loci, 2 * h - 1))
    locj2 = _unifint(diff_lb, diff_ub, (locj, w - 1))
    go = [row[locj:locj2 + 1] for row in g[loci:loci2 + 1]]
    gi = [row[:] for row in g]
    for i in range(loci, loci2 + 1):
        for j in range(locj, locj2 + 1):
            gi[i][j] = 3
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(np.array(gi), -k).tolist()   # rot90 (CW) applied k times
    go = np.rot90(np.array(go), -k).tolist()
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared with the code-golf entry): the picture is symmetric up<->down and
    # left<->right. Each 3-cell takes back the colour of its mirror cell. First pass:
    # fill from the up/down mirror wherever that mirror cell is not 3 itself. Second
    # pass: fill whatever is still 3 from the left/right mirror. Then crop to the block.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    rs, cs = np.where(I == 3)
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())
    block = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    cur = I.copy()

    def fill_pass(mirror):
        # Group the remaining 3-cells by the colour their mirror cell has in I.
        # One Color op per group: "3-cells whose mirror cell is colour k".
        groups = {}
        for (r, c) in block:
            if cur[r, c] != 3:
                continue
            mr, mc = mirror(r, c)
            v = int(I[mr, mc])
            if v == 3:
                continue
            groups.setdefault(v, []).append((r, c))
        for v in sorted(groups):
            cells = groups[v]
            ops.append(v); sels.append(sel_of(cells))
            for (r, c) in cells:
                cur[r, c] = v

    # pass 1: up/down mirror counterpart
    fill_pass(lambda r, c: (hi - 1 - r, c))
    # pass 2: left/right mirror counterpart for anything still occluded
    fill_pass(lambda r, c: (r, wi - 1 - c))

    # crop the canvas to the formerly occluded rectangle (full rect intended)
    bh, bw = r1 - r0 + 1, c1 - c0 + 1
    ops.append(33); sels.append([r0, c0, bh - 1, bw - 1])
    ops.append(34); sels.append([0, 0, bh - 1, bw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task dc0a314f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task dc0a314f"
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
                                f"for task dc0a314f"
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
                    f"Failed to build a complete episode for task dc0a314f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"dc0a314f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
