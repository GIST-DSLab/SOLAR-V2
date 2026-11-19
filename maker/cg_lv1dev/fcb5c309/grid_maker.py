"""
ARC Task: fcb5c309 (RE-ARC) — LLM-generated grid_maker
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

try:
    from dsl import *
except Exception:
    pass
try:
    from utils import unifint
except Exception:
    pass


def sample_colors(num_examples=None) -> dict:
    bgc, dotc, sqc = random.sample(range(10), 3)
    return {"bgc": bgc, "dotc": dotc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc, sqc) -> dict:
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    numsq = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    maxtr = 4 * numsq
    tr = 0
    succ = 0
    numcells = None
    take = False
    go = None
    while tr < maxtr and succ < numsq:
        oh = randint(3, 7)
        ow = randint(3, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        loci, locj = loc
        sq = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        bd = backdrop(sq)
        if bd.issubset(inds):
            gi = fill(gi, sqc, sq)
            ib = backdrop(inbox(sq))
            if numcells is None:
                numcells = unifint(diff_lb, diff_ub, (1, len(ib)))
                cells = sample(totuple(ib), numcells)
                take = True
            else:
                nc = unifint(diff_lb, diff_ub, (0, min(max(0, numcells - 1), len(ib))))
                cells = sample(totuple(ib), nc)
            gi = fill(gi, dotc, cells)
            if take:
                go = replace(subgrid(sq, gi), sqc, dotc)
                take = False
            inds = (inds - bd) - outbox(bd)
            succ += 1
        tr += 1
    if go is None:
        raise ValueError("no square placed")
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, len(inds) // 2 - 1)))
    noise = sample(totuple(inds), nnoise)
    gi = fill(gi, dotc, noise)
    return {"input": gi, "output": go}


def _components(I, col):
    h, w = I.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == col and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                comps.append(cells)
    return comps


def _is_box(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    ring = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)}
    return set(cells) == ring


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    cnt = Counter(I.flatten().tolist())
    colors = list(cnt.keys())

    sqc = dotc = None
    # Episode convention from the demonstrations: the frame colour is the input
    # colour that never shows up in any output; the dot colour is the output's border colour.
    if examples:
        try:
            ex_in = [np.asarray(a, dtype=int) for a, _ in examples]
            ex_out = [np.asarray(b, dtype=int) for _, b in examples]
            in_cols = set()
            for a in ex_in:
                in_cols |= set(a.flatten().tolist())
            out_cols = set()
            for b in ex_out:
                out_cols |= set(b.flatten().tolist())
            cand_sq = [c for c in in_cols if c not in out_cols]
            if len(cand_sq) == 1 and cand_sq[0] in cnt:
                sqc = cand_sq[0]
                dotc = int(ex_out[0][0, 0])
        except Exception:
            sqc = dotc = None

    if sqc is None:
        # Fallback from the verifier: the frame colour is the one whose components are all boxes
        best_n = -1
        for col in colors:
            comps = _components(I, col)
            if comps and all(_is_box(cc) for cc in comps) and len(comps) > best_n:
                best_n = len(comps)
                sqc = col
        remaining = [c for c in colors if c != sqc]
        dotc = min(remaining, key=lambda c: cnt[c])

    boxes = [cc for cc in _components(I, sqc) if _is_box(cc)]

    def dots_inside(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return int(np.sum(I[min(rs):max(rs) + 1, min(cs):max(cs) + 1] == dotc))

    chosen = max(boxes, key=dots_inside)
    rs = [r for r, _ in chosen]
    cs = [c for _, c in chosen]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    bh, bw = r1 - r0 + 1, c1 - c0 + 1

    ops, sels = [], []
    # 1. Crop the canvas to the chosen box's full rectangle (frame + interior).
    ops.append(33); sels.append([r0, c0, bh - 1, bw - 1])
    # 2. The cropped grid's outer ring is the frame: repaint it in the dot colour.
    ring = [(r, c) for r in range(bh) for c in range(bw)
            if r in (0, bh - 1) or c in (0, bw - 1)]
    ops.append(int(dotc)); sels.append(sel_of(ring))
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
                        f"num_examples+1 ({num_examples + 1}) for task fcb5c309"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task fcb5c309"
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
                                f"for task fcb5c309"
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
                    f"Failed to build a complete episode for task fcb5c309 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"fcb5c309-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
