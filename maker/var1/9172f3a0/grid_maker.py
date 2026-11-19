"""
ARC Task: 9172f3a0 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import Counter, deque

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The rule is a pure 3x upscale: it is color-agnostic for the foreground
    # pixels, so only the canvas/background color has to stay fixed.
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = interval(0, 10, 1)
    # output is a 3x upscale of the input -> input dims bounded by max_h//3, max_w//3
    hcap = max(1, min(10, max_h // 3))
    wcap = max(1, min(10, max_w // 3))
    h = unifint(diff_lb, diff_ub, (1, hcap))
    w = unifint(diff_lb, diff_ub, (1, wcap))
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        num = min(num, len(inds))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = upscale(gi, 3)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Blow every input pixel up into a k x k block.

    The rule is fixed; the ROUTE through it is read off the input, the way a
    person's route would be: do you lay the dominant colour down as a base
    first, do you work colour by colour / shape by shape / pixel by pixel, from
    which corner, along which axis, and do you stamp a repeated block instead
    of painting it again.  Nothing here looks at O beyond its size.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    # Upscale factor: read it off the demonstrations when they are handed over,
    # otherwise off the output's geometry. Nothing else here consults O.
    k = None
    for ei, eo in (examples or []):
        ei, eo = np.asarray(ei), np.asarray(eo)
        if ei.shape[0] and eo.shape[0] % ei.shape[0] == 0:
            k = eo.shape[0] // ei.shape[0]
            break
    if not k:
        k = np.asarray(O).shape[0] // hi
    ho, wo = hi * k, wi * k

    ops, sels = [], []

    # ── route choices, every one of them measured on the INPUT ────────────────
    cnt = Counter(I.flatten().tolist())
    base = min(cnt, key=lambda col: (-cnt[col], col))      # dominant colour
    # A colour that owns most of the grid reads as a background worth laying
    # down in one stroke; otherwise it is just one more colour to draw.
    use_base = cnt[base] * 20 >= hi * wi * 11              # >= 55% of the grid

    nonbase = [(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) != base]

    # Sweep along the longer axis, starting from the busier side of the grid.
    along_rows = wi >= hi
    top = sum(1 for r, _ in nonbase if 2 * r < hi)
    left = sum(1 for _, c in nonbase if 2 * c < wi)
    row_fwd = 2 * top >= len(nonbase)
    col_fwd = 2 * left >= len(nonbase)

    def order_key(rc):
        r, c = rc
        rr = r if row_fwd else hi - 1 - r
        cc = c if col_fwd else wi - 1 - c
        return (rr, cc) if along_rows else (cc, rr)

    def components(cells):
        """Same-colour 4-connected groups of `cells`, in sweep order."""
        pool = set(cells)
        seen = set()
        out = []
        for cell in sorted(pool, key=order_key):
            if cell in seen:
                continue
            col = int(I[cell])
            comp, dq = [], deque([cell])
            seen.add(cell)
            while dq:
                r, c = dq.popleft()
                comp.append((r, c))
                for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if nb in pool and nb not in seen and int(I[nb]) == col:
                        seen.add(nb)
                        dq.append(nb)
            out.append(comp)
        return out

    nb_comps = components(nonbase)
    biggest = max([len(cp) for cp in nb_comps], default=0)
    distinct = len({int(I[r, c]) for r, c in nonbase})
    if biggest >= 3:
        mode = "object"       # there are real shapes -> blow up a shape at a time
    elif 0 < distinct <= 4:
        mode = "colour"       # few colours, all scattered -> all of one colour at once
    else:
        mode = "sweep"        # confetti -> pixel by pixel

    # ── 1. open the k x k canvas (bbox == the whole new canvas: exact) ────────
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])
    # ResizeGrid is a transparent copy: the un-upscaled input still sits in the
    # top-left hi x wi corner, everything else is 0.
    G = np.zeros((ho, wo), dtype=int)
    G[:hi, :wi] = I

    def block(r, c):
        return [(rr, cc) for rr in range(r * k, r * k + k)
                for cc in range(c * k, c * k + k)]

    def sel_for(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1):
            return [r0, c0, r1 - r0, c1 - c0]   # exactly this full rectangle
        return sel_of(cells)

    def paint(cells, col):
        cells = sorted(set(cells))
        if all(G[r, c] == col for r, c in cells):
            return False                         # would change nothing: no op
        ops.append(int(col))
        sels.append(sel_for(cells))
        for r, c in cells:
            G[r, c] = col
        return True

    # ── 2. optionally lay the dominant colour down as a base ─────────────────
    targets = [(r, c) for r in range(hi) for c in range(wi)]
    if use_base:
        paint([(r, c) for r in range(ho) for c in range(wo)], int(base))
        targets = list(nonbase)

    # ── 3. draw the rest, grouped the way this input suggests ────────────────
    if mode == "sweep":
        for (r, c) in sorted(targets, key=order_key):
            paint(block(r, c), int(I[r, c]))

    elif mode == "colour":
        groups = {}
        for rc in targets:
            groups.setdefault(int(I[rc]), []).append(rc)
        for col in sorted(groups, key=lambda x: (-len(groups[x]), x)):
            cells_rc = sorted(groups[col], key=order_key)
            todo = [rc for rc in cells_rc[1:]
                    if any(G[r, c] != col for r, c in block(*rc))]
            if col != 0 and len(cells_rc) >= 3 and todo:
                # paint the first block, then stamp that same block on the rest
                r0, c0 = cells_rc[0]
                paint(block(r0, c0), col)
                ops.append(29)                       # CopyO: the painted block
                sels.append([r0 * k, c0 * k, k - 1, k - 1])
                for (r, c) in todo:
                    ops.append(30)                   # Paste at the block corner
                    sels.append([r * k, c * k, 0, 0])
                    for rr, cc in block(r, c):
                        G[rr, cc] = col
            else:
                paint([x for rc in cells_rc for x in block(*rc)], col)

    else:  # "object"
        comps = components(targets)
        comps.sort(key=lambda cp: (-len(cp), order_key(min(cp, key=order_key))))
        for comp in comps:
            paint([x for rc in comp for x in block(*rc)], int(I[comp[0]]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 9172f3a0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9172f3a0"
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
                                f"for task 9172f3a0"
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
                    f"Failed to build a complete episode for task 9172f3a0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9172f3a0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
