"""
ARC Task: b60334d2 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- colors
def sample_colors(num_examples=None) -> dict:
    """bgc = canvas colour, fgc = dot colour. The ring colour is the hardcoded
    1 (never sampled, never equal to bgc/fgc) so it is NOT returned here."""
    cols = [c for c in range(10) if c != 1]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


# ---------------------------------------------------------------- generator
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, fgc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 9)))
    bx = box(frozenset({(0, 0), (2, 2)}))
    bd = backdrop(bx)
    while True:
        gi = canvas(bgc, (h, w))
        go = canvas(bgc, (h, w))
        inds = asindices(gi)
        maxtrials = 4 * num
        succ = 0
        tr = 0
        while succ < num and tr < maxtrials:
            loc = choice(totuple(inds))
            bxs = shift(bx, loc)
            if bxs.issubset(set(inds)):
                cen = center(bxs)
                gi = fill(gi, fgc, {cen})
                go = fill(go, fgc, ineighbors(cen))
                go = fill(go, 1, dneighbors(cen))
                inds = difference(inds, shift(bd, loc))
                succ += 1
            tr += 1
        if succ > 0:          # never emit an empty (identical I/O) instance
            break
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- trajectory
def derive_operations(I, O, examples=None):
    """Rule: every isolated dot is replaced by one and the same 3x3 stamp

            fgc  ring  fgc
            ring  bgc  ring
            fgc  ring  fgc

    The generator guarantees the 3x3 blocks never overlap, so the honest
    account is: build the stamp ONCE on the first dot, put it on the
    clipboard, and stamp it onto every other dot.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]                 # canvas colour (>= 8/9 of cells)
    others = [c for c in cnt if c != bgc]
    if not others:                                 # no dots at all -> nothing to do
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    fgc = min(others, key=lambda c: cnt[c])        # dot colour

    # The plus-arm colour is a task convention: read it from the demonstrations
    # (it is the one colour that appears in an example output but not its input).
    ring = 1
    if examples:
        for ei, eo in examples:
            ea = np.asarray(ei); eb = np.asarray(eo)
            extra = set(eb.flatten().tolist()) - set(ea.flatten().tolist())
            if len(extra) == 1:
                ring = int(extra.pop())
                break

    centers = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == fgc]
    centers.sort()

    # ---- 1. build the stamp on the first dot -------------------------------
    r0, c0 = centers[0]
    plus0 = [(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)]
    corn0 = [(r0 - 1, c0 - 1), (r0 - 1, c0 + 1), (r0 + 1, c0 - 1), (r0 + 1, c0 + 1)]
    ops.append(ring); sels.append(sel_of(plus0))          # the plus arms
    ops.append(fgc);  sels.append(sel_of(corn0))          # the four corners
    ops.append(bgc);  sels.append(sel_of([(r0, c0)]))     # the dot itself is consumed

    rest = centers[1:]
    if rest:
        # ---- 2. that finished 3x3 block IS the stamp -----------------------
        # full rectangle intended (background centre included) -> bbox is exact
        ops.append(29); sels.append([r0 - 1, c0 - 1, 2, 2])   # CopyO

        # ---- 3. stamp it on every remaining dot ----------------------------
        for (r, c) in rest:
            ops.append(30); sels.append([r - 1, c - 1, 0, 0])  # Paste at block origin

        # ---- 4. Paste is transparent to colour 0 ---------------------------
        # If one of the stamp's colours happens to be 0 those cells were not
        # carried by the clipboard; paint exactly those cells (known from the
        # rule, not from O).
        if bgc == 0:
            ops.append(0); sels.append(sel_of(rest))           # centres stay bgc
        elif fgc == 0:
            cells = []
            for (r, c) in rest:
                cells += [(r - 1, c - 1), (r - 1, c + 1),
                          (r + 1, c - 1), (r + 1, c + 1)]
            ops.append(0); sels.append(sel_of(cells))          # corners stay fgc

    ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b60334d2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b60334d2"
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
                                f"for task b60334d2"
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
                    f"Failed to build a complete episode for task b60334d2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b60334d2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
