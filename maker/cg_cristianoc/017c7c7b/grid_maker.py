"""
ARC Task: 017c7c7b (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c not in (0, 2)]
    return {"fgc": random.choice(cols)}

def generate(diff_lb, diff_ub, max_h, max_w, fgc=None) -> dict:
    from dsl import canvas, asindices, totuple, fill, shift, replace
    from utils import unifint
    if fgc is None:
        fgc = random.choice([c for c in range(10) if c not in (0, 2)])
    for _attempt in range(200):
        hmax = max(3, min(10, (max_h * 2) // 6))
        h = unifint(diff_lb, diff_ub, (3, hmax))
        w = unifint(diff_lb, diff_ub, (2, min(30, max_w)))
        h += h
        go = canvas(0, (h + h // 2, w))
        oh = unifint(diff_lb, diff_ub, (2, max(2, h // 3 * 2)))
        ow = unifint(diff_lb, diff_ub, (1, w))
        locj = random.randint(0, w - ow)
        bounds = asindices(canvas(-1, (oh, ow)))
        ncellsd = unifint(diff_lb, diff_ub, (1, oh * ow))
        ncells = min(max(1, ncellsd), oh * ow)
        obj = random.sample(totuple(bounds), ncells)
        obj = {(r, c + locj) for (r, c) in obj}
        for k in range((2 * h) // oh):
            go = fill(go, 2, shift(obj, (k * oh, 0)))
        gi = replace(go[:h], 2, fgc)
        # keep only instances whose continuation is readable from the input's period
        G = np.array(go); Ia = np.array(gi)
        p = h
        for cand in range(1, h + 1):
            if all(np.array_equal(Ia[r], Ia[r - cand]) for r in range(cand, h)):
                p = cand
                break
        if all(np.array_equal(G[r], G[r - p]) for r in range(p, len(G))):
            return {"input": gi, "output": go}
    return {"input": gi, "output": go}

def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Reading: the picture is a vertically periodic strip that continues for
    # another half of its height; the continuation is a band of the input itself.
    # The pattern colour changes to 2 (convention read from the examples).
    ho, wo = hi + hi // 2, wi
    ext = hi // 2
    fg_vals = sorted({int(v) for v in I.flatten().tolist() if v != 0})
    fgc = fg_vals[0] if fg_vals else 0
    new_col = 2
    if examples:
        for ei, eo in examples:
            eo = np.asarray(eo)
            vals = {int(v) for v in eo.flatten().tolist() if v != 0}
            if vals:
                new_col = vals.pop(); break
    # vertical period of I
    p = hi
    for cand in range(1, hi + 1):
        if all(np.array_equal(I[r], I[r - cand]) for r in range(cand, hi)):
            p = cand
            break
    ops, sels = [], []
    s = hi % p
    band = I[s:s + ext]
    # 1) the pattern changes colour: every cell of the pattern colour in I
    fg_cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == fgc]
    if fg_cells and fgc != new_col:
        ops.append(new_col); sels.append(sel_of(fg_cells))
    # 2) extend the canvas downward by half the height (zero padding == bgc 0)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    # 3) the strip continues: the band of ext rows starting a whole number of
    #    periods above row hi (s = hi mod p) is the continuation; copy that
    #    full-width rectangle from the (recoloured) working grid and paste it
    #    onto the newly added rows in one piece
    if band.any():
        ops.append(29); sels.append([s, 0, ext - 1, wi - 1])   # full rectangle band
        ops.append(30); sels.append([hi, 0, 0, 0])             # the rows Resize added
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 017c7c7b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 017c7c7b"
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
                                f"for task 017c7c7b"
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
                    f"Failed to build a complete episode for task 017c7c7b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"017c7c7b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
