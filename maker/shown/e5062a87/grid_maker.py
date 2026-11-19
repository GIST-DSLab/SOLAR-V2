"""
ARC Task: e5062a87 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # eligcol (background) and objc (the marker / fill colour) are sampled from 1..9
    # by the generator; both are fixed for the whole episode so the rule ("fill every
    # hidden copy of the marked shape with THIS colour") is learnable from the examples.
    cols = list(range(1, 10))
    bgc = random.choice(cols)
    objc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "objc": objc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, objc=None) -> dict:
    cols = interval(1, 10, 1)
    if bgc is None or objc is None:
        a, b = sample(cols, 2)
        if bgc is None:
            bgc = a
        if objc is None or objc == bgc:
            objc = b if b != bgc else a
    eligcol = bgc

    mh = max(10, min(30, int(max_h)))
    mw = max(10, min(30, int(max_w)))
    h = unifint(diff_lb, diff_ub, (10, mh))
    w = unifint(diff_lb, diff_ub, (10, mw))

    gi = canvas(eligcol, (h, w))
    inds = asindices(gi)
    sp = choice(totuple(inds))
    obj = {sp}
    ncells = unifint(diff_lb, diff_ub, (3, 9))
    for k in range(ncells - 1):
        obj.add(choice(totuple((inds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    nnoise = unifint(diff_lb, diff_ub, (int(0.2 * h * w), int(0.5 * h * w)))
    locs = sample(totuple(inds), nnoise)
    gi = fill(gi, 0, locs)
    noccs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // (len(obj) * 3))))
    oh, ow = shape(obj)
    for k in range(noccs):
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        loc = (loci, locj)
        gi = fill(gi, objc if k == noccs - 1 else 0, shift(obj, loc))
    occs = occurrences(gi, recolor(0, obj))
    res = mapply(lbind(shift, obj), occs)
    go = fill(gi, objc, res)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule (read from I + the demonstrations, never from O):
      * one small shape in I is drawn in the marker colour;
      * every place in I where that same shape appears as a hole of colour 0
        is repainted in the marker colour.
    The marker colour is the episode-fixed fill colour: it is the rarest colour
    of the input, and it is the colour every demonstration paints with.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # ---- marker / fill colour, measured from I (rarest colour: the single marked shape)
    cnt = Counter(I.flatten().tolist())
    objc = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]

    # ---- corroborate with the demonstrations (the colour the rule always fills with)
    if examples:
        seen = set()
        for pair in examples:
            try:
                ei, eo = pair[0], pair[1]
            except Exception:
                continue
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            if ei.shape != eo.shape:
                continue
            added = set(eo[ei != eo].tolist())
            if len(added) == 1:
                seen.add(added.pop())
        if len(seen) == 1:
            objc = seen.pop()

    ops, sels = [], []

    marker = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == objc]
    if marker:
        r0 = min(r for r, _ in marker)
        c0 = min(c for _, c in marker)
        norm = sorted((r - r0, c - c0) for r, c in marker)
        oh = max(dr for dr, _ in norm) + 1
        ow = max(dc for _, dc in norm) + 1

        # every placement where the shape sits in I as a hole of colour 0
        occs = []
        for i in range(hi - oh + 1):
            for j in range(wi - ow + 1):
                if all(I[i + dr, j + dc] == 0 for dr, dc in norm):
                    occs.append((i, j))

        # paint outward from the marked shape: nearest hidden copy first
        occs.sort(key=lambda p: (abs(p[0] - r0) + abs(p[1] - c0), p[0], p[1]))

        painted = set()
        for (i, j) in occs:
            cells = [(i + dr, j + dc) for dr, dc in norm]
            if all(cell in painted for cell in cells):
                continue  # already entirely objc -> would be a no-op
            ops.append(int(objc))
            sels.append(sel_of(cells))
            painted.update(cells)

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
                        f"num_examples+1 ({num_examples + 1}) for task e5062a87"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e5062a87"
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
                                f"for task e5062a87"
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
                    f"Failed to build a complete episode for task e5062a87 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e5062a87-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
