"""
ARC Task: 27a28665 (RE-ARC) — LLM-generated grid_maker
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

MAPPING = [
    (1, {(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)}),
    (2, {(0, 0), (1, 1), (2, 0), (0, 2), (2, 2)}),
    (3, {(2, 0), (0, 1), (0, 2), (1, 1), (1, 2)}),
    (6, {(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)}),
]
PATTERN_SET = {frozenset(s) for _, s in MAPPING}
# Golf reading: only the two TOP corners of the 3x3 stamp matter.
CORNER_TABLE = {(True, False): 1, (True, True): 2, (False, True): 3, (False, False): 6}


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    n_ex = num_examples if num_examples else 3
    labels = [m[0] for m in MAPPING]
    if n_ex >= len(labels):
        ex = list(labels) + [random.choice(labels) for _ in range(n_ex - len(labels))]
        random.shuffle(ex)
    else:
        ex = random.sample(labels, n_ex)
    plan = [{"label": l} for l in ex]
    plan.append({"label": random.choice(ex)})
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, label=None, objc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round(lb * (b - a))
        hi = a + round(ub * (b - a))
        return random.randint(min(lo, hi), max(lo, hi))

    if label is None:
        label = random.choice([m[0] for m in MAPPING])
    obj = dict(MAPPING)[label]
    if objc is None:
        objc = random.choice([c for c in range(10) if c != bgc])
    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))
    fac = unifint(diff_lb, diff_ub, (1, max(1, min(h, w) // 3)))
    gi = [[bgc] * w for _ in range(h)]
    loci = random.randint(0, h - 3 * fac)
    locj = random.randint(0, w - 3 * fac)
    for (i, j) in obj:
        for a in range(fac):
            for b in range(fac):
                gi[loci + i * fac + a][locj + j * fac + b] = objc
    return {"input": gi, "output": [[label]]}


def _find_stamp(I, bgc=None):
    """Return (r0, c0, f, blocks) of the upscaled 3x3 stamp."""
    cands = sorted(set(I.flatten().tolist()), key=lambda c: (c == bgc, c))
    for cand in cands:
        pos = np.argwhere(I == cand)
        r0, c0 = pos.min(0).tolist()
        r1, c1 = pos.max(0).tolist()
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh != bw or bh % 3 != 0:
            continue
        f = bh // 3
        blocks = np.zeros((3, 3), dtype=bool)
        ok = True
        for i in range(3):
            for j in range(3):
                blk = I[r0 + i * f:r0 + (i + 1) * f, c0 + j * f:c0 + (j + 1) * f]
                vals = set(blk.flatten().tolist())
                if vals == {cand}:
                    blocks[i, j] = True
                elif cand in vals:
                    ok = False
        if not ok:
            continue
        cellset = frozenset((i, j) for i in range(3) for j in range(3) if blocks[i, j])
        if cellset in PATTERN_SET:
            return r0, c0, f, blocks
    # fallback
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    pos = np.argwhere(I != bgc)
    r0, c0 = pos.min(0).tolist()
    r1, c1 = pos.max(0).tolist()
    f = max(1, (r1 - r0 + 1) // 3)
    blocks = np.array([[I[min(r0 + i * f, I.shape[0] - 1), min(c0 + j * f, I.shape[1] - 1)] != bgc
                        for j in range(3)] for i in range(3)], dtype=bool)
    return r0, c0, f, blocks


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape

    # background: the colour shared by every demonstration input, most frequent overall
    bgc = None
    table = dict(CORNER_TABLE)
    if examples:
        exs = [(np.asarray(a, dtype=int), np.asarray(b, dtype=int)) for a, b in examples]
        common = set(range(10))
        tot = Counter()
        for a, _ in exs:
            common &= set(a.flatten().tolist())
            tot.update(a.flatten().tolist())
        if common:
            bgc = max(common, key=lambda c: tot[c])
        # read the corner -> colour convention from the demonstrations
        for a, b in exs:
            _, _, _, bl = _find_stamp(a, bgc)
            table[(bool(bl[0, 0]), bool(bl[0, 2]))] = int(b[0, 0])

    r0, c0, f, blocks = _find_stamp(I, bgc)
    color = table[(bool(blocks[0, 0]), bool(blocks[0, 2]))]

    ops, sels = [], []
    # 1. Focus on the stamp's top band (the row of blocks holding the two deciding corners).
    #    Full rectangle crop: f rows x 3f cols.
    ops.append(33); sels.append([r0, c0, f - 1, 3 * f - 1])
    band = I[r0:r0 + f, c0:c0 + 3 * f].copy()
    band[band == 0] = 0  # transparent crop keeps zeros as zero
    # 2. Write the verdict read off the two top corners into the band's top-left cell
    if band[0, 0] != color:
        ops.append(color); sels.append(sel_of([(0, 0)]))
    # 3. Shrink the canvas to that single verdict cell
    ops.append(33); sels.append([0, 0, 0, 0])
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
                        f"num_examples+1 ({num_examples + 1}) for task 27a28665"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 27a28665"
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
                                f"for task 27a28665"
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
                    f"Failed to build a complete episode for task 27a28665 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"27a28665-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
