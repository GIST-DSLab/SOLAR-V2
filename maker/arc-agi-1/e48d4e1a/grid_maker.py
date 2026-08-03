"""
ARC Task: e48d4e1a (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    import random
    ROTS = ["identity", "rot90", "rot180", "rot270"]
    cols = list(range(10))
    bgc, fgc, dotc = random.sample(cols, 3)
    n = num_examples if num_examples else 3
    if n >= len(ROTS):
        ex = [{"rotf_name": r} for r in ROTS]
        ex += [{"rotf_name": random.choice(ROTS)} for _ in range(n - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rotf_name": r} for r in random.sample(ROTS, n)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "fgc": fgc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, dotc, rotf_name=None) -> dict:
    import random
    rotmap = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}
    if rotf_name is None:
        rotf_name = random.choice(list(rotmap.keys()))
    rotf = rotmap[rotf_name]

    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))
    loci = randint(1, h - 2)
    locj = randint(1, w - 2)
    inds = asindices(canvas(-1, (loci, locj)))
    maxn = min(min(h - loci - 1, w - locj - 1), len(inds))
    maxn = max(1, maxn)
    nn = unifint(diff_lb, diff_ub, (1, maxn))
    ss = sample(totuple(inds), nn)

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    gi = fill(gi, fgc, hfrontier((loci, 0)) | vfrontier((0, locj)))
    gi = fill(gi, dotc, ss)
    go = fill(go, fgc, hfrontier((loci + nn, 0)) | vfrontier((0, locj + nn)))
    gi = rotf(gi)
    go = rotf(go)

    inp = [list(r) for r in gi]
    out = [list(r) for r in go]
    return {"input": inp, "output": out}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    def sign(x):
        return 0 if x == 0 else (1 if x > 0 else -1)

    # background = most common color in I
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # horizontal frontier: a full row of a single non-bgc color
    old_hrow = None
    for r in range(hi):
        if I[r, 0] != bgc and np.all(I[r, :] == I[r, 0]):
            old_hrow = r
            fgc = int(I[r, 0])
            break
    # vertical frontier: a full column of the frontier color
    old_vcol = None
    for c in range(wi):
        if I[0, c] != bgc and np.all(I[:, c] == I[0, c]):
            old_vcol = c
            break

    # dot color = the color that is neither bgc nor fgc
    colors = set(I.flatten().tolist())
    dotc = [c for c in colors if c != bgc and c != fgc][0]
    dots = [(int(r), int(c)) for r, c in np.argwhere(I == dotc)]
    nn = len(dots)

    # direction: cross shifts by nn toward the center from the dots
    mean_r = sum(r for r, _ in dots) / nn
    mean_c = sum(c for _, c in dots) / nn
    di = sign(old_hrow - mean_r)
    dj = sign(old_vcol - mean_c)

    new_hrow = old_hrow + di * nn
    new_vcol = old_vcol + dj * nn

    ops, sels = [], []
    # erase old cross to bgc
    ops.append(int(bgc)); sels.append([old_hrow, 0, 0, wi - 1])
    ops.append(int(bgc)); sels.append([0, old_vcol, hi - 1, 0])
    # erase dots to bgc
    for (r, c) in dots:
        ops.append(int(bgc)); sels.append([r, c, 0, 0])
    # draw new cross at shifted center
    ops.append(int(fgc)); sels.append([new_hrow, 0, 0, wi - 1])
    ops.append(int(fgc)); sels.append([0, new_vcol, hi - 1, 0])

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task e48d4e1a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e48d4e1a"
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
                                f"for task e48d4e1a"
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
                    f"Failed to build a complete episode for task e48d4e1a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e48d4e1a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
