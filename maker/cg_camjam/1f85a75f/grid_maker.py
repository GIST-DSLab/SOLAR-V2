"""
ARC Task: 1f85a75f (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, objc = random.sample(cols, 2)
    remcols = [c for c in cols if c not in (bgc, objc)]
    random.shuffle(remcols)
    # noise palette order fixed per episode; generate() takes the first numc entries
    return {"bgc": bgc, "objc": objc, "noise_pool": remcols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, noise_pool=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, int(round(a + (b - a) * d))))

    if bgc is None or objc is None:
        bgc, objc = random.sample(range(10), 2)
    if noise_pool is None:
        noise_pool = [c for c in range(10) if c not in (bgc, objc)]
        random.shuffle(noise_pool)
    noise_pool = [c for c in noise_pool if c not in (bgc, objc)]

    H = max(10, min(30, max_h))
    W = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
    oh = random.randint(3, min(8, h // 2))
    ow = random.randint(3, min(8, w // 2))
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    ncells = random.randint(max(oh, ow), oh * ow)
    sp = random.choice(sorted(bounds))
    obj = {sp}
    for _ in range(ncells - 1):
        nb = set()
        for (i, j) in obj:
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb.add((i + di, j + dj))
        cands = sorted((bounds - obj) & nb)
        obj.add(random.choice(cands))
    mi = min(i for i, _ in obj)
    mj = min(j for _, j in obj)
    obj = {(i - mi, j - mj) for i, j in obj}
    oh = max(i for i, _ in obj) + 1
    ow = max(j for _, j in obj) + 1
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = noise_pool[:max(1, min(numc, len(noise_pool)))]
    bd = {(loci + i, locj + j) for i in range(oh) for j in range(ow)}
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, (h * w - len(bd)) // 4)))
    gi = [[bgc] * w for _ in range(h)]
    for (i, j) in obj:
        gi[loci + i][locj + j] = objc
    others = [(i, j) for i in range(h) for j in range(w) if (i, j) not in bd]
    for (i, j) in random.sample(others, nnoise):
        gi[i][j] = random.choice(ccols)
    go = [[bgc] * ow for _ in range(oh)]
    for (i, j) in obj:
        go[i][j] = objc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the canvas colour (majority colour of the input)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Unit of reasoning = a COLOUR (all its cells), as in the colour-wise reading:
    # the object colour is the one whose cells form a single connected piece
    # (noise colours are scattered singletons); among such colours, the one
    # with the most cells.
    def n_components(col):
        mask = (I == col)
        seen = np.zeros_like(mask)
        n = 0
        for r in range(hi):
            for c in range(wi):
                if mask[r, c] and not seen[r, c]:
                    n += 1
                    stack = [(r, c)]
                    seen[r, c] = True
                    while stack:
                        y, x = stack.pop()
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < hi and 0 <= nx < wi and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
        return n

    counts = Counter(I.flatten().tolist())
    cands = [col for col in counts if col != bgc and n_components(col) == 1]
    if not cands:
        cands = [col for col in counts if col != bgc]
    objc = max(cands, key=lambda col: counts[col])

    rs, cs = np.nonzero(I == objc)
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())
    oh, ow = r1 - r0 + 1, c1 - c0 + 1

    ops, sels = [], []
    # Crop to the bounding rectangle of all object-colour cells. The selection is
    # intentionally that FULL rectangle (background cells inside it are kept).
    ops.append(33); sels.append([r0, c0, oh - 1, ow - 1])
    ops.append(34); sels.append([0, 0, oh - 1, ow - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 1f85a75f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1f85a75f"
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
                                f"for task 1f85a75f"
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
                    f"Failed to build a complete episode for task 1f85a75f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1f85a75f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
