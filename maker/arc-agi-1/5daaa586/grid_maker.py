"""
ARC Task: 5daaa586 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    c1, c2, c3, c4 = random.sample(rem, 4)
    n_ex = num_examples if num_examples else 3
    variants = [{"rot": r} for r in range(4)]      # identity / rot90 / rot180 / rot270
    if n_ex >= len(variants):
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(variants, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=0, c1=1, c2=2, c3=3, c4=4, rot=None) -> dict:
    if rot is None:
        rot = choice((0, 1, 2, 3))
    # after a 90/270 rotation the grid dims swap -> respect caller bounds
    if rot in (1, 3):
        hub, wub = max_w, max_h
    else:
        hub, wub = max_h, max_w
    hub = max(7, min(30, hub))
    wub = max(7, min(30, wub))

    h = unifint(diff_lb, diff_ub, (7, hub))
    w = unifint(diff_lb, diff_ub, (7, wub))
    loci1 = randint(1, h - 4)
    locj1 = randint(1, w - 4)
    loci1dev = unifint(diff_lb, diff_ub, (0, loci1 - 1))
    locj1dev = unifint(diff_lb, diff_ub, (0, locj1 - 1))
    loci1 -= loci1dev
    locj1 -= locj1dev
    loci2 = unifint(diff_lb, diff_ub, (loci1 + 2, h - 2))
    locj2 = unifint(diff_lb, diff_ub, (locj1 + 2, w - 2))

    f1 = recolor(c1, hfrontier(toivec(loci1)))
    f2 = recolor(c2, hfrontier(toivec(loci2)))
    f3 = recolor(c3, vfrontier(tojvec(locj1)))
    f4 = recolor(c4, vfrontier(tojvec(locj2)))
    base = canvas(bgc, (h, w))
    fronts = [f1, f2, f3, f4]
    shuffle(fronts)
    for fr in fronts:
        base = paint(base, fr)

    cands = totuple(ofcolor(base, bgc))
    nn = len(cands)

    def acceptable(g):
        # the two frontier rows / cols must be the ONLY bgc-free lines,
        # and bgc must stay the majority color (both are what derive relies on)
        for i in range(h):
            if i in (loci1, loci2):
                continue
            if all(g[i][j] != bgc for j in range(w)):
                return False
        for j in range(w):
            if j in (locj1, locj2):
                continue
            if all(g[i][j] != bgc for i in range(h)):
                return False
        cnt = {}
        for i in range(h):
            for j in range(w):
                cnt[g[i][j]] = cnt.get(g[i][j], 0) + 1
        top = max(cnt.values())
        return cnt[bgc] == top and list(cnt.values()).count(top) == 1

    gi = None
    for _ in range(60):
        nnoise = unifint(diff_lb, diff_ub, (1, max(1, nn // 3)))
        noise = sample(cands, nnoise)
        cand = fill(base, c1, noise)
        if len(frontiers(cand)) > 4:
            continue
        if acceptable(cand):
            gi = cand
            break
    if gi is None:
        gi = fill(base, c1, sample(cands, 1))

    go = crop(gi, (loci1, locj1), (loci2 - loci1 + 1, locj2 - locj1 + 1))
    ns = ofcolor(go, c1)
    go = fill(go, c1, mapply(rbind(shoot, (-1, 0)), ns))

    rotf = (identity, rot90, rot180, rot270)[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # the box is delimited by the four full frontier lines: the only bgc-free rows/cols
    frows = [r for r in range(hi) if not np.any(I[r, :] == bgc)]
    fcols = [c for c in range(wi) if not np.any(I[:, c] == bgc)]
    r0, r1 = min(frows), max(frows)
    c0, c1b = min(fcols), max(fcols)
    ho, wo = r1 - r0 + 1, c1b - c0 + 1
    sub = I[r0:r1 + 1, c0:c1b + 1]

    ops, sels = [], []
    # 1. keep only the framed box
    ops.append(33); sels.append([r0, c0, ho - 1, wo - 1])

    # 2. the interior speck color; its rays run toward the frontier of the same color
    interior = sub[1:ho - 1, 1:wo - 1]
    cand_cols = [int(v) for v in np.unique(interior) if int(v) != bgc]
    nc, direction = None, None
    for v in cand_cols:
        if all(sub[0, j] == v for j in range(1, wo - 1)):
            nc, direction = v, 'U'; break
        if all(sub[ho - 1, j] == v for j in range(1, wo - 1)):
            nc, direction = v, 'D'; break
        if all(sub[i, 0] == v for i in range(1, ho - 1)):
            nc, direction = v, 'L'; break
        if all(sub[i, wo - 1] == v for i in range(1, ho - 1)):
            nc, direction = v, 'R'; break

    if nc is not None:
        if direction in ('U', 'D'):
            for j in range(1, wo - 1):                      # one ray per column
                rs = [i for i in range(1, ho - 1) if sub[i, j] == nc]
                if not rs:
                    continue
                a, b = (1, max(rs)) if direction == 'U' else (min(rs), ho - 2)
                if all(sub[i, j] == nc for i in range(a, b + 1)):
                    continue                                # ray already complete
                ops.append(nc); sels.append([a, j, b - a, 0])
        else:
            for i in range(1, ho - 1):                      # one ray per row
                cs = [j for j in range(1, wo - 1) if sub[i, j] == nc]
                if not cs:
                    continue
                a, b = (1, max(cs)) if direction == 'L' else (min(cs), wo - 2)
                if all(sub[i, j] == nc for j in range(a, b + 1)):
                    continue
                ops.append(nc); sels.append([i, a, 0, b - a])

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
                        f"num_examples+1 ({num_examples + 1}) for task 5daaa586"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5daaa586"
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
                                f"for task 5daaa586"
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
                    f"Failed to build a complete episode for task 5daaa586 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5daaa586-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
