"""
ARC Task: db93a21d (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c not in (1, 3)]
    bgc, fgc = sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, fgc: int) -> dict:
    # output is cropped: rows h-1, cols w-2
    hu = max(12, min(31, max_h + 1))
    wu = max(12, min(32, max_w + 2))
    h = unifint(diff_lb, diff_ub, (12, hu))
    w = unifint(diff_lb, diff_ub, (12, wu))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = randint(1, h // 4)
        ow = oh
        fullh = 4 * oh
        fullw = 4 * ow
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - fullh and ij[1] < w - fullw))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        bigobj = backdrop(frozenset({(loci, locj), (loci + fullh - 1, locj + fullw - 1)}))
        smallobj = backdrop(frozenset({(loci + oh, locj + ow), (loci + fullh - 1 - oh, locj + fullw - 1 - ow)}))
        if bigobj.issubset(indss | ofcolor(go, 3)):
            gi = fill(gi, fgc, smallobj)
            go = fill(go, 3, bigobj)
            go = fill(go, fgc, smallobj)
            strp = mapply(rbind(shoot, (1, 0)), connect(lrcorner(smallobj), llcorner(smallobj)))
            go = fill(go, 1, ofcolor(go, bgc) & strp)
            succ += 1
            indss = indss - bigobj
        tr += 1
    gi = gi[1:]
    go = go[1:]
    gi = tuple(r[1:-1] for r in gi)
    go = tuple(r[1:-1] for r in go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (measured from I only):
      1) every foreground cell shoots a ray DOWNWARD; every background-in-I cell on
         those rays becomes 1.
      2) every foreground component's bounding box, expanded by pad = (bbox width)//2
         on all four sides, becomes 3 wherever it is background in I  (3 over 1).
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- foreground components (8-connectivity, same as objects(I,T,T,T)) ---
    seen = np.zeros((h, w), dtype=bool)
    objs = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                comp = []
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] != bgc:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                objs.append(sorted(comp))

    objs.sort(key=lambda cs: (cs[0][0], cs[0][1]))

    ops, sels = [], []

    # --- pass 1: downward beams (color 1), one op per object ---
    painted1 = set()
    for comp in objs:
        rmax = max(r for r, _ in comp)
        cols = sorted({c for _, c in comp})
        beam = []
        for c in cols:
            for r in range(rmax + 1, h):
                if I[r, c] == bgc and (r, c) not in painted1:
                    beam.append((r, c))
        if beam:
            beam.sort()
            painted1.update(beam)
            ops.append(1)
            sels.append(sel_of(beam))

    # --- pass 2: halos (color 3), one op per object; 3 covers 1 where they overlap ---
    painted3 = set()
    for comp in objs:
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        pad = (c1 - c0 + 1) // 2
        own = set(comp)
        halo = []
        for r in range(max(0, r0 - pad), min(h, r1 + pad + 1)):
            for c in range(max(0, c0 - pad), min(w, c1 + pad + 1)):
                if (r, c) in own:
                    continue
                if I[r, c] != bgc:
                    continue
                if (r, c) in painted3:
                    continue
                halo.append((r, c))
        if halo:
            painted3.update(halo)
            ops.append(3)
            sels.append(sel_of(halo))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task db93a21d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task db93a21d"
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
                                f"for task db93a21d"
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
                    f"Failed to build a complete episode for task db93a21d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"db93a21d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
