"""
ARC Task: c3f564a4 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [
    {"straight": True,  "rotated": False},
    {"straight": True,  "rotated": True},
    {"straight": False, "rotated": False},
    {"straight": False, "rotated": True},
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    fixc = random.choice(cols)
    rest = [c for c in cols if c != fixc]
    random.shuffle(rest)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"fixc": fixc, "ccols_all": rest, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, fixc=0, ccols_all=None,
             straight=None, rotated=None, **kwargs) -> dict:
    if ccols_all is None:
        ccols_all = [c for c in range(10) if c != fixc]
    if straight is None:
        straight = choice((True, False))
    if rotated is None:
        rotated = choice((True, False))
    hb, wb = (max_w, max_h) if rotated else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (7, max(7, min(30, hb))))
    w = unifint(diff_lb, diff_ub, (7, max(7, min(30, wb))))
    p = unifint(diff_lb, diff_ub, (2, min(9, min(h // 3, w // 3))))
    ccols = list(ccols_all[:p])
    c = canvas(-1, (h, w))
    baseobj = {(cc, (0, jj)) for cc, jj in zip(ccols, range(p))}
    obj = {x for x in baseobj}
    while rightmost(obj) < 2 * max(w, h):
        obj = obj | shift(obj, (0, p))
    if straight:
        obj = mapply(lbind(shift, obj), {(jj, 0) for jj in interval(0, h, 1)})
    else:
        obj = mapply(lbind(shift, obj), {(jj, -jj) for jj in interval(0, h, 1)})
    go = paint(c, obj)
    gi = tuple(e for e in go)
    nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    maxtr = 4 * nsq
    tr = 0
    succ = 0
    while succ < nsq and tr < maxtr:
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (2, 5))
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        tmpg = fill(gi, fixc, backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})))
        if len(occurrences(tmpg, baseobj)) > 1 and len([r for r in tmpg if fixc not in r]) > 0 and len([r for r in dmirror(tmpg) if fixc not in r]) > 0:
            gi = tmpg
            succ += 1
        tr += 1
    if rotated:
        gi = rot90(gi)
        go = rot90(go)
    return {'input': gi, 'output': go}


def _fit(I, f):
    """Find (a, b, p, colormap) such that every non-f cell satisfies
    I[i,j] == cmap[(a*i + b*j) % p], with every residue class observed."""
    h, w = I.shape
    for p in range(2, 10):
        for a, b in ((0, 1), (1, 0), (1, 1), (1, -1)):
            cmap = {}
            ok = True
            for i in range(h):
                for j in range(w):
                    v = int(I[i, j])
                    if v == f:
                        continue
                    k = (a * i + b * j) % p
                    if cmap.setdefault(k, v) != v:
                        ok = False
                        break
                if not ok:
                    break
            if ok and len(cmap) == p and len(set(cmap.values())) == p:
                return (a, b, p, cmap)
    return None


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # hole colour: present in demo inputs but never in demo outputs
    cands = []
    if examples:
        ins, outs = set(), set()
        for ei, eo in examples:
            ins |= set(np.asarray(ei).flatten().tolist())
            outs |= set(np.asarray(eo).flatten().tolist())
        cands = [c for c in sorted(ins - outs) if c in set(I.flatten().tolist())]
    cnt = Counter(I.flatten().tolist())
    cands += [c for c, _ in sorted(cnt.items(), key=lambda t: t[1])]
    fit = None
    fixc = None
    for f in cands:
        fit = _fit(I, f)
        if fit is not None:
            fixc = f
            break
    ops, sels = [], []
    if fit is not None:
        a, b, p, cmap = fit
        # connected components of the hole colour (each erased block region)
        seen = set()
        comps = []
        for i in range(h):
            for j in range(w):
                if I[i, j] == fixc and (i, j) not in seen:
                    st = [(i, j)]; seen.add((i, j)); comp = []
                    while st:
                        r, c = st.pop(); comp.append((r, c))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and I[nr, nc] == fixc:
                                seen.add((nr, nc)); st.append((nr, nc))
                    comps.append(comp)
        # restore each hole region, one pattern colour at a time
        for comp in comps:
            bycol = {}
            for r, c in comp:
                bycol.setdefault(cmap[(a * r + b * c) % p], []).append((r, c))
            for col in sorted(bycol):
                ops.append(int(col)); sels.append(sel_of(bycol[col]))
    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c3f564a4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c3f564a4"
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
                                f"for task c3f564a4"
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
                    f"Failed to build a complete episode for task c3f564a4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c3f564a4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
