"""
ARC Task: 57aa92db (RE-ARC) — LLM-generated grid_maker
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
from dsl import *
from utils import *
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, fixc, mainc = random.sample(range(10), 3)
    return {"bgc": bgc, "fixc": fixc, "mainc": mainc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, fixc=1, mainc=2) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    oh = randint(2, 5)
    ow = randint(2, 5)
    bounds = asindices(canvas(-1, (oh, ow)))
    obj = {choice(totuple(bounds))}
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    fixp = choice(totuple(obj))
    remcols = difference(cols, (bgc, fixc, mainc))
    gi = canvas(bgc, (h, w))
    obj = {(fixc, fixp)} | recolor(mainc, remove(fixp, obj))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    plcd = shift(obj, (loci, locj))
    gi = paint(gi, plcd)
    go = tuple(e for e in gi)
    inds = ofcolor(gi, bgc) - mapply(neighbors, toindices(plcd))
    nocc = unifint(diff_lb, diff_ub, (1, (h * w) // (4 * len(obj))))
    tr = 0
    succ = 0
    maxtr = 5 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = randint(1, 4)
        objups = upscale(obj, fac)
        hh, ww = shape(objups)
        cands = sfilter(inds, lambda ij: ij[0] <= h - hh and ij[1] <= w - ww)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        objupsplcd = shift(objups, loc)
        objupsplcdi = toindices(objupsplcd)
        if objupsplcdi.issubset(inds):
            succ += 1
            newc = choice(remcols)
            fixp2 = sfilter(objupsplcd, lambda cij: cij[0] == fixc)
            inds = inds - mapply(neighbors, objupsplcdi)
            gi = paint(gi, fixp2)
            go = paint(go, fixp2)
            remobjfull = toindices(objupsplcd - fixp2)
            ntorem = unifint(diff_lb, diff_ub, (0, max(0, len(remobjfull) - 1)))
            ntokeep = len(remobjfull) - ntorem
            tokeep = {choice(totuple(remobjfull & outbox(fixp2)))}
            fixp2i = toindices(fixp2)
            for k in range(ntokeep - 1):
                fullopts = remobjfull & mapply(neighbors, tokeep | fixp2i)
                remopts = fullopts - tokeep
                tokeep.add(choice(totuple(remopts)))
            gi = fill(gi, newc, tokeep)
            go = fill(go, newc, remobjfull)
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    # background = canvas colour (majority; objects cover a small fraction)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected multicolour components of non-background
    seen = np.zeros_like(I, dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if I[r, c] == bgc or seen[r, c]:
                continue
            stack = [(r, c)]; seen[r, c] = True; cells = []
            while stack:
                a, b = stack.pop(); cells.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        x, y = a + da, b + db
                        if 0 <= x < H and 0 <= y < W and not seen[x, y] and I[x, y] != bgc:
                            seen[x, y] = True; stack.append((x, y))
            comps.append(cells)

    # anchor colour = the colour present in the most objects
    presence = Counter()
    for cells in comps:
        for col in {int(I[a, b]) for a, b in cells}:
            presence[col] += 1
    fixc = presence.most_common(1)[0][0]

    def fixcount(cells):
        return sum(1 for a, b in cells if I[a, b] == fixc)
    comps = [cl for cl in comps if fixcount(cl) > 0]
    mn = min(fixcount(cl) for cl in comps)
    # template = the complete (largest) object at unit scale
    tmpl = max([cl for cl in comps if fixcount(cl) == mn], key=len)
    tr0 = min(a for a, b in tmpl); tc0 = min(b for a, b in tmpl)
    tnorm = [(a - tr0, b - tc0) for a, b in tmpl]
    tfix = [(a - tr0, b - tc0) for a, b in tmpl if I[a, b] == fixc][0]
    tmain = [p for p in tnorm if p != tfix]

    ops, sels = [], []
    for cl in comps:
        if cl is tmpl:
            continue
        fx = [(a, b) for a, b in cl if I[a, b] == fixc]
        others = [int(I[a, b]) for a, b in cl if I[a, b] != fixc]
        if not others:
            continue
        newc = Counter(others).most_common(1)[0][0]
        br = min(a for a, b in fx); bc = min(b for a, b in fx)
        fac = max(b for a, b in fx) - bc + 1          # scale = anchor block width
        orr = br - tfix[0] * fac; oc = bc - tfix[1] * fac
        full = set()
        for (a, b) in tmain:
            for i in range(fac):
                for j in range(fac):
                    full.add((orr + a * fac + i, oc + b * fac + j))
        # complete this object's scaled template body in its own colour
        todo = [(a, b) for a, b in full
                if 0 <= a < H and 0 <= b < W and I[a, b] != newc]
        if todo:
            ops.append(newc); sels.append(sel_of(todo))
    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 57aa92db"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 57aa92db"
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
                                f"for task 57aa92db"
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
                    f"Failed to build a complete episode for task 57aa92db "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"57aa92db-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
