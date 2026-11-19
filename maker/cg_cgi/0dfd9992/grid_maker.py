"""
ARC Task: 0dfd9992 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    return {"bgc": bgc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, **kw) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or noisec is None:
        bgc, noisec = sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
    hp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    wp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    pinds = asindices(canvas(-1, (hp, wp)))
    remcols = remove(noisec, cols)
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numc)
    pobj = frozenset({(choice(ccols), ij) for ij in pinds})
    go = canvas(bgc, (h, w))
    locs = set()
    for a in range(h // hp + 1):
        for b in range(w // wp + 1):
            loci = hp * a
            locj = wp * b
            locs.add((loci, locj))
            mf1 = identity if a % 2 == 0 else hmirror
            mf2 = identity if b % 2 == 0 else vmirror
            mf = compose(mf1, mf2)
            go = paint(go, shift(mf(pobj), (loci, locj)))
    numpatches = unifint(diff_lb, diff_ub, (1, int((h * w) ** 0.5 // 2)))
    gi = tuple(e for e in go)
    places = apply(lbind(shift, pinds), locs)
    succ = 0
    tr = 0
    maxtr = 5 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph = randint(2, 6)
        pw = randint(2, 6)
        loci = randint(0, h - ph)
        locj = randint(0, w - pw)
        ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
        gi2 = fill(gi, noisec, ptch)
        candset = apply(normalize, apply(rbind(toobject, gi2), places))
        if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and \
           len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2 and \
           (pobj in candset or hmirror(pobj) in candset or vmirror(pobj) in candset
                or hmirror(vmirror(pobj)) in candset):
            succ += 1
            gi = gi2
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Line-by-line repair (the code-golf reading).  The wallpaper is built
    from only a few distinct rows and a few distinct columns, each repeated
    many times.  A row hidden under a patch of the occluding colour is the
    same line as its sister rows elsewhere in the grid.  So: punch the
    occluding colour out to holes (Color0), then for every damaged row take a
    sister row (CopyO of that whole row) and lay it over the damaged one
    (Paste; the copy's own holes are 0 and carry nothing, so only the target's
    holes get filled).  Do the same with columns, alternating until every line
    is whole.  Sister lines are found from the fold structure the visible
    cells obey.  O is never consulted."""
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # -- occluding colour: in every demo input, absent from its output
    noise = None
    if examples:
        cand = None
        for a, b in examples:
            a = np.asarray(a, dtype=int); b = np.asarray(b, dtype=int)
            s = set(np.unique(a).tolist()) - set(np.unique(b).tolist())
            if s:
                cand = s if cand is None else (cand & s or cand)
        if cand:
            inI = [c for c in cand if (I == c).any()]
            if inI:
                noise = min(inI, key=lambda c: int((I == c).sum()))
    if noise is None:
        def per(ok, axis):
            n = I.shape[axis]
            for p in range(1, n):
                if axis == 0:
                    a, b, ka, kb = I[p:], I[:n - p], ok[p:], ok[:n - p]
                else:
                    a, b, ka, kb = I[:, p:], I[:, :n - p], ok[:, p:], ok[:, :n - p]
                if not ((a != b) & ka & kb).any():
                    return p
            return n
        best = None
        for col in sorted(set(I.flatten().tolist())):
            m = (I == col)
            if (~m.any(axis=1)).sum() < 2 or (~m.any(axis=0)).sum() < 2:
                continue
            pr, pc = per(~m, 0), per(~m, 1)
            if pr >= hi or pc >= wi:
                continue
            key = (pr * pc, int(m.sum()), col)
            if best is None or key < best:
                best = key
        noise = best[2] if best else None
    if noise is None or not (I == noise).any():
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    g = I.copy()
    known = (I != noise)

    # -- punch each occluding patch out to holes (maximal rectangles)
    left = (~known).copy()
    for r in range(hi):
        for c in range(wi):
            if not left[r, c]:
                continue
            w = 0
            while c + w < wi and left[r, c + w]:
                w += 1
            h = 1
            while r + h < hi and left[r + h, c:c + w].all():
                h += 1
            left[r:r + h, c:c + w] = False
            g[r:r + h, c:c + w] = 0
            if noise != 0:
                ops.append(0); sels.append([r, c, h - 1, w - 1])

    # -- which lines are the same line of the wallpaper.  Along each axis
    #    the wallpaper runs  0,1,..,p-1,p-1,..,1,0,0,1,..  (a strip of p lines
    #    and its mirror image, alternating), from some phase o.  Take the
    #    tightest (p, o) that every pair of visible cells agrees with.
    def fold_index(i, p, o):
        x = (i + o) % (2 * p)
        return x if x < p else 2 * p - 1 - x

    def classes(axis):
        n = I.shape[axis]
        A = I if axis == 0 else I.T
        K = known if axis == 0 else known.T
        for p in range(1, n):
            for o in range(2 * p):
                groups = {}
                for i in range(n):
                    groups.setdefault(fold_index(i, p, o), []).append(i)
                ok = True
                for mem in groups.values():
                    if len(mem) < 2:
                        continue
                    ref = np.full(A.shape[1], -1)
                    for i in mem:
                        k = K[i]
                        if ((ref >= 0) & k & (ref != A[i])).any():
                            ok = False
                            break
                        ref = np.where((ref < 0) & k, A[i], ref)
                    if not ok:
                        break
                if ok:
                    cls = [None] * n
                    for mem in groups.values():
                        for i in mem:
                            cls[i] = mem
                    return cls
        return [[i] for i in range(n)]

    row_cls, col_cls = classes(0), classes(1)

    # -- line repair: lay a sister line over each damaged line.
    def line(arr, axis, i):
        return arr[i, :] if axis == 0 else arr[:, i]

    def repair(axis):
        n = hi if axis == 0 else wi
        cls = row_cls if axis == 0 else col_cls
        progress = False
        for i in range(n):
            for j in sorted((j for j in cls[i] if j != i), key=lambda j: (abs(j - i), j)):
                ki = line(known, axis, i)
                if ki.all():
                    break
                kj, gj = line(known, axis, j), line(g, axis, j)
                gain = np.where(~ki & kj)[0]
                if not len(gain):
                    continue
                if (gj[gain] != 0).any():
                    # CopyO the whole sister line, Paste it over the damaged one
                    if axis == 0:
                        ops.extend([29, 30]); sels.extend([[j, 0, 0, wi - 1], [i, 0, 0, 0]])
                    else:
                        ops.extend([29, 30]); sels.extend([[0, j, hi - 1, 0], [0, i, 0, 0]])
                if axis == 0:
                    g[i, gain] = g[j, gain]; known[i, gain] = True
                else:
                    g[gain, i] = g[gain, j]; known[gain, i] = True
                progress = True
        return progress

    for _ in range(12):
        if known.all():
            break
        a = repair(0)
        b = repair(1)
        if not (a or b):
            break

    # -- drop any op the final grid does not depend on
    def run(o_list, s_list):
        gg = I.copy()
        clip = None
        for op, (r, c, h, w) in zip(o_list, s_list):
            if op <= 9:
                gg[r:r + h + 1, c:c + w + 1] = op
            elif op == 29:
                clip = gg[r:r + h + 1, c:c + w + 1].copy()
            elif op == 30 and clip is not None:
                p = clip[:hi - r, :wi - c]
                tgt = gg[r:r + p.shape[0], c:c + p.shape[1]]
                np.copyto(tgt, p, where=(p > 0))
        return gg

    target = run(ops, sels)
    k = 0
    while k < len(ops):
        if (run(ops[:k] + ops[k + 1:], sels[:k] + sels[k + 1:]) == target).all():
            ops.pop(k); sels.pop(k)
            k = 0
        else:
            k += 1

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 0dfd9992"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0dfd9992"
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
                                f"for task 0dfd9992"
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
                    f"Failed to build a complete episode for task 0dfd9992 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0dfd9992-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
