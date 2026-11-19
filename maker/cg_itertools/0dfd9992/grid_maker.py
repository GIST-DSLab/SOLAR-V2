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
    """Whole lines are the unit (the golf reading: a hole is filled by whatever
    follows the same run of cells elsewhere).  The wallpaper is made of only a
    handful of distinct rows (and distinct columns), each repeated many times.
    A row damaged by patches of the occluding colour is the same line as some
    other row that agrees with it everywhere both are visible -- found by
    comparing rows, not by knowing the period.  So: punch out the occluding
    patches (Color0 -> holes), then for each damaged row lay the best-agreeing
    other row over it (CopyO that row, Paste onto the damaged one: holes in the
    copy are 0 and carry nothing).  Then columns, alternating until whole.
    A patch is only punched if the final grid depends on it.  O is never read."""
    import numpy as np
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # ── the occluding colour: present in every demo input, gone from its output
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
        # no demos: the colour whose cells, treated as unknown, leave every
        # row matched by some other row (lines repeat) with the fewest cells
        best = None
        for col in sorted(set(I.flatten().tolist())):
            m = (I == col)
            if (~m.any(axis=1)).sum() < 2 or (~m.any(axis=0)).sum() < 2:
                continue
            key = (int(m.sum()), col)
            ok = True
            for A, K in ((I, ~m), (I.T, (~m).T)):
                for i in range(A.shape[0]):
                    if not any(((K[i] & K[j]).sum() > 0) and
                               not ((K[i] & K[j]) & (A[i] != A[j])).any()
                               for j in range(A.shape[0]) if j != i):
                        ok = False
                        break
                if not ok:
                    break
            if ok and (best is None or key < best):
                best = key
        noise = best[1] if best else None
    if noise is None or not (I == noise).any():
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    g = I.copy()
    known = (I != noise)

    # ── punch out each occluding patch (maximal rectangles of the colour)
    if noise != 0:
        left = (I == noise).copy()
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
                ops.append(0); sels.append([r, c, h - 1, w - 1])  # full rectangle of intruder
        g[I == noise] = 0

    # ── one pass of line repair along an axis (rows if axis == 0)
    def relates(G, K, i, j):
        n = G.shape[0]
        for f in (lambda x: i + j - x, lambda x: x + (j - i)):
            ok = True
            for x in range(n):
                y = f(x)
                if 0 <= y < n and y != x:
                    both = K[x] & K[y]
                    if (both & (G[x] != G[y])).any():
                        ok = False
                        break
            if ok:
                return 1
        return 0

    def repair(axis, strict):
        G = g if axis == 0 else g.T          # views: writes reach g / known
        K = known if axis == 0 else known.T
        n = G.shape[0]
        progress = False
        for i in range(n):
            if K[i].all():
                continue
            best, mates = None, []
            for j in range(n):
                if j == i:
                    continue
                both = K[i] & K[j]
                if not both.any() or (both & (G[i] != G[j])).any():
                    continue          # not the same line
                mates.append(j)
                gain = int((~K[i] & K[j]).sum())
                if gain == 0:
                    continue
                key = (int(both.sum()), gain, -abs(j - i))
                if not strict:
                    key = (relates(G, K, i, j),) + key
                if best is None or key > best[0]:
                    best = (key, j)
            if best is None:
                continue
            if strict:
                clash = any(((K[a] & K[b]) & (G[a] != G[b]) & ~K[i]).any()
                            for x, a in enumerate(mates) for b in mates[x + 1:])
                if clash or 2 * best[0][0] < G.shape[1]:
                    continue
            j = best[1]
            gain = ~K[i] & K[j]
            if (G[j][gain] != 0).any():
                if axis == 0:
                    ops.extend([29, 30]); sels.extend([[j, 0, 0, wi - 1], [i, 0, 0, 0]])
                else:
                    ops.extend([29, 30]); sels.extend([[0, j, hi - 1, 0], [0, i, 0, 0]])
            G[i][gain] = G[j][gain]
            K[i] |= gain
            progress = True
            if not strict:
                return True       # one guess, then back to sure moves
        return progress

    for _ in range(4 * (hi + wi)):
        if known.all():
            break
        a = repair(0, True)
        b = repair(1, True)
        if not (a or b):
            if not (repair(0, False) or repair(1, False)):
                break

    # ── a patch whose cells all get covered by repaired lines needed no
    #    punching; drop every op the final grid does not depend on (judged by
    #    replaying on I alone)
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
