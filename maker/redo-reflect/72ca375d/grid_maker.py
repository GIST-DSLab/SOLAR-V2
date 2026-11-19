"""
ARC Task: 72ca375d (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # bgc and trgc are the only randomly sampled colours the rule depends on
    # (distractor colours are irrelevant - the rule is about mirror symmetry).
    # Both are kept NON-ZERO on purpose: the extracted object's bounding box is then
    # entirely non-zero, so CopyI / Paste / FlipH (all of which treat 0 as "nothing")
    # act on the whole region exactly as intended.
    cols = list(range(1, 10))
    bgc = random.choice(cols)
    trgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "trgc": trgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, trgc) -> dict:
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)
    while True:
        h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
        w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
        nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))

        # ---- the target object: a half glued to its own vmirror ----
        while True:
            srcobjh = unifint(diff_lb, diff_ub, (2, 8))
            srcobjwh = unifint(diff_lb, diff_ub, (1, 4))
            bnds = asindices(canvas(-1, (srcobjh, srcobjwh)))
            spi = randint(0, srcobjh - 1)
            sp = (spi, srcobjwh - 1)
            half = {sp}
            bnds = remove(sp, bnds)
            ncellsd = unifint(diff_lb, diff_ub, (0, (srcobjh * srcobjwh) // 2))
            ncells1 = choice((ncellsd, srcobjh * srcobjwh - ncellsd))
            ncells2 = unifint(diff_lb, diff_ub, (1, srcobjh * srcobjwh))
            ncells = (ncells1 + ncells2) // 2
            ncells = min(max(1, ncells), srcobjh * srcobjwh, (h * w) // 2 - 1)
            for _k in range(ncells - 1):
                cands = totuple((bnds - half) & mapply(neighbors, half))
                if len(cands) == 0:
                    break
                half.add(choice(cands))
            half = normalize(half)
            # the generating half must itself be asymmetric, so that reflecting it is
            # a visible act (a palindromic half would make FlipH a no-op)
            if half != vmirror(half):
                break

        srcobj = half | shift(vmirror(half), (0, width(half)))
        srcobjh, srcobjw = shape(srcobj)
        if srcobjh > h or srcobjw > w:
            continue

        go = canvas(bgc, (srcobjh, srcobjw))
        go = fill(go, trgc, srcobj)

        loci = randint(0, h - srcobjh)
        locj = randint(0, w - srcobjw)
        gi = canvas(bgc, (h, w))
        shftd = shift(srcobj, (loci, locj))
        gi = fill(gi, trgc, shftd)

        indss = asindices(gi)
        # keep the whole bounding box of the target (plus a one-cell margin) free of
        # distractors, so the extracted subgrid really is the symmetric object alone
        prot = frozenset(
            (i, j)
            for i in range(loci - 1, loci + srcobjh + 1)
            for j in range(locj - 1, locj + srcobjw + 1)
        )
        indss = indss - prot

        remcands = asindices(canvas(-1, (8, 8))) - srcobj
        maxtrials = 4 * nobjs
        tr = 0
        succ = 0
        while succ < nobjs and tr <= maxtrials:
            if len(indss) == 0:
                break
            newobj = None
            for _try in range(50):
                cand = {e for e in srcobj}
                numperti = unifint(diff_lb, diff_ub, (1, 63))
                numpert = 64 - numperti
                for _p in range(numpert):
                    isadd = choice((True, False))
                    if isadd and len(cand) < 64:
                        cndds = totuple((remcands - cand) & mapply(neighbors, cand))
                        if len(cndds) == 0:
                            break
                        cand.add(choice(cndds))
                    if not isadd and len(cand) > 2:
                        cand = remove(choice(totuple(cand)), cand)
                cand = normalize(cand)
                a, b = shape(cand)
                cc = canvas(-1, (a + 2, b + 2))
                cc2 = compress(fill(cc, -2, shift(cand, (1, 1))))
                cand = toindices(argmax(colorfilter(objects(cc2, T, T, F), -2), size))
                if cand != vmirror(cand):          # distractors are never symmetric
                    newobj = cand
                    break
            if newobj is None:
                break
            col = choice(remcols)
            loccands = sfilter(indss, lambda ij: shift(newobj, ij).issubset(indss))
            if len(loccands) == 0:
                tr += 1
                continue
            locc = choice(totuple(loccands))
            newobj = shift(newobj, locc)
            gi = fill(gi, col, newobj)
            succ += 1
            indss = (indss - newobj) - mapply(neighbors, newobj)

        # background must stay dominant (derive_operations infers bgc that way)
        cnt = Counter([v for row in gi for v in row])
        if cnt.most_common(1)[0][0] != bgc or cnt[bgc] * 2 <= h * w:
            continue
        # the target's bounding box must show exactly the symmetric object
        sub = tuple(tuple(gi[i][locj:locj + srcobjw]) for i in range(loci, loci + srcobjh))
        if sub != go:
            continue
        return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule: exactly one object in I is its own left-right mirror; the answer is that
    object's bounding box.  The route performs that reflection instead of merely
    landing on its result: crop down to the object's LEFT HALF (the half that
    generates it), widen the canvas, copy that half into the empty right side and
    FlipH it - which is literally `half | vmirror(half)`.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background = the colour the generator paints the canvas with (dominant colour)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected, single-colour components of the non-background cells
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                col = I[r, c]
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
                comps.append(cells)

    # the target is the component equal to its own vmirror
    target = None
    for cells in comps:
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        s = set(cells)
        if all((y, c0 + c1 - x) in s for (y, x) in cells):
            if target is None or len(cells) > len(target[0]):
                target = (cells, (r0, c0, r1 - r0 + 1, c1 - c0 + 1))

    ops, sels = [], []

    if target is None:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    cells, (r0, c0, h, w) = target
    hw = w // 2
    left = I[r0:r0 + h, c0:c0 + hw] if hw > 0 else None

    if hw > 0 and w % 2 == 0 and not np.array_equal(left, np.fliplr(left)):
        # 1. crop the canvas down to the object's generating half
        #    (full rectangle, background included -> bbox selection is exact)
        ops.append(33); sels.append([r0, c0, h - 1, hw - 1])
        # 2. widen the canvas to the object's full width; the right half is empty
        ops.append(33); sels.append([0, 0, h - 1, w - 1])
        # 3. take the half from the input ...
        ops.append(28); sels.append([r0, c0, h - 1, hw - 1])
        # 4. ... and lay it into the empty right half
        ops.append(30); sels.append([0, hw, 0, 0])
        # 5. reflect that copy in place: the reflection the rule is about
        #    (full rectangle of the right half, background included)
        ops.append(26); sels.append([0, hw, h - 1, hw - 1])
    else:
        # degenerate half (a reflection would be invisible): just take the object
        ops.append(33); sels.append([r0, c0, h - 1, w - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task 72ca375d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 72ca375d"
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
                                f"for task 72ca375d"
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
                    f"Failed to build a complete episode for task 72ca375d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"72ca375d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
