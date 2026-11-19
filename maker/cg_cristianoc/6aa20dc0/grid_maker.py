"""
ARC Task: 6aa20dc0 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc, fgc, c1, c2 = random.sample(range(10), 4)
    return {"bgc": bgc, "fgc": fgc, "c1": c1, "c2": c2}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, c1, c2, **_ignored) -> dict:
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    od = unifint(diff_lb, diff_ub, (2, 4))
    ncellsextra = randint(1, max(1, (od ** 2 - 2) // 2))
    sinds = asindices(canvas(-1, (od, od)))
    extracells = set(sample(totuple(sinds - {(0, 0), (od - 1, od - 1)}), ncellsextra))
    extracells.add(choice(totuple(dneighbors((0, 0)) & sinds)))
    extracells.add(choice(totuple(dneighbors((od - 1, od - 1)) & sinds)))
    extracells = frozenset(extracells)
    obj = frozenset({(c1, (0, 0)), (c2, (od - 1, od - 1))}) | recolor(fgc, extracells)
    obj = obj | dmirror(obj)
    if choice((True, False)):
        obj = hmirror(obj)
    gi = canvas(bgc, (h, w))
    loci = randint(0, h - od)
    locj = randint(0, w - od)
    plcd = shift(obj, (loci, locj))
    gi = paint(gi, plcd)
    go = tuple(e for e in gi)
    inds = asindices(gi)
    inds = inds - backdrop(outbox(plcd))
    nocc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (od ** 2 * 2))))
    succ = 0
    tr = 0
    maxtr = 4 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = randint(1, 4)
        mf1 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        mf2 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        mf = compose(mf2, mf1)
        cobj = normalize(upscale(mf(obj), fac))
        ohx, owx = shape(cobj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - ohx and ij[1] <= w - owx)
        if len(cands) == 0:
            continue
        locc = choice(totuple(cands))
        cobjo = shift(cobj, locc)
        cobji = sfilter(cobjo, lambda cij: cij[0] != fgc)
        cobjoi = toindices(cobjo)
        if cobjoi.issubset(inds):
            succ += 1
            inds = inds - backdrop(outbox(cobjoi))
            gi = paint(gi, cobji)
            go = paint(go, cobjo)
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    """Reading: the multicolour TEMPLATE is a stamp. Every other place where
    its corner markers reappear (possibly mirrored / rotated 180 / upscaled)
    with an empty body gets a full copy of the template stamped onto it.
    Same-size copies are literally copied: CopyI the template's square,
    Paste it on the site, then flip the pasted copy in place into the site's
    orientation. Upscaled copies cannot be produced by copy/paste, so their
    body is painted with the template's body colour."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # background: canvas colour the generator paints first (all examples share it)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- template: 8-connected multicolour component with most colours
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            stack, comp = [(r, c)], []
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] != bgc:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(comp)
    if not comps:
        return [34], [[0, 0, h - 1, w - 1]]
    ncol = [len({int(I[r, c]) for r, c in comp}) for comp in comps]
    mx = max(ncol)
    tcells = [p for comp, n in zip(comps, ncol) if n == mx for p in comp]
    tr0 = min(p[0] for p in tcells); tr1 = max(p[0] for p in tcells)
    tc0 = min(p[1] for p in tcells); tc1 = max(p[1] for p in tcells)
    T = I[tr0:tr1 + 1, tc0:tc1 + 1].copy()
    BH, BW = T.shape
    vals = [int(v) for v in T.flatten() if v != bgc]
    if not vals:
        return [34], [[0, 0, h - 1, w - 1]]
    fgc = Counter(vals).most_common(1)[0][0]

    # orientations as flip sequences realisable by ARCLE (26=fliplr, 27=flipud)
    orients = [((), lambda a: a), ((26,), np.fliplr), ((27,), np.flipud),
               ((26, 27), lambda a: np.flipud(np.fliplr(a))),
               (None, lambda a: a.T), (None, lambda a: np.rot90(a, 2).T)]

    sites = []
    seen_sets = set()
    for f in range(1, 5):
        U = np.kron(T, np.ones((f, f), dtype=int))
        for flips, fn in orients:
            A = fn(U)
            UH, UW = A.shape
            marks = [(r, c, int(A[r, c])) for r in range(UH) for c in range(UW)
                     if A[r, c] != bgc and A[r, c] != fgc]
            body = [(r, c) for r in range(UH) for c in range(UW) if A[r, c] == fgc]
            if not marks or not body:
                continue
            mset = {(r, c) for r, c, _ in marks}
            ring = set()
            for r, c in mset:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if (r + dr, c + dc) not in mset:
                        ring.add((r + dr, c + dc))
            for r0 in range(h - UH + 1):
                for c0 in range(w - UW + 1):
                    if any(I[r0 + r, c0 + c] != v for r, c, v in marks):
                        continue
                    if any(I[r0 + r, c0 + c] != bgc for r, c in body):
                        continue
                    if any(0 <= r0 + r < h and 0 <= c0 + c < w and I[r0 + r, c0 + c] != bgc
                           for r, c in ring):
                        continue
                    filled = frozenset((r0 + r, c0 + c) for r, c in body)
                    if filled in seen_sets:
                        continue
                    seen_sets.add(filled)
                    if flips is None and f == 1:
                        # transposes of a dmirror-symmetric template equal a flip variant
                        for fl2, fn2 in orients[:4]:
                            if np.array_equal(fn2(T), A):
                                flips = fl2
                                break
                    mcells = frozenset((r0 + r, c0 + c) for r, c in mset)
                    # a copy's square may overlap other objects' cells; then it
                    # cannot be pasted/flipped as a whole square
                    clean = all(I[r0 + r, c0 + c] == bgc for r in range(UH) for c in range(UW)
                                if (r, c) not in mset)
                    sites.append((r0, c0, UH, UW, f, flips if clean else None,
                                  sorted(filled), mcells))

    # Copies never share marker cells: when stray matches overlap real ones,
    # keep the set of sites that uses every marker cell exactly once.
    universe = set().union(*[s[7] for s in sites]) if sites else set()

    def cover(left, used, chosen):
        if not left:
            return list(chosen)
        cell = min(left, key=lambda p: sum(1 for s in sites if p in s[7] and not (s[7] & used)))
        for s in sites:
            if cell in s[7] and not (s[7] & used):
                res = cover(left - s[7], used | s[7], chosen + [s])
                if res is not None:
                    return res
        return None

    chosen = cover(frozenset(universe), frozenset(), [])
    if chosen is not None:
        sites = chosen

    sites.sort(key=lambda s: (s[0], s[1]))
    # Paste of 0 is transparent: only copy when no template colour is 0 (bgc aside)
    copyable = all(int(v) != 0 for v in T.flatten() if v != bgc)
    copied = False
    for r0, c0, UH, UW, f, flips, filled, _m in sites:
        if f == 1 and flips is not None and copyable:
            if not copied:
                # the template's full square, background included
                ops.append(28); sels.append([tr0, tc0, BH - 1, BW - 1])
                copied = True
            ops.append(30); sels.append([r0, c0, 0, 0])
            # the cells the Paste just wrote: the whole square when bgc != 0,
            # only the template's coloured cells when bgc == 0 (0 is transparent)
            landed = [(r0 + r, c0 + c) for r in range(BH) for c in range(BW) if T[r, c] != 0]
            for k, fl in enumerate(flips):
                # first flip grabs the pasted copy, a second one keeps it grabbed
                ops.append(fl); sels.append(sel_of(landed) if k == 0 else sel_of([]))
        else:
            ops.append(int(fgc)); sels.append(sel_of(filled))

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
                        f"num_examples+1 ({num_examples + 1}) for task 6aa20dc0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6aa20dc0"
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
                                f"for task 6aa20dc0"
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
                    f"Failed to build a complete episode for task 6aa20dc0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6aa20dc0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
