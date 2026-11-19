"""
ARC Task: 29ec7d0e (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    return {"bgc": bgc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or noisec is None:
        bgc, noisec = sample(cols, 2)
    rotf = choice((identity, rot90, rot180, rot270))
    swap = rotf in (rot90, rot270)
    lim_h = max(10, min(30, max_w if swap else max_h))
    lim_w = max(10, min(30, max_h if swap else max_w))
    h = unifint(diff_lb, diff_ub, (10, lim_h))
    w = unifint(diff_lb, diff_ub, (10, lim_w))
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
            loci = (a + 1) + hp * a
            locj = (b + 1) + wp * b
            locs.add((loci, locj))
            go = paint(go, shift(pobj, (loci, locj)))
    numpatches = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
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
        if pobj in apply(normalize, apply(rbind(toobject, gi2), places)):
            if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                succ += 1
                gi = gi2
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def _period(I, noise, axis):
    A = I if axis == 1 else I.T
    N = noise if axis == 1 else noise.T
    n = A.shape[1]
    for p in range(1, n):
        a, b = A[:, :-p], A[:, p:]
        ok = ~N[:, :-p] & ~N[:, p:]
        if np.all(a[ok] == b[ok]):
            return p
    return n


def _noise_color(I, examples):
    if examples:
        cand = None
        for ei, eo in examples:
            ei = np.asarray(ei); eo = np.asarray(eo)
            s = set(np.unique(ei).tolist()) - set(np.unique(eo).tolist())
            cand = s if cand is None else (cand & s)
        if cand:
            present = [c for c in cand if (I == c).any()]
            if len(present) == 1:
                return present[0]
    # fallback: the noise colour is the one whose removal leaves the smallest
    # consistent 2-D period with every residue class still visible
    best = None
    h, w = I.shape
    for col in np.unique(I).tolist():
        nz = (I == col)
        pc = _period(I, nz, 1); pr = _period(I, nz, 0)
        cov = set((r % pr, c % pc) for r in range(h) for c in range(w) if not nz[r, c])
        key = (pr * pc, pr * pc - len(cov), int(nz.sum()))
        if best is None or key < best[0]:
            best = (key, col)
    return best[1]


def _bg_color(I, examples):
    # bgc is fixed per episode: the separator-line colour, i.e. a colour filling
    # a whole row AND a whole column of every demonstration output
    cand = None
    for ei, eo in (examples or []):
        eo = np.asarray(eo)
        rows = {int(r[0]) for r in eo if (r == r[0]).all()}
        cols = {int(c[0]) for c in eo.T if (c == c[0]).all()}
        s = rows & cols
        cand = s if cand is None else (cand & s)
    if cand:
        return min(cand)
    return Counter(I.flatten().tolist()).most_common(1)[0][0]


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    nc = _noise_color(I, examples)
    noise = (I == nc)
    pc = _period(I, noise, 1)
    pr = _period(I, noise, 0)
    # periodic reference value per residue class, read from clean cells of I
    ref = {}
    for r in range(h):
        for c in range(w):
            if not noise[r, c]:
                ref.setdefault((r % pr, c % pc), int(I[r, c]))
    bgc = None
    ops, sels = [], []
    seen = np.zeros_like(noise)
    for r0 in range(h):
        for c0 in range(w):
            if noise[r0, c0] and not seen[r0, c0]:
                comp = []
                st = [(r0, c0)]; seen[r0, c0] = True
                while st:
                    a, b = st.pop(); comp.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < h and 0 <= y < w and noise[x, y] and not seen[x, y]:
                            seen[x, y] = True; st.append((x, y))
                # restore this noise patch: group its cells by the periodic colour
                groups = {}
                for (a, b) in comp:
                    key = (a % pr, b % pc)
                    if key not in ref:
                        # residue hidden everywhere: only separator lines can be -> bgc
                        if bgc is None:
                            bgc = _bg_color(I, examples)
                        ref[key] = bgc
                    groups.setdefault(ref[key], []).append((a, b))
                for col in sorted(groups, key=lambda k: -len(groups[k])):
                    ops.append(col); sels.append(sel_of(sorted(groups[col])))
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
                        f"num_examples+1 ({num_examples + 1}) for task 29ec7d0e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 29ec7d0e"
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
                                f"for task 29ec7d0e"
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
                    f"Failed to build a complete episode for task 29ec7d0e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"29ec7d0e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
