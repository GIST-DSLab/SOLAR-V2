"""
ARC Task: dc433765 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from maker.sel_helpers import sel_of

VARIANTS = [{"diagonal": True}, {"diagonal": False}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, src = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "src": src, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, src=1, diagonal=None) -> dict:
    if diagonal is None:
        diagonal = random.choice([True, False])
    mfs = ["identity", "dmirror", "cmirror", "vmirror", "hmirror", "rot90", "rot180", "rot270"]
    nmfs = random.choice((1, 2))
    fns = random.sample(mfs, nmfs)
    transposing = any(f in ("dmirror", "cmirror", "rot90", "rot270") for f in fns)
    if transposing:
        mh = mw = min(max_h, max_w)
    else:
        mh, mw = max_h, max_w
    h = _unifint(diff_lb, diff_ub, (4, max(4, mh)))
    w = _unifint(diff_lb, diff_ub, (4, max(4, mw)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    if diagonal:
        starts = [(ii, 0) for ii in range(h - 2)] + [(0, jj) for jj in range(1, w - 2)]
        opts = []
        for (r0, c0) in starts:
            ln = []
            r, c = r0, c0
            while r < h and c < w:
                ln.append((r, c))
                r += 1
                c += 1
            opts.append(ln)
        opts.sort(key=len)
        k = len(opts)
        opt = _unifint(diff_lb, diff_ub, (0, k - 1))
        ln = sorted(opts[opt])
        epi = _unifint(diff_lb, diff_ub, (2, len(ln) - 1))
        ep = ln[epi]
        ln2 = ln[:epi - 1][::-1]
        spi = _unifint(diff_lb, diff_ub, (0, len(ln2) - 1))
        sp = ln2[spi]
        gi[sp[0]][sp[1]] = src
        gi[ep[0]][ep[1]] = 4
        go[sp[0] + 1][sp[1] + 1] = src
        go[ep[0]][ep[1]] = 4
    else:
        loci = random.randint(0, h - 1)
        objw = _unifint(diff_lb, diff_ub, (3, w))
        locj1 = random.randint(0, w - objw)
        locj2 = locj1 + objw - 1
        gi[loci][locj1] = src
        gi[loci][locj2] = 4
        go[loci][locj1 + 1] = src
        go[loci][locj2] = 4
    gi = np.array(gi)
    go = np.array(go)

    def apply(fn, g):
        if fn == "identity":
            return g
        if fn == "dmirror":
            return g.T
        if fn == "cmirror":
            return np.rot90(g, 2).T
        if fn == "vmirror":
            return np.fliplr(g)
        if fn == "hmirror":
            return np.flipud(g)
        if fn == "rot90":
            return np.rot90(g, 3)
        if fn == "rot180":
            return np.rot90(g, 2)
        if fn == "rot270":
            return np.rot90(g, 1)
        return g

    for fn in fns:
        gi = apply(fn, gi)
        go = apply(fn, go)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    four = None
    srcpos = None
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v == bgc:
                continue
            if v == 4:
                four = (r, c)
            else:
                srcpos = (r, c)
    ops, sels = [], []
    if four is not None and srcpos is not None:
        dr = int(np.sign(four[0] - srcpos[0]))
        dc = int(np.sign(four[1] - srcpos[1]))
        steps = []
        if dr != 0:
            steps.append((21 if dr > 0 else 20, dr, 0))
        if dc != 0:
            steps.append((22 if dc > 0 else 23, 0, dc))
        cur = [srcpos]
        first = True
        for op, sr, sc in steps:
            ops.append(op)
            sels.append(sel_of(cur) if first else sel_of([]))
            first = False
            cur = [(r + sr, c + sc) for r, c in cur]
        hole = sorted(set([srcpos]) - set(cur))
        if bgc != 0 and hole and steps:
            ops.append(bgc)
            sels.append(sel_of(hole))
    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task dc433765"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task dc433765"
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
                                f"for task dc433765"
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
                    f"Failed to build a complete episode for task dc433765 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"dc433765-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
