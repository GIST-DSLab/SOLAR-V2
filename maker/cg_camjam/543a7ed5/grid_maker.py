"""
ARC Task: 543a7ed5 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque

import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (3, 4)]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numc = random.randint(1, 7)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round((b - a) * lb))
        hi = a + int(round((b - a) * ub))
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(lo, hi)

    cols = [c for c in range(10) if c not in (3, 4)]
    if bgc is None:
        bgc = random.choice(cols)
    if ccols is None:
        remcols = [c for c in cols if c != bgc]
        ccols = random.sample(remcols, random.randint(1, 7))
    ccols = [c for c in ccols if c != bgc] or [random.choice([c for c in cols if c != bgc])]

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    indss = {(i, j) for i in range(h) for j in range(w)}
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if not indss:
            break
        oh = random.randint(4, 8)
        ow = random.randint(4, 8)
        subs = [ij for ij in indss if ij[0] < h - oh and ij[1] < w - ow]
        if not subs:
            tr += 1
            continue
        loci, locj = random.choice(subs)
        bd = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
        col = random.choice(ccols)
        if bd.issubset(indss):
            for i in range(loci, loci + oh):
                for j in range(locj, locj + ow):
                    border = i in (loci, loci + oh - 1) or j in (locj, locj + ow - 1)
                    if border:
                        go[i][j] = 3
                    else:
                        go[i][j] = col
                        gi[i][j] = col
            if oh > 5 and ow > 5 and random.randint(1, 10) != 1:
                ulci, ulcj = loci + 1, locj + 1
                lrci, lrcj = loci + oh - 2, locj + ow - 2
                aa = random.randint(ulci + 1, lrci - 1)
                aa = random.randint(ulci + 1, aa)
                bb = random.randint(ulcj + 1, lrcj - 1)
                bb = random.randint(ulcj + 1, bb)
                cc = random.randint(aa, lrci - 1)
                dd = random.randint(bb, lrcj - 1)
                cc = random.randint(cc, lrci - 1)
                dd = random.randint(dd, lrcj - 1)
                for i in range(aa, cc + 1):
                    for j in range(bb, dd + 1):
                        go[i][j] = 4
                        gi[i][j] = bgc
            succ += 1
            indss -= bd
        tr += 1
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (shared by maker, verifier and the golf sweep): every object gets a
    # colour-3 frame on the ring of cells hugging it (its outbox), and any background
    # pocket enclosed inside an object turns 4.
    # The golf entry finds the ring and the pocket with whole-grid directional sweeps.
    # Here the unit is the object: its outbox ring is painted in ONE Color3 op, and
    # its enclosed pocket, a connected region, is filled with ONE FloodFill4 from a
    # seed. The maker uses four line ops and a rectangle op instead.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the canvas colour the generator paints first. It is the majority
    # colour; check it against the example inputs when they are given.
    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(counts)])
    if examples:
        from collections import Counter
        tally = Counter()
        for ex in examples:
            ei = np.asarray(ex[0], dtype=int)
            v, c = np.unique(ei, return_counts=True)
            tally[int(v[np.argmax(c)])] += 1
        if bgc not in tally and tally:
            pass  # keep the per-grid majority; the examples share the same canvas idea

    # Univalued, 4-connected objects that are not background
    seen = np.zeros_like(I, dtype=bool)
    objs = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                col = I[r, c]
                comp = []
                dq = deque([(r, c)])
                seen[r, c] = True
                while dq:
                    a, b = dq.popleft()
                    comp.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < hi and 0 <= y < wi and not seen[x, y] and I[x, y] == col:
                            seen[x, y] = True
                            dq.append((x, y))
                objs.append(comp)

    ops, sels = [], []
    for comp in objs:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)

        # Outbox ring: the one-cell frame just outside the object's bounding box
        ring = []
        for r in range(r0 - 1, r1 + 2):
            for c in range(c0 - 1, c1 + 2):
                if r in (r0 - 1, r1 + 1) or c in (c0 - 1, c1 + 1):
                    if 0 <= r < hi and 0 <= c < wi:
                        ring.append((r, c))
        if ring:
            ops.append(3)
            sels.append(sel_of(ring))

        # Enclosed background pocket inside the object: fill it from one seed
        seed = None
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if I[r, c] == bgc:
                    seed = (r, c)
                    break
            if seed:
                break
        if seed is not None:
            ops.append(14)
            sels.append([seed[0], seed[1], 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 543a7ed5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 543a7ed5"
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
                                f"for task 543a7ed5"
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
                    f"Failed to build a complete episode for task 543a7ed5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"543a7ed5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
