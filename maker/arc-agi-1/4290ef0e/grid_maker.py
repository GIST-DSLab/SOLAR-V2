"""
ARC Task: 4290ef0e (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"dot": True}, {"dot": False}]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc=0, dot=None) -> dict:
    cols = interval(0, 10, 1)
    dmax = max(2, min(7, max_h // 4, max_w // 4))
    while True:
        d = unifint(diff_lb, diff_ub, (2, dmax))
        h, w = d, d
        fullh = unifint(diff_lb, diff_ub, (min(4 * d, max_h), max_h))
        fullw = unifint(diff_lb, diff_ub, (min(4 * d, max_w), max_w))
        remcols = remove(bgc, cols)
        ccols = sample(remcols, d)
        quad = canvas(bgc, (d + 1, d + 1))
        for idx, c in enumerate(ccols):
            linlen = randint(2, w - idx + 1)
            quad = fill(quad, c, (connect((idx, idx), (idx + linlen - 1, idx))))
            quad = fill(quad, c, (connect((idx, idx), (idx, idx + linlen - 1))))
        go = canvas(bgc, (d + 1, 2 * d + 1))
        qobj1 = asobject(quad)
        qobj2 = shift(asobject(vmirror(quad)), (0, d))
        go = paint(go, qobj1)
        go = paint(go, qobj2)
        go = vconcat(go, hmirror(go)[1:])
        usedot = choice((True, False)) if dot is None else bool(dot)
        if usedot:
            go = fill(go, choice(difference(remcols, ccols)), {center(asindices(go))})
        objs = partition(go)
        objs = sfilter(objs, lambda o: color(o) != bgc)
        gi = canvas(bgc, (fullh, fullw))
        objs = order(objs, width)
        fullinds = asindices(gi)
        inds = asindices(gi)
        fullsuc = True
        for obj in objs:
            objn = normalize(obj)
            obji = toindices(objn)
            dd = width(obj)
            dh = max(0, dd // 2 - 1)
            cands = sfilter(fullinds, lambda ij: ij[0] <= fullh - dd and ij[1] <= fullw - dd)
            cands = cands | shift(cands, (-dh, 0)) | shift(cands, (0, -dh)) | shift(cands, (dh, 0)) | shift(cands, (0, dh))
            maxtr = 10
            tr = 0
            succ = False
            if len(cands) == 0:
                break
            while tr < maxtr and not succ:
                tr += 1
                loc = choice(totuple(cands))
                if (shift(obji, loc) & fullinds).issubset(inds):
                    succ = True
                    break
            if not succ:
                fullsuc = False
                break
            gi = paint(gi, shift(objn, loc))
            inds = inds - shift(obji, loc)
        if not fullsuc:
            continue
        break
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)

    def runs(vals):
        out = []
        for v in vals:
            if out and v == out[-1][-1] + 1:
                out[-1].append(v)
            else:
                out.append([v])
        return out

    # background = the canvas colour the objects are scattered on
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Each foreground colour in I is one square, 4-fold-symmetric frame, possibly
    # clipped by a grid edge. Recover its true size and complete it by symmetry.
    pats = []
    for c in sorted(set(I.flatten().tolist())):
        if c == bgc:
            continue
        rs, cs = np.where(I == c)
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        hf, wf = r1 - r0 + 1, c1 - c0 + 1
        s = max(hf, wf)                      # unclipped axis gives the real frame size
        ar = ac = 0
        if hf < s:                           # rows clipped: the dense edge row is a frame edge
            if int((I[r0, c0:c1 + 1] == c).sum()) <= int((I[r1, c0:c1 + 1] == c).sum()):
                ar = s - hf
        if wf < s:
            if int((I[r0:r1 + 1, c0] == c).sum()) <= int((I[r0:r1 + 1, c1] == c).sum()):
                ac = s - wf
        F = np.zeros((s, s), dtype=bool)
        F[ar + (rs - r0), ac + (cs - c0)] = True
        P = F | np.rot90(F, 1) | np.rot90(F, 2) | np.rot90(F, 3)
        pats.append((s, int(c), P))

    pats.sort(key=lambda t: -t[0])
    n = pats[0][0]                           # biggest frame sets the output canvas

    ops, sels = [], []
    ops.append(33); sels.append([0, 0, n - 1, n - 1])          # canvas -> n x n
    if not bool(np.all(I[:n, :n] == bgc)):
        ops.append(int(bgc)); sels.append([0, 0, n - 1, n - 1])  # bgc canvas

    for s, c, P in pats:                     # nest frames concentrically, outermost first
        o = (n - s) // 2
        cells = {(o + int(r), o + int(x)) for r, x in zip(*np.where(P))}
        painted = set()
        for r in sorted({o, o + s - 1}):                    # horizontal frame segments
            xs = sorted(x for (rr, x) in cells if rr == r)
            for run in runs(xs):
                ops.append(c); sels.append([r, run[0], 0, len(run) - 1])
                painted.update((r, x) for x in run)
        for x in sorted({o, o + s - 1}):                    # vertical frame segments
            ys = sorted(rr for (rr, cc) in cells if cc == x and (rr, cc) not in painted)
            for run in runs(ys):
                ops.append(c); sels.append([run[0], x, len(run) - 1, 0])
                painted.update((y, x) for y in run)

    ops.append(34); sels.append([0, 0, n - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 4290ef0e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4290ef0e"
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
                                f"for task 4290ef0e"
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
                    f"Failed to build a complete episode for task 4290ef0e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4290ef0e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
