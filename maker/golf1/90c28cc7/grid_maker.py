"""
ARC Task: 90c28cc7 (RE-ARC) — LLM-generated grid_maker
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

try:
    from dsl import *
except Exception:
    pass
try:
    from utils import unifint
except Exception:
    def unifint(diff_lb, diff_ub, bounds):
        a, b = bounds
        if a >= b:
            return a
        lo = a + (b - a) * diff_lb
        hi = a + (b - a) * diff_ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))


def sample_colors(num_examples=None) -> dict:
    # Background/frame is hardcoded 0; cell colours are random per cell and the
    # rule (collapse repeated rows/cols, drop the 0 frame) never depends on them.
    return {}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **kwargs) -> dict:
    cols = list(range(1, 10))
    h = unifint(diff_lb, diff_ub, (2, min(10, max_h)))
    w = unifint(diff_lb, diff_ub, (2, min(10, max_w)))
    nc = unifint(diff_lb, diff_ub, (2, 9))
    colss = random.sample(cols, nc)
    g = [[random.choice(colss) for _ in range(w)] for _ in range(h)]

    def dedupe(rows):
        out = [rows[0]]
        for r in rows[1:]:
            if r != out[-1]:
                out.append(r)
        return out

    def T(rows):
        return [list(x) for x in zip(*rows)]

    g = T(dedupe(T(dedupe(g))))
    go = [list(r) for r in g]
    h, w = len(g), len(g[0])
    fullh = unifint(diff_lb, diff_ub, (h, max_h))
    fullw = unifint(diff_lb, diff_ub, (w, max_w))
    inh = unifint(diff_lb, diff_ub, (h, fullh))
    inw = unifint(diff_lb, diff_ub, (w, fullw))
    while h < inh or w < inw:
        if h < inh:
            idx = random.randint(0, h - 1)
            g = g[:idx + 1] + [list(g[idx])] + g[idx + 1:]
        else:
            g = T(g)
            idx = random.randint(0, w - 1)
            g = g[:idx + 1] + [list(g[idx])] + g[idx + 1:]
            g = T(g)
        h, w = len(g), len(g[0])
    while h < fullh or w < fullw:
        if h < fullh:
            z = [[0] * w]
            g = z + g if random.random() < 0.5 else g + z
        else:
            g = T(g)
            z = [[0] * h]
            g = z + g if random.random() < 0.5 else g + z
            g = T(g)
        h, w = len(g), len(g[0])
    return {"input": [list(r) for r in g], "output": go}


def derive_operations(I, O=None, examples=None):
    """
    Rule (read from I alone): the picture sits in a frame of 0 rows/cols; inside it every
    row of the answer is stretched into a run of identical rows, and likewise for columns.
    Keep one row per run and one column per run, then crop.

    Big strokes: the surviving rows from each run can be any member of the run (they are
    identical). Pick, for each run, its first or last row so that as many survivors as
    possible are adjacent — each maximal adjacent group then moves as ONE block
    (CopyI + Paste). The first group stays put and the final CropGrid starts there, so no
    op is spent on it. Then the same for column groups on the row-collapsed strip
    (CopyO + Paste), then one CropGrid.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    nz = np.argwhere(I != 0)
    r0, c0 = int(nz[:, 0].min()), int(nz[:, 1].min())
    r1, c1 = int(nz[:, 0].max()), int(nz[:, 1].max())
    core = I[r0:r1 + 1, c0:c1 + 1]

    def runs_of(lines):
        rs, s = [], 0
        for i in range(1, len(lines) + 1):
            if i == len(lines) or not np.array_equal(lines[i], lines[i - 1]):
                rs.append((s, i - 1))
                s = i
        return rs

    def pick(rs):
        # DP: per run choose its start (0) or end (1) as representative, minimising
        # the number of breaks between consecutive representatives.
        n = len(rs)
        INF = 10 ** 9
        cost = [[0, 0]] + [[INF, INF] for _ in range(n - 1)]
        back = [[None, None] for _ in range(n)]
        for k in range(1, n):
            for t in (0, 1):
                for u in (0, 1):
                    brk = 0 if rs[k - 1][u] + 1 == rs[k][t] else 1
                    v = cost[k - 1][u] + brk
                    if v < cost[k][t]:
                        cost[k][t] = v
                        back[k][t] = u
        t = 0 if cost[n - 1][0] <= cost[n - 1][1] else 1
        ch = [0] * n
        for k in range(n - 1, -1, -1):
            ch[k] = rs[k][t]
            if k:
                t = back[k][t]
        groups, s = [], 0
        for k in range(1, n + 1):
            if k == n or ch[k] != ch[k - 1] + 1:
                groups.append((s, k - 1))
                s = k
        return ch, groups

    rch, rgr = pick(runs_of(core))
    cch, cgr = pick(runs_of(core.T))
    ho, wo = len(rch), len(cch)
    R0, C0 = r0 + rch[0], c0 + cch[0]   # the collapsed block's top-left (first group stays)
    cw = core.shape[1]

    # rows: each later group of kept rows slides up as one block (source = input)
    for a, b in rgr[1:]:
        ops.append(28); sels.append([r0 + rch[a], c0, b - a, cw - 1])   # CopyI row block
        ops.append(30); sels.append([R0 + a, c0, 0, 0])                 # Paste under previous

    # cols: each later group of kept columns slides left, on the collapsed row strip
    for a, b in cgr[1:]:
        ops.append(29); sels.append([R0, c0 + cch[a], ho - 1, b - a])   # CopyO column block
        ops.append(30); sels.append([R0, C0 + a, 0, 0])                 # Paste right of previous

    # keep only the collapsed block (full rectangle)
    if (R0, C0, ho, wo) != (0, 0, hi, wi):
        ops.append(33); sels.append([R0, C0, ho - 1, wo - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task 90c28cc7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 90c28cc7"
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
                                f"for task 90c28cc7"
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
                    f"Failed to build a complete episode for task 90c28cc7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"90c28cc7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
