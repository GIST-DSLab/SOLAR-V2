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
from itertools import product

def sample_colors(num_examples=None) -> dict:
    # Frame colour 0 is hardcoded; picture colours are resampled per cell and the rule
    # (dedupe rows/cols inside the frame) does not depend on them.
    return {}

def _dedupe(g):
    out = [g[0]]
    for r in g[1:]:
        if r != out[-1]:
            out.append(r)
    return out

def _dm(g):
    return [list(x) for x in zip(*g)]

def generate(diff_lb, diff_ub, max_h=30, max_w=30, **kw) -> dict:
    def unifint(lo, hi):
        a, b = lo, hi
        lb = a + (b - a) * diff_lb
        ub = a + (b - a) * diff_ub
        lb, ub = int(round(lb)), int(round(ub))
        lb, ub = max(a, min(lb, b)), max(a, min(ub, b))
        if lb > ub:
            lb, ub = ub, lb
        return random.randint(lb, ub)
    cols = list(range(1, 10))
    h = unifint(2, min(10, max_h))
    w = unifint(2, min(10, max_w))
    nc = unifint(2, 9)
    colss = random.sample(cols, nc)
    gi = [[random.choice(colss) for _ in range(w)] for _ in range(h)]
    gi = _dm(_dedupe(_dm(_dedupe(gi))))
    go = [list(r) for r in gi]
    h, w = len(gi), len(gi[0])
    fullh = unifint(h, max_h)
    fullw = unifint(w, max_w)
    inh = unifint(h, fullh)
    inw = unifint(w, fullw)
    while h < inh or w < inw:
        if h < inh:
            idx = random.randint(0, h - 1)
            gi = gi[:idx + 1] + gi[idx:]
        else:
            gi = _dm(gi)
            idx = random.randint(0, w - 1)
            gi = gi[:idx + 1] + gi[idx:]
            gi = _dm(gi)
        h, w = len(gi), len(gi[0])
    while h < fullh or w < fullw:
        if h < fullh:
            z = [[0] * w]
            gi = z + gi if random.random() < 0.5 else gi + z
        else:
            gi = _dm(gi)
            z = [[0] * len(gi[0])]
            gi = z + gi if random.random() < 0.5 else gi + z
            gi = _dm(gi)
        h, w = len(gi), len(gi[0])
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Rule: inside a 0-frame, every distinct row of the picture is stretched into a run of
    identical rows, likewise for columns; O keeps one row per run and one col per run.

    Plan: pick a window (rows p.., cols q..) of output size anywhere on the grid. Output
    line k is already "in place" if grid line p+k lies inside run k. Every other maximal
    group of output lines that can be served by one contiguous source strip (constant
    shift into their runs) is fetched with one Copy + one Paste. First one axis from the
    input (CopyI), then the other axis from the working grid (CopyO), then CropGrid.
    All window offsets, both axis orders are searched and replay-verified; the shortest wins.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    nz = np.argwhere(I != 0)
    r0, c0 = int(nz[:, 0].min()), int(nz[:, 1].min())
    r1, c1 = int(nz[:, 0].max()), int(nz[:, 1].max())
    core = I[r0:r1 + 1, c0:c1 + 1]
    ch, cw = core.shape

    def run_spans(M, off):
        spans, s = [], 0
        for i in range(1, M.shape[0] + 1):
            if i == M.shape[0] or not np.array_equal(M[i], M[i - 1]):
                spans.append((off + s, off + i - 1))
                s = i
        return spans

    rowspans = run_spans(core, r0)
    colspans = run_spans(core.T, c0)

    def plan_axis(spans, p):
        n = len(spans)
        INF = 10 ** 9
        best = [INF] * (n + 1)
        choice = [None] * (n + 1)
        best[0] = 0
        for b in range(1, n + 1):
            lo, hi_ = -INF, INF
            for a in range(b - 1, -1, -1):
                s, e = spans[a]
                lo = max(lo, s - p - a)
                hi_ = min(hi_, e - p - a)
                if lo > hi_:
                    break
                if lo <= 0 <= hi_:
                    cost, d = 0, 0
                else:
                    cost = 1
                    d = lo if lo > 0 else hi_
                if best[a] + cost < best[b]:
                    best[b] = best[a] + cost
                    choice[b] = (a, d, cost)
        blocks, b = [], n
        while b > 0:
            a, d, cost = choice[b]
            if cost:
                blocks.append((a, b - 1, d))  # output idx a..b-1, src = p+idx+d
            b = a
        return best[n], blocks[::-1]

    def simulate(ops, sels):
        g = I.copy()
        clip = None
        for op, s in zip(ops, sels):
            r, c, h, w = s
            if op == 28 or op == 29:
                src = I if op == 28 else g
                clip = src[r:r + h + 1, c:c + w + 1].copy()
            elif op == 30:
                ch_, cw_ = clip.shape
                reg = g[r:r + ch_, c:c + cw_]
                cc = clip[:reg.shape[0], :reg.shape[1]]
                reg[cc != 0] = cc[cc != 0]
            elif op == 33:
                sub = g[r:r + h + 1, c:c + w + 1]
                ng = np.zeros_like(sub)
                ng[sub != 0] = sub[sub != 0]
                g = ng
        return g

    def order_blocks(blocks, p):
        # blocks read from the working grid: a block's source must not be overwritten first
        m = len(blocks)
        rng = [((p + a + d, p + b + d), (p + a, p + b)) for a, b, d in blocks]
        before = {i: set() for i in range(m)}
        for x in range(m):
            for y in range(m):
                if x != y:
                    (sx0, sx1), _ = rng[x]
                    _, (dy0, dy1) = rng[y]
                    if sx0 <= dy1 and dy0 <= sx1:
                        before[y].add(x)
        done, order = set(), []
        while len(order) < m:
            ready = [i for i in range(m) if i not in done and before[i] <= done]
            if not ready:
                return None
            i = ready[0]
            done.add(i); order.append(i)
        return [blocks[i] for i in order]

    def build(rows_first, p, q):
        ops, sels = [], []
        if rows_first:
            _, rb = plan_axis(rowspans, p)
            _, cb = plan_axis(colspans, q)
            for a, b, d in rb:
                ops.append(28); sels.append([p + a + d, c0, b - a, cw - 1])
                ops.append(30); sels.append([p + a, c0, 0, 0])
            cb = order_blocks(cb, q)
            if cb is None:
                return None
            for a, b, d in cb:
                ops.append(29); sels.append([p, q + a + d, ho - 1, b - a])
                ops.append(30); sels.append([p, q + a, 0, 0])
        else:
            _, cb = plan_axis(colspans, q)
            _, rb = plan_axis(rowspans, p)
            for a, b, d in cb:
                ops.append(28); sels.append([r0, q + a + d, ch - 1, b - a])
                ops.append(30); sels.append([r0, q + a, 0, 0])
            rb = order_blocks(rb, p)
            if rb is None:
                return None
            for a, b, d in rb:
                ops.append(29); sels.append([p + a + d, q, b - a, wo - 1])
                ops.append(30); sels.append([p + a, q, 0, 0])
        if (p, q, ho, wo) != (0, 0, hi, wi):
            ops.append(33); sels.append([p, q, ho - 1, wo - 1])
        return ops, sels

    cands = []
    for p in range(0, hi - ho + 1):
        cr, _ = plan_axis(rowspans, p)
        for q in range(0, wi - wo + 1):
            cc, _ = plan_axis(colspans, q)
            cands.append((cr + cc, p, q))
    cands.sort()
    best = None
    for cost, p, q in cands:
        if best is not None and 2 * cost >= len(best[0]):
            break
        for rf in (True, False):
            res = build(rf, p, q)
            if res is None:
                continue
            g = simulate(*res)
            if g.shape == O.shape and np.array_equal(g, O):
                if best is None or len(res[0]) < len(best[0]):
                    best = res
    ops, sels = best
    # drop any Copy+Paste pair whose paste leaves the grid unchanged (e.g. the window line
    # already holds an identical, non-adjacent run)
    k = 0
    while k < len(ops):
        if ops[k] in (28, 29) and k + 1 < len(ops) and ops[k + 1] == 30:
            before = simulate(ops[:k], sels[:k])
            after = simulate(ops[:k + 2], sels[:k + 2])
            if before.shape == after.shape and np.array_equal(before, after):
                del ops[k:k + 2]; del sels[k:k + 2]
                continue
        k += 2 if ops[k] in (28, 29) else 1
    ops = ops + [34]
    sels = sels + [[0, 0, ho - 1, wo - 1]]
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
