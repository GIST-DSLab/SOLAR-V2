"""
ARC Task: beb8660c (RE-ARC) — LLM-generated grid_maker
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

# ── LLM-generated ──
import random
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of

ROTS = ['identity', 'rot90', 'rot180', 'rot270']


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{'rotf': v} for v in ROTS]
        examples += [{'rotf': random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{'rotf': v} for v in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {'bgc': bgc, 'instance_plan': plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rotf=None) -> dict:
    if rotf is None:
        rotf = random.choice(ROTS)
    cols = remove(8, interval(0, 10, 1))
    lim = max(3, min(max_h, max_w))
    w = unifint(diff_lb, diff_ub, (3, lim))
    h = unifint(diff_lb, diff_ub, (w, lim))
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    k = min(8, w - 1)
    k = unifint(diff_lb, diff_ub, (1, k))
    co = sample(remcols, k)
    wds = sorted(sample(interval(1, w, 1), k))
    for j, (c, l) in enumerate(zip(co, wds)):
        jj = h - k - 1 + j
        gi = fill(gi, c, connect((jj, 0), (jj, l - 1)))
    gi = fill(gi, 8, connect((h - 1, 0), (h - 1, w - 1)))
    go = vmirror(gi)
    gi = list(list(r) for r in gi[:-1])
    shuffle(gi)
    gi = tuple(tuple(r) for r in gi)
    gi = gi + go[-1:]
    gif = tuple()
    for r in gi:
        nbc = r.count(bgc)
        ofs = randint(0, nbc)
        gif = gif + (r[-ofs:] + r[:-ofs],)
    gi = vmirror(gif)
    rf = {'identity': identity, 'rot90': rot90, 'rot180': rot180, 'rot270': rot270}[rotf]
    return {'input': rf(gi), 'output': rf(go)}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: dominant non-8 colour (the generator's canvas colour)
    cnt = Counter(v for v in I.flatten().tolist() if v != 8)
    bgc = cnt.most_common(1)[0][0]

    # which edge carries the full 8-line -> canonical frame (8-line at bottom)
    if np.all(I[hi - 1, :] == 8):
        H, Wc = hi, wi
        f = lambda r, c: (r, c)
    elif np.all(I[0, :] == 8):
        H, Wc = hi, wi
        f = lambda r, c: (H - 1 - r, Wc - 1 - c)
    elif np.all(I[:, 0] == 8):
        H, Wc = wi, hi
        f = lambda r, c: (c, H - 1 - r)
    else:
        H, Wc = wi, hi
        f = lambda r, c: (Wc - 1 - c, r)

    def op_for(dr, dc):
        a = f(5, 5); b = f(5 + dr, 5 + dc)
        d = (b[0] - a[0], b[1] - a[1])
        return {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[d]

    def act(cells):
        return sel_of([f(r, c) for r, c in cells])

    C = np.array([[I[f(r, c)] for c in range(Wc)] for r in range(H)])

    # bars: one per canonical row (excluding the 8-line)
    bars = []
    for r in range(H - 1):
        cs = [c for c in range(Wc) if C[r, c] != bgc]
        if cs:
            bars.append({'color': int(C[r, cs[0]]), 'cells': [(r, c) for c in cs]})
    # longest bar next to the 8-line, all aligned to the canonical right edge
    bars.sort(key=lambda b: -len(b['cells']))
    for i, b in enumerate(bars):
        L = len(b['cells'])
        b['dest'] = [(H - 2 - i, c) for c in range(Wc - L, Wc)]
        b['done'] = set(b['cells']) == set(b['dest'])
        b['erased'] = False
        b['temped'] = False

    ops, sels = [], []

    def slide(b, target):
        src = list(b['cells'])
        dr = target[0][0] - src[0][0]
        dc = target[0][1] - src[0][1]
        steps = [(1 if dr > 0 else -1, 0)] * abs(dr) + [(0, 1 if dc > 0 else -1)] * abs(dc)
        cur = src
        for k, (sr, sc) in enumerate(steps):
            ops.append(op_for(sr, sc))
            sels.append(act(cur) if k == 0 else sel_of([]))
            cur = [(r + sr, c + sc) for r, c in cur]
        hole = sorted(set(src) - set(cur))
        if bgc != 0 and hole:
            ops.append(bgc); sels.append(act(hole))
        b['cells'] = cur

    def erase(b):
        ops.append(bgc); sels.append(act(b['cells']))
        b['cells'] = []
        b['erased'] = True

    def occupied_by_others(b, cells):
        s = set(cells)
        for o in bars:
            if o is b:
                continue
            if s & set(o['cells']):
                return True
        return False

    guard = 0
    while any(not b['done'] for b in bars):
        guard += 1
        if guard > 500:
            raise ValueError('scheduling failed')
        progressed = False
        for b in bars:
            if b['done'] or occupied_by_others(b, b['dest']):
                continue
            if b['color'] == 0 or b['erased']:
                # colour 0 cannot be carried by Move (transparent); erase then paint
                if not b['erased']:
                    erase(b)
                ops.append(b['color']); sels.append(act(b['dest']))
                b['cells'] = list(b['dest'])
            else:
                slide(b, b['dest'])
            b['done'] = True
            progressed = True
            break
        if progressed:
            continue
        # cycle: break it
        pend = [b for b in bars if not b['done']]
        zb = [b for b in pend if b['color'] == 0 and not b['erased']]
        if zb:
            erase(zb[0]); continue
        cand = None
        for b in pend:
            if b['temped']:
                continue
            L = len(b['cells'])
            r0, c0 = b['cells'][0]
            other_dest = set()
            for o in pend:
                if o is not b:
                    other_dest |= set(o['dest'])
            best = None
            for r in range(H - 1):
                for c in range(Wc - L + 1):
                    cells = [(r, cc) for cc in range(c, c + L)]
                    if occupied_by_others(b, cells) or set(cells) & other_dest:
                        continue
                    if set(cells) == set(b['cells']):
                        continue
                    d = abs(r - r0) + abs(c - c0)
                    if best is None or d < best[0]:
                        best = (d, cells)
            if best is not None:
                cand = (b, best[1]); break
        if cand is not None:
            b, cells = cand
            slide(b, cells)
            b['temped'] = True
        else:
            erase(pend[0])

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
                        f"num_examples+1 ({num_examples + 1}) for task beb8660c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task beb8660c"
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
                                f"for task beb8660c"
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
                    f"Failed to build a complete episode for task beb8660c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"beb8660c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
