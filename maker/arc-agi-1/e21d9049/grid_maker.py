"""
ARC Task: e21d9049 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    import random
    VARIANTS = [{"cross": True}, {"cross": False}]
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


def generate(diff_lb, diff_ub, max_h, max_w, bgc, cross=None) -> dict:
    import random
    if cross is None:
        cross = random.choice((True, False))
    # foreground colors: never 0 (0 is "transparent" for Copy/Paste), never bgc
    remcols = [c for c in range(1, 10) if c != bgc]

    lo_h = min(10, max_h)
    hi_h = max(lo_h, min(30, max_h))
    lo_w = min(10, max_w)
    hi_w = max(lo_w, min(30, max_w))

    while True:
        h = unifint(diff_lb, diff_ub, (lo_h, hi_h))
        w = unifint(diff_lb, diff_ub, (lo_w, hi_w))
        ph = unifint(diff_lb, diff_ub, (2, max(2, min(9, h - 1))))
        pw = unifint(diff_lb, diff_ub, (2, max(2, min(9, w - 1))))
        if ph > h or pw > w:
            continue
        locih = random.randint(0, h - ph)     # vertical bar rows locih..locih+ph-1
        locjw = random.randint(0, w - pw)     # horizontal bar cols locjw..locjw+pw-1
        if cross:
            lociw = random.randint(locih, locih + ph - 1)
            locjh = random.randint(locjw, locjw + pw - 1)
        else:
            lociw = random.randint(0, h - 1)  # horizontal bar row
            locjh = random.randint(0, w - 1)  # vertical bar column
            a_row = locih <= lociw <= locih + ph - 1
            a_col = locjw <= locjh <= locjw + pw - 1
            if a_row and a_col:
                continue
            # forbid a lone shared cell merging into the other bar's run
            # (would make the bar length ambiguous from the input alone)
            if a_row and (locjh == locjw - 1 or locjh == locjw + pw):
                continue
            if a_col and (lociw == locih - 1 or lociw == locih + ph):
                continue
        break

    vc = [random.choice(remcols) for _ in range(ph)]
    hc = [random.choice(remcols) for _ in range(pw)]
    col = random.choice(remcols)
    vc[(lociw - locih) % ph] = col
    hc[(locjh - locjw) % pw] = col

    gi = [[bgc] * w for _ in range(h)]
    for k in range(ph):
        gi[locih + k][locjh] = vc[k]
    for k in range(pw):
        gi[lociw][locjw + k] = hc[k]

    go = [row[:] for row in gi]
    for r in range(h):
        go[r][locjh] = vc[(r - locih) % ph]
    for c in range(w):
        go[lociw][c] = hc[(c - locjw) % pw]

    return {
        'input': tuple(tuple(r) for r in gi),
        'output': tuple(tuple(r) for r in go),
    }


def derive_operations(I, O):
    import numpy as np
    from collections import Counter

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    def longest_run(line):
        best_s, best_l, i, n = 0, 0, 0, len(line)
        while i < n:
            if line[i] != bgc:
                j = i
                while j < n and line[j] != bgc:
                    j += 1
                if j - i > best_l:
                    best_s, best_l = i, j - i
                i = j
            else:
                i += 1
        return best_s, best_l

    def plan(start, L, limit):
        # full-length repeats of a bar of length L anchored at `start`, period L
        origins = []
        t = start - L
        while t >= 0:
            origins.append(t)
            t -= L
        t = start + L
        while t + L <= limit:
            origins.append(t)
            t += L
        origins.sort()
        s = start % L                      # cells left over before the first full repeat
        head = (start + L - s, s) if s > 0 else None
        tmax = start
        while tmax + 2 * L <= limit:
            tmax += L
        nxt = tmax + L
        tail = (start, limit - nxt, nxt) if nxt < limit else None
        return origins, head, tail

    # the bar row / bar column are the only lines carrying more than one fg cell
    cr = [int((I[r] != bgc).sum()) for r in range(hi)]
    cc = [int((I[:, c] != bgc).sum()) for c in range(wi)]
    i0 = max(range(hi), key=lambda r: cr[r])
    j0 = max(range(wi), key=lambda c: cc[c])
    r0, ph = longest_run([int(v) for v in I[:, j0]])
    c0, pw = longest_run([int(v) for v in I[i0, :]])

    ops, sels = [], []

    # --- vertical bar: repeat it down column j0 with period ph ---
    origins, head, tail = plan(r0, ph, hi)
    if origins:
        ops.append(28); sels.append([r0, j0, ph - 1, 0])
        for t in origins:
            ops.append(30); sels.append([t, j0, 0, 0])
    if head:
        ops.append(28); sels.append([head[0], j0, head[1] - 1, 0])
        ops.append(30); sels.append([0, j0, 0, 0])
    if tail:
        ops.append(28); sels.append([tail[0], j0, tail[1] - 1, 0])
        ops.append(30); sels.append([tail[2], j0, 0, 0])

    # --- horizontal bar: repeat it across row i0 with period pw ---
    origins, head, tail = plan(c0, pw, wi)
    if origins:
        ops.append(28); sels.append([i0, c0, 0, pw - 1])
        for t in origins:
            ops.append(30); sels.append([i0, t, 0, 0])
    if head:
        ops.append(28); sels.append([i0, head[0], 0, head[1] - 1])
        ops.append(30); sels.append([i0, 0, 0, 0])
    if tail:
        ops.append(28); sels.append([i0, tail[0], 0, tail[1] - 1])
        ops.append(30); sels.append([i0, tail[2], 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task e21d9049"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e21d9049"
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
                                f"for task e21d9049"
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
                    f"Failed to build a complete episode for task e21d9049 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e21d9049-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
