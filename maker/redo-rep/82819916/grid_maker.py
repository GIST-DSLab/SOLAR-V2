"""
ARC Task: 82819916 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


# ---------------------------------------------------------------- 1. colours
def sample_colors(num_examples=None) -> dict:
    # only the background is a structural colour role here; the line colours are
    # free (the rule is about which positions repeat, not about which hues).
    bgc = random.choice(list(range(10)))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]   # test variant was shown
    return {"bgc": bgc, "instance_plan": plan}


# ---------------------------------------------------------------- 2. generate
def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None) -> dict:
    if rot is None:
        rot = random.choice([0, 1, 2, 3])
    if rot in (1, 3):                     # rot90/rot270 swap the canvas dims
        hmax, wmax = max(5, max_w), max(5, max_h)
    else:
        hmax, wmax = max(5, max_h), max(5, max_w)

    cols = list(range(10))
    h = _unifint(diff_lb, diff_ub, (5, hmax))
    w = _unifint(diff_lb, diff_ub, (5, wmax))
    remcols = [c for c in cols if c != bgc]

    itv = list(range(w))
    na = random.randint(2, w - 2)
    alocs = sorted(random.sample(itv, na))
    blocs = sorted(set(itv) - set(alocs))
    if min(alocs) > min(blocs):
        alocs, blocs = blocs, alocs
    ass, bss = random.sample(remcols, 2)

    llocs = random.randint(0, h - 1)
    gi = [[bgc] * w for _ in range(h)]
    for j in alocs:
        gi[llocs][j] = ass
    for j in blocs:
        gi[llocs][j] = bss

    numl = _unifint(diff_lb, diff_ub, (1, max(1, (h - 1) // 2)))
    remlocs = [r for r in range(h) if r != llocs]
    for _ in range(numl):
        lloc = random.choice(remlocs)
        remlocs.remove(lloc)
        a, b = random.sample(remcols, 2)
        for j in alocs:
            gi[lloc][j] = a
        for j in blocs:
            gi[lloc][j] = b

    cutoff = min(blocs) + 1
    go = [row[:] for row in gi]
    for r in range(h):
        for c in range(cutoff, w):
            gi[r][c] = bgc
    for j in alocs:
        gi[llocs][j] = ass
    for j in blocs:
        gi[llocs][j] = bss

    gia = np.array(gi, dtype=int)
    goa = np.array(go, dtype=int)
    if rot:
        gia = np.rot90(gia, rot)
        goa = np.rot90(goa, rot)
    return {"input": gia.tolist(), "output": goa.tolist()}


# ---------------------------------------------------------------- 3. ops
def derive_operations(I, O):
    """
    Rule (read from I alone):
      * exactly one line of the canvas (a row -- a column when the scene is
        turned) carries no background at all: the KEY line.  Its two colours
        split the perpendicular positions into two GROUPS.
      * every position belonging to a group holds exactly the same strip of
        colours.  Only the strips at one end survived in the input; the rest of
        the canvas was wiped down to the key line.
      * so each group's surviving strip is REPLICATED onto every wiped position
        of that group: CopyI the strip, Paste it at each of them.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # orientation: when the lines run horizontally at least one row is pure
    # background; when they run vertically every row is crossed by them.
    horiz = any(len(set(I[r].tolist())) == 1 for r in range(hi))
    G = I if horiz else I.T
    H, W = G.shape

    uniform = [int(G[r, 0]) for r in range(H) if len(set(G[r].tolist())) == 1]
    if uniform:
        bgc = Counter(uniform).most_common(1)[0][0]
    else:
        bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    line_rows = [r for r in range(H) if bool((G[r] != bgc).any())]
    key = max(line_rows, key=lambda r: int((G[r] != bgc).sum()))
    others = [r for r in line_rows if r != key]

    # a position is still fully drawn iff every drawn line has a colour there
    full = [c for c in range(W) if all(G[r, c] != bgc for r in line_rows)]
    wiped = [c for c in range(W) if c not in full]

    def cell(r, c):
        return (r, c) if horiz else (c, r)

    ops, sels = [], []
    groups = sorted({int(G[key, c]) for c in range(W)},
                    key=lambda v: min([c for c in full if int(G[key, c]) == v] + [W]))

    for v in groups:
        srcs = [c for c in full if int(G[key, c]) == v]
        tgts = [c for c in wiped if int(G[key, c]) == v]
        if not tgts:
            continue
        if not srcs:                        # degenerate: nothing to replicate from
            for c in tgts:
                for r in others:
                    if int(O[cell(r, c)]) != int(I[cell(r, c)]):
                        ops.append(int(O[cell(r, c)]))
                        sels.append(sel_of([cell(r, c)]))
            continue
        src = srcs[0]
        strip = {r: int(G[r, src]) for r in others}
        # Paste never carries a cell holding colour 0, so it only helps while the
        # strip still has something else to give.
        if any(val != 0 for val in strip.values()):
            ops.append(28)                                        # CopyI the strip
            sels.append(sel_of([cell(r, src) for r in range(H)]))
            for c in tgts:
                ops.append(30)                                    # Paste at each
                sels.append(sel_of([cell(0, c)]))
        # the strip's colour-0 cells are the ones Paste could not carry
        zeros = [cell(r, c) for c in tgts for r in others if strip[r] == 0]
        if zeros:
            ops.append(0)
            sels.append(sel_of(zeros))

    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(O.shape[0]) for c in range(O.shape[1])]))
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
                        f"num_examples+1 ({num_examples + 1}) for task 82819916"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 82819916"
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
                                f"for task 82819916"
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
                    f"Failed to build a complete episode for task 82819916 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"82819916-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
