"""
ARC Task: 746b3537 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of

VARIANTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    """The rule is colour-blind (it is about runs of identical lines), so the stripe
    colours stay free; one palette per episode keeps the instances looking like one
    puzzle, and the two orientations (row stripes / column stripes) are pre-planned
    so both are demonstrated before the test asks for one."""
    palette = random.sample(range(10), random.randint(5, 10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"palette": palette, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, palette=None, mirrored=None) -> dict:
    pool = [int(c) for c in palette] if palette and len(palette) >= 2 else list(range(10))
    if mirrored is None:
        mirrored = random.choice([True, False])

    def ui(lo, up):
        if up < lo:
            up = lo
        try:
            return unifint(diff_lb, diff_ub, (lo, up))    # noqa: F821  (re-arc utils)
        except NameError:
            return random.randint(lo, up)

    # canonical layout: stripes stacked as rows, transposed at the end if mirrored
    A = max(3, min(30, max_w if mirrored else max_h))     # extent along the stripe axis
    B = max(1, min(30, max_h if mirrored else max_w))     # length of one stripe

    h = max(2, min(ui(2, min(15, A - 1)), A - 1))
    w = max(1, min(ui(1, B), B))

    cols, lastc = [], -1
    for _ in range(h):
        c = random.choice([x for x in pool if x != lastc])
        cols.append(c)
        lastc = c

    go = tuple((c,) for c in cols)
    gi = tuple(tuple([c] * w) for c in cols)

    numinserts = max(1, min(ui(1, A - h), A - h))
    for _ in range(numinserts):
        loc = random.randint(0, len(gi) - 1)
        gi = gi[:loc + 1] + gi[loc:]

    if mirrored:
        gi = tuple(zip(*gi))
        go = tuple(zip(*go))

    return {"input":  [[int(v) for v in r] for r in gi],
            "output": [[int(v) for v in r] for r in go]}


def derive_operations(I, O):
    """The grid is a stack of stripes, some of them repeated; the answer keeps one
    cell per run of identical stripes.  So: slide stripes over their own copies to
    close the gaps, then crop to the single line that is left.

    Which route that is, is read off the input, so two instances are solved
    differently while one instance is always solved the same way:
      * the colours of the two end stripes say which end the stack collapses
        towards (the survivors gather at the top/left or at the bottom/right),
      * how much duplication there is says whether to shove the whole remaining
        block over each duplicated run -- starting at the near end or at the far
        end -- or to drag the stripes into place one at a time.
    Routes are tried in that order and the first sound one is used: sound means
    every step visibly changes the grid and no step could be dropped.

    A stripe of colour 0 is nothing to ARCLE (Move carries only non-zero cells),
    so wherever such a stripe lands it is put down explicitly with one Color0 at
    the end; an instance without a 0 stripe emits no such op.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # ---- read the stripe structure out of I ---------------------------------
    row_mode = len(set(I[0].tolist())) == 1           # stripes run along the rows
    seq = [int(v) for v in (I[:, 0] if row_mode else I[0, :])]
    N = len(seq)                                      # extent along the stripe axis
    cross = wi if row_mode else hi                    # length of one stripe

    runs = []                                         # [colour, number of copies]
    for v in seq:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    n = len(runs)

    def framed(to_far):
        return (list(reversed(runs)), list(reversed(seq))) if to_far else (list(runs), list(seq))

    def run_prog(prog, line0):
        """Replay a route on the 1-D stripe model with ARCLE's own object rules:
        a Move with cells grabs (background = grid with those cells cleared), a
        Move with no cells keeps sliding that same grab, colour 0 never pastes."""
        line = list(line0)
        held = None
        states = [(len(line), tuple(line))]
        for op in prog:
            if op[0] == "mv":
                if op[1] is not None:
                    a, b = op[1], op[2]
                    held = [[line[i] if a <= i <= b else 0 for i in range(len(line))],
                            [0 if a <= i <= b else line[i] for i in range(len(line))], 0]
                if held is not None:
                    held[2] -= 1
                    obj, bg, off = held
                    line = list(bg)
                    for i, v in enumerate(obj):
                        if v and 0 <= i + off < len(line):
                            line[i + off] = v
            elif op[0] == "crop":
                line = line[op[1]:op[2] + 1]
                held = None
            else:                                     # colour
                line = list(line)
                for i in op[2]:
                    line[i] = op[1]
                held = None
            states.append((len(line), tuple(line)))
        return line, states

    def plan(to_far, style, truncate=True):
        """Build the route in frame coordinates (frame index 0 = the end the stack
        collapses towards).  Returns the 1-D program, or None if it cannot be made
        sound."""
        f_runs, line = framed(to_far)
        starts, acc = [], 0
        for _c, L in f_runs:
            starts.append(acc)
            acc += L
        prog = []

        def slide(cur, a, b, want):
            """Grab frame cells a..b and slide them `want` cells towards 0. Emit
            only steps that visibly change the grid; usable only if those reach
            the same picture as the whole slide."""
            goal, _ = run_prog([("mv", a, b)] + [("mv", None, None)] * (want - 1), cur)
            if not truncate:
                return want, goal
            got, good = cur, 0
            for k in range(1, want + 1):
                step, _ = run_prog([("mv", a, b)] + [("mv", None, None)] * (k - 1), cur)
                if step == got:
                    break                             # this step would change nothing
                got, good = step, k
            if got != goal:
                return None                           # the visible part misses the goal
            return good, got

        if style == 3:
            pass                                      # nothing to slide: see below
        elif style == 2:
            # drag the stripes into place one at a time, nearest gap first
            for j in range(n):
                p = starts[j]
                if p == j:
                    continue
                res = slide(line, p, p, p - j)
                if res is None:
                    return None
                steps, line = res
                if steps:
                    prog.append(("mv", p, p))
                    prog += [("mv", None, None)] * (steps - 1)
        else:
            # shove everything past a duplicated run over that run's copies
            pos, live = list(starts), N
            for i in (range(n) if style == 0 else range(n - 1, -1, -1)):
                e = f_runs[i][1] - 1
                if e == 0:
                    continue
                s = pos[i] + f_runs[i][1]
                if s >= live:                         # trailing copies: the crop drops them
                    live = pos[i] + 1
                    continue
                res = slide(line, s, live - 1, e)
                if res is None:
                    return None
                steps, line = res
                if steps:
                    prog.append(("mv", s, live - 1))
                    prog += [("mv", None, None)] * (steps - 1)
                for j in range(n):
                    if pos[j] >= s:
                        pos[j] -= e
                live -= e

        prog.append(("crop", 0, n - 1))
        fixes = []
        for k in range(n):
            if line[k] != f_runs[k][0]:
                if f_runs[k][0] != 0:                 # only 0 stripes may need laying down
                    return None
                fixes.append(k)
        if fixes:
            prog.append(("color", 0, tuple(fixes)))
        return prog

    # ---- the routes a solver could take, in the order the input suggests ----
    to_far_pref = (runs[0][0] + runs[-1][0]) % 2 == 1
    style_pref = (sum(L for _, L in runs if L > 1) + n) % 3
    order = [(to_far_pref, (style_pref + k) % 3) for k in range(3)]
    order += [(not to_far_pref, (style_pref + k) % 3) for k in range(3)]
    # last: the stack that needs no sliding at all, because the stripes the crop
    # keeps already carry the right colours once the 0 stripes are laid down
    order += [(to_far_pref, 3), (not to_far_pref, 3)]

    def sound(to_far, prog, nodrop):
        if prog is None:
            return False
        f_runs, line0 = framed(to_far)
        want = [c for c, _L in f_runs]
        final, states = run_prog(prog, line0)
        if final != want or len(set(states)) != len(states):
            return False                              # wrong, or a step changed nothing
        if nodrop and any(run_prog(prog[:k] + prog[k + 1:], line0)[0] == want
                          for k in range(len(prog))):
            return False                              # some op could be dropped
        return True

    chosen = None
    for nodrop in (True, False):
        for to_far, style in order:
            prog = plan(to_far, style)
            if sound(to_far, prog, nodrop):
                chosen = (to_far, prog)
                break
        if chosen:
            break
    if chosen is None:                                # never seen: plain full slides
        chosen = (to_far_pref, plan(to_far_pref, 0, truncate=False))
    to_far, prog = chosen

    # ---- emit -----------------------------------------------------------------
    ops, sels = [], []
    mv = (21 if row_mode else 22) if to_far else (20 if row_mode else 23)
    for op in prog:
        if op[0] == "mv":
            ops.append(mv)
            if op[1] is None:
                sels.append(sel_of([]))               # keep sliding the same grab
            else:
                a, b = op[1], op[2]
                ra, rb = (N - 1 - b, N - 1 - a) if to_far else (a, b)
                # exactly the full rectangle of stripes that slides
                sels.append([ra, 0, rb - ra, cross - 1] if row_mode
                            else [0, ra, cross - 1, rb - ra])
        elif op[0] == "crop":
            fa = (N - n) if to_far else 0             # keep one cell per distinct stripe
            ops.append(33)
            sels.append([fa, 0, n - 1, 0] if row_mode else [0, fa, 0, n - 1])
        else:
            cells = []
            for k in op[2]:
                m = (n - 1 - k) if to_far else k
                cells.append((m, 0) if row_mode else (0, m))
            ops.append(0)                             # 0 stripes cannot ride a Move
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, n - 1, 0] if row_mode else [0, 0, 0, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 746b3537"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 746b3537"
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
                                f"for task 746b3537"
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
                    f"Failed to build a complete episode for task 746b3537 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"746b3537-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
