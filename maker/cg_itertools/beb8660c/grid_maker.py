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

# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
from collections import Counter
from maker.sel_helpers import sel_of

ROT_VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROT_VARIANTS):
        examples = [dict(v) for v in ROT_VARIANTS]
        examples += [dict(random.choice(ROT_VARIANTS)) for _ in range(n_ex - len(ROT_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(ROT_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, rot=None, **kwargs) -> dict:
    cols = remove(8, interval(0, 10, 1))
    if bgc is None:
        bgc = choice(cols)
    if rot is None:
        rot = randint(0, 3)
    mx = min(max_h, max_w)
    w = unifint(diff_lb, diff_ub, (3, mx))
    h = unifint(diff_lb, diff_ub, (w, max(w, mx)))
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    k = min(8, w - 1)
    k = unifint(diff_lb, diff_ub, (1, k))
    co = sample(remcols, k)
    wds = sorted(sample(interval(1, w, 1), k))
    for j, (c, l) in enumerate(zip(co, wds)):
        j = h - k - 1 + j
        gi = fill(gi, c, connect((j, 0), (j, l - 1)))
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
    rotf = (identity, rot90, rot180, rot270)[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background: the canvas colour shared by every input of the episode
    cnt = Counter()
    for X in [I] + [np.asarray(a, dtype=int) for a, _ in (examples or [])]:
        cnt.update(X.flatten().tolist())
    cnt.pop(8, None)
    bgc = cnt.most_common(1)[0][0]

    # the 8-line is a full edge row/column -> anchor edge
    if np.all(I[0, :] == 8) and wi > 1:
        edge = 'top'
    elif np.all(I[hi - 1, :] == 8) and wi > 1:
        edge = 'bottom'
    elif np.all(I[:, 0] == 8):
        edge = 'left'
    else:
        edge = 'right'

    # normalized frame: lane = coordinate across the lines, pos = along them
    if edge in ('top', 'bottom'):
        NL, P = hi, wi
        to_rc = lambda lane, pos: (lane, pos)
        lane_op = {1: 21, -1: 20}
        pos_op = {1: 22, -1: 23}
    else:
        NL, P = wi, hi
        to_rc = lambda lane, pos: (pos, lane)
        lane_op = {1: 22, -1: 23}
        pos_op = {1: 21, -1: 20}
    if edge in ('top', 'left'):
        slot_lane = lambda i: 1 + i
        anchor_lane = 0
    else:
        slot_lane = lambda i: NL - 2 - i
        anchor_lane = NL - 1
    align_low = edge in ('top', 'right')   # corner CCW-adjacent to the 8-edge

    # every other colour is one straight line segment (one object)
    lines = {}
    for col in sorted(set(I.flatten().tolist()) - {bgc, 8}):
        rr, cc = np.nonzero(I == col)
        if edge in ('top', 'bottom'):
            lane, ps = int(rr[0]), cc
        else:
            lane, ps = int(cc[0]), rr
        lines[col] = {"lane": lane, "pos": int(ps.min()), "L": len(ps)}
    # sort lines by length: longest goes next to the 8-line
    order = sorted(lines, key=lambda c: (-lines[c]["L"], c))
    target_lane = {c: slot_lane(i) for i, c in enumerate(order)}
    aligned = lambda L: 0 if align_low else P - L
    opposite = lambda L: P - L if align_low else 0

    def cells(c, lane=None, pos=None):
        d = lines[c]
        lane = d["lane"] if lane is None else lane
        pos = d["pos"] if pos is None else pos
        return [to_rc(lane, p) for p in range(pos, pos + d["L"])]

    # a 0-coloured line (possible when bgc != 0) cannot be carried by Move:
    # ARCLE pastes only nonzero cells. It is painted once at its slot at the end.
    movable = [c for c in order if c != 0]

    def plan(park_shortest):
        if park_shortest and not movable:
            return None
        st = {c: dict(lines[c]) for c in movable}
        steps = []

        def occ(exclude):
            s = set()
            for c in st:
                if c != exclude:
                    s.update((st[c]["lane"], p) for p in range(st[c]["pos"], st[c]["pos"] + st[c]["L"]))
            return s

        def fits(c, lane):
            o = occ(c)
            return all((lane, p) not in o for p in range(st[c]["pos"], st[c]["pos"] + st[c]["L"]))

        def mv(c, axis, new):
            cur = st[c][axis]
            if new != cur:
                steps.append((c, axis, new))
                st[c][axis] = new

        S = movable[-1] if movable else None
        # stage 1: slide every line along its own lane to the anchored side
        # (sort within the row); the shortest may wait at the far end when
        # every lane is occupied, so it never blocks another line.
        for c in sorted(st, key=lambda c: st[c]["lane"]):
            tgt = opposite(st[c]["L"]) if (park_shortest and c == S) else aligned(st[c]["L"])
            mv(c, "pos", tgt)
        # stage 2: slide lines across lanes into their length-ordered slots
        pending = [c for c in movable if not (park_shortest and c == S)]
        guard = 0
        while True:
            pending = [c for c in pending if st[c]["lane"] != target_lane[c]]
            if not pending:
                break
            guard += 1
            if guard > 200:
                return None
            ready = [c for c in pending if fits(c, target_lane[c])]
            if ready:
                mv(ready[0], "lane", target_lane[ready[0]])
                continue
            # break a cycle: park one line in a free lane
            tl = {target_lane[c] for c in pending}
            done = False
            for pref in (True, False):
                for c in pending:
                    for lane in range(NL):
                        if lane == anchor_lane or lane == st[c]["lane"]:
                            continue
                        if pref and lane in tl:
                            continue
                        if fits(c, lane):
                            mv(c, "lane", lane)
                            done = True
                            break
                    if done:
                        break
                if done:
                    break
            if not done:
                return None
        if park_shortest:
            if not fits(S, target_lane[S]):
                return None
            mv(S, "lane", target_lane[S])
            mv(S, "pos", aligned(st[S]["L"]))
        return steps

    steps = plan(False)
    if steps is None:
        steps = plan(True)

    ops, sels = [], []
    holes = set()          # cells left reading 0 by a grab (vacated footprints)
    for c, axis, new in steps:
        d = lines[c]
        src = cells(c)
        if axis == "lane":
            dst = cells(c, lane=new)
            delta = 1 if new > d["lane"] else -1
            n = abs(new - d["lane"])
            op = lane_op[delta]
        else:
            dst = cells(c, pos=new)
            delta = 1 if new > d["pos"] else -1
            n = abs(new - d["pos"])
            op = pos_op[delta]
        ops.append(op); sels.append(sel_of(src))      # grab the line
        for _ in range(n - 1):
            ops.append(op); sels.append(sel_of([]))   # keep sliding it
        holes |= set(src)
        holes -= set(dst)
        d[axis] = new
    if 0 in lines:
        src0 = cells(0)
        lines[0]["lane"] = target_lane[0]
        lines[0]["pos"] = aligned(lines[0]["L"])
        dst0 = cells(0)
        # draw the 0-line in its slot (its cells that do not already read 0)
        paint = [rc for rc in dst0 if rc not in holes and rc not in set(src0)]
        if paint:
            ops.append(0); sels.append(sel_of(paint))
        holes |= set(src0)
    if bgc != 0:
        covered = set()
        for c in lines:
            covered |= set(cells(c))
        rest = sorted(holes - covered)
        if rest:
            ops.append(bgc); sels.append(sel_of(rest))   # restore vacated footprints

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
