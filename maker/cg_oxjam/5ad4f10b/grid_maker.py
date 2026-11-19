"""
ARC Task: 5ad4f10b (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc, noisec, objc = random.sample(list(range(10)), 3)
    return {"bgc": bgc, "noisec": noisec, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, objc=None) -> dict:
    if bgc is None or noisec is None or objc is None:
        bgc, noisec, objc = sample(interval(0, 10, 1), 3)
    nbh = {(0, 0), (1, 0), (0, 1), (1, 1)}
    nbhs = apply(lbind(shift, nbh), {(0, 0), (-1, 0), (0, -1), (-1, -1)})
    oh = unifint(diff_lb, diff_ub, (2, max(2, min(6, (max_h - 2) // 2))))
    ow = unifint(diff_lb, diff_ub, (2, max(2, min(6, (max_w - 2) // 2))))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(1, ncells), oh * ow - 1)
    obj = set(sample(totuple(bounds), ncells))
    while len(sfilter(obj, lambda ij: sum([len(obj & shift(nbh, ij)) < 4 for nbh in nbhs]) > 0)) == 0:
        ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
        ncells = choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow)
        obj = set(sample(totuple(bounds), ncells))
    obj = normalize(obj)
    oh, ow = shape(obj)
    if oh == ow and len(obj) == oh * ow:
        # a solid square has no recoverable scale; draw again
        return generate(diff_lb, diff_ub, max_h, max_w, bgc=bgc, noisec=noisec, objc=objc)
    go = canvas(bgc, (oh, ow))
    go = fill(go, noisec, obj)
    fmax = max(2, min((max_h - 2) // oh, (max_w - 2) // ow))
    fac = unifint(diff_lb, diff_ub, (2, fmax))
    gobj = asobject(upscale(replace(go, noisec, objc), fac))
    oh, ow = shape(gobj)
    h = unifint(diff_lb, diff_ub, (oh + 2, max(oh + 2, max_h)))
    w = unifint(diff_lb, diff_ub, (ow + 2, max(ow + 2, max_w)))
    loci = randint(1, h - oh - 1)
    locj = randint(1, w - ow - 1)
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shift(gobj, (loci, locj)))
    cands = ofcolor(gi, bgc)
    namt = unifint(diff_lb, diff_ub, (2, max(2, len(cands) // 4)))
    noise = sample(totuple(cands), namt)
    gi = fill(gi, noisec, noise)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (the code-golf reading, which differs from the maker's):
    # The big object made of blocks is a scaled-up picture. Within its box,
    # keep one row per block (row r <- row r*fac), then one column per block
    # (col c <- col c*fac). Crop away the duplicate rows and columns, then
    # swap the palette: stray noise -> background, object pieces -> noise colour.
    # The maker instead repaints a shrunk copy in place, piece by piece. Here the
    # downscale is done by copying the sampled lines, not by redrawing the pattern.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    border = I[0, :].tolist() + I[-1, :].tolist() + I[:, 0].tolist() + I[:, -1].tolist()
    bgc = Counter(border).most_common(1)[0][0]
    others = [int(c) for c in np.unique(I).tolist() if int(c) != bgc]

    def block_score(c):
        cs = set(map(tuple, np.argwhere(I == c).tolist()))
        iso = sum(1 for (r, cc) in cs
                  if not any((r + dr, cc + dc) in cs
                             for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))))
        return (1 if iso == 0 else 0, len(cs))

    objc = max(others, key=block_score)
    rest = [c for c in others if c != objc]
    if rest:
        noisec = rest[0]
    else:
        noisec = None
        for (ei, eo) in (examples or []):
            ev = [int(v) for v in np.unique(np.asarray(eo)).tolist()
                  if int(v) != Counter(np.asarray(ei)[0, :].tolist()).most_common(1)[0][0]]
            if ev:
                noisec = ev[0]
                break
        if noisec is None:
            noisec = objc

    cells = np.argwhere(I == objc)
    r0, c0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    r1, c1 = int(cells[:, 0].max()), int(cells[:, 1].max())
    H, W = r1 - r0 + 1, c1 - c0 + 1
    M = (I[r0:r1 + 1, c0:c1 + 1] == objc)
    fac = 1
    for k in range(min(H, W, max(H, W) - 1), 0, -1):
        if H % k or W % k:
            continue
        small = M.reshape(H // k, k, W // k, k).any(axis=(1, 3))
        if np.array_equal(small.repeat(k, 0).repeat(k, 1), M):
            fac = k
            break
    ph, pw = H // fac, W // fac

    def to_cells(sel):
        if isinstance(sel, dict):
            return [tuple(p) for p in sel["cells"]]
        r, c, h, w = sel
        return [(r + i, c + j) for i in range(h + 1) for j in range(w + 1)]

    def simulate(o_list, s_list):
        # tiny replay of the op subset used below (Color, FloodFill, CopyO, Paste, Crop)
        g = I.copy()
        cl = None
        for op, sel in zip(o_list, s_list):
            if op <= 9:
                for p in to_cells(sel):
                    if p[0] < g.shape[0] and p[1] < g.shape[1]:
                        g[p] = op
            elif op <= 19:
                r, c = sel[0], sel[1]
                if r >= g.shape[0] or c >= g.shape[1]:
                    continue
                col = g[r, c]
                st, seen_ = [(r, c)], {(r, c)}
                while st:
                    y, x = st.pop()
                    g[y, x] = op - 10
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        q = (y + dy, x + dx)
                        if (0 <= q[0] < g.shape[0] and 0 <= q[1] < g.shape[1]
                                and q not in seen_ and g[q] == col):
                            seen_.add(q)
                            st.append(q)
            elif op == 29:
                r, c, h, w = sel
                cl = g[r:r + h + 1, c:c + w + 1].copy()
            elif op == 30:
                if cl is None:
                    continue
                r, c = sel[0], sel[1]
                for i in range(cl.shape[0]):
                    for j in range(cl.shape[1]):
                        if cl[i, j] != 0 and r + i < g.shape[0] and c + j < g.shape[1]:
                            g[r + i, c + j] = cl[i, j]
            elif op == 33:
                r, c, h, w = sel
                ng = np.zeros((h + 1, w + 1), dtype=int)
                sub = g[r:r + h + 1, c:c + w + 1]
                ng[:sub.shape[0], :sub.shape[1]] = sub
                g = ng
        return g

    def build():
        ops, sels = [], []
        G = I.copy()
        clip = [None]   # values currently on the clipboard
        clip_shape = [None]
        # 1. stray noise inside the object's box is not part of the picture: erase it
        strays = [(r0 + y, c0 + x) for y in range(H) for x in range(W)
                  if G[r0 + y, c0 + x] == noisec]
        if strays and noisec != bgc:
            ops.append(bgc); sels.append(sel_of(strays))
            for p in strays:
                G[p] = bgc

        def canon(v):
            # stray noise is headed for the background, so it is not content
            return bgc if v == noisec else v

        def copy_line(dst_cells, src_cells, src_rect, dst_tl, relevant):
            # copy one sampled line onto its destination line, only when a kept cell changes
            diff = [i for i in relevant if canon(G[src_cells[i]]) != canon(G[dst_cells[i]])]
            if not diff:
                return
            if any(G[src_cells[i]] == 0 for i in diff):
                # Paste is transparent to 0: clear the destination line first
                ops.append(0); sels.append(sel_of(dst_cells))
                for d in dst_cells:
                    G[d] = 0
            vals = [int(G[s_]) for s_ in src_cells]
            if any(vals[i] != 0 and canon(vals[i]) != canon(G[dst_cells[i]]) for i in relevant):
                old = clip[0]

                def landed(cv, i):
                    return cv[i] if cv[i] != 0 else G[dst_cells[i]]
                same = (old is not None and clip_shape[0] == tuple(src_rect[2:])
                        and len(old) == len(vals)
                        and all(canon(landed(old, i)) == canon(landed(vals, i))
                                for i in relevant))
                if not same:
                    ops.append(29); sels.append(src_rect)      # CopyO the sampled line
                    clip[0] = vals
                    clip_shape[0] = tuple(src_rect[2:])
                ops.append(30); sels.append([dst_tl[0], dst_tl[1], 0, 0])
                for d, v in zip(dst_cells, clip[0]):
                    if v != 0:
                        G[d] = v

        def rows_pass(width, rel):
            # keep one row per block: row r <- row r*fac
            for r in range(1, ph):
                sr = r * fac
                copy_line([(r0 + r, c0 + x) for x in range(width)],
                          [(r0 + sr, c0 + x) for x in range(width)],
                          [r0 + sr, c0, 0, width - 1], (r0 + r, c0), rel)

        def cols_pass(height, rel):
            # keep one column per block: col c <- col c*fac
            for c in range(1, pw):
                sc = c * fac
                copy_line([(r0 + y, c0 + c) for y in range(height)],
                          [(r0 + y, c0 + sc) for y in range(height)],
                          [r0, c0 + sc, height - 1, 0], (r0, c0 + c), rel)

        # 2. rows first (across the box width, judged at the sampled columns),
        #    then columns within the kept rows
        rows_pass(W, [c * fac for c in range(pw)])
        cols_pass(ph, list(range(ph)))
        # 3. crop to the block samples
        ops.append(33); sels.append([r0, c0, ph - 1, pw - 1])
        G = G[r0:r0 + ph, c0:c0 + pw].copy()
        # 4. stray noise that still made it in becomes background
        nz = [tuple(p) for p in np.argwhere(G == noisec).tolist()]
        if nz and noisec != bgc:
            ops.append(bgc); sels.append(sel_of(nz))
            for p in nz:
                G[p] = bgc
        # 5. each connected piece of the shrunk object takes the noise colour
        if objc != noisec:
            seen = np.zeros(G.shape, dtype=bool)
            for (r, c) in sorted(map(tuple, np.argwhere(G == objc).tolist())):
                if seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < ph and 0 <= nx < pw and not seen[ny, nx]
                                and G[ny, nx] == objc):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                ops.append(10 + noisec); sels.append([r, c, 0, 0])
        return ops, sels

    def prune(o_list, s_list):
        # self-check from I alone: drop any op whose removal leaves the result unchanged
        o_list, s_list = list(o_list), list(s_list)
        ref = simulate(o_list, s_list)
        changed = True
        while changed:
            changed = False
            for k in range(len(o_list)):
                g = simulate(o_list[:k] + o_list[k + 1:], s_list[:k] + s_list[k + 1:])
                if g.shape == ref.shape and np.array_equal(g, ref):
                    del o_list[k]
                    del s_list[k]
                    changed = True
                    break
        return o_list, s_list

    ops, sels = prune(*build())
    ops.append(34); sels.append([0, 0, ph - 1, pw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 5ad4f10b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5ad4f10b"
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
                                f"for task 5ad4f10b"
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
                    f"Failed to build a complete episode for task 5ad4f10b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5ad4f10b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
