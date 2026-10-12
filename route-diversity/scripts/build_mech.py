"""분할·병합 변환을 실제 maker로 굳힌다. LLM은 들지 않는다.

mechroutes.py가 이미 과제별로 "이 변환이 기존 경로와 다른 새 경로를 낸다"를
재놨다. 분할 40개, 병합 117개(겹침 16). 둘을 한 arm에 합치면 과제당 하나만
남으므로 arm을 둘로 나눈다 -- 그래야 157개가 다 산다.

만드는 방식은 원본 maker 파일을 그대로 베끼고, derive_operations의 이름만
_base_derive_operations로 바꾼 뒤 변환을 씌운 새 derive_operations를 뒤에
붙이는 것이다. GridMaker가 부르는 이름은 호출 시점에 풀리므로 자동으로
변환본을 쓴다. 인자 수는 원본과 맞춘다 -- call_derive가 개수로 분기한다.
"""
import json, re, shutil, sys, inspect
from pathlib import Path

ROOT = Path("/hdd_data/yunho/solar-traj/maker")
BASE = ROOT / "arc-agi-1"
KINDS = {"분할": "mech_split", "병합": "mech_merge"}

HELPERS = '''

# ── 기계적 변환 ──────────────────────────────────────────────────────────
# 원본 경로를 받아 한 자리만 바꾼다. 입력 격자도 답 격자도 보지 않고
# (ops, sels)만 읽으므로, 원본이 유도 가능하면 이것도 유도 가능하다.
import numpy as _np

_MECH_MAX = (30, 30)
_MECH_MERGEABLE = {f"Color{_i}" for _i in range(10)} | {f"FloodFill{_i}" for _i in range(10)}
_MECH_NAME = %(name_map)r


def _mech_cells(sel):
    if isinstance(sel, dict) and "cells" in sel:
        return frozenset((int(r), int(c)) for r, c in sel["cells"])
    r, c, h, w = (int(v) for v in sel)        # h, w는 크기가 아니라 오프셋이다
    return frozenset((rr, cc) for rr in range(r, r + h + 1)
                     for cc in range(c, c + w + 1))


def _mech_blobs(cs):
    todo, out = set(cs), []
    while todo:
        seed = todo.pop(); comp = {seed}; stack = [seed]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in todo:
                        todo.discard(p); comp.add(p); stack.append(p)
        out.append(sorted(comp))
    return out


def _mech_apply(ops, sels, kind):
    n = len(ops)
    if kind == "분할":
        # 한 번에 칠하던 떨어진 덩어리들을 덩어리별로 나눠 칠한다.
        for i in range(n - 1):
            bl = _mech_blobs(_mech_cells(sels[i]))
            if len(bl) > 1:
                return (list(ops[:i]) + [ops[i]] * len(bl) + list(ops[i + 1:]),
                        list(sels[:i]) + [{"cells": [list(p) for p in b]} for b in bl]
                        + list(sels[i + 1:]))
    else:
        # 같은 색을 연달아 두 번 칠하면 한 번에 칠한다.
        for i in range(n - 2):
            if ops[i] == ops[i + 1] and _MECH_NAME.get(int(ops[i])) in _MECH_MERGEABLE:
                u = sorted(_mech_cells(sels[i]) | _mech_cells(sels[i + 1]))
                return (list(ops[:i]) + [ops[i]] + list(ops[i + 2:]),
                        list(sels[:i]) + [{"cells": [list(p) for p in u]}]
                        + list(sels[i + 2:]))
    return None


_MECH_KIND = %(kind)r


def derive_operations(%(args)s):
    ops, sels = _base_derive_operations(%(args)s)
    out = _mech_apply(ops, sels, _MECH_KIND)
    # 변환이 걸리지 않는 인스턴스에서는 원본 경로를 그대로 쓴다.
    return out if out is not None else (ops, sels)
'''


def main():
    sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
    import choose_maker as cm
    name_map = {int(k): v for k, v in cm.NAME.items()}

    rows = {r["task"]: r["new"] for r in json.load(
        open("/hdd_data/yunho/cctmp/mechroutes.json"))}
    made = {v: 0 for v in KINDS.values()}
    skipped = []
    for t, new in sorted(rows.items()):
        src_path = BASE / t / "grid_maker.py"
        if not src_path.is_file():
            continue
        src = src_path.read_text()
        m = re.search(r"^def derive_operations\(([^)]*)\)", src, re.M)
        if not m:
            skipped.append((t, "no def")); continue
        args = ", ".join(a.split("=")[0].split(":")[0].strip()
                         for a in m.group(1).split(",") if a.strip())
        body = re.sub(r"^def derive_operations\(", "def _base_derive_operations(",
                      src, count=1, flags=re.M)
        for kind, arm in KINDS.items():
            if not new.get(kind):
                continue
            out = ROOT / arm / t
            out.mkdir(parents=True, exist_ok=True)
            (out / "grid_maker.py").write_text(
                body + HELPERS % {"kind": kind, "args": args, "name_map": name_map})
            made[arm] += 1
    print("만듦:", made, " 합계", sum(made.values()))
    if skipped:
        print("건너뜀:", skipped[:10])


main()
