"""배포 에피소드 기준으로 전수 갈래의 경로 수를 센다.

cm.draw는 re-arc/generators.py에서 새로 뽑는데, maker는 자기 안에 든
generate() 복사본이 만든 인스턴스에 맞춰져 있다. 둘이 다르면 멀쩡한 maker가
못 푸는 것으로 세어진다. 실제로 학습에 쓰이는 것은 배포 에피소드이므로,
그걸 잣대로 삼는다.
"""
import sys, os, json, collections
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
sys.path.insert(0, "/hdd_data/yunho/cctmp")
import isolate
import numpy as np
import choose_maker as cm
from pathlib import Path

ROOT = Path("/hdd_data/yunho/solar-traj/maker")
OUT = Path("/hdd_data/yunho/cctmp/eval_dep.json")
CHUNK = int(os.environ.get("SOLAR_CHUNK", "25"))
NEP = 5                       # 배포 에피소드 앞 5개

BASE = "arc-agi-1"
FULL = ["arc-agi1-noref", "resample1",
        "cg_cgi", "cg_cristianoc", "cg_aggregate", "cg_oxjam",
        "cg_camjam", "cg_itertools", "cg_lv1dev"]

def sig(derive, eps):
    out = []
    for ex, (I, O) in eps:
        g, ops, sels = cm.replay(derive, I, O, ex)
        if g is None or g.shape != O.shape or not bool((g == O).all()):
            return None
        steps = []
        for op, sel in zip(ops, sels):
            m = cm.solar_utils.to_sel_mask(sel, cm.MAX_GRID_DIM).astype(bool)
            rr, cc = np.where(m)
            steps.append((cm.NAME.get(int(op)), frozenset(zip(rr.tolist(), cc.tolist()))))
        out.append(tuple(steps))
    return tuple(out)

tasks = sorted(p.name for p in (ROOT / BASE).iterdir()
               if p.is_dir() and p.name != "__pycache__")
rows = json.loads(OUT.read_text()) if OUT.is_file() else []
done = {r["task"] for r in rows}
todo = [t for t in tasks if t not in done][:CHUNK]
if not todo:
    print("done"); raise SystemExit

def measure(t):
    eps = cm.episodes_from_draw("/hdd_data/yunho/ARC_rearc_draw20", t)
    if not eps:
        return {"task": t, "skip": True}
    eps = eps[:NEP]
    got = {}
    for a in [BASE] + FULL:
        p = ROOT / a / t / "grid_maker.py"
        if not p.is_file(): continue
        d = cm.load_derive(p)
        if d is None: continue
        s = sig(d, eps)
        if s: got[a] = s
    cg = [a for a in FULL if a.startswith("cg_")]
    sub = lambda arms: len({got[a] for a in arms if a in got})
    return {"task": t, "total": len(set(got.values())), "solved": sorted(got),
            "base_only": sub([BASE]),
            "base_resample": sub([BASE, "resample1", "arc-agi1-noref"]),
            "base_cg": sub([BASE] + cg)}

for t in todo:
    got, why = isolate.run(measure, t, gib=3.0, seconds=900)
    rows.append(got if got else {"task": t, "failed": why})
    tmp = OUT.with_suffix(".part"); tmp.write_text(json.dumps(rows)); tmp.replace(OUT)
print(f"{len(rows)}/400")
