"""짧아진 것이 다른 풀이인가, 마스크에 밀어넣은 것인가.

op을 줄이는 가장 싼 길은 계산을 선택 마스크로 옮기는 것이다. 궤적은 짧아지고
거기서 배울 것은 없어진다. 그래서 두 가지를 따로 센다.

  임의 마스크   선택이 자기 bbox의 진부분집합이다. bbox를 통째로 고르면
                "이 영역"이라고 말할 수 있지만, 그 안에서 칸을 골라내면
                고르는 일 자체가 궤적 바깥에서 한 계산이다.
  어휘 변화     쓰는 op의 종류가 기존과 다른가. 같은 op을 덜 쓰는 것과
                다른 op으로 푸는 것은 다른 이야기다.
"""
import sys, collections, json
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
import numpy as np
import choose_maker as cm
from pathlib import Path

ROOT = Path("/hdd_data/yunho/solar-traj/maker")
ARMS = ["arc-agi-1", "golf1", "golf2"]
tasks = sorted(p.name for p in (ROOT / "golf1").iterdir()
               if p.is_dir() and p.name != "__pycache__")

def stats(derive, pairs, eps):
    arb = tot = 0
    names = collections.Counter()
    for i, (I, O) in enumerate(pairs):
        shown = eps[i % len(eps)][0] if eps else None
        g, ops, sels = cm.replay(derive, I, O, shown)
        if ops is None:
            continue
        for op, sel in zip(ops, sels):
            m = cm.solar_utils.to_sel_mask(sel, cm.MAX_GRID_DIM).astype(bool)
            if not m.any():
                continue
            rr, cc = np.where(m)
            box = (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)
            tot += 1
            if m.sum() < box:
                arb += 1
            names[cm.NAME.get(int(op))] += 1
    return (arb / tot if tot else None), names

out = []
for t in tasks:
    pairs = cm.draw(t, 10, [0])
    if not pairs:
        continue
    eps = cm.episodes_from_draw("/hdd_data/yunho/ARC_rearc_draw20", t) or []
    row = {"task": t}
    for a in ARMS:
        p = ROOT / a / t / "grid_maker.py"
        if not p.exists():
            continue
        d = cm.load_derive(p)
        if d is None:
            continue
        r, names = stats(d, pairs, eps)
        row[a] = {"arb": r, "ops": dict(names)}
    out.append(row)

json.dump(out, open("/hdd_data/yunho/cctmp/golfdiff.json", "w"))

def share(rows, a):
    v = [r[a]["arb"] for r in rows if a in r and r[a]["arb"] is not None]
    return sum(v) / len(v) if v else None

print(f"{'task':10s} {'배포본':>7s} {'golf1':>7s} {'golf2':>7s}   어휘변화")
for r in out:
    b = r.get("arc-agi-1", {}); g1 = r.get("golf1", {}); g2 = r.get("golf2", {})
    def f(x): return "  -  " if not x or x.get("arb") is None else f"{x['arb']:5.2f}"
    bn = set(b.get("ops", {}))
    ch = []
    for a, g in (("g1", g1), ("g2", g2)):
        gn = set(g.get("ops", {}))
        if gn and (gn - bn or bn - gn):
            ch.append(f"{a}:+{len(gn-bn)}/-{len(bn-gn)}")
    print(f"{r['task']:10s} {f(b):>7s} {f(g1):>7s} {f(g2):>7s}   {' '.join(ch)}")
print(f"\n평균 임의마스크  배포본 {share(out,'arc-agi-1'):.0%}  "
      f"golf1 {share(out,'golf1'):.0%}  golf2 {share(out,'golf2'):.0%}")
for a in ("golf1", "golf2"):
    ch = sum(1 for r in out if a in r and
             (set(r[a]["ops"]) - set(r.get("arc-agi-1", {}).get("ops", {}))
              or set(r.get("arc-agi-1", {}).get("ops", {})) - set(r[a]["ops"])))
    worse = sum(1 for r in out if a in r and r[a]["arb"] is not None
                and r.get("arc-agi-1", {}).get("arb") is not None
                and r[a]["arb"] > r["arc-agi-1"]["arb"] + 0.25)
    print(f"{a}: 어휘가 바뀐 과제 {ch}/{len(out)},  마스크가 0.25 넘게 나빠진 과제 {worse}")
