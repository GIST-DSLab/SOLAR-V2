"""배포 maker + code-golf 참조 7종. 과제당 경로 수 히스토그램.

참조 없이 재생성(arc-agi1-noref)과 재표집(resample1)은 뺀다. 전자는 배포하지
않는 로컬 실험이고, 둘 다 이 발표의 주제인 "참조를 넣으면 경로가 갈리는가"와
다른 축이다. 남는 갈래는 8개이므로 과제당 경로는 최대 8개다.
"""
import json, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

for f in ("/home/ubuntu/.local/share/fonts/NotoSansKR-Regular.otf",
          "/home/ubuntu/.local/share/fonts/NotoSansKR-Bold.otf"):
    try: fm.fontManager.addfont(f)
    except Exception: pass
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

rows = [r for r in json.load(open("/hdd_data/yunho/cctmp/eval_dep.json")) if "total" in r]
n = len(rows)
base = [r["base_only"] for r in rows]
cg   = [r["base_cg"] for r in rows]

mx = max(cg)
centers = np.arange(1, mx + 1)
fig, ax = plt.subplots(figsize=(10.5, 5.2))
w = 0.38
for k, (lab, vals, col) in enumerate(
        [("배포 maker 1개 (기준)", base, "#BDBDBD"),
         ("기준 + code-golf 참조 7종", cg, "#1A5FB4")]):
    c = collections.Counter(vals)
    h = [c.get(int(x), 0) for x in centers]
    ax.bar(centers + (k - 0.5)*w, h, width=w, color=col,
           label=f"{lab}   평균 {np.mean(vals):.2f}", edgecolor="white", linewidth=0.6)
    for x, y in zip(centers, h):
        if y:
            ax.text(x + (k - 0.5)*w, y + 3, str(y), ha="center", fontsize=7.5, color="#555")

ax.set_xticks(centers)
ax.set_xlabel("과제 하나가 가진 서로 다른 경로 수", fontsize=11)
ax.set_ylabel("과제 수", fontsize=11)
ax.set_title("배포 maker에 code-golf 참조 7종을 더했을 때  (400개 과제 전부 대상, 갈래 8개)",
             fontsize=12.5, fontweight="bold", pad=12)
ax.legend(fontsize=10.5, frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#DDD", linewidth=0.7); ax.set_axisbelow(True)
ax.text(0.99, 0.80,
        f"경로 2개 이상 {sum(1 for v in cg if v>=2)}/{n}    총 경로 {sum(cg)}개",
        transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#444")
fig.tight_layout()
fig.savefig("/hdd_data/yunho/cctmp/hist_cg.png", dpi=170, facecolor="white")

print("기준 ", collections.Counter(base), "평균", round(np.mean(base),2), "총", sum(base))
print("+cg  ", dict(sorted(collections.Counter(cg).items())), "평균", round(np.mean(cg),2), "총", sum(cg))
print("2개 이상", sum(1 for v in cg if v>=2), "/", n, " 1개뿐", sum(1 for v in cg if v==1))
