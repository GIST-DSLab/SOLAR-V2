"""
ARC Trajectory Visualizer — Streamlit app
Usage: streamlit run viz_trajectories.py --server.port 8501
"""
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path
import streamlit as st

# ── ARC color palette ──────────────────────────────────────────────────────────
ARC_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 grey
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 sky
    "#870C25",  # 9 maroon
    "#F0F0F0",  # 10 padding (out-of-bounds)
]
ARC_CMAP = mcolors.ListedColormap(ARC_COLORS)
ARC_NORM = mcolors.BoundaryNorm(boundaries=list(range(12)), ncolors=11)

OP_NAMES = [
    "Color0","Color1","Color2","Color3","Color4","Color5","Color6","Color7","Color8","Color9",
    "FloodFill0","FloodFill1","FloodFill2","FloodFill3","FloodFill4",
    "FloodFill5","FloodFill6","FloodFill7","FloodFill8","FloodFill9",
    "MoveU","MoveD","MoveR","MoveL","Rotate90(CCW)","Rotate270(CW)","FlipH","FlipV",
    "CopyI","CopyO","Paste","CopyInput","ResetGrid","ResizeGrid","Submit","None",
]

# op id -> (category, label color), for a quick visual read of what kind of
# action each step is without having to read the full op name every time.
_OP_CATEGORY = {}
for _i in range(10):  _OP_CATEGORY[_i] = "Color"
for _i in range(10, 20): _OP_CATEGORY[_i] = "FloodFill"
for _i in range(20, 24): _OP_CATEGORY[_i] = "Move"
for _i in (24, 25): _OP_CATEGORY[_i] = "Rotate"
for _i in (26, 27): _OP_CATEGORY[_i] = "Flip"
for _i in (28, 29, 30, 31): _OP_CATEGORY[_i] = "Copy/Paste"
_OP_CATEGORY[32] = "ResetGrid"
_OP_CATEGORY[33] = "ResizeGrid"
_OP_CATEGORY[34] = "Submit"

CATEGORY_COLOR = {
    "Color": "#7FDBFF",
    "FloodFill": "#2ECC40",
    "Move": "#FFDC00",
    "Rotate": "#B10DC9",
    "Flip": "#F012BE",
    "Copy/Paste": "#FF851B",
    "ResetGrid": "#FF4136",
    "ResizeGrid": "#39CCCC",
    "Submit": "#FFFFFF",
}

_default_root = Path(__file__).resolve().parents[1] / "ARC_Single_llm_v5_obj" / "whole"
DATA_ROOT = Path(os.environ.get("VIZ_DATA_ROOT", _default_root))

# Other trajectory sets worth flipping between without restarting the app.
# Anything the sidebar's free-text box accepts also works; these are just shortcuts.
KNOWN_ROOTS = [
    DATA_ROOT,
    # arc-1d: 1D-ARC makers — one maker per category (18), rolled out. Grids are 1×N;
    # use the Task filter to step through each category's solution trajectory.
    Path("/hdd_data/yunho/ARC_1d/whole"),
    # arc-fail: ONLY the samples whose rolled-out output is wrong (arc-best makers,
    # seed 0, 10 samples/task). Each trajectory here ends on an INCORRECT grid — use it
    # to see where the maker diverges. Empty for tasks that never fail.
    Path("/hdd_data/yunho/ARC_fail/whole"),
    # arc-best: the FINAL best-of winners. One trajectory per task, chosen by the
    # comparative judge across v9/v10/v11 (bestof_final.json). v9 204 / v11 126 / v10 69.
    # Use the "winner: vN" filters to inspect which version won each task.
    Path("/hdd_data/yunho/ARC_best/whole"),
    # v11 snapshot: the 102 trajectory-defect tasks regenerated after the prompt fixes
    # (op count is not a cost / translations must use Move / repair Move's 0-trail).
    # Partial — snapshot taken mid-run. Compare against v10 below on the same task ids.
    # Full v11: all 96 makers that survived, rolled out after the background-carving
    # fix. Supersedes the ARC_v11snap 34-task snapshot below.
    Path("/hdd_data/yunho/ARC_v11_full/whole"),
    Path("/hdd_data/yunho/ARC_v11snap/whole"),
    # v10: mask selections + solver-concept critic (CONCEPT_NOT_LEGIBLE).
    Path("/hdd_data/yunho/ARC_Single_llm_v10_obj/whole"),
    # v9: previous composite, the "before" for v10's 28 regenerated tasks.
    Path("/hdd_data/yunho/ARC_Single_llm_v9_obj/whole"),
    Path("/hdd_data/yunho/ARC_Single_llm_v7_obj/whole"),
    # v8 pilot 3-way comparison (same 18 flagged tasks): baseline v7 above vs
    # solver-as-concept vs model-states-rule-first.
    Path("/hdd_data/yunho/ARC_Single_llm_v8solver_obj/whole"),
    Path("/hdd_data/yunho/ARC_Single_llm_v8rulefirst_obj/whole"),
    Path(__file__).resolve().parents[1] / "ARC_Single" / "whole",
]

# Critic audit JSONs — verdict + findings per task, shown alongside the trajectory.
KNOWN_CRITIQUES = [
    # Newest first. The v10 loop rounds (r1-r3) came from a critic that still believed
    # selections are rectangles and anchored on the solver's primitives; it passed
    # 387/400. Kept for comparison only — movefix supersedes them.
    Path("/hdd_data/yunho/critique_v10_movefix.json"),   # 400, fixed critic (2026-07-20)
    Path("/hdd_data/yunho/v10_work/critique_r1.json"),   # 399, stale critic
    Path("/hdd_data/yunho/v10_work/critique_r2.json"),   # 11 only (r1 leftovers)
    Path("/hdd_data/yunho/v10_work/critique_r3.json"),   # 4 only (r2 leftovers)
    Path("/hdd_data/yunho/critic_check_v10.json"),       # 5, small-set validation
    Path("/hdd_data/yunho/v9_work/critique_r1.json"),
    Path("/hdd_data/yunho/v9_work/critique_r2.json"),
]


def _discover_critiques() -> list:
    """KNOWN_CRITIQUES that exist, then any other critique*.json under /hdd_data/yunho
    (newest first) — so a fresh audit appears without editing this file."""
    out = [p for p in KNOWN_CRITIQUES if p.exists()]
    seen = {p.resolve() for p in out}
    root, extra = Path("/hdd_data/yunho"), []
    if root.is_dir():
        for p in list(root.glob("critique*.json")) + list(root.glob("*/critique*.json")):
            if p.resolve() not in seen:
                seen.add(p.resolve())
                extra.append(p)
    extra.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out + extra


@st.cache_data(show_spinner=False)
def load_critique(path_str: str) -> dict:
    """task_id -> critic record ({verdict, findings:[{code,severity,evidence}]})."""
    recs = json.load(open(path_str))
    return {r["task_id"]: r for r in recs if "task_id" in r}


def render_grid(ax, grid_data, h, w, title="", sel_mask=None, highlight_color="lime"):
    """Render one ARC grid on ax with optional selection overlay."""
    arr = np.array(grid_data, dtype=int)[:h, :w]
    ax.imshow(arr, cmap=ARC_CMAP, norm=ARC_NORM, interpolation="nearest")
    # Cell separators, drawn explicitly rather than with ax.grid(which="minor"):
    # that renders the vertical lines only, so a uniform-coloured grid read as one
    # undivided block (e.g. 1190e5a7's all-6 answer looked like 4 tall bars).
    for x in np.arange(-0.5, w, 1):
        ax.axvline(x, color="white", linewidth=0.4, zorder=3)
    for y in np.arange(-0.5, h, 1):
        ax.axhline(y, color="white", linewidth=0.4, zorder=3)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=8, pad=3)
    # selection overlay
    if sel_mask is not None:
        mask = np.array(sel_mask, dtype=bool)[:h, :w]
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = mcolors.to_rgba(highlight_color, alpha=0.35)
        ax.imshow(overlay, interpolation="nearest")


def load_all_tasks(root: Path):
    tasks = {}
    if not root.is_dir():
        return tasks
    for d in sorted(root.iterdir()):
        # folder: test.<tid>.s30.<date>
        if not d.is_dir(): continue
        parts = d.name.split(".")
        if len(parts) < 2: continue
        tid = parts[1]
        files = sorted(d.glob("*.json"))
        if files:
            tasks[tid] = files
    return tasks


def _step_task(task_ids, delta):
    """Move the Task selectbox by delta, wrapping. Runs as an on_click callback,
    i.e. before the rerun draws the widget, so the selectbox picks the new value
    up through its key."""
    i = task_ids.index(st.session_state.task_select)
    st.session_state.task_select = task_ids[(i + delta) % len(task_ids)]


def render_drift_view():
    """Generator-drift inspector. Two datasets:
      • All drift        — maker O vs RE-ARC verifier O (2-way)
      • Maker-wrong (3-oracle) — instances where verifier AND llm-solver AGREE but maker
                                 differs, i.e. two independent oracles convict the maker.
    """
    import json
    DS = {
        "All drift (maker vs verifier)": "/hdd_data/yunho/drift_compare.json",
        "Maker-wrong (3-oracle consensus)": "/hdd_data/yunho/maker_wrong_compare.json",
    }
    which = st.sidebar.radio("Dataset", list(DS), index=1)
    p = Path(DS[which])
    if not p.is_file():
        st.error(f"data not found: `{p}`")
        return
    data = json.loads(p.read_text())
    three = "maker_wrong" in p.name

    def rate_val(d):
        try:
            a, b = d["rate"].split()[0].split("/"); return int(a) / int(b)
        except Exception:
            return 1.0
    data.sort(key=rate_val)

    if three:
        st.subheader("Maker-wrong — two oracles agree, maker differs")
        st.caption("For each instance the RE-ARC DSL verifier and the re-arc-llm solver "
                   "independently produce the SAME output, and the maker's O differs — so the "
                   "maker genuinely failed to solve it (not a verifier artifact). "
                   "red overlay = cells where maker O differs from that consensus.")
    else:
        st.subheader("Generator drift — maker output vs verifier")
        st.caption("maker writes its own O; the RE-ARC verifier is one (imperfect on OOD inputs) "
                   "oracle. red overlay = cells where maker O and verifier O differ. Use the "
                   "3-oracle dataset to see which of these the maker truly got wrong.")

    labels = [f'{d["task"]}   {d["rate"]}' for d in data]
    pick = st.sidebar.selectbox(f"Task ({len(data)})", labels, index=0)
    d = data[labels.index(pick)]
    st.markdown(f"### `{d['task']}` — generated instances matching verifier: {d['rate']}")

    for ex in d["examples"]:
        I = np.array(ex["I"], dtype=int)
        mO = np.array(ex["maker_O"], dtype=int)
        vO = np.array(ex["verifier_O"], dtype=int)
        cols = [("input", I, None, "lime")]
        # consensus target for highlighting maker's error: verifier (== llm in 3-oracle set)
        same_v = mO.shape == vO.shape
        diff_v = (mO != vO) if same_v else None
        cols.append(("maker O", mO, diff_v, "red"))
        cols.append(("verifier O", vO, diff_v, "red"))
        if three and "llm_O" in ex:
            lO = np.array(ex["llm_O"], dtype=int)
            same_l = mO.shape == lO.shape
            diff_l = (mO != lO) if same_l else None
            cols.append(("llm-solver O", lO, diff_l, "red"))
        fig, axes = plt.subplots(1, len(cols), figsize=(2.8 * len(cols), 3.2))
        if len(cols) == 1:
            axes = [axes]
        for ax, (lab, g, dm, hc) in zip(axes, cols):
            render_grid(ax, g, g.shape[0], g.shape[1],
                        title=f"{lab}  {g.shape[0]}×{g.shape[1]}", sel_mask=dm, highlight_color=hc)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig)
        plt.close(fig)
        if not same_v:
            st.warning(f"maker/verifier shape differs — {mO.shape[0]}×{mO.shape[1]} "
                       f"vs {vO.shape[0]}×{vO.shape[1]}")
        st.divider()


def main():
    st.set_page_config(page_title="ARC Trajectory Viewer", layout="wide")
    st.title("ARC Trajectory Viewer")

    mode = st.sidebar.radio("Mode", ["Trajectories", "Drift compare"], index=0)
    if mode == "Drift compare":
        render_drift_view()
        return

    # ── sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Data")
        root_options = []
        for p in KNOWN_ROOTS:                       # de-dup, keep order
            if p not in root_options:
                root_options.append(p)
        labels = [f"{'✓' if p.is_dir() else '✗'} {p}" for p in root_options] + ["other…"]
        picked = st.selectbox("Trajectory set", labels, index=0)
        if picked == "other…":
            root = Path(st.text_input("Path", str(DATA_ROOT)))
        else:
            root = root_options[labels.index(picked)]

        tasks = load_all_tasks(root)
        if not tasks:
            st.error(f"No trajectory data found in `{root}`")
            st.stop()

        st.header("Select")
        task_ids = sorted(tasks.keys())

        # Named filter sets: step through only a curated subset of task ids.
        _FILTERS = {
            "all": None,
            "winner: v9": "/hdd_data/yunho/best_v9.txt",
            "winner: v10": "/hdd_data/yunho/best_v10.txt",
            "winner: v11": "/hdd_data/yunho/best_v11.txt",
            "newly regenerated": "/hdd_data/yunho/v11_new86.txt",
            "answer-in-selection (residual)": "/hdd_data/yunho/v11_answersel20.txt",
        }
        _fopts = ["all"] + [
            k for k, p in _FILTERS.items()
            if p and Path(p).is_file()
            and any(t in set(Path(p).read_text().split()) for t in task_ids)
        ]
        _pick = st.selectbox("Filter", _fopts, index=0)
        if _FILTERS.get(_pick):
            keep = set(Path(_FILTERS[_pick]).read_text().split())
            task_ids = [t for t in task_ids if t in keep]
            st.caption(f"{len(task_ids)} task(s)")
            if st.session_state.get("task_select") not in task_ids:
                st.session_state.task_select = task_ids[0]
        if st.session_state.get("task_select") not in task_ids:
            st.session_state.task_select = task_ids[0]   # root changed under us

        prev_col, pos_col, next_col = st.columns([1, 1.4, 1])
        prev_col.button("◀", width="stretch", on_click=_step_task,
                        args=(task_ids, -1), help="Previous task")
        next_col.button("▶", width="stretch", on_click=_step_task,
                        args=(task_ids, 1), help="Next task")
        pos_col.markdown(
            f"<div style='text-align:center;padding-top:6px'>"
            f"{task_ids.index(st.session_state.task_select) + 1} / {len(task_ids)}</div>",
            unsafe_allow_html=True,
        )

        tid = st.selectbox("Task", task_ids, key="task_select")

        files = tasks[tid]
        file_labels = [f.stem for f in files]
        # Keyed per task so switching tasks resets sample and step instead of
        # carrying over an index the new task may not have.
        chosen_file = st.selectbox("Sample", file_labels, index=0, key=f"sample_{tid}")
        chosen_path = files[file_labels.index(chosen_file)]

        data = json.load(open(chosen_path))
        n_steps = len(data["step"])

        step = (st.slider("Step", 0, n_steps - 1, 0, key=f"step_{tid}_{chosen_file}")
                if n_steps > 1 else 0)

        st.markdown("---")
        st.markdown(f"**Task:** `{tid}`")
        st.markdown(f"**ID:** `{data['desc']['id']}`")
        st.markdown(f"**Total steps:** {n_steps}")
        reward_at_step = data["reward"][step]
        terminated = data["terminated"][step]
        # operation/operation_name have length N (actions); grid & state arrays are N+1.
        # The final state is terminal and carries no action.
        if step < len(data["operation"]):
            op_name = data["operation_name"][step]
            op_id = data["operation"][step]
            st.markdown(f"**Op:** `{op_name}` (id={op_id})")
        else:
            op_id, op_name = None, "(terminal — no action)"
            st.markdown(f"**Op:** `{op_name}`")
        st.markdown(f"**Reward:** {reward_at_step}  **Done:** {terminated}")
        sel = data["selection"][step]
        if isinstance(sel, dict) and "cells" in sel:
            st.markdown(f"**Selection:** {len(sel['cells'])} cells (object shape)")
        else:
            st.markdown(f"**Selection:** `[r={sel[0]}, c={sel[1]}, dh={sel[2]}, dw={sel[3]}]`")

        st.markdown("---")
        st.header("Critic")
        crit_paths = _discover_critiques()
        crit_labels = [f"{'✓' if p.exists() else '✗'} {p.name}" for p in crit_paths] + ["none"]
        crit_pick = st.selectbox("Critique file", crit_labels, index=0)
        critique = {}
        if crit_pick != "none":
            cpath = crit_paths[crit_labels.index(crit_pick)]
            if cpath.exists():
                critique = load_critique(str(cpath))

    # ── main area ──────────────────────────────────────────────────────────────
    st.caption(f"`{root}` — task `{tid}` "
               f"({task_ids.index(tid) + 1} of {len(task_ids)}), sample `{chosen_file}`")

    # ── critic audit (verdict + findings for this task) ─────────────────────────
    rec = critique.get(tid) if critique else None
    if rec is not None:
        verdict = rec.get("verdict", "?")
        color = {"PASS": "#2f9e6f", "REVISE": "#c47f16", "FAIL": "#d1493f"}.get(verdict, "#888")
        st.markdown(
            "### Critic audit &nbsp;"
            f"<span style='background:{color};color:#fff;padding:2px 10px;border-radius:6px;"
            f"font-size:0.7em;vertical-align:middle'>{verdict}</span>",
            unsafe_allow_html=True,
        )
        findings = rec.get("findings", [])
        if not findings:
            st.caption("no findings — clean derivation")
        for f in findings:
            st.markdown(f"**`{f.get('code','?')}`** &nbsp;·&nbsp; _{f.get('severity','?')}_")
            st.markdown(f"> {f.get('evidence','').strip()}")
        st.markdown("---")
    elif critique:
        st.caption(f"`{tid}` not present in the selected critique file")

    in_h, in_w = data["grid_dim"][0]
    in_grid = data["in_grid"]
    out_grid = data["out_grid"]
    out_h, out_w = data["grid_dim"][-1]

    # examples
    ex_in_dims = data.get("ex_in_grid_dim", [])
    ex_outs = data.get("ex_out", [])
    ex_ins = data.get("ex_in", [])
    ex_out_dims = data.get("ex_out_grid_dim", [])

    n_ex = len(ex_ins)

    # ── examples row (in -> out pairs, lined up) ────────────────────────────────
    if n_ex > 0:
        st.subheader("Examples")
        fig_ex, axes_ex = plt.subplots(1, 2 * n_ex, figsize=(2.8 * n_ex, 3.0))
        fig_ex.patch.set_facecolor("#1e1e1e")
        axes_ex = np.atleast_2d(axes_ex).reshape(-1)
        for ax in axes_ex:
            ax.set_facecolor("#1e1e1e")
            for spine in ax.spines.values():
                spine.set_visible(False)
        for i in range(n_ex):
            h_in, w_in = ex_in_dims[i] if i < len(ex_in_dims) else (5, 5)
            h_out, w_out = ex_out_dims[i] if i < len(ex_out_dims) else (5, 5)
            render_grid(axes_ex[2 * i], ex_ins[i], h_in, w_in, title=f"Ex{i+1} in")
            render_grid(axes_ex[2 * i + 1], ex_outs[i], h_out, w_out, title=f"Ex{i+1} out")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig_ex, width="stretch")
        plt.close(fig_ex)

    # ── figure ────────────────────────────────────────────────────────────────
    # The test instance, read the same way as the Examples row above: just the
    # problem and its answer. The per-step view lives in the trajectory strip
    # below, which shows every step at once instead of one at a time; the
    # clipboard / object / background panels that used to sit here said nothing
    # about the test problem itself.
    st.subheader("Test")
    fig, axes = plt.subplots(1, 2, figsize=(2.8 * 2, 3.2))
    fig.patch.set_facecolor("#1e1e1e")
    for ax in axes.flat:
        ax.set_facecolor("#1e1e1e")
        for spine in ax.spines.values():
            spine.set_visible(False)

    render_grid(axes[0], in_grid, in_h, in_w, title="Input (I)")
    render_grid(axes[1], out_grid, out_h, out_w, title="Target (O)")

    # color legend
    legend_patches = [
        mpatches.Patch(color=ARC_COLORS[i], label=f"{i}")
        for i in range(10)
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=10,
               fontsize=7, framealpha=0, labelcolor="white",
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(pad=0.5)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # ── full trajectory strip ────────────────────────────────────────────────
    with st.expander("Full trajectory (all steps)", expanded=True):
        legend_line = "  ".join(
            f"<span style='color:{c}'>&#9632;</span> {cat}"
            for cat, c in CATEGORY_COLOR.items()
        )
        st.markdown(legend_line, unsafe_allow_html=True)
        step_cols = min(n_steps, 5)
        step_rows = math.ceil(n_steps / step_cols)
        fig2 = plt.figure(figsize=(2.6 * step_cols, 2.9 * step_rows))
        fig2.patch.set_facecolor("#1e1e1e")
        gs2 = fig2.add_gridspec(step_rows, step_cols)
        for i in range(n_steps):
            r, c = divmod(i, step_cols)
            ax = fig2.add_subplot(gs2[r, c])
            ax.set_facecolor("#1e1e1e")
            is_current = i == step
            for spine in ax.spines.values():
                spine.set_visible(is_current)
                spine.set_edgecolor("yellow")
                spine.set_linewidth(2.5)
            gd_h_i, gd_w_i = data["grid_dim"][i]
            if i < len(data["operation"]):
                op_id_i = data["operation"][i]
                op_name_i = data["operation_name"][i]
                op_label = f"{op_id_i}  {op_name_i}"
                op_color = CATEGORY_COLOR.get(_OP_CATEGORY.get(op_id_i, "?"), "white")
            else:
                op_label, op_color = "(terminal)", "gray"
            render_grid(ax, data["grid"][i], gd_h_i, gd_w_i,
                        title=f"step {i}", sel_mask=data["selection_mask"][i])
            ax.text(0.5, -0.08, op_label, ha="center", va="top",
                    transform=ax.transAxes, fontsize=9, fontweight="bold",
                    color=op_color)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig2, width="stretch")
        plt.close(fig2)

    # ── op history ────────────────────────────────────────────────────────────
    with st.expander("Full op sequence"):
        rows = []
        for i, (op_id_, op_name_, sel_, rew_) in enumerate(zip(
                data["operation"], data["operation_name"],
                data["selection"], data["reward"])):
            marker = "→ " if i == step else "  "
            rows.append(f"{marker}[{i:2d}] {op_name_:18s}  sel={sel_}  r={rew_}")
        st.code("\n".join(rows))


if __name__ == "__main__":
    main()
