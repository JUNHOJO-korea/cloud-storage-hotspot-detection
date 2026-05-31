import json

with open('sample2.ipynb', 'r') as f:
    nb = json.load(f)

# User's EXACT VIZ code with safety wrappers added
fixed_src = r'''# NEW BLOCK VIZ) VISUALIZATIONS
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── VIZ-1) SPC Control Chart — 대표 stg 1개, imb_norm_top1_share__throughput ──
print("VIZ-1: SPC control chart ...")
metric_plot = "imb_norm_top1_share__throughput"
rw_plot = "read"
viz1_done = False

try:
    for stg in stg_list[:3]:
        sub = project_long[
            (project_long["stg_ip"]==stg) &
            (project_long["metric"]==metric_plot) &
            (project_long["rw"]==rw_plot) &
            (project_long["entity_id"]=="group")
        ].sort_values("timestamp").copy()
        if len(sub) < 30: continue
        lim_sub = limits_imb[
            (limits_imb["stg_ip"]==stg) &
            (limits_imb["metric"]==metric_plot) &
            (limits_imb["rw"]==rw_plot)
        ]
        if lim_sub.empty: continue

        ucl = float(lim_sub["UCL"].iloc[0])
        cl  = float(lim_sub["CL"].iloc[0])
        lcl = float(lim_sub["LCL"].iloc[0])

        ev_sub = events_imb[
            (events_imb["stg_ip"]==stg) &
            (events_imb["metric"]==metric_plot) &
            (events_imb["rw"]==rw_plot)
        ].copy()

        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(sub["timestamp"], sub["value"], lw=0.8, color="#2196F3", label="value")
        ax.axhline(ucl, color="red",    lw=1.2, ls="--", label=f"UCL={ucl:.3f}")
        ax.axhline(cl,  color="green",  lw=1.0, ls=":",  label=f"CL={cl:.3f}")
        ax.axhline(lcl, color="orange", lw=1.0, ls="--", label=f"LCL={lcl:.3f}")
        if not ev_sub.empty:
            ax.scatter(ev_sub["timestamp"], ev_sub["value"],
                       color="red", s=20, zorder=5, label="SPC event")
        ax.set_title(f"SPC Chart — {stg} | {metric_plot} | {rw_plot}", fontsize=11)
        ax.set_xlabel("timestamp"); ax.set_ylabel("norm_top1_share")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.legend(fontsize=8, loc="upper left"); plt.tight_layout()
        fig.savefig(PLOT_DIR / f"spc_chart_{stg[:12]}_{rw_plot}.png", dpi=120)
        plt.close(fig)
        print(f"  saved spc_chart_{stg[:12]}_{rw_plot}.png")
        viz1_done = True
        break
    if not viz1_done:
        print("⚠️ VIZ-1: not enough data to plot")
except Exception as e:
    print(f"⚠️ VIZ-1 ERROR: {type(e).__name__}: {e}")

# ── VIZ-2) Imbalance 지표 비교 (멀티라인) ──────────────────────────────
print("VIZ-2: Imbalance multi-metric comparison ...")
viz2_done = False

try:
    for stg in stg_list[:3]:
        fig, axes = plt.subplots(len(IMB_METRICS_REVISED), 1, figsize=(14, 3*len(IMB_METRICS_REVISED)), sharex=True)
        if len(IMB_METRICS_REVISED)==1: axes = [axes]
        has_any = False
        for ax, imb_m in zip(axes, IMB_METRICS_REVISED):
            m_key = f"{imb_m}__throughput"
            sub = project_long[
                (project_long["stg_ip"]==stg) &
                (project_long["metric"]==m_key) &
                (project_long["rw"]=="read") &
                (project_long["entity_id"]=="group")
            ].sort_values("timestamp").copy()
            if sub.empty:
                ax.set_ylabel(imb_m, fontsize=8); ax.set_visible(False); continue
            has_any = True
            ax.plot(sub["timestamp"], sub["value"], lw=0.8)
            ax.set_ylabel(imb_m, fontsize=8)
            ax.grid(True, ls=":", alpha=0.4)
        if not has_any: plt.close(fig); continue
        axes[-1].set_xlabel("timestamp")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        fig.suptitle(f"Imbalance Metrics — {stg} | read", fontsize=11, y=1.01)
        plt.tight_layout()
        fig.savefig(PLOT_DIR / f"imbalance_compare_{stg[:12]}.png", dpi=120)
        plt.close(fig)
        print(f"  saved imbalance_compare_{stg[:12]}.png")
        viz2_done = True
        break
    if not viz2_done: print("⚠️ VIZ-2: not enough data")
except Exception as e:
    print(f"⚠️ VIZ-2 ERROR: {type(e).__name__}: {e}")

# ── VIZ-3) Hotspot segment 히트맵 ──────────────────────────────────────
print("VIZ-3: Hotspot heatmap ...")

try:
    if not segments_final.empty and "hotspot_score" in segments_final.columns:
        seg_h = segments_final.copy()
        seg_h["peak_ts"] = pd.to_datetime(seg_h["peak_ts"], errors="coerce")
        seg_h["hour"] = seg_h["peak_ts"].dt.floor("h")
        pivot_data = (seg_h.groupby(["stg_ip","hour"])["hotspot_score"]
                      .max().unstack(fill_value=0))
        if pivot_data.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(min(18, max(10, pivot_data.shape[1]//2)),
                                            max(3, pivot_data.shape[0]*0.4+1)))
            im = ax.imshow(pivot_data.values, aspect="auto", cmap="YlOrRd",
                           interpolation="nearest", vmin=0, vmax=3)
            ax.set_yticks(range(len(pivot_data.index)))
            ax.set_yticklabels(pivot_data.index, fontsize=7)
            step = max(1, pivot_data.shape[1]//20)
            ax.set_xticks(range(0, pivot_data.shape[1], step))
            ax.set_xticklabels([str(c)[:13] for c in pivot_data.columns[::step]],
                               rotation=45, ha="right", fontsize=7)
            plt.colorbar(im, ax=ax, label="hotspot_score (0=none,1=weak,2=medium,3=strong)")
            ax.set_title("Hotspot Heatmap (per storage × hour)", fontsize=11)
            plt.tight_layout()
            fig.savefig(PLOT_DIR / "hotspot_heatmap.png", dpi=120)
            plt.close(fig)
            print("  saved hotspot_heatmap.png")
        else:
            print("⚠️ VIZ-3: not enough time range")
    else:
        print("⚠️ VIZ-3: segments_final empty or missing hotspot_score")
except Exception as e:
    print(f"⚠️ VIZ-3 ERROR: {type(e).__name__}: {e}")

# ── VIZ-4) Tier 분포 막대그래프 ───────────────────────────────────────
print("VIZ-4: Tier distribution bar ...")

try:
    if not segments_final.empty and "hotspot_tier" in segments_final.columns:
        tier_counts = segments_final["hotspot_tier"].value_counts().reindex(
            ["strong","medium","weak"], fill_value=0)
        fig, ax = plt.subplots(figsize=(6,4))
        colors = {"strong":"#d32f2f","medium":"#f57c00","weak":"#fbc02d"}
        bars = ax.bar(tier_counts.index, tier_counts.values,
                      color=[colors[t] for t in tier_counts.index], edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, tier_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, str(val),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title("Hotspot Tier Distribution", fontsize=12)
        ax.set_xlabel("Tier"); ax.set_ylabel("Count")
        ax.grid(True, ls=":", alpha=0.4, axis="y")
        plt.tight_layout()
        fig.savefig(PLOT_DIR / "tier_distribution.png", dpi=120)
        plt.close(fig)
        print("  saved tier_distribution.png")
    else:
        print("⚠️ VIZ-4: no segment data")
except Exception as e:
    print(f"⚠️ VIZ-4 ERROR: {type(e).__name__}: {e}")

print("✅ VIZ block done. Plots saved to:", PLOT_DIR)
'''

nb['cells'][32]['source'] = fixed_src.split('\n')

with open('sample2.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Validate
with open('sample2.ipynb', 'r') as f:
    v = json.load(f)
src_check = ''.join(v['cells'][32]['source'])
assert 'try:' in src_check
assert 'except Exception' in src_check
assert '.floor("h")' in src_check
assert '.floor("H")' not in src_check
print(f"✅ Fixed Cell[32]:")
print(f"   - .floor('h') lowercase ✅")
print(f"   - 4× try-except safety wrappers ✅")
print(f"   - Total lines: {len(fixed_src.split(chr(10)))}")
print(f"   - JSON valid: OK")
