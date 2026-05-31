# BLOCK 7-C) FINAL EPISODE EXTRACTION
# - baseline(train) 기준 threshold 생성
# - learned / equal / gini 세 방식으로 hotspot point 생성
# - episode로 묶기
# ============================================================

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 0) final config
# ------------------------------------------------------------
FINAL_SIGMA_K = 2.5
FINAL_MIN_DURATION_BINS = 2
FINAL_GAP_ALLOW_BINS = 2

# episode 단위 entity
# 너무 크게 stg_ip만 쓰면 rw/src_metric 혼합됨
# 따라서 논문용으로는 stg_ip|level|rw|src_metric context 단위 추천
point_score_df["episode_entity"] = (
    point_score_df["stg_ip"].astype(str) + "|" +
    point_score_df["level"].astype(str) + "|" +
    point_score_df["rw"].astype(str) + "|" +
    point_score_df["src_metric"].astype(str)
)

# ------------------------------------------------------------
# 1) frequency inference
# ------------------------------------------------------------
def infer_freq_minutes_from_points(df):
    s = pd.to_datetime(df["timestamp"], errors="coerce").dropna().sort_values().diff().dropna()
    if len(s) == 0:
        return 30
    med_sec = float(np.median(s.dt.total_seconds().values))
    if not np.isfinite(med_sec) or med_sec <= 0:
        return 30
    return max(1, int(round(med_sec / 60.0)))


FREQ_MIN = infer_freq_minutes_from_points(point_score_df)
ALLOW_GAP_MIN = FINAL_GAP_ALLOW_BINS * FREQ_MIN

print("FREQ_MIN =", FREQ_MIN)
print("ALLOW_GAP_MIN =", ALLOW_GAP_MIN)


# ------------------------------------------------------------
# 2) threshold 함수
# ------------------------------------------------------------
def make_threshold_from_train(df, score_col, sigma_k=2.5):
    base = df.loc[df["split"] == "train", score_col].dropna().astype(float)
    if len(base) == 0:
        raise ValueError(f"No train baseline found for {score_col}")

    mu = float(base.mean())
    sd = float(base.std(ddof=0))
    ucl = mu + sigma_k * sd
    return mu, sd, ucl


# ------------------------------------------------------------
# 3) episode extraction 함수
# ------------------------------------------------------------
def extract_episodes(df, score_col, hot_col,
                     entity_col="episode_entity",
                     time_col="timestamp",
                     min_bins=2,
                     gap_allow_min=60):
    x = df[[entity_col, "stg_ip", "level", "rw", "src_metric", time_col, score_col, hot_col]].copy()
    x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
    x = x.dropna(subset=[entity_col, time_col]).sort_values([entity_col, time_col]).reset_index(drop=True)

    rows = []
    allow_gap = pd.Timedelta(minutes=gap_allow_min)

    for ent, g in x.groupby(entity_col):
        hot = g[g[hot_col] == 1].copy()
        if hot.empty:
            continue

        hot["gap"] = hot[time_col].diff()
        hot["new_ep"] = ((hot["gap"].isna()) | (hot["gap"] > allow_gap)).astype(int)
        hot["ep_id"] = hot["new_ep"].cumsum()

        for _, ge in hot.groupby("ep_id"):
            if len(ge) < min_bins:
                continue

            peak_idx = ge[score_col].idxmax()
            peak_row = ge.loc[peak_idx]

            rows.append({
                "method": score_col.replace("score_", ""),
                "episode_entity": ent,
                "stg_ip": str(peak_row["stg_ip"]).lower(),
                "level": str(peak_row["level"]).lower(),
                "rw": str(peak_row["rw"]).lower(),
                "src_metric": str(peak_row["src_metric"]).lower(),
                "start_ts": ge[time_col].min(),
                "end_ts": ge[time_col].max(),
                "duration_bins": int(len(ge)),
                "duration_minutes": int(len(ge) * FREQ_MIN),
                "peak_ts": peak_row[time_col],
                "peak_score": float(peak_row[score_col]),
                "n_points_in_episode": int(len(ge))
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 4) 세 방식 모두 실행
# ------------------------------------------------------------
summary_rows = []
episode_list = []

for score_col in ["score_gini", "score_equal", "score_learned"]:
    mu, sd, ucl = make_threshold_from_train(point_score_df, score_col, sigma_k=FINAL_SIGMA_K)
    hot_col = f"{score_col}_hot"

    point_score_df[f"{score_col}_mu"] = mu
    point_score_df[f"{score_col}_sd"] = sd
    point_score_df[f"{score_col}_ucl"] = ucl
    point_score_df[hot_col] = (point_score_df[score_col] > ucl).astype(int)

    eps = extract_episodes(
        point_score_df,
        score_col=score_col,
        hot_col=hot_col,
        entity_col="episode_entity",
        time_col="timestamp",
        min_bins=FINAL_MIN_DURATION_BINS,
        gap_allow_min=ALLOW_GAP_MIN
    )

    if not eps.empty:
        episode_list.append(eps)

    summary_rows.append({
        "method": score_col.replace("score_", ""),
        "ucl": ucl,
        "n_hot_points": int(point_score_df[hot_col].sum()),
        "point_ratio": float(point_score_df[hot_col].mean()),
        "n_episodes": int(len(eps)),
        "avg_duration_bins": float(eps["duration_bins"].mean()) if not eps.empty else np.nan,
        "median_duration_bins": float(eps["duration_bins"].median()) if not eps.empty else np.nan,
        "avg_peak_score": float(eps["peak_score"].mean()) if not eps.empty else np.nan,
        "max_peak_score": float(eps["peak_score"].max()) if not eps.empty else np.nan,
    })

episode_df = pd.concat(episode_list, ignore_index=True) if episode_list else pd.DataFrame()
episode_summary = pd.DataFrame(summary_rows).sort_values("method").reset_index(drop=True)

print("=" * 80)
print("EPISODE SUMMARY")
print("=" * 80)
print(episode_summary)


# ------------------------------------------------------------
# 5) culprit feature 붙이기
# - peak 시점에서 |z| * weight 기준
# ------------------------------------------------------------
def add_peak_culprit_features(ep_df, score_df, feature_cols, method_name):
    if ep_df.empty:
        return ep_df.copy()

    out = ep_df.copy()

    if method_name == "gini":
        weight_map = {f: 0.0 for f in feature_cols}
        gcol = "raw_imb_gini" if "raw_imb_gini" in feature_cols else feature_cols[0]
        weight_map[gcol] = 1.0
    elif method_name == "equal":
        weight_map = {f: 1.0 / len(feature_cols) for f in feature_cols}
    elif method_name == "learned":
        weight_map = dict(zip(final_weight_df["feature"], final_weight_df["weight"]))
    else:
        weight_map = {f: 0.0 for f in feature_cols}

    top1_list = []
    top2_list = []

    for _, r in out.iterrows():
        m = (
                (score_df["episode_entity"] == r["episode_entity"]) &
                (pd.to_datetime(score_df["timestamp"]) == pd.to_datetime(r["peak_ts"]))
        )
        gp = score_df.loc[m]
        if gp.empty:
            top1_list.append(None)
            top2_list.append(None)
            continue

        row = gp.iloc[0]
        contrib = []

        for f in feature_cols:
            zc = f"scaled__{f}"
            zabs = abs(float(row[zc])) if zc in row.index and pd.notna(row[zc]) else 0.0
            wv = float(weight_map.get(f, 0.0))
            contrib.append((f, zabs * wv))

        contrib = sorted(contrib, key=lambda x: x[1], reverse=True)
        top1_list.append(contrib[0][0] if len(contrib) >= 1 else None)
        top2_list.append(contrib[1][0] if len(contrib) >= 2 else None)

    out["culprit_feature_top1"] = top1_list
    out["culprit_feature_top2"] = top2_list
    return out


episode_parts = []
for m in ["gini", "equal", "learned"]:
    sub = episode_df[episode_df["method"] == m].copy()
    sub = add_peak_culprit_features(sub, point_score_df, WL_FEATURE_COLS, m)
    episode_parts.append(sub)

episode_df = pd.concat(episode_parts, ignore_index=True) if episode_parts else pd.DataFrame()

# ------------------------------------------------------------
# 6) 저장
# ------------------------------------------------------------
point_score_df.to_parquet(FINAL_OUT_DIR / "POINT_SCORE_WITH_HOT_FLAGS.parquet", index=False)
episode_df.to_parquet(FINAL_OUT_DIR / "EPISODES_FINAL.parquet", index=False)
episode_summary.to_excel(FINAL_OUT_DIR / "EPISODE_SUMMARY_FINAL.xlsx", index=False)

with pd.ExcelWriter(FINAL_OUT_DIR / "EPISODE_DETAIL_FINAL.xlsx", engine="openpyxl") as writer:
    episode_summary.to_excel(writer, sheet_name="SUMMARY", index=False)
    episode_df.to_excel(writer, sheet_name="EPISODES", index=False)

print("final episode extraction saved")
