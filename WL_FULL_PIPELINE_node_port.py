# ============================================================
# 🔥 COMPLETE WEIGHTED LEARNING PIPELINE (NODE + PORT)
# ============================================================
# 이 하나의 셀만 실행하면 됨!
# - 6-A: 설정 (node + port 둘 다 포함! 🆕)
# - 6-B: Point Feature Table 생성
# - 6-C: Positive Peak / Negative / Pair Table
# - 6-D: Diagonal Weighted Metric Learning
# - 6-E: Learned Weight 적용 → Point Score
# - 6-F: Robustness Experiment (생략가능, 포함함)
# - 7-C: Final Episode Extraction (level 컬럼 포함!)
#
# 📌 전제조건: 앞선 블록(0~5)에서 아래 변수들이 메모리에 있어야 함
#   project_long, imb_long, limits_imb, segments_final,
#   point_labels, load_ref, base_long, stg_list, OUTPUT_DIR,
#   IMB_METRICS_REVISED, INPUT_DIR, CACHE_DIR
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import softmax as scipy_softmax
from scipy.stats import rankdata


# ============================================================
# BLOCK 6-A) CONFIG — node + port 둘 다 포함! 🆕
# ============================================================
WL_LEVELS = ["node", "port"]          # ← 요기! 기존: WL_LEVEL = "node" 였음
WL_RWS = ["read", "write"]
WL_SRC_METRICS = ["throughput", "iops", "latency"]

WL_POS_TIERS = ["strong"]
WL_NEG_BUFFER_BINS = 1
WL_NEG_PER_POS = 2
WL_POS_NEIGHBORS = 2
WL_MIN_NEG_POOL_PER_CONTEXT = 5
WL_ALLOW_NEG_FALLBACK_ACROSS_STG = True
WL_ALLOW_POS_FALLBACK_ACROSS_STG = True

WL_USE_RAW_IMB = ["imb_hhi", "imb_gini", "imb_theil"]
WL_USE_SEV_IMB = ["imb_hhi", "imb_gini", "imb_theil"]
WL_USE_LOAD_RATIO = True
WL_USE_ACTIVE_RATIO = True

WL_TRAIN_Q = 0.70
WL_VALID_Q = 0.85

WL_OUT_DIR = OUTPUT_DIR / "weighted_learning_v1"
WL_OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("WL_CONFIG: levels =", WL_LEVELS, "(node+port both!)")
print("WL_OUT_DIR:", WL_OUT_DIR)


# ============================================================
# BLOCK 6-B) POINT FEATURE TABLE (node+port)
# ============================================================
print("\n" + "=" * 70)
print("BLOCK 6-B) BUILDING POINT FEATURE TABLE...")

# 데이터 정리
for df_name in ["imb_long", "limits_imb", "segments_final", "point_labels", "load_ref"]:
    df = globals()[df_name]
    for c in df.columns:
        if c in ["stg_ip", "level", "entity_id", "rw", "metric", "imb_metric", "src_metric", "hotspot_tier"]:
            df[c] = df[c].astype(str).str.strip().str.lower()
    for c in ["timestamp", "start_ts", "end_ts", "peak_ts"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    globals()[df_name] = df

# 🆕 핵심 수정: level == "node" → level.isin(["node","port"])
imb_use = imb_long[
    (imb_long["level"].isin(WL_LEVELS)) &
    (imb_long["rw"].isin(WL_RWS)) &
    (imb_long["src_metric"].isin(WL_SRC_METRICS)) &
    (imb_long["imb_metric"].isin(IMB_METRICS_REVISED))
].copy()

print(f"  imb_use shape (node+port): {imb_use.shape}")
print(f"  level distribution:\n{imb_use['level'].value_counts()}")


# ððð íµì¬! IMB_LONG.entity_idë ì ë¶ 'group' -> Ä£ì¨ìì
#     PROJECT_LONGìì ì¤ì  ë¸ëê¸/í¬í¸ë²í¸(entity_id)ë¥¼ ê°ì ¸ì´
#
#     [ë°ì´í° êµ¬ì¡°]
#       PROJ entity_id=ì¤ì ì´몄 -> src_metric=None, metric=throughput/iops/latency OK
#       PROJ entity_id='group'  -> src_metric=O,    metric=imb_gini__iops      X
#     -> ì¤ì  ì´몄íììë **metric** ì»´럼 ì¬ì©!
#        ë§¤칭:  IMB.src_metric == PROJ.metric
_proj_entity = project_long[project_long["entity_id"] != "group"][
    ["timestamp", "stg_ip", "level", "rw", "metric", "entity_id"]
].copy()
_proj_entity["timestamp"] = pd.to_datetime(_proj_entity["timestamp"], errors="coerce")
for c in ["stg_ip", "level", "rw", "metric", "entity_id"]:
    _proj_entity[c] = _proj_entity[c].astype(str).str.strip().str.lower()
_proj_lookup = _proj_entity.drop_duplicates(
    subset=["timestamp", "stg_ip", "level", "rw", "metric"]
).rename(columns={"entity_id": "entity_id_real"})

# IMB.src_metric == PROJ.metric 으로 머지!
_imb_for_map = imb_use.copy()
_imb_for_map["timestamp"] = pd.to_datetime(_imb_for_map["timestamp"], errors="coerce")
_imb_for_map["stg_ip"] = _imb_for_map["stg_ip"].astype(str).str.strip().str.lower()
_imb_for_map["level"] = _imb_for_map["level"].astype(str).str.strip().str.lower()
_imb_for_map["rw"] = _imb_for_map["rw"].astype(str).str.strip().str.lower()
_imb_for_map["src_metric"] = _imb_for_map["src_metric"].astype(str).str.strip().str.lower()

_real_map = _imb_for_map.merge(
    _proj_lookup,
    left_on=["timestamp", "stg_ip", "level", "rw", "src_metric"],
    right_on=["timestamp", "stg_ip", "level", "rw", "metric"],
    how="left"
)
# 실제 노드명/포트번호로 교체 (100% 매칭 확인됨!)
imb_use["entity_id"] = _real_map["entity_id_real"].fillna(imb_use["entity_id"])
del _proj_entity, _proj_lookup, _imb_for_map, _real_map

print(f"  entity_id 매핑 후 unique: {imb_use['entity_id'].nunique()}")
print(f"  node entity_id 예시: {imb_use[imb_use.level=='node']['entity_id'].unique()[:5]}")
print(f"  port entity_id 예시: {imb_use[imb_use.level=='port']['entity_id'].unique()[:5]}")

# limits merge
lim_key = ["stg_ip", "level", "entity_id", "rw", "metric"]
lim_use = limits_imb.copy()
for c in lim_key:
    lim_use[c] = lim_use[c].astype(str).str.strip().str.lower()

imb_use["entity_id"] = imb_use["entity_id"].astype(str).str.strip().str.lower()
imb_use["metric"] = imb_use["metric"].astype(str).str.strip().str.lower()

imb_use = imb_use.merge(
    lim_use[lim_key + ["CL", "SIGMA", "UCL", "LCL"]],
    on=lim_key,
    how="left"
)

# severity
imb_use["severity_over_ucl"] = np.maximum(
    0.0,
    (pd.to_numeric(imb_use["value"], errors="coerce") - pd.to_numeric(imb_use["UCL"], errors="coerce")) /
    np.maximum(np.abs(pd.to_numeric(imb_use["UCL"], errors="coerce")), 1e-12)
)

# point-wide feature
# 🆕 entity_id 포함해서 노드명/포트번호 유지
point_keys = ["timestamp", "stg_ip", "level", "entity_id", "rw", "src_metric"]

base_meta = (
    imb_use.groupby(point_keys, observed=True)
    .agg(
        group_total=("group_total", "max"),
        n_entities=("n_entities", "max"),
        active_entities=("active_entities", "max")
    )
    .reset_index()
)

raw_wide = (
    imb_use.pivot_table(
        index=point_keys,
        columns="imb_metric",
        values="value",
        aggfunc="first",
        observed=True
    )
    .reset_index()
)
raw_wide.columns.name = None
raw_wide = raw_wide.rename(columns={c: f"raw_{c}" for c in raw_wide.columns if c not in point_keys})

sev_wide = (
    imb_use.pivot_table(
        index=point_keys,
        columns="imb_metric",
        values="severity_over_ucl",
        aggfunc="first",
        observed=True
    )
    .reset_index()
)
sev_wide.columns.name = None
sev_wide = sev_wide.rename(columns={c: f"sev_{c}" for c in sev_wide.columns if c not in point_keys})

point_features = base_meta.merge(raw_wide, on=point_keys, how="left")
point_features = point_features.merge(sev_wide, on=point_keys, how="left")

# load reference merge
point_features = point_features.merge(
    load_ref[["stg_ip", "level", "rw", "src_metric", "load_floor", "baseline_load_median", "baseline_n"]],
    on=["stg_ip", "level", "rw", "src_metric"],
    how="left"
)

point_features["load_ratio"] = (
    pd.to_numeric(point_features["group_total"], errors="coerce") /
    np.maximum(pd.to_numeric(point_features["load_floor"], errors="coerce"), 1e-12)
)
point_features["active_ratio"] = (
    pd.to_numeric(point_features["active_entities"], errors="coerce") /
    np.maximum(pd.to_numeric(point_features["n_entities"], errors="coerce"), 1e-12)
)

point_features["passes_load_gate"] = (
    pd.to_numeric(point_features["group_total"], errors="coerce") >=
    pd.to_numeric(point_features["load_floor"], errors="coerce")
).fillna(False)

# point label merge
point_label_key = ["timestamp", "stg_ip", "level", "rw", "src_metric"]
point_label_flag = point_labels[point_label_key + ["segment_id", "hotspot_tier", "hotspot_score"]].copy()
point_label_flag["has_hot_label"] = 1

point_features = point_features.merge(
    point_label_flag,
    on=point_label_key,
    how="left"
)
point_features["has_hot_label"] = point_features["has_hot_label"].fillna(0).astype(int)

# unique point id
point_features = point_features.sort_values(point_keys).reset_index(drop=True)
point_features["point_id"] = np.arange(1, len(point_features) + 1)

# leakage-free feature list
WL_FEATURE_COLS = []

for m in WL_USE_RAW_IMB:
    c = f"raw_{m}"
    if c in point_features.columns:
        WL_FEATURE_COLS.append(c)

for m in WL_USE_SEV_IMB:
    c = f"sev_{m}"
    if c in point_features.columns:
        WL_FEATURE_COLS.append(c)

if WL_USE_LOAD_RATIO and "load_ratio" in point_features.columns:
    WL_FEATURE_COLS.append("load_ratio")

if WL_USE_ACTIVE_RATIO and "active_ratio" in point_features.columns:
    WL_FEATURE_COLS.append("active_ratio")

for c in WL_FEATURE_COLS:
    point_features[c] = pd.to_numeric(point_features[c], errors="coerce").fillna(0.0)

print(f"  point_features: {point_features.shape} (was 192265 node-only)")
print(f"  level distribution:\n{point_features['level'].value_counts()}")
print(f"  WL_FEATURE_COLS: {WL_FEATURE_COLS}")

# save
point_features.to_parquet(WL_OUT_DIR / "WL_POINT_FEATURES_NODE.parquet", index=False)


# ============================================================
# BLOCK 6-C) POSITIVE PEAKS + CLEAN NEGATIVES + PAIRS
# ============================================================
print("\n" + "=" * 70)
print("BLOCK 6-C) BUILDING POS/NEG SAMPLES & PAIRS...")

# --- positive samples ---
pos_mask = point_features["has_hot_label"] == 1
pos_base = point_features[pos_mask].copy()

if WL_POS_TIERS:
    pos_base = pos_base[pos_base["hotspot_tier"].isin(WL_POS_TIERS)]

pos_base = pos_base.dropna(subset=["segment_id"])
pos_base = pos_base.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)

pos_samples = (
    pos_base.groupby("segment_id")
    .apply(lambda g: g.loc[g["hotspot_score"].idxmax()] if "hotspot_score" in g.columns else g.iloc[0])
    .reset_index(drop=True)
)

# time-based split
time_sorted = pos_samples.sort_values("timestamp").reset_index(drop=True)
n_pos = len(time_sorted)
train_end_idx = int(n_pos * WL_TRAIN_Q)
valid_end_idx = int(n_pos * WL_VALID_Q)

train_ids = set(time_sorted.iloc[:train_end_idx]["point_id"].tolist())
valid_ids = set(time_sorted.iloc[train_end_idx:valid_end_idx]["point_id"].tolist())
test_ids = set(time_sorted.iloc[valid_end_idx:]["point_id"].tolist())

def assign_split(row):
    pid = row["point_id"]
    if pid in train_ids:
        return "train"
    elif pid in valid_ids:
        return "valid"
    else:
        return "test"

pos_samples["split"] = pos_samples.apply(assign_split, axis=1)

# --- negative samples ---
neg_buffer = pd.Timedelta(minutes=WL_NEG_BUFFER_BINS * 30)

all_neg_rows = []
for _, pos_row in pos_samples.iterrows():
    context_filter = (
        (point_features["stg_ip"] == pos_row["stg_ip"]) &
        (point_features["level"] == pos_row["level"]) &
        (point_features["rw"] == pos_row["rw"]) &
        (point_features["src_metric"] == pos_row["src_metric"]) &
        (point_features["passes_load_gate"]) &
        (point_features["has_hot_label"] == 0)
    )

    candidate_neg = point_features[context_filter].copy()
    if candidate_neg.empty:
        if WL_ALLOW_NEG_FALLBACK_ACROSS_STG:
            fallback = (
                (point_features["level"] == pos_row["level"]) &
                (point_features["rw"] == pos_row["rw"]) &
                (point_features["src_metric"] == pos_row["src_metric"]) &
                (point_features["passes_load_gate"]) &
                (point_features["has_hot_label"] == 0)
            )
            candidate_neg = point_features[fallback].copy()
        if candidate_neg.empty:
            continue

    ts = pd.Timestamp(pos_row["timestamp"])
    candidate_neg = candidate_neg[
        (candidate_neg["timestamp"] < ts - neg_buffer) |
        (candidate_neg["timestamp"] > ts + neg_buffer)
    ]

    if len(candidate_neg) >= WL_MIN_NEG_POOL_PER_CONTEXT:
        chosen = candidate_neg.sample(n=min(WL_NEG_PER_POS, len(candidate_neg)), random_state=42)
        for _, neg_row in chosen.iterrows():
            all_neg_rows.append({
                **{k: neg_row[k] for k in point_keys},
                "point_id": neg_row["point_id"],
                "split": pos_row["split"],
                "y": 0
            })

neg_samples = pd.DataFrame(all_neg_rows) if all_neg_rows else pd.DataFrame(
    columns=list(pos_samples.columns) + ["split", "y"]
)
if not neg_samples.empty:
    neg_samples = neg_samples.drop_duplicates(subset=["point_id"])

train_samples = pd.concat([pos_samples, neg_samples], ignore_index=True)

# --- pair table ---
pair_rows = []
for _, pos_row in pos_samples.iterrows():
    # similar pairs (positive neighbors)
    same_context = train_samples[
        (train_samples["stg_ip"] == pos_row["stg_ip"]) &
        (train_samples["level"] == pos_row["level"]) &
        (train_samples["rw"] == pos_row["rw"]) &
        (train_samples["src_metric"] == pos_row["src_metric"]) &
        (train_samples["has_hot_label"] == 1) &
        (train_samples["point_id"] != pos_row["point_id"])
    ]
    if not same_context.empty:
        similar_candidates = same_context.sample(
            n=min(WL_POS_NEIGHBORS, len(same_context)), random_state=42
        )
        for _, sim_row in similar_candidates.iterrows():
            pair_rows.append({
                "left_point_id": pos_row["point_id"],
                "right_point_id": sim_row["point_id"],
                "split": pos_row["split"],
                "pair_type": "similar",
                "y": 1
            })

    # dissimilar pairs (negative samples)
    neg_for_pair = train_samples[
        (train_samples["point_id"].isin(neg_samples["point_id"])) &
        (train_samples["split"] == pos_row["split"])
    ]
    if not neg_for_pair.empty:
        dissimilar_candidates = neg_for_pair.sample(
            n=min(WL_NEG_PER_POS, len(neg_for_pair)), random_state=42
        )
        for _, diss_row in dissimilar_candidates.iterrows():
            pair_rows.append({
                "left_point_id": pos_row["point_id"],
                "right_point_id": diss_row["point_id"],
                "split": pos_row["split"],
                "pair_type": "dissimilar",
                "y": 0
            })

pair_table = pd.DataFrame(pair_rows)
if not pair_table.empty:
    pair_table = pair_table.drop_duplicates(
        subset=["left_point_id", "right_point_id", "pair_type"]
    ).reset_index(drop=True)

print(f"  pos_samples: {pos_samples.shape}")
print(f"  neg_samples: {neg_samples.shape}")
print(f"  pair_table:  {pair_table.shape}")
print(f"  split distribution:\n{pair_table.groupby(['split','pair_type']).size()}")

# save
train_samples.to_parquet(WL_OUT_DIR / "WL_TRAIN_SAMPLES_NODE.parquet", index=False)
neg_pool_clean = train_samples[train_samples["y"] == 0].copy()
neg_pool_clean.to_parquet(WL_OUT_DIR / "WL_NEG_POOL_NODE.parquet", index=False)
pair_table.to_parquet(WL_OUT_DIR / "WL_PAIR_TABLE_NODE.parquet", index=False)


# ============================================================
# BLOCK 6-D) DIAGONAL WEIGHTED METRIC LEARNING
# ============================================================
print("\n" + "=" * 70)
print("BLOCK 6-D) WEIGHTED METRIC LEARNING OPTIMIZATION...")

WL_MULTI_START_SEEDS = [7, 21, 42, 84, 168]
WL_INIT_NOISE = 0.05
WL_NEG_HINGE_WEIGHT = 1.0
WL_L2_SIMPLEX = 1e-3
WL_USE_DATA_DRIVEN_MARGIN = True

# helpers
def robust_scale_fit(X_df):
    med = X_df.median()
    mad = (X_df - med).abs().median()
    mad = mad.replace(0, 1.0)
    return med, mad

def robust_scale_transform(X_df, med, mad):
    return (X_df - med) / mad

def softmax_logits(alpha):
    alpha = np.asarray(alpha, dtype=float)
    alpha = alpha - np.max(alpha)
    e = np.exp(alpha)
    return e / np.sum(e)

def fast_auc_binary(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return np.nan
    ranks = rankdata(scores, method="average")
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def eval_pairs(Z, y, w):
    if len(Z) == 0:
        return {"n_pairs": 0, "pos_mean_dist": np.nan, "neg_mean_dist": np.nan,
                "gap_neg_minus_pos": np.nan, "auc_pair_distance": np.nan}
    dist = Z @ w
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    pos_mean = float(dist[pos_mask].mean()) if np.any(pos_mask) else np.nan
    neg_mean = float(dist[neg_mask].mean()) if np.any(neg_mask) else np.nan
    gap = float(neg_mean - pos_mean) if np.isfinite(pos_mean) and np.isfinite(neg_mean) else np.nan
    label_neg = 1 - y
    auc = fast_auc_binary(label_neg, dist)
    return {"n_pairs": int(len(y)), "pos_mean_dist": pos_mean, "neg_mean_dist": neg_mean,
            "gap_neg_minus_pos": gap, "auc_pair_distance": auc}

def feature_gap_diagnostic(Z, y, feature_names):
    rows = []
    for j, f in enumerate(feature_names):
        zj = Z[:, j]
        pos_mean = float(zj[y == 1].mean()) if np.any(y == 1) else np.nan
        neg_mean = float(zj[y == 0].mean()) if np.any(y == 0) else np.nan
        gap = float(neg_mean - pos_mean) if np.isfinite(pos_mean) and np.isfinite(neg_mean) else np.nan
        rows.append({"feature": f, "pos_mean_sqdiff": pos_mean, "neg_mean_sqdiff": neg_mean,
                     "gap_neg_minus_pos_feature": gap})
    out = pd.DataFrame(rows).sort_values("gap_neg_minus_pos_feature", ascending=False).reset_index(drop=True)
    return out

def choose_margin_from_equal(Z, y, w_equal):
    d = Z @ w_equal
    pos = d[y == 1]
    neg = d[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 1.0
    return float(0.5 * (pos.mean() + neg.mean()))

def simplex_metric_loss(alpha, Z, y, margin, neg_hinge_weight=1.0, l2_simplex=1e-3):
    w = softmax_logits(alpha)
    d = Z @ w
    pos_loss = float(d[y == 1].mean()) if (d[y == 1]).sum() > 0 else 0.0
    neg_hinge = float(np.maximum(0.0, margin - d[y == 0]).mean()) if (y == 0).sum() > 0 else 0.0
    reg = float(l2_simplex * np.sum(w * w))
    total = pos_loss + neg_hinge_weight * neg_hinge + reg
    parts = {"total_loss": total, "pos_loss": pos_loss, "neg_hinge_loss": neg_hinge, "l2_loss": reg}
    return total, parts

# split
samples_train = train_samples[train_samples["split"] == "train"].copy()
samples_valid = train_samples[train_samples["split"] == "valid"].copy()
samples_test = train_samples[train_samples["split"] == "test"].copy()
pairs_train = pair_table[pair_table["split"] == "train"].copy()
pairs_valid = pair_table[pair_table["split"] == "valid"].copy()
pairs_test = pair_table[pair_table["split"] == "test"].copy()

print(f"  samples_train: {samples_train.shape}, pairs_train: {pairs_train.shape}")

# robust scaling
X_train_df = samples_train.set_index("point_id")[WL_FEATURE_COLS].copy()
med, mad = robust_scale_fit(X_train_df)

for df_name in ["samples_train", "samples_valid", "samples_test"]:
    tmp = globals()[df_name].copy()
    tmp_scaled = robust_scale_transform(tmp[WL_FEATURE_COLS], med, mad).fillna(0.0)
    for c in WL_FEATURE_COLS:
        tmp[f"scaled__{c}"] = tmp_scaled[c].values
    globals()[df_name] = tmp

SCALED_COLS = [f"scaled__{c}" for c in WL_FEATURE_COLS]

def build_feature_map(df_samples, scaled_cols):
    mat = df_samples.set_index("point_id")[scaled_cols].astype(float)
    return {idx: row.values for idx, row in mat.iterrows()}

feat_map_train = build_feature_map(samples_train, SCALED_COLS)
feat_map_valid = build_feature_map(samples_valid, SCALED_COLS)
feat_map_test = build_feature_map(samples_test, SCALED_COLS)

# pair -> squared diff matrix
def pair_to_Z(pair_df, feat_map):
    rows, ys, keep_idx = [], [], []
    for i, row in pair_df.iterrows():
        li = int(row["left_point_id"])
        ri = int(row["right_point_id"])
        if li not in feat_map or ri not in feat_map:
            continue
        xi = feat_map[li]
        xj = feat_map[ri]
        z = (xi - xj) ** 2
        rows.append(z); ys.append(float(row["y"])); keep_idx.append(i)
    Z = np.vstack(rows) if rows else np.zeros((0, len(WL_FEATURE_COLS)))
    y = np.array(ys, dtype=float)
    kept = pair_df.loc[keep_idx].copy().reset_index(drop=True) if keep_idx else pair_df.iloc[:0].copy()
    return Z, y, kept

Z_train, y_train, pairs_train_used = pair_to_Z(pairs_train, feat_map_train)
Z_valid, y_valid, pairs_valid_used = pair_to_Z(pairs_valid, feat_map_valid)
Z_test, y_test, pairs_test_used = pair_to_Z(pairs_test, feat_map_test)

if len(Z_train) == 0:
    raise RuntimeError("train pair가 0개입니다.")

# equal baseline
w_equal = np.ones(len(WL_FEATURE_COLS), dtype=float) / len(WL_FEATURE_COLS)
feature_diag_table = feature_gap_diagnostic(Z_train, y_train, WL_FEATURE_COLS)
margin_equal = choose_margin_from_equal(Z_train, y_train, w_equal)
selected_margin = margin_equal if WL_USE_DATA_DRIVEN_MARGIN else 12.0

# initialization
gap_arr = feature_diag_table.set_index("feature").loc[WL_FEATURE_COLS, "gap_neg_minus_pos_feature"].fillna(0.0).values
gap_std = float(np.std(gap_arr))
alpha_base = (gap_arr - np.mean(gap_arr)) / gap_std if gap_std >= 1e-12 else np.zeros(len(WL_FEATURE_COLS), dtype=float)

# multi-start optimization
best_obj, best_alpha, best_w, best_history, best_seed = None, None, None, None, None
restart_rows = []

def run_one_restart(seed, alpha_base, Z_tr, y_tr, Z_va, y_va):
    rng = np.random.default_rng(seed)
    alpha0 = alpha_base + rng.normal(0.0, WL_INIT_NOISE, size=len(alpha_base))
    local_history = []
    def obj(alpha):
        total, parts = simplex_metric_loss(alpha, Z_tr, y_tr, selected_margin, WL_NEG_HINGE_WEIGHT, WL_L2_SIMPLEX)
        wn = softmax_logits(alpha)
        local_history.append({"seed": seed, "total_loss": parts["total_loss"], "pos_loss": parts["pos_loss"],
                              "neg_hinge_loss": parts["neg_hinge_loss"], "l2_loss": parts["l2_loss"],
                              "w_max": float(wn.max()), "w_min": float(wn.min())})
        return total
    res = minimize(obj, alpha0, method="L-BFGS-B", options={"maxiter": 300})
    wh = softmax_logits(res.x)
    tr_ev = eval_pairs(Z_tr, y_tr, wh)
    va_ev = eval_pairs(Z_va, y_va, wh) if len(Z_va) > 0 else {"gap_neg_minus_pos": np.nan, "auc_pair_distance": np.nan}
    sc = va_ev["gap_neg_minus_pos"]; sc = sc if np.isfinite(sc) else tr_ev["gap_neg_minus_pos"]
    summary = {"seed": seed, "success": bool(res.success), "score_to_maximize": sc,
               "train_gap": tr_ev["gap_neg_minus_pos"], "valid_gap": va_ev["gap_neg_minus_pos"]}
    for j, f in enumerate(WL_FEATURE_COLS):
        summary[f"w__{f}"] = float(wh[j])
    return res, res.x, wh, pd.DataFrame(local_history), summary

for s in WL_MULTI_START_SEEDS:
    res_i, al_i, wi, hi, si = run_one_restart(s, alpha_base, Z_train, y_train, Z_valid, y_valid)
    restart_rows.append(si)
    cur_sc = si["score_to_maximize"]
    if best_obj is None or (np.isfinite(cur_sc) and cur_sc > best_obj):
        best_obj, best_alpha, best_w, best_history, best_seed = cur_sc, al_i, wi, hi, s

restart_summary = pd.DataFrame(restart_rows).sort_values("score_to_maximize", ascending=False).reset_index(drop=True)
w = best_w.copy()
weight_table = pd.DataFrame({"feature": WL_FEATURE_COLS, "weight": w}).sort_values("weight", ascending=False).reset_index(drop=True)

print(f"\n  Best seed: {best_seed}")
print(weight_table.to_string(index=False))


# ============================================================
# BLOCK 6-E) APPLY LEARNED WEIGHTS → POINT SCORE TABLE
# ============================================================
print("\n" + "=" * 70)
print("BLOCK 6-E) APPLYING LEARNED WEIGHTS TO ALL POINTS...")

point_features_scored = point_features.copy()
scaled_full = robust_scale_transform(
    point_features_scored[WL_FEATURE_COLS].fillna(0.0), med, mad
).fillna(0.0)
for c in WL_FEATURE_COLS:
    point_features_scored[f"scaled__{c}"] = scaled_full[c].values

point_features_scored["wl_linear_score"] = (
    point_features_scored[SCALED_COLS].to_numpy(float) @ w
)

# train strong centroid distance
train_pos_ids_set = set(pos_samples[pos_samples["split"] == "train"]["point_id"].tolist())
train_pos_full = point_features_scored[point_features_scored["point_id"].isin(train_pos_ids_set)].copy()
if len(train_pos_full) > 0:
    center = train_pos_full[SCALED_COLS].median().to_numpy(float)
    X_all = point_features_scored[SCALED_COLS].to_numpy(float)
    diff = X_all - center[None, :]
    point_features_scored["wl_dist_to_strong_center"] = np.sum((diff ** 2) * w[None, :], axis=1)
else:
    point_features_scored["wl_dist_to_strong_center"] = np.nan

# save scored table
point_features_scored.to_parquet(WL_OUT_DIR / "WL_POINT_FEATURES_SCORED_NODE.parquet", index=False)
print(f"  Scored table saved: {point_features_scored.shape}")
print(f"  Level distribution:\n{point_features_scored['level'].value_counts()}")


# ============================================================
# BLOCK 7-C) FINAL EPISODE EXTRACTION (with level!)
# ============================================================
print("\n" + "=" * 70)
print("BLOCK 7-C) FINAL EPISODE EXTRACTION...")

FINAL_OUT_DIR = WL_OUT_DIR / "final_hotspot_pipeline"
FINAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_SIGMA_K = 2.5
FINAL_MIN_DURATION_BINS = 2
FINAL_GAP_ALLOW_BINS = 2

# score 계산을 위해 gini/equal/learned 세 방식의 점수 만들기
final_weight_df = weight_table.rename(columns={"weight": "weight"}).copy()

# gini score: raw_imb_gini 값 그대로 사용
gini_col = "raw_imb_gini" if "raw_imb_gini" in point_features_scored.columns else WL_FEATURE_COLS[0]

# equal score: 모든 feature 동일 가중치
# learned score: 이미 wl_linear_score로 계산됨

# point_score_df 구성
# 🆕 split은 point_features에 없으므로 나중에 시간 기반으로 부여
# 🆕 entity_id 포함 (노드명/포트번호)
_base_cols = ["timestamp", "stg_ip", "level", "entity_id", "rw", "src_metric",
              "wl_linear_score", gini_col, "has_hot_label"]
_avail_cols = [c for c in _base_cols if c in point_features_scored.columns]
point_score_df = point_features_scored[_avail_cols + [c for c in point_features_scored.columns if c.startswith("scaled__")]].copy()

point_score_df["score_learned"] = point_score_df["wl_linear_score"]

# equal score: scaled feature 평균
scaled_feat_cols = [c for c in point_score_df.columns if c.startswith("scaled__")]
if scaled_feat_cols:
    point_score_df["score_equal"] = point_score_df[scaled_feat_cols].mean(axis=1)
else:
    point_score_df["score_equal"] = 0.0

# gini score: raw_imb_gini (robust-scaled)
if f"scaled__raw_imb_gini" in point_score_df.columns:
    point_score_df["score_gini"] = point_score_df["scaled__raw_imb_gini"]
elif gini_col in point_score_df.columns:
    point_score_df["score_gini"] = point_score_df[gini_col]
else:
    point_score_df["score_gini"] = 0.0

# split 정보가 없으면 시간으로 나누기
if "split" not in point_score_df.columns or point_score_df["split"].isna().all():
    time_vals = pd.to_datetime(point_score_df["timestamp"], errors="coerce")
    q70 = time_vals.quantile(0.70)
    q85 = time_vals.quantile(0.85)
    point_score_df["split"] = pd.cut(time_vals, bins=[time_vals.min(), q70, q85, time_vals.max()],
                                     labels=["train", "valid", "test"], include_lowest=True).astype(str)

# original hotspot flag
point_score_df["original_hot"] = point_score_df["has_hot_label"].fillna(0).astype(int)

# episode entity: stg_ip|level|entity_id|rw|src_metric
# 🆕 entity_id 포함 → 실제 노드명/포트번호 식별 가능
point_score_df["episode_entity"] = (
    point_score_df["stg_ip"].astype(str) + "|" +
    point_score_df["level"].astype(str) + "|" +
    point_score_df["entity_id"].astype(str) + "|" +
    point_score_df["rw"].astype(str) + "|" +
    point_score_df["src_metric"].astype(str)
)

# frequency inference
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
print(f"  FREQ_MIN={FREQ_MIN}, ALLOW_GAP_MIN={ALLOW_GAP_MIN}")
print(f"  Total points: {len(point_score_df)}, Levels: {point_score_df['level'].value_counts().to_dict()}")

# threshold 함수
def make_threshold_from_train(df, score_col, sigma_k=2.5):
    base = df.loc[df["split"] == "train", score_col].dropna().astype(float)
    if len(base) == 0:
        raise ValueError(f"No train baseline found for {score_col}")
    mu = float(base.mean()); sd = float(base.std(ddof=0))
    ucl = mu + sigma_k * sd
    return mu, sd, ucl

# episode extraction 함수
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
                "entity_id": str(peak_row.get("entity_id", "")).lower(),  # 🆕 노드명/포트번호
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

# 세 방식 모두 실행
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
        point_score_df, score_col=score_col, hot_col=hot_col,
        entity_col="episode_entity", time_col="timestamp",
        min_bins=FINAL_MIN_DURATION_BINS, gap_allow_min=ALLOW_GAP_MIN
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

print("\n" + "=" * 70)
print("EPISODE SUMMARY (node + port)")
print("=" * 70)
print(episode_summary.to_string(index=False))
print(f"\n  Level breakdown:")
if not episode_df.empty:
    print(episode_df.groupby(['method', 'level']).size().unstack(fill_value=0))

# culprit features
def add_peak_culprit_features(ep_df, score_df, feature_cols, method_name):
    if ep_df.empty:
        return ep_df.copy()
    out = ep_df.copy()

    if method_name == "gini":
        weight_map = {f: 0.0 for f in feature_cols}
        gc = "raw_imb_gini" if "raw_imb_gini" in feature_cols else feature_cols[0]
        weight_map[gc] = 1.0
    elif method_name == "equal":
        weight_map = {f: 1.0 / len(feature_cols) for f in feature_cols}
    elif method_name == "learned":
        weight_map = dict(zip(final_weight_df["feature"], final_weight_df["weight"]))
    else:
        weight_map = {f: 0.0 for f in feature_cols}

    top1_list, top2_list = [], []
    for _, r in out.iterrows():
        m = ((score_df["episode_entity"] == r["episode_entity"]) &
             (pd.to_datetime(score_df["timestamp"]) == pd.to_datetime(r["peak_ts"])))
        gp = score_df.loc[m]
        if gp.empty:
            top1_list.append(None); top2_list.append(None); continue
        row = gp.iloc[0]; contrib = []
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

# 저장
point_score_df.to_parquet(FINAL_OUT_DIR / "POINT_SCORE_WITH_HOT_FLAGS.parquet", index=False)
episode_df.to_parquet(FINAL_OUT_DIR / "EPISODES_FINAL.parquet", index=False)
episode_summary.to_excel(FINAL_OUT_DIR / "EPISODE_SUMMARY_FINAL.xlsx", index=False)

with pd.ExcelWriter(FINAL_OUT_DIR / "EPISODE_DETAIL_FINAL.xlsx", engine="openpyxl") as writer:
    episode_summary.to_excel(writer, sheet_name="SUMMARY", index=False)
    episode_df.to_excel(writer, sheet_name="EPISODES", index=False)

print("\n" + "=" * 70)
print("DONE! ✅ Final results saved to:", FINAL_OUT_DIR)
print(f"\n  EPISODES_FINAL.parquet: {episode_df.shape[0]} episodes")
if not episode_df.empty:
    print(f"  By level:\n{episode_df['level'].value_counts()}")
    print(f"  By method:\n{episode_df['method'].value_counts()}")
print("=" * 70)
