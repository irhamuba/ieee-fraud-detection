"""
IEEE-CIS Fraud Detection — FULL pipeline, Kaggle GPU-ready (v2)
================================================================
- 100% data dipakai: train 590.540 + test 506.691 + identity train 144.233 + identity test 141.907
- Split validasi berdasarkan waktu (5 blok waktu), sesuai ancaman utama dataset ini
- Model: LightGBM (CPU) + XGBoost (GPU via device="cuda") — kooperatif recommendation
- CatBoost bisa ditambahkan nanti, terpisah, commanding GPU sama
- Run-time: 2–4 jam di Kaggle P100/T4x2

CARA PAKAI DI KAGGLE:
1. Notebook baru, copy seluruh isi file ini ke 1 cell ATAU add Utility Script
2. Add Input: kompetisi "ieee-fraud-detection"
3. Settings -> Accelerator -> GPU T4 x2 (atau P100)
4. Save & Run All
Output: /kaggle/working/submission.csv
"""

import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

pd.set_option("display.max_columns", 500)

import os
DATA_DIR = None
for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
    if "train_transaction.csv" in filenames:
        DATA_DIR = dirpath
        break
print("DATA_DIR =", DATA_DIR)
if DATA_DIR is None:
    raise FileNotFoundError(
        f"train_transaction.csv tidak ketemu di /kaggle/input. Isi /kaggle/input: {os.listdir('/kaggle/input')}"
    )
N_FOLDS = 5
SEED = 42

# ============================================================
# 1. LOAD FULL DATA (train+test digabung, agar freq-encoding konsisten)
# ============================================================
print("1) Loading full data ...")
train_tx = pd.read_csv(f"{DATA_DIR}/train_transaction.csv", index_col="TransactionID")
test_tx = pd.read_csv(f"{DATA_DIR}/test_transaction.csv", index_col="TransactionID")
train_id = pd.read_csv(f"{DATA_DIR}/train_identity.csv", index_col="TransactionID")
test_id = pd.read_csv(f"{DATA_DIR}/test_identity.csv", index_col="TransactionID")

test_id.columns = [c.replace("-", "_") for c in test_id.columns]
print("    train_tx:", train_tx.shape, " test_tx:", test_tx.shape)

# --- SIMPAN LABEL SEBELUM concat (punya NaN) ---
y_raw = train_tx["isFraud"].astype(np.int8).values
train_tx = train_tx.drop(columns=["isFraud"])
test_tx["isFraud"] = np.nan
all_tx = pd.concat([train_tx, test_tx], sort=False).reset_index().rename(columns={"index": "_row"})
all_tx["TransactionID"] = np.concatenate(
    [train_tx.index.values, test_tx.index.values]
)
train_tx = None; test_tx = None; gc.collect()
print("    all_tx:", all_tx.shape, " y kept:", y_raw.shape)

# ============================================================
# 2. FEATURES
# ============================================================
print("2) Feature engineering ...")
# --- waktu ---
all_tx["DT_min"]  = all_tx.TransactionDT // 60
all_tx["hour"]    = (all_tx.TransactionDT // 3600) % 24
all_tx["dow"]     = (all_tx.TransactionDT // 86400) % 7
all_tx["week_dt"] = all_tx.TransactionDT // (7 * 86400)
all_tx["DT_inter"] = all_tx.TransactionDT % 3600    # detik sisa setelah menit penuh

# --- amount ---
all_tx["Amt_log"] = np.log1p(all_tx.TransactionAmt)
all_tx["Amt_decimal"] = np.mod(np.round(all_tx.TransactionAmt * 1000, 1), 1000).astype(np.int16)
all_tx["Amt_decimal2"] = all_tx.TransactionAmt - all_tx.TransactionAmt.astype(np.int32)

# --- kombinasi kartu = "client" ---
all_tx["card1_addr1"] = all_tx.card1.astype(np.int32).astype(str) + "_" + all_tx.addr1.fillna(-1).astype(np.int32).astype(str)
all_tx["has_addr"] = all_tx.addr1.notna().astype(np.int8)
all_tx["card1_addr1_card4"] = all_tx.card1_addr1 + "_" + all_tx.card4.fillna("na").astype(str)
all_tx["card_id1"] = all_tx.card1.astype(str) + "_" + all_tx.card2.fillna(-1).astype(str)
all_tx["card_id2"] = all_tx.card1.astype(str) + "_" + all_tx.card3.fillna(-1).astype(str) + "_" + all_tx.card5.fillna(-1).astype(str)

# --- email root ---
for col in ["P_emaildomain", "R_emaildomain"]:
    root = all_tx[col].fillna("none").astype(str).str.split(".").str[0]
    sfx = all_tx[col].fillna("none").astype(str).str.split(".").str[-1]
    all_tx[f"{col}_root"] = root
    all_tx[f"{col}_sfx"] = sfx
print("    done.")
gc.collect()

# ============================================================
# 3. AGGREGATIONS (dilatih train+test gabungan, aman dari leakage)
# ============================================================
print("3) Aggregations ...")
AGG_KEYS = ["card1", "card1_addr1", "card_id1", "card_id2", "card1_addr1_card4",
            "P_emaildomain_root", "R_emaildomain_root", "addr1"]
for key in AGG_KEYS:
    g = all_tx.groupby(key, dropna=False)
    all_tx[f"{key}_freq"] = g.TransactionID.transform("size")
    agg_dict = g["TransactionAmt"].agg(["mean", "std"])
    all_tx[f"{key}_amt_mean"] = all_tx[key].map(agg_dict["mean"])
    all_tx[f"{key}_amt_std"]  = all_tx[key].map(agg_dict["std"])
    agg2 = g["hour"].mean()   # mode lambat di jutaan grup; mean lebih aman waktu
    all_tx[f"{key}_hour_mean"] = all_tx[key].map(agg2)
    del g, agg_dict, agg2; gc.collect()

# --- identity features di bagian lain ---
all_id = pd.concat([train_id, test_id], sort=False)
all_id["DeviceInfo"] = all_id.DeviceInfo.fillna("none").astype(str).str.replace(r"[^a-zA-Z0-9]", "", regex=True)
all_id["OS_clean"] = all_id.id_30.fillna("none").astype(str).str.replace(r"\s", "_", regex=True).str.replace(r"[^\w]+\Z", "", regex=True)
all_id["id_31_clean"] = all_id.id_31.fillna("none").astype(str).str.split(" ").str[0]
all_id["id_33_num"] = all_id.id_33.fillna("0x0").astype(str).str.lower().str.replace("x", "0").astype("float32")
train_id = None; test_id = None; gc.collect()

all_tx = all_tx.merge(all_id, on="TransactionID", how="left")
all_id = None; gc.collect()

# ============================================================
# 4. Kategorikal + Object -> category dtype
# ============================================================
print("4) Encoding categories ...")
cat_cols = [
    c for c in all_tx.columns
    if all_tx[c].dtype == "object" or str(all_tx[c].dtype) == "str"
]
for c in cat_cols:
    all_tx[c] = all_tx[c].fillna("none").astype("category")
print("    cat columns:", len(cat_cols))

gc.collect()

# ============================================================
# 5. Split waktu (5 blok kronologis), label disisipkan KEMBALI
# ============================================================
train_df = all_tx.iloc[:len(y_raw)].copy()
train_df["isFraud"] = y_raw
test_df = all_tx.iloc[len(y_raw):].drop(columns=["isFraud"]).copy()
del all_tx
gc.collect()

tds = train_df.TransactionDT.values
sorted_idx = np.argsort(tds)
folds = np.array_split(sorted_idx, N_FOLDS)

print("5) Time folds ready:",
      [len(f) for f in folds],
      " train:", train_df.shape, " test:", test_df.shape)

# ============================================================
# 6. MODEL TRAINING (LightGBM + XGBoost GPU)
# ============================================================
params_lgb = {
    "n_estimators": 5000,
    "learning_rate": 0.007,
    "num_leaves": 255,
    "max_depth": 11,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.55,
    "reg_alpha": 0.9,
    "reg_lambda": 6,
    "min_split_gain": 0.06,
    "min_child_weight": 39,
    "objective": "binary",
    "metric": "auc",
    "n_jobs": -1,
    "seed": SEED,
    "verbose": -1,
}

params_xgb = {
    "n_estimators": 5000,
    "learning_rate": 0.007,
    "tree_method": "hist",
    "device": "cuda",           # GPU; ubah ke "cpu" untuk laptop
    "max_depth": 12,
    "subsample": 0.75,
    "colsample_bytree": 0.5,
    "min_child_weight": 110,
    "gamma": 0.03,
    "alpha": 0.06,
    "lambda": 5.1,
    "eval_metric": "auc",
    "objective": "binary:logistic",
    "early_stopping_rounds": 90,   # XGBoost 3.x: early stop pindah ke constructor
    "enable_categorical": True,
    "seed": SEED,
}

categorical_cols_idxs = [train_df.columns.get_loc(c) for c in cat_cols]

oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
pred_lgb = np.zeros(len(test_df))
pred_xgb = np.zeros(len(test_df))

y = train_df["isFraud"].values.astype(np.int8)
X_full = train_df.drop(columns=["isFraud"])
X_test_full = test_df

for f_i in range(N_FOLDS):
    valid_pos = folds[f_i]
    train_pos = np.concatenate([folds[j] for j in range(N_FOLDS) if j != f_i])

    X_trn, y_trn = X_full.iloc[train_pos], y[train_pos]
    X_vld, y_vld = X_full.iloc[valid_pos], y[valid_pos]
    print(f"\n===== FOLD {f_i+1}/{N_FOLDS} =====  train={len(X_trn):,}  val={len(X_vld):,}")

    # ---------- LightGBM (CPU) ----------
    m_lgb = lgb.LGBMClassifier(**params_lgb)
    m_lgb.fit(
        X_trn, y_trn,
        eval_set=[(X_vld, y_vld)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(90, verbose=False)],
    )
    oof_lgb[valid_pos] = m_lgb.predict_proba(X_vld)[:, 1]
    auc_l = roc_auc_score(y_vld, oof_lgb[valid_pos])
    pred_lgb += m_lgb.predict_proba(test_df)[:, 1] / N_FOLDS
    print(f"    LGB  fold-AUC = {auc_l:.5f}")
    del m_lgb; gc.collect()

    # ---------- XGBoost (GPU) ----------
    m_xgb = xgb.XGBClassifier(**params_xgb)
    m_xgb.fit(
        X_trn, y_trn,
        eval_set=[(X_vld, y_vld)],
        verbose=False,
    )
    oof_xgb[valid_pos] = m_xgb.predict_proba(X_vld)[:, 1]
    auc_x = roc_auc_score(y_vld, oof_xgb[valid_pos])
    pred_xgb += m_xgb.predict_proba(test_df)[:, 1] / N_FOLDS
    print(f"    XGB  fold-AUC = {auc_x:.5f}")
    del m_xgb, X_trn, X_vld, y_trn, y_vld
    gc.collect()

print("\n================ INNER-AUC ================")
print("  OOF LGB :", roc_auc_score(y, oof_lgb))
print("  OOF XGB :", roc_auc_score(y, oof_xgb))
print("  OOF mean:", roc_auc_score(y, (oof_lgb + oof_xgb) / 2))

# ============================================================
# 7. SUBMISSION
# ============================================================
sub = pd.DataFrame({
    "TransactionID": test_df["TransactionID"].values,
    "isFraud": 0.5 * pred_lgb + 0.5 * pred_xgb,
})
sub.to_csv("submission.csv", index=False)
print("\nSaved submission.csv", sub.shape)
