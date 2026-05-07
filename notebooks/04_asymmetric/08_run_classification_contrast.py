#!/usr/bin/env python
# coding: utf-8


"""
Predictability Contrast: SFI vs SSN burst classification using
planetary features informed by CEOS findings.

修改说明 (2026-03):
- 用 TimeSeriesSplit CV 在训练集内选最佳配置，测试集只做最终评估
- 加入置换检验 (permutation test) 计算 p 值
- 移除按 test_auc 挑 best config 的数据泄漏

Output: results summary + CSV for plotting.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
from sklearn.model_selection import TimeSeriesSplit
import ephem
import os
import json
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

print("=" * 70)
print("  SFI vs SSN Burst Predictability Contrast (CEOS-Informed)")
print("=" * 70)

BASE = str(Path(__file__).resolve().parents[2]) + '/'
os.makedirs(os.path.join(BASE, 'results', '04_asymmetric'), exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1] Loading data...")
sfi = pd.read_csv(os.path.join(BASE, 'data/ready/sfi_1975-2024.csv'))
sfi['date'] = pd.to_datetime(sfi['date'])

ssn = pd.read_csv(os.path.join(BASE, 'data/ready/ssn_daily_1849_2025.csv'))
ssn['date'] = pd.to_datetime(ssn['date'])

df = pd.merge(sfi, ssn, on='date', how='inner')
df = df[(df['date'] >= '1976-01-01') & (df['date'] <= '2024-12-31')].copy()
df.sort_values('date', inplace=True)
df.reset_index(drop=True, inplace=True)
df['sfi'] = df['sfi'].fillna(0)
df['ssn'] = df['ssn'].fillna(0)
print(f"  Data: {df['date'].min().date()} to {df['date'].max().date()}, {len(df)} days")

# ============================================================
# 2. DEFINE BURST TARGETS (Y)
# ============================================================
print("\n[2] Defining burst events...")

TRAIN_END = pd.Timestamp('2010-12-31')

for col in ['sfi', 'ssn']:
    rm = df[col].shift(1).rolling(window=7, min_periods=1).mean()
    rs = df[col].shift(1).rolling(window=7, min_periods=1).std().fillna(0)
    z_thresh_2 = rm + 2.0 * rs
    min_val = df.loc[df['date'] <= TRAIN_END, col].quantile(0.50)
    df[f'{col}_burst_2s'] = ((df[col] > z_thresh_2) & (df[col] > min_val)).astype(int)
    z_thresh_15 = rm + 1.5 * rs
    df[f'{col}_burst_15s'] = ((df[col] > z_thresh_15) & (df[col] > min_val)).astype(int)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

for col in ['sfi', 'ssn']:
    for s in ['2s', '15s']:
        n = df[f'{col}_burst_{s}'].sum()
        pct = df[f'{col}_burst_{s}'].mean()
        print(f"  {col.upper()} burst ({s}): {n} days ({pct:.2%})")

# ============================================================
# 3. COMPUTE PLANETARY FEATURES
# ============================================================
print("\n[3] Computing planetary positions...")

planets = {
    'Mercury': ephem.Mercury(),
    'Venus':   ephem.Venus(),
    'Mars':    ephem.Mars(),
    'Jupiter': ephem.Jupiter(),
    'Saturn':  ephem.Saturn(),
    'Uranus':  ephem.Uranus(),
    'Neptune': ephem.Neptune(),
}
planet_names = list(planets.keys())
n = len(df)

r_mat = {p: np.zeros(n) for p in planet_names}
lon_mat = {p: np.zeros(n) for p in planet_names}

for i, date_obj in enumerate(df['date']):
    d_ephem = ephem.Date(date_obj)
    for pname, pobj in planets.items():
        pobj.compute(d_ephem)
        r_mat[pname][i] = pobj.sun_distance
        lon_mat[pname][i] = float(pobj.hlon)

for p in planet_names:
    df[f'r_{p}'] = r_mat[p]
    df[f'cos_l_{p}'] = np.cos(lon_mat[p])
    df[f'sin_l_{p}'] = np.sin(lon_mat[p])

for i in range(len(planet_names)):
    for j in range(i + 1, len(planet_names)):
        p1, p2 = planet_names[i], planet_names[j]
        diff = lon_mat[p1] - lon_mat[p2]
        df[f'cos_{p1}_{p2}'] = np.cos(diff)
        df[f'sin_{p1}_{p2}'] = np.sin(diff)

# ============================================================
# 4. DEFINE FEATURE SETS
# ============================================================
ceos_top = ['Venus', 'Mars', 'Jupiter']
feature_sets = {}

feat_A = []
for p in ceos_top:
    feat_A.extend([f'r_{p}', f'cos_l_{p}', f'sin_l_{p}'])
feature_sets['A_top3_pos'] = feat_A

feat_B = []
for p in planet_names:
    feat_B.extend([f'r_{p}', f'cos_l_{p}', f'sin_l_{p}'])
feature_sets['B_all7_pos'] = feat_B

feat_C = list(feat_A)
for i in range(len(ceos_top)):
    for j in range(i + 1, len(ceos_top)):
        p1, p2 = ceos_top[i], ceos_top[j]
        feat_C.extend([f'cos_{p1}_{p2}', f'sin_{p1}_{p2}'])
feature_sets['C_top3_full'] = feat_C

feat_D = list(feat_B)
for i in range(len(planet_names)):
    for j in range(i + 1, len(planet_names)):
        p1, p2 = planet_names[i], planet_names[j]
        feat_D.extend([f'cos_{p1}_{p2}', f'sin_{p1}_{p2}'])
feature_sets['D_all7_full'] = feat_D

for name, cols in feature_sets.items():
    print(f"  Feature set {name}: {len(cols)} features")

# ============================================================
# 5. MODEL PARAMS
# ============================================================
LGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=42,
    n_jobs=4,
    verbose=-1,
)

N_CV_SPLITS = 5
N_PERM = 1000
PERM_SEED = 123

# ============================================================
# 6. STEP 1 — CV 选最佳配置 (仅在训练集内部)
# ============================================================
print("\n[5] Selecting best config via TimeSeriesSplit CV (train set only)...")

train_mask = df['date'] <= TRAIN_END
test_mask = df['date'] > TRAIN_END
train_df = df[train_mask].copy()
test_df = df[test_mask].copy()

tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
cv_results = []

for burst_type in ['2s', '15s']:
    for target_name in ['sfi', 'ssn']:
        target_col = f'{target_name}_burst_{burst_type}'
        y_all_train = train_df[target_col].values

        for feat_name, feat_cols in feature_sets.items():
            X_all_train = train_df[feat_cols].values
            fold_aucs = []

            for fold_train_idx, fold_val_idx in tscv.split(X_all_train):
                X_tr, X_val = X_all_train[fold_train_idx], X_all_train[fold_val_idx]
                y_tr, y_val = y_all_train[fold_train_idx], y_all_train[fold_val_idx]

                if y_tr.sum() == 0 or y_val.sum() == 0:
                    continue

                spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
                model = lgb.LGBMClassifier(scale_pos_weight=spw, **LGB_PARAMS)
                model.fit(X_tr, y_tr)
                y_prob = model.predict_proba(X_val)[:, 1]
                fold_aucs.append(roc_auc_score(y_val, y_prob))

            mean_cv_auc = np.mean(fold_aucs) if fold_aucs else 0.5
            cv_results.append({
                'target': target_name,
                'burst_threshold': burst_type,
                'feature_set': feat_name,
                'n_features': len(feat_cols),
                'cv_auc_mean': mean_cv_auc,
                'cv_auc_std': np.std(fold_aucs) if fold_aucs else 0,
            })
            print(f"  {target_name}/{burst_type}/{feat_name:15s}  CV AUC = {mean_cv_auc:.4f} ± {cv_results[-1]['cv_auc_std']:.4f}")

cv_df = pd.DataFrame(cv_results)

# 按 target 分别选 CV 最优配置
best_configs = {}
for target_name in ['sfi', 'ssn']:
    sub = cv_df[cv_df['target'] == target_name]
    best_row = sub.sort_values('cv_auc_mean', ascending=False).iloc[0]
    best_configs[target_name] = (best_row['burst_threshold'], best_row['feature_set'])
    print(f"\n  Best CV config for {target_name.upper()}: "
          f"burst={best_row['burst_threshold']}, feat={best_row['feature_set']}, "
          f"CV AUC={best_row['cv_auc_mean']:.4f}")

# ============================================================
# 7. STEP 2 — 用 CV 选定的配置在测试集做最终评估 + 置换检验
# ============================================================
print(f"\n[6] Final evaluation on test set + permutation test (N_perm={N_PERM})...")

rng_perm = np.random.default_rng(PERM_SEED)
final_results = []
all_roc_data = {}

for target_name, target_label in [('sfi', 'SFI (Flares)'), ('ssn', 'SSN (Sunspots)')]:
    best_bt, best_fs = best_configs[target_name]
    target_col = f'{target_name}_burst_{best_bt}'
    feat_cols = feature_sets[best_fs]

    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    X_train = train_df[feat_cols].values
    X_test = test_df[feat_cols].values

    spw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    # 真实模型
    model = lgb.LGBMClassifier(scale_pos_weight=spw, **LGB_PARAMS)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    real_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr)

    # 置换检验: 打乱训练标签, 重新训练, 测试集评估
    print(f"\n  {target_label} ({best_fs}, {best_bt}): real AUC = {real_auc:.4f}")
    print(f"    Running {N_PERM} permutations...", end='', flush=True)
    null_aucs = np.zeros(N_PERM)
    for i in range(N_PERM):
        y_perm = rng_perm.permutation(y_train)
        spw_p = (len(y_perm) - y_perm.sum()) / max(y_perm.sum(), 1)
        m_perm = lgb.LGBMClassifier(scale_pos_weight=spw_p, **LGB_PARAMS)
        m_perm.fit(X_train, y_perm)
        yp = m_perm.predict_proba(X_test)[:, 1]
        null_aucs[i] = roc_auc_score(y_test, yp)
        if (i + 1) % 200 == 0:
            print(f" {i+1}", end='', flush=True)

    p_perm = (np.sum(null_aucs >= real_auc) + 1) / (N_PERM + 1)
    print(f"\n    Permutation p = {p_perm:.4f}  (null mean={null_aucs.mean():.4f}, std={null_aucs.std():.4f})")

    # Feature importance
    imp = pd.DataFrame({'Feature': feat_cols, 'Importance': model.feature_importances_})
    imp = imp.sort_values('Importance', ascending=False).head(8)
    print(f"    Top features:")
    for _, row in imp.iterrows():
        print(f"      {row['Feature']:20s}: {row['Importance']}")

    y_pred_labels = (y_proba > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred_labels, zero_division=0)

    final_results.append({
        'target': target_name,
        'burst_threshold': best_bt,
        'feature_set': best_fs,
        'n_features': len(feat_cols),
        'cv_auc': cv_df[(cv_df['target'] == target_name) &
                        (cv_df['burst_threshold'] == best_bt) &
                        (cv_df['feature_set'] == best_fs)].iloc[0]['cv_auc_mean'],
        'test_auc': real_auc,
        'test_f1': f1,
        'perm_p': p_perm,
        'perm_null_mean': null_aucs.mean(),
        'perm_null_std': null_aucs.std(),
        'label': f"{target_label} {best_bt} {best_fs}"
    })

    key = f"{target_name}_{best_bt}_{best_fs}"
    all_roc_data[key] = (fpr, tpr, roc_auc_val)

# ============================================================
# 8. ALSO RUN ALL 16 COMBOS ON TEST SET (for full CSV, no model selection)
# ============================================================
print("\n[7] Running all 16 combos on test set (for reference CSV)...")
all_results = []
all_roc_data_full = {}

for burst_type in ['2s', '15s']:
    for target_name, target_label in [('sfi', 'SFI (Flares)'), ('ssn', 'SSN (Sunspots)')]:
        target_col = f'{target_name}_burst_{burst_type}'
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values

        for feat_name, feat_cols in feature_sets.items():
            X_train = train_df[feat_cols].values
            X_test = test_df[feat_cols].values

            if y_train.sum() == 0 or y_test.sum() == 0:
                continue

            spw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
            model = lgb.LGBMClassifier(scale_pos_weight=spw, **LGB_PARAMS)
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)

            # 查找对应的 CV AUC
            cv_row = cv_df[(cv_df['target'] == target_name) &
                           (cv_df['burst_threshold'] == burst_type) &
                           (cv_df['feature_set'] == feat_name)]
            cv_auc_val = cv_row.iloc[0]['cv_auc_mean'] if len(cv_row) > 0 else np.nan

            all_results.append({
                'target': target_name,
                'burst_threshold': burst_type,
                'feature_set': feat_name,
                'n_features': len(feat_cols),
                'cv_auc': cv_auc_val,
                'test_auc': test_auc,
                'label': f"{target_label} {burst_type} {feat_name}"
            })

            key = f"{target_name}_{burst_type}_{feat_name}"
            all_roc_data_full[key] = (fpr, tpr, auc(fpr, tpr))

# ============================================================
# 9. SAVE RESULTS
# ============================================================
res_df = pd.DataFrame(all_results)
res_path = os.path.join(BASE, 'results', '04_asymmetric', '08_classification_results.csv')
res_df.to_csv(res_path, index=False)
print(f"\nAll results saved to: {res_path}")

# Best config ROC data
best_bt_sfi, best_fs_sfi = best_configs['sfi']
roc_save = {}
for target in ['sfi', 'ssn']:
    bt, fs = best_configs[target]
    key = f"{target}_{bt}_{fs}"
    if key in all_roc_data:
        fpr, tpr, roc_auc_val = all_roc_data[key]
        roc_save[target] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc_val}

roc_path = os.path.join(BASE, 'results', '04_asymmetric', '08_best_roc_data.json')
with open(roc_path, 'w') as f:
    json.dump({
        'config': {t: {'feature_set': best_configs[t][1], 'burst_threshold': best_configs[t][0]}
                   for t in ['sfi', 'ssn']},
        'roc': roc_save,
        'permutation': {r['target']: {'p': r['perm_p'], 'null_mean': r['perm_null_mean'],
                                       'null_std': r['perm_null_std']}
                        for r in final_results}
    }, f)
print(f"Best ROC data saved to: {roc_path}")

# All ROC data
all_roc_save = {}
for key, (fpr, tpr, roc_auc_val) in all_roc_data_full.items():
    all_roc_save[key] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc_val}
all_roc_path = os.path.join(BASE, 'results', '04_asymmetric', '08_all_roc_data.json')
with open(all_roc_path, 'w') as f:
    json.dump(all_roc_save, f)

# ============================================================
# 10. FINAL SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("  FINAL SUMMARY")
print("=" * 70)
print(f"  {'Target':<6} {'Config':>20}  {'CV AUC':>8} {'Test AUC':>9} {'Perm p':>8}")
print("-" * 60)
for r in final_results:
    cfg = f"{r['feature_set']}/{r['burst_threshold']}"
    sig = '*' if r['perm_p'] < 0.05 else ''
    print(f"  {r['target'].upper():<6} {cfg:>20}  {r['cv_auc']:>8.4f} {r['test_auc']:>9.4f} {r['perm_p']:>7.4f}{sig}")

print(f"\n--- All 16 combos (test AUC, for reference) ---")
for burst_type in ['2s', '15s']:
    print(f"\n  Burst threshold: {burst_type}")
    sub = res_df[res_df['burst_threshold'] == burst_type]
    for feat_name in feature_sets.keys():
        fsub = sub[sub['feature_set'] == feat_name]
        sfi_row = fsub[fsub['target'] == 'sfi']
        ssn_row = fsub[fsub['target'] == 'ssn']
        if len(sfi_row) > 0 and len(ssn_row) > 0:
            sa = sfi_row.iloc[0]['test_auc']
            na = ssn_row.iloc[0]['test_auc']
            print(f"    {feat_name:15s}  SFI={sa:.4f}  SSN={na:.4f}  Δ={sa-na:+.4f}")

print("\nDone!")
