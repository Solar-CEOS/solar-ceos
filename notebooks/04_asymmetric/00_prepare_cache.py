#!/usr/bin/env python3
"""
00_prepare_cache.py — 缓存数据预处理
=====================================
从原始 CSV/Parquet 生成 CEOS 计算所需的缓存文件:
  - ready_All.parquet (黑子全量, 含插值星历)
  - ready_Onset/Duration/Dissipation/Daily.parquet (黑子各阶段)
  - ready_Flare_All.parquet (耀斑, 含插值星历)
  - ephem_matrix_8p.npy (8 行星每日经度矩阵)
  - kepler_prob_maps.pkl (行星经度直方图概率映射)

输入:
  data/ready/sg_1874_2025_*.csv           (黑子 5 个阶段)
  data/ready/flare_1975_2017.csv          (耀斑)
  data/ready/planets_satellites_lonlat.parquet  (星历)

输出:
  results/04_asymmetric/sg/cache_data/          (黑子缓存)
  results/04_asymmetric/sf/cache_data/          (耀斑缓存)

用法:
  python 00_prepare_cache.py
"""

import sys
import os
import time
import gc
import pickle
import numpy as np
import pandas as pd

# =====================================================================
# 路径配置
# =====================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))

DATA_DIR       = os.path.join(_PROJECT_ROOT, 'data', 'ready')
CACHE_DIR_SG   = os.path.join(_PROJECT_ROOT, 'results', '04_asymmetric', 'sg', 'cache_data')
CACHE_DIR_SF   = os.path.join(_PROJECT_ROOT, 'results', '04_asymmetric', 'sf', 'cache_data')

# 星历文件 (优先 781 天体版本, Algo 3 需要全部天体列)
EPHEMERIS_FILE = os.path.join(DATA_DIR, '781_planets_dwarfs_asteroids_lonlat.parquet')
if not os.path.exists(EPHEMERIS_FILE):
    EPHEMERIS_FILE = os.path.join(DATA_DIR, 'planets_satellites_lonlat.parquet')

# 黑子数据文件
SG_FILES = {
    'sg_1874_2025_all.csv':   'All',
    'sg_1874_2025_daily.csv': 'Daily',
    'sg_1874_2025_diss.csv':  'Dissipation',
    'sg_1874_2025_dur.csv':   'Duration',
    'sg_1874_2025_onset.csv': 'Onset',
}

# 耀斑数据文件
SF_FILE = 'flare_1975_2017.csv'

# 8 大行星列名
PLANET_COLS = ['199_lon', '299_lon', '399_lon', '499_lon',
               '599_lon', '699_lon', '799_lon', '899_lon']


# =====================================================================
# 工具函数
# =====================================================================
def interpolate_angle(a1, a2, frac):
    """跨 0-360 安全角度插值 (向量化)"""
    r1, r2 = np.deg2rad(a1), np.deg2rad(a2)
    delta = (r2 - r1 + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(r1 + frac * delta) % 360.0


def categorize_area(area):
    """黑子面积分类"""
    if area < 100:  return 'Small <100'
    elif area < 500: return 'Medium 100-500'
    elif area < 2000: return 'Large 500-2000'
    else: return 'XLarge >2000'


def categorize_flare(xray_class):
    """耀斑分级"""
    if not isinstance(xray_class, str):
        return 'Other'
    c = xray_class[0].upper()
    return f"{c}-Class" if c in 'ABCMX' else 'Other'


def convert_to_intensity(xray_class):
    """将 GOES 耀斑等级转换为相对强度代理值 (Algo 3 需要 area 列)"""
    multiplier = {'A': 1, 'B': 10, 'C': 100, 'M': 1000, 'X': 10000}
    try:
        if not isinstance(xray_class, str) or len(xray_class) < 1:
            return np.nan
        raw = xray_class.strip().upper()
        flare_type = raw[0]
        if flare_type not in multiplier:
            return np.nan
        level = float(raw[1:]) if len(raw) > 1 else 1.0
        return multiplier[flare_type] * level
    except (ValueError, TypeError):
        return np.nan


def load_ephemeris():
    """加载星历数据, 返回 (df_ephem, all_body_cols, ephem_matrix_all)"""
    print(f"[星历] 加载: {EPHEMERIS_FILE}")
    if not os.path.exists(EPHEMERIS_FILE):
        raise FileNotFoundError(f"星历文件不存在: {EPHEMERIS_FILE}")

    df = pd.read_parquet(EPHEMERIS_FILE)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)

    all_body_cols = [c for c in df.columns if str(c).endswith('_lon')]
    ephem_matrix = df[all_body_cols].values.astype(np.float32)

    print(f"  星历天数: {len(df)}, 天体数: {len(all_body_cols)}")
    return df, all_body_cols, ephem_matrix


def save_kepler_and_matrix(df_ephem, cache_dir):
    """保存 8 行星每日矩阵和 Kepler 概率映射"""
    os.makedirs(cache_dir, exist_ok=True)

    # 8 行星每日经度矩阵
    path_npy = os.path.join(cache_dir, 'ephem_matrix_8p.npy')
    np.save(path_npy, df_ephem[PLANET_COLS].values.astype(np.float64))
    print(f"  保存: ephem_matrix_8p.npy ({df_ephem[PLANET_COLS].shape})")

    # Kepler 概率映射
    prob_maps = {}
    for col in PLANET_COLS:
        hist, _ = np.histogram(df_ephem[col], bins=360, range=(0, 360), density=True)
        prob_maps[col] = hist
    path_pkl = os.path.join(cache_dir, 'kepler_prob_maps.pkl')
    with open(path_pkl, 'wb') as f:
        pickle.dump(prob_maps, f)
    print(f"  保存: kepler_prob_maps.pkl ({len(prob_maps)} 行星)")


# =====================================================================
# Step 1: 黑子缓存生成
# =====================================================================
def prepare_sg_cache():
    """生成黑子缓存 (5 个阶段)"""
    print("\n" + "=" * 60)
    print("Step 1: 黑子缓存生成")
    print("=" * 60)

    start = time.time()
    os.makedirs(CACHE_DIR_SG, exist_ok=True)

    # 加载星历
    df_ephem, all_body_cols, ephem_matrix_all = load_ephemeris()

    # 保存 8P 矩阵和 Kepler 映射
    save_kepler_and_matrix(df_ephem, CACHE_DIR_SG)

    # 处理 All 阶段 (重插值)
    file_all = 'sg_1874_2025_all.csv'
    path_all = os.path.join(DATA_DIR, file_all)
    if not os.path.exists(path_all):
        print(f"  ✗ {file_all} 不存在, 跳过黑子缓存")
        return

    print(f"\n  处理 All 阶段 (插值)...")
    df_sun = pd.read_csv(path_all, usecols=['date', 'hme_lon', 'area'])
    df_sun['date'] = pd.to_datetime(df_sun['date'])
    df_sun['Group'] = df_sun['area'].apply(categorize_area)

    # 去重
    len_before = len(df_sun)
    df_sun.drop_duplicates(subset=['date', 'hme_lon', 'area'], inplace=True)
    if len(df_sun) < len_before:
        print(f"    去重: {len_before} → {len(df_sun)}")

    # 过滤时间范围
    min_date, max_date = df_ephem.index.min(), df_ephem.index.max()
    df_sun = df_sun[(df_sun['date'] >= min_date) & (df_sun['date'] <= max_date)].copy()

    # 插值计算
    day_t = df_sun['date'].dt.normalize()
    fraction = (df_sun['date'] - day_t).dt.total_seconds() / 86400.0
    idx_t = df_ephem.index.searchsorted(day_t)

    valid_mask = (idx_t < len(df_ephem) - 1)
    if not valid_mask.all():
        df_sun = df_sun[valid_mask]
        idx_t = idx_t[valid_mask]
        fraction = fraction[valid_mask]

    pos_t = ephem_matrix_all[idx_t]
    pos_t1 = ephem_matrix_all[idx_t + 1]
    frac_vals = fraction.values[:, np.newaxis].astype(np.float32)
    interp_matrix = interpolate_angle(pos_t, pos_t1, frac_vals)

    # 构建结果
    df_sun['ephem_idx_daily'] = idx_t
    df_bodies = pd.DataFrame(interp_matrix, columns=all_body_cols, index=df_sun.index)
    df_final_all = pd.concat([df_sun, df_bodies], axis=1)

    save_path = os.path.join(CACHE_DIR_SG, 'ready_All.parquet')
    df_final_all.to_parquet(save_path)
    print(f"    保存: ready_All.parquet ({len(df_final_all)} 条)")

    # 处理其他阶段 (通过 merge lookup)
    other_files = {k: v for k, v in SG_FILES.items() if v != 'All'}
    print(f"\n  处理子阶段 ({len(other_files)} 个)...")

    for fname, stage_name in other_files.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"    ✗ {fname} 不存在, 跳过")
            continue

        df_sub = pd.read_csv(fpath, usecols=['date', 'hme_lon', 'area'])
        df_sub['date'] = pd.to_datetime(df_sub['date'])

        # Inner join: 使用 All 阶段已计算的插值结果
        df_merged = pd.merge(
            df_sub,
            df_final_all,
            on=['date', 'hme_lon', 'area'],
            how='inner'
        )

        save_path = os.path.join(CACHE_DIR_SG, f'ready_{stage_name}.parquet')
        df_merged.to_parquet(save_path)
        print(f"    保存: ready_{stage_name}.parquet ({len(df_merged)} 条)")

    del df_ephem, ephem_matrix_all, df_final_all
    gc.collect()
    print(f"\n  黑子缓存完成. 耗时: {time.time() - start:.1f}s")


# =====================================================================
# Step 2: 耀斑缓存生成
# =====================================================================
def prepare_sf_cache():
    """生成耀斑缓存"""
    print("\n" + "=" * 60)
    print("Step 2: 耀斑缓存生成")
    print("=" * 60)

    start = time.time()
    os.makedirs(CACHE_DIR_SF, exist_ok=True)

    # 加载星历
    df_ephem, all_body_cols, ephem_matrix_all = load_ephemeris()

    # 保存 8P 矩阵和 Kepler 映射 (sf 目录也保存一份)
    save_kepler_and_matrix(df_ephem, CACHE_DIR_SF)

    # 加载耀斑数据
    path_flare = os.path.join(DATA_DIR, SF_FILE)
    if not os.path.exists(path_flare):
        print(f"  ✗ {SF_FILE} 不存在, 跳过耀斑缓存")
        return

    print(f"\n  加载耀斑: {SF_FILE}")
    df = pd.read_csv(path_flare)

    # 时间映射
    df['date'] = pd.to_datetime(df['datetime_start'])

    # 分级
    df['Group'] = df['xray_class'].apply(categorize_flare)
    df = df[df['Group'] != 'Other']

    # 生成强度代理值 (Algo 3 的振幅比率指标需要 area 列)
    df['area'] = df['xray_class'].apply(convert_to_intensity)
    df.dropna(subset=['area'], inplace=True)

    # 去重
    len_before = len(df)
    df.drop_duplicates(subset=['date', 'xray_class', 'hme_lon'], inplace=True)
    df.dropna(subset=['hme_lon', 'date'], inplace=True)
    if len(df) < len_before:
        print(f"    去重: {len_before} → {len(df)}")

    # 过滤时间范围
    min_date, max_date = df_ephem.index.min(), df_ephem.index.max()
    df = df[(df['date'] >= min_date) & (df['date'] <= max_date)].copy()

    # 插值计算
    print(f"  插值 {len(df)} 条耀斑记录...")
    day_t = df['date'].dt.normalize()
    fraction = (df['date'] - day_t).dt.total_seconds() / 86400.0
    idx_t = df_ephem.index.searchsorted(day_t)

    valid_mask = (idx_t < len(df_ephem) - 1)
    if not valid_mask.all():
        df = df[valid_mask]
        idx_t = idx_t[valid_mask]
        fraction = fraction[valid_mask]

    pos_t = ephem_matrix_all[idx_t]
    pos_t1 = ephem_matrix_all[idx_t + 1]
    frac_vals = fraction.values[:, np.newaxis].astype(np.float32)
    interp_matrix = interpolate_angle(pos_t, pos_t1, frac_vals)

    # 构建结果
    df['ephem_idx_daily'] = idx_t
    df_bodies = pd.DataFrame(interp_matrix, columns=all_body_cols, index=df.index)
    df_final = pd.concat([df, df_bodies], axis=1)

    save_path = os.path.join(CACHE_DIR_SF, 'ready_Flare_All.parquet')
    df_final.to_parquet(save_path)
    print(f"    保存: ready_Flare_All.parquet ({len(df_final)} 条)")

    del df_ephem, ephem_matrix_all
    gc.collect()
    print(f"\n  耀斑缓存完成. 耗时: {time.time() - start:.1f}s")


# =====================================================================
# 主流程
# =====================================================================
def main():
    print("=" * 70)
    print("  CEOS 缓存数据预处理")
    print(f"  输入: {DATA_DIR}")
    print(f"  黑子输出: {CACHE_DIR_SG}")
    print(f"  耀斑输出: {CACHE_DIR_SF}")
    print("=" * 70)

    overall_start = time.time()

    prepare_sg_cache()
    prepare_sf_cache()

    elapsed = time.time() - overall_start
    print("\n" + "=" * 70)
    print(f"  全部预处理完成! 耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 70)

    # 列出输出文件
    print("\n输出文件清单:")
    for label, cache_dir in [('黑子', CACHE_DIR_SG), ('耀斑', CACHE_DIR_SF)]:
        print(f"\n  [{label}] {cache_dir}/")
        if os.path.exists(cache_dir):
            for f in sorted(os.listdir(cache_dir)):
                fp = os.path.join(cache_dir, f)
                sz = os.path.getsize(fp) / 1024
                print(f"    {f:40s} {sz:>8.1f} KB")


if __name__ == '__main__':
    main()
