"""
ceos_engine.py
==============
CEOS 分析统一工具模块 — 合并 algo_workers + ceos_utils 的所有功能

功能清单:
  - 角度计算: angular_diff, interpolate_angle, _is_phase, _kepler_idx
  - 分组函数: categorize_area, categorize_flare
  - 数据加载: load_sunspot_data, load_flare_data, load_sunspot_stage, load_solar_cycles
  - 单星 CEOS: compute_single_planet_ceos, compute_single_planet_ratio_only
  - 子集扫描: compute_bits, scan_all_subsets, cts_selected
  - 联合相位: compute_joint_phase
  - 统计工具: binom_2tail, poisson_ci, fdr_correction, sign_test_p, sig_stars
  - Kuiper 检验: run_kuiper_test
  - 效应量: cohens_h, bootstrap_ratio_ci
"""

import numpy as np
import pandas as pd
import os
import pickle
import time
import gc
import warnings
import scipy.stats as stats
import algo_workers
from multiprocessing import Pool, cpu_count

# 尝试导入 astropy Kuiper 检验
try:
    from astropy.stats import kuiper as _astropy_kuiper
except ImportError:
    _astropy_kuiper = None

# 尝试导入 FDR 校正
try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None

# ============================================================
# 路径配置（基于本文件所在目录推算）
# ============================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))

DEFAULT_DATA_DIR   = os.path.join(_PROJECT_ROOT, 'data', 'ready')
DEFAULT_CACHE_SG   = os.path.join(_PROJECT_ROOT, 'results', '04_asymmetric', 'sg', 'cache_data')
DEFAULT_CACHE_SF   = os.path.join(_PROJECT_ROOT, 'results', '04_asymmetric', 'sf', 'cache_data')
DEFAULT_OUTPUT_SG  = os.path.join(_PROJECT_ROOT, 'results', '04_asymmetric', 'sg')
DEFAULT_OUTPUT_SF  = os.path.join(_PROJECT_ROOT, 'results', '04_asymmetric', 'sf')

# ============================================================
# 常量
# ============================================================
PLANET_MAP = {
    '199_lon': 'Mercury', '299_lon': 'Venus',  '399_lon': 'Earth',   '499_lon': 'Mars',
    '599_lon': 'Jupiter', '699_lon': 'Saturn',  '799_lon': 'Uranus',  '899_lon': 'Neptune',
}
PLANET_COLS = list(PLANET_MAP.keys())          # 8 行星 (含 Earth)
PLANET_COLS_7P = [c for c in PLANET_COLS if c != '399_lon']  # 7 行星 (排除 Earth)
PLANET_MAP_7P = {k: v for k, v in PLANET_MAP.items() if k != '399_lon'}
PLANET_SHORT = ['Mer', 'Ven', 'Ear', 'Mar', 'Jup', 'Sat', 'Ura', 'Nep']
POWERS = (1 << np.arange(8)).astype(np.uint16)
EARTH_BIT = 4  # bit 2 = Earth (399)

# 联合检验场景
SCENARIOS = [
    ('A', 'Conjunction', 'Conjunction', '均合'),
    ('B', 'Opposition',  'Opposition',  '均冲'),
    ('C', 'Conjunction', 'Opposition',  'A合B冲'),
    ('D', 'Opposition',  'Conjunction', 'A冲B合'),
]

# ============================================================
# 基础工具
# ============================================================
def angular_diff(a, b):
    """角度差 → [-180, 180]"""
    return np.mod(a - b + 180, 360) - 180

def interpolate_angle(a1, a2, frac):
    """跨 0-360 安全角度插值"""
    r1, r2 = np.deg2rad(a1), np.deg2rad(a2)
    delta = (r2 - r1 + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(r1 + frac * delta) % 360.0

def _is_phase(delta, etype, w):
    """判定是否在相位窗口内"""
    if etype == 'Conjunction':
        return np.abs(delta) <= w
    else:
        return np.abs(np.abs(delta) - 180) <= w

def _kepler_idx(sun_idx, etype):
    """Kepler 基线索引"""
    if etype == 'Conjunction':
        return sun_idx
    else:
        return (sun_idx + 180) % 360

# ============================================================
# 分类函数
# ============================================================
def categorize_area(area):
    """黑子面积分类: Small(<100) / Medium(100-500) / Large(500-2000) / XLarge(≥2000)"""
    if area < 100: return 'Small <100'
    elif area < 500: return 'Medium 100-500'
    elif area < 2000: return 'Large 500-2000'
    else: return 'XLarge >2000'

def categorize_flare(xray_class):
    """耀斑分级: B/C/M/X-Class"""
    if not isinstance(xray_class, str): return 'Other'
    c = xray_class[0].upper()
    return f"{c}-Class" if c in 'ABCMX' else 'Other'



# ============================================================
# 数据加载
# ============================================================
def load_sunspot_data(cache_dir=None):
    """
    加载黑子 All 数据和每日星历矩阵。
    返回 (df, ephem_daily_8p)
    """
    cache_dir = cache_dir or DEFAULT_CACHE_SG
    cache_all = os.path.join(cache_dir, 'ready_All.parquet')
    matrix_file = os.path.join(cache_dir, 'ephem_matrix_8p.npy')

    if not os.path.exists(cache_all):
        raise FileNotFoundError(f"缓存文件不存在: {cache_all}")

    print(f"[数据] 加载黑子缓存: ready_All.parquet")
    df = pd.read_parquet(cache_all)
    if 'Group' not in df.columns:
        df['Group'] = df['area'].apply(categorize_area)

    ephem_daily = np.load(matrix_file)
    print(f"  黑子记录数: {len(df)}, 星历天数: {ephem_daily.shape[0]}")
    return df, ephem_daily

def load_flare_data(cache_dir_sf=None, cache_dir_sg=None):
    """
    加载耀斑数据和每日星历矩阵。
    返回 (df, ephem_daily_8p)
    """
    cache_dir_sf = cache_dir_sf or DEFAULT_CACHE_SF
    cache_dir_sg = cache_dir_sg or DEFAULT_CACHE_SG
    cache_flare = os.path.join(cache_dir_sf, 'ready_Flare_All.parquet')

    if not os.path.exists(cache_flare):
        raise FileNotFoundError(f"缓存文件不存在: {cache_flare}")

    print(f"[数据] 加载耀斑缓存: ready_Flare_All.parquet")
    df = pd.read_parquet(cache_flare)
    if 'Group' not in df.columns and 'xray_class' in df.columns:
        df['Group'] = df['xray_class'].apply(categorize_flare)
    df = df[df['Group'].isin(['B-Class', 'C-Class', 'M-Class', 'X-Class'])]

    # 加载星历矩阵（优先从 sg 目录，也支持 sf 目录）
    for cd in [cache_dir_sg, cache_dir_sf]:
        mf = os.path.join(cd, 'ephem_matrix_8p.npy')
        if os.path.exists(mf):
            ephem_daily = np.load(mf)
            break
    else:
        raise FileNotFoundError("ephem_matrix_8p.npy 不存在")

    print(f"  耀斑记录数: {len(df)}, 星历天数: {ephem_daily.shape[0]}")
    return df, ephem_daily

def load_sunspot_stage(cache_dir=None, stage='Onset'):
    """加载黑子生命周期阶段数据"""
    cache_dir = cache_dir or DEFAULT_CACHE_SG
    fname = f'ready_{stage}.parquet'
    path = os.path.join(cache_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"阶段文件不存在: {path}")
    print(f"[数据] 加载黑子阶段: {fname}")
    df = pd.read_parquet(path)
    if 'Group' not in df.columns and 'area' in df.columns:
        df['Group'] = df['area'].apply(categorize_area)
    print(f"  {stage} 记录数: {len(df)}")
    return df

def load_solar_cycles(csv_path=None):
    """
    读取太阳周期起止表。
    返回 list of (SC_label, start_date, end_date)
    """
    csv_path = csv_path or os.path.join(DEFAULT_DATA_DIR, 'solar_cycle_minmax.csv')
    df = pd.read_csv(csv_path)
    df['start_date'] = pd.to_datetime(df['start_Min'], format='%Y-%m')
    cycles = []
    for i in range(len(df)):
        sc = df['SC'].iloc[i]
        start = df['start_date'].iloc[i]
        if i + 1 < len(df):
            end = df['start_date'].iloc[i + 1] - pd.Timedelta(days=1)
        else:
            end = pd.Timestamp('2025-12-31')
        cycles.append((sc, start, end))
    return cycles

def load_kepler_prob_maps(cache_dir):
    """加载 Kepler 概率映射"""
    pkl_path = os.path.join(cache_dir, 'kepler_prob_maps.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    return None

# ============================================================
# 单星 CEOS
# ============================================================
def compute_single_planet_ceos(sun_lons, planet_single, planet_daily,
                                sun_idxs, w, n_sim=1000, rng=None):
    """
    计算单颗行星的 CEOS 效应 (含 CTS 模拟)。
    返回 dict: Conj/Opp 的 k_obs, k_exp(CTS均值), k_exp_kepler, Ratio, p, Z
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T = len(planet_daily)
    hist_prob, _ = np.histogram(planet_daily, bins=360, range=(0, 360), density=True)
    sun_idx_int = np.floor(sun_lons).astype(int) % 360
    delta = angular_diff(sun_lons, planet_single)
    result = {}

    for etype in ['Conjunction', 'Opposition']:
        is_ev = _is_phase(delta, etype, w)
        ti = _kepler_idx(sun_idx_int, etype)

        k_obs = int(np.sum(is_ev))
        # Kepler 解析基线 (2w bins, 供交叉检查)
        k_exp_kepler = 0.0
        for d in range(-w, w):
            k_exp_kepler += float(np.sum(hist_prob[(ti + d) % 360]))

        # CTS 模拟
        k_sims = np.empty(n_sim, dtype=np.int64)
        for i in range(n_sim):
            shift = rng.integers(0, T)
            sp = planet_daily[(sun_idxs + shift) % T]
            d = angular_diff(sun_lons, sp)
            k_sims[i] = int(np.sum(_is_phase(d, etype, w)))

        # k_exp = CTS 模拟均值 (无偏, 自动处理时间聚集性)
        k_exp = float(k_sims.mean())
        ratio = (k_obs / k_exp * 100) if k_exp > 0 else 0.0

        pl = (np.sum(k_sims <= k_obs) + 1) / (n_sim + 1)
        pr = (np.sum(k_sims >= k_obs) + 1) / (n_sim + 1)
        pv = min(2 * min(pl, pr), 1.0)
        std = k_sims.std()
        z = (k_obs - k_exp) / std if std > 0 else 0.0

        pfx = 'Conj' if etype == 'Conjunction' else 'Opp'
        result[f'{pfx}_k_obs'] = k_obs
        result[f'{pfx}_k_exp'] = round(k_exp, 2)
        result[f'{pfx}_k_exp_kepler'] = round(k_exp_kepler, 2)
        result[f'{pfx}_Ratio'] = round(ratio, 2)
        result[f'{pfx}_p'] = round(pv, 4)
        result[f'{pfx}_Z'] = round(z, 2)

    return result

def compute_single_planet_ratio_only(sun_lons, planet_single, planet_daily, w, etype):
    """仅计算 Ratio（无 CTS），用于太阳周分段等轻量级场景"""
    delta = angular_diff(sun_lons, planet_single)
    k_obs = int(np.sum(_is_phase(delta, etype, w)))
    hist, _ = np.histogram(planet_daily, bins=360, range=(0, 360), density=True)
    idx = _kepler_idx(np.floor(sun_lons).astype(int) % 360, etype)
    # 多 bin 积分 (2w 个 bin)
    k_exp = 0.0
    for d in range(-w, w):
        k_exp += float(np.sum(hist[(idx + d) % 360]))
    ratio = (k_obs / k_exp * 100) if k_exp > 0 else 0.0
    return k_obs, k_exp, ratio

# ============================================================
# Algo 1/2 主流程 (默认 7 行星, 排除 Earth)
# ============================================================
def run_algo12(cache_dir, output_dir, thresholds, n_sim, n_workers,
               stage_files=None, group_sort_key=None, algo_type='algo1',
               planet_cols=None):
    """
    运行 Algo 1 (Total Pairs) 或 Algo 2 (At Least One) 计算。

    参数:
        cache_dir: 缓存目录
        output_dir: 输出目录
        thresholds: 窗口列表 (如 range(1,21))
        n_sim: CTS 模拟次数
        n_workers: 并行进程数
        stage_files: 缓存中的 parquet 文件名列表 (None=自动发现)
        group_sort_key: 排序 key 函数
        algo_type: 'algo1' 或 'algo2'
        planet_cols: 使用的行星列名列表 (默认 7P, 排除 Earth)
    """
    if planet_cols is None:
        planet_cols = PLANET_COLS_7P
    import algo_workers

    suffix = 'total_pairs' if algo_type == 'algo1' else 'at_least_one'
    prefix = os.path.basename(output_dir)  # 'sg' or 'sf'
    output_file = os.path.join(output_dir, f'{prefix}_algo1_{suffix}.csv' if algo_type == 'algo1'
                               else f'{prefix}_algo2_{suffix}.csv')

    # 清除旧文件
    if os.path.exists(output_file):
        os.remove(output_file)

    # 加载缓存
    ephem_matrix_daily_8p = np.load(os.path.join(cache_dir, 'ephem_matrix_8p.npy'))
    # 按 planet_cols 选择对应的列索引
    col_indices = [PLANET_COLS.index(c) for c in planet_cols]
    ephem_matrix_daily = ephem_matrix_daily_8p[:, col_indices]
    prob_maps = load_kepler_prob_maps(cache_dir)
    n_planets_used = len(planet_cols)
    print(f"  使用 {n_planets_used} 行星: {[PLANET_MAP[c] for c in planet_cols]}")

    if stage_files is None:
        stage_files = [f for f in os.listdir(cache_dir) if f.startswith('ready_') and f.endswith('.parquet')]

    worker_fn = algo_workers.cts_worker_algo1 if algo_type == 'algo1' else algo_workers.cts_worker_algo2
    count_fn_name = 'count_events_vectorized' if algo_type == 'algo1' else 'count_events_at_least_once'
    count_fn = getattr(algo_workers, count_fn_name)

    results_buffer = []
    total_start = time.time()

    for f in stage_files:
        stage_name = f.replace('ready_', '').replace('.parquet', '')
        print(f"  处理阶段: {stage_name} ...")

        cols_to_load = ['hme_lon', 'ephem_idx_daily', 'Group'] + list(planet_cols)
        df = pd.read_parquet(os.path.join(cache_dir, f), columns=cols_to_load)

        if group_sort_key:
            groups = sorted(df['Group'].unique(), key=group_sort_key)
        else:
            groups = sorted(df['Group'].unique())
        groups.append('Total')

        for group in groups:
            subset = df if group == 'Total' else df[df['Group'] == group]
            if subset.empty: continue

            sun_lons = subset['hme_lon'].values.astype(np.float64)
            obs_planets = subset[list(planet_cols)].values.astype(np.float64)
            sun_idxs = subset['ephem_idx_daily'].values.astype(int)
            n_recs = len(subset)

            print(f"    分组: {group} (N={n_recs})")

            for w in thresholds:
                for etype in ['Conjunction', 'Opposition']:
                    # 1. 计算观测值
                    k_obs = count_fn(sun_lons, obs_planets, w, etype)

                    # 2. CTS 模拟 (GPU 优先, 自动退回 CPU)
                    # v3 修订(D 项):seed 由 (stage,group,w,etype,algo) 派生,
                    # 跨条目 Monte Carlo 误差独立。
                    cts_seed = algo_workers.derive_seed(
                        stage_name, group, w, etype, algo_type)
                    k_sims = algo_workers.run_cts_simulation(
                        sun_lons, ephem_matrix_daily, sun_idxs, w, etype,
                        n_sim, algo_type=algo_type, n_workers=n_workers,
                        seed=cts_seed)

                    # 3. 统计 — k_exp = CTS 模拟均值
                    k_exp = float(k_sims.mean())
                    p_left = (np.sum(k_sims <= k_obs) + 1) / (n_sim + 1)
                    p_right = (np.sum(k_sims >= k_obs) + 1) / (n_sim + 1)
                    p_val = min(2 * min(p_left, p_right), 1.0)
                    effect = 'Suppression' if k_obs < k_exp else 'Enhancement'
                    z_score = (k_obs - k_exp) / k_sims.std() if k_sims.std() > 0 else 0
                    ratio = (k_obs / k_exp * 100) if k_exp > 0 else 0

                    # 4. Kepler 解析基线 (供交叉检查)
                    lon_indices = np.floor(sun_lons).astype(int) % 360
                    target_indices = _kepler_idx(lon_indices, etype)

                    if algo_type == 'algo1':
                        k_exp_kepler = 0.0
                        for col in planet_cols:
                            for d in range(-w, w):
                                k_exp_kepler += np.sum(prob_maps[col][(target_indices + d) % 360])
                    else:
                        p_mat = np.zeros((n_recs, len(planet_cols)))
                        for i, col in enumerate(planet_cols):
                            for d in range(-w, w):
                                p_mat[:, i] += prob_maps[col][(target_indices + d) % 360]
                        k_exp_kepler = np.sum(1.0 - np.prod(1.0 - p_mat, axis=1))

                    results_buffer.append({
                        'Stage': stage_name, 'Group': group, 'Window': w, 'Type': etype,
                        'N_Records': n_recs, 'k_obs': k_obs,
                        'k_exp': round(k_exp, 2), 'k_exp_kepler': round(float(k_exp_kepler), 2),
                        'Ratio': round(ratio, 2), 'p_val': p_val,
                        'Z_score': round(z_score, 2), 'Effect': effect
                    })

        # 每阶段保存
        if results_buffer:
            new_df = pd.DataFrame(results_buffer)
            hdr = not os.path.exists(output_file)
            new_df.to_csv(output_file, mode='a', header=hdr, index=False)
            results_buffer = []

    elapsed = time.time() - total_start
    print(f"  Algo {'1' if algo_type == 'algo1' else '2'} 完成. 耗时: {elapsed:.1f}s")
    print(f"  输出文件: {output_file}")
    return output_file

# ============================================================
# Algo 3 (单体 781 天体扫描)
# ============================================================
def run_algo3(cache_dir, output_dir, thresholds,
              stage_files=None, prefix='sg'):
    """运行 Algo 3: 单体 781 天体扫描 (使用解析二项检验，无 CTS 模拟)"""

    output_file = os.path.join(output_dir, f'{prefix}_algo3_single_body_781.csv')
    if os.path.exists(output_file):
        os.remove(output_file)

    if stage_files is None:
        stage_files = [f for f in os.listdir(cache_dir) if f.startswith('ready_') and f.endswith('.parquet')]

    total_start = time.time()

    for f in stage_files:
        stage_name = f.replace('ready_', '').replace('.parquet', '')
        print(f"  处理阶段: {stage_name} ...")

        df = pd.read_parquet(os.path.join(cache_dir, f))

        # 识别天体列
        meta_cols = {'date', 'hme_lon', 'area', 'ephem_idx_daily', 'Group',
                     'lat_lon', 'hg_lon', 'hgc_lon', 'lon', 'lat',
                     'xray_class', 'datetime_start', 'datetime_peak', 'datetime_end',
                     'hg_lat', 'goes_class', 'noaa_ar'}
        body_cols = [c for c in df.columns if c not in meta_cols and c.endswith('_lon')]

        print(f"    分析 {len(body_cols)} 天体 vs {len(df)} 条记录...")

        sun_lons = df['hme_lon'].values.astype(np.float32)
        sun_areas = df['area'].values.astype(np.float32) if 'area' in df.columns else np.ones(len(df))
        n_recs = len(sun_lons)
        global_avg_area = np.mean(sun_areas) if n_recs > 0 else 0

        results_flat = []
        for col in body_cols:
            body_lons = df[col].values.astype(np.float32)
            hist_prob, _ = np.histogram(body_lons, bins=360, range=(0, 360), density=True)
            sun_idx_conj = np.floor(sun_lons).astype(int) % 360
            sun_idx_opp = (sun_idx_conj + 180) % 360

            for w in thresholds:
                for etype in ['Conjunction', 'Opposition']:
                    if etype == 'Conjunction':
                        delta = np.abs(sun_lons - body_lons)
                        delta = np.where(delta > 180, 360 - delta, delta)
                        is_event = (delta <= w)
                        target_indices = sun_idx_conj
                    else:
                        delta = np.abs(np.abs(sun_lons - body_lons) - 180)
                        is_event = (delta <= w)
                        target_indices = sun_idx_opp

                    k_obs = int(np.sum(is_event))
                    # 多 bin 积分 (2w 个 bin)
                    k_exp = 0.0
                    for offset in range(-w, w):
                        k_exp += float(np.sum(hist_prob[(target_indices + offset) % 360]))
                    ratio_freq = (k_obs / k_exp * 100) if k_exp > 0 else 0

                    # 振幅指标
                    if k_obs > 0:
                        event_avg_area = float(np.mean(sun_areas[is_event]))
                        ratio_amp = (event_avg_area / global_avg_area * 100) if global_avg_area > 0 else 0
                    else:
                        event_avg_area = 0
                        ratio_amp = 0

                    # P 值（二项近似）
                    p_avg = k_exp / n_recs if n_recs > 0 else 0
                    if 0 < p_avg < 1:
                        p_val = binom_2tail(k_obs, n_recs, p_avg)
                    else:
                        p_val = 1.0

                    effect = 'Suppression' if k_obs < k_exp else 'Enhancement'

                    results_flat.append({
                        'Body': col, 'Window': w, 'Type': etype,
                        'k_obs': k_obs, 'k_exp': round(k_exp, 2),
                        'Ratio_Freq': round(ratio_freq, 2),
                        'Avg_Area': round(event_avg_area, 2),
                        'Ratio_Amp': round(ratio_amp, 2),
                        'p_val': p_val, 'Effect': effect
                    })

        print(f"    处理 {len(body_cols)} 天体完成.")

        if results_flat:
            df_res = pd.DataFrame(results_flat)
            df_res['Stage'] = stage_name
            hdr = not os.path.exists(output_file)
            df_res.to_csv(output_file, mode='a', header=hdr, index=False)

    elapsed = time.time() - total_start
    print(f"  Algo 3 完成. 耗时: {elapsed:.1f}s")
    print(f"  输出文件: {output_file}")
    return output_file

# ============================================================
# Kuiper 检验
# ============================================================
def run_kuiper_test(cache_dir, output_dir, stage_files=None, prefix='sg'):
    """运行 Kuiper 检验"""
    if _astropy_kuiper is None:
        print("  [跳过] Kuiper 检验: astropy 未安装")
        return None

    output_file = os.path.join(output_dir, f'{prefix}_algo_kuiper_test.csv')

    if stage_files is None:
        stage_files = [f for f in os.listdir(cache_dir) if f.startswith('ready_') and f.endswith('.parquet')]

    results = []
    for f in stage_files:
        stage_name = f.replace('ready_', '').replace('.parquet', '')
        print(f"  Kuiper: {stage_name} ...")
        df = pd.read_parquet(os.path.join(cache_dir, f))

        groups_dict = {'Total': df}
        if 'Group' in df.columns:
            for g in df['Group'].unique():
                groups_dict[g] = df[df['Group'] == g]

        for group_name, subset_df in groups_dict.items():
            if len(subset_df) < 10: continue
            for planet in PLANET_COLS:
                phases = (subset_df[planet] - subset_df['hme_lon']) % 360.0
                phases = phases.dropna().values
                if len(phases) == 0: continue
                data = phases / 360.0
                try:
                    V, p_val = _astropy_kuiper(data)
                    results.append({
                        'Stage': stage_name, 'Group': group_name, 'Planet': planet,
                        'N': len(phases), 'V_statistic': round(V, 4),
                        'p_value': p_val, 'Sig': '**' if p_val < 0.01 else ('*' if p_val < 0.05 else '')
                    })
                except Exception as e:
                    print(f"    [Kuiper 警告] {stage_name}/{group_name}/{planet}: {e}")

    if results:
        df_k = pd.DataFrame(results).sort_values('p_value')
        df_k.to_csv(output_file, index=False)
        print(f"  Kuiper 结果: {output_file}")
        return output_file
    return None

# ============================================================
# 子集扫描 (255 子集)
# ============================================================
def _subset_label(m):
    """将位掩码转换为可读标签"""
    parts = []
    for i in range(8):
        if m & (1 << i):
            parts.append(PLANET_SHORT[i])
    return '+'.join(parts)

def _subset_np(m):
    """掩码中包含的行星数量"""
    return bin(m).count('1')

def compute_bits(sun_lons, planet_matrix, w):
    """对每行计算冲/合命中位图 → (bits_conj, bits_opp) uint16 数组"""
    delta = np.mod(sun_lons[:, None] - planet_matrix + 180, 360) - 180
    conj = np.abs(delta) <= w
    opp = np.abs(np.abs(delta) - 180) <= w
    bits_c = np.zeros(len(sun_lons), dtype=np.uint16)
    bits_o = np.zeros(len(sun_lons), dtype=np.uint16)
    for i in range(planet_matrix.shape[1]):
        bits_c |= conj[:, i].astype(np.uint16) << i
        bits_o |= opp[:, i].astype(np.uint16) << i
    return bits_c, bits_o

def scan_all_subsets(sun_lons, sun_idxs, planet_matrix, ephem_8p, w,
                     n_sim=50000, rng=None):
    """
    扫描全部 255 个非空子集 (8 行星)，返回 list[dict]。
    使用位掩码加速。
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T = ephem_8p.shape[0]
    bits_c, bits_o = compute_bits(sun_lons, planet_matrix, w)

    # Kepler 基线
    hist_probs = []
    for j in range(8):
        h, _ = np.histogram(ephem_8p[:, j], bins=360, range=(0, 360), density=True)
        hist_probs.append(h)
    sun_idx_int = np.floor(sun_lons).astype(int) % 360
    idx_conj = sun_idx_int
    idx_opp = (sun_idx_int + 180) % 360

    # 预计算模拟 — GPU 批量或 CPU 循环
    if algo_workers.HAS_GPU:
        import cupy as cp
        try:
            print(f"    [子集CTS模拟] GPU 批量 (n_sim={n_sim}, N={len(sun_lons)}, P=8)")
            T = ephem_8p.shape[0]
            N = len(sun_lons)
            P = 8
            d_sun = cp.asarray(sun_lons, dtype=cp.float32)
            d_ephem = cp.asarray(ephem_8p, dtype=cp.float32)
            d_idx = cp.asarray(sun_idxs, dtype=cp.int32)

            all_shifts = rng.integers(0, T, size=n_sim)

            # 自动计算 batch_size
            free_mem = cp.cuda.Device(0).mem_info[0]
            bytes_per = N * P * 12 + N * 4
            batch_sz = max(50, min(n_sim, int(free_mem * 0.5) // max(bytes_per, 1)))

            sim_bits_c_list = []
            sim_bits_o_list = []
            for start in range(0, n_sim, batch_sz):
                end = min(start + batch_sz, n_sim)
                b_shifts = cp.asarray(all_shifts[start:end], dtype=cp.int32)
                shifted_idx = (d_idx[cp.newaxis, :] + b_shifts[:, cp.newaxis]) % T
                sp = d_ephem[shifted_idx]  # (batch, N, P)
                delta = cp.mod(d_sun[cp.newaxis, :, cp.newaxis] - sp + 180, 360) - 180
                del sp
                conj = cp.abs(delta) <= w
                opp = cp.abs(cp.abs(delta) - 180) <= w
                del delta
                # 构建位图 (batch, N) — uint16
                bc = cp.zeros((end - start, N), dtype=cp.uint16)
                bo = cp.zeros((end - start, N), dtype=cp.uint16)
                for i in range(P):
                    bc |= conj[:, :, i].astype(cp.uint16) << i
                    bo |= opp[:, :, i].astype(cp.uint16) << i
                del conj, opp
                bc_np = cp.asnumpy(bc)
                bo_np = cp.asnumpy(bo)
                del bc, bo
                for b in range(end - start):
                    sim_bits_c_list.append(bc_np[b])
                    sim_bits_o_list.append(bo_np[b])
            del d_sun, d_ephem, d_idx
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            print(f"    [子集GPU失败, 退回CPU] {e}")
            sim_bits_c_list = []
            sim_bits_o_list = []
            for _ in range(n_sim):
                shift = rng.integers(0, T)
                sp = ephem_8p[(sun_idxs + shift) % T]
                sc, so = compute_bits(sun_lons, sp, w)
                sim_bits_c_list.append(sc)
                sim_bits_o_list.append(so)
    else:
        sim_bits_c_list = []
        sim_bits_o_list = []
        for _ in range(n_sim):
            shift = rng.integers(0, T)
            sp = ephem_8p[(sun_idxs + shift) % T]
            sc, so = compute_bits(sun_lons, sp, w)
            sim_bits_c_list.append(sc)
            sim_bits_o_list.append(so)

    # 堆叠为 2D 数组: (n_sim, N)
    all_sim_c = np.stack(sim_bits_c_list)  # (n_sim, N) uint16
    all_sim_o = np.stack(sim_bits_o_list)
    del sim_bits_c_list, sim_bits_o_list

    # 255 子集掩码计数 — GPU 分批加速或 CPU 向量化
    use_gpu_mask = algo_workers.HAS_GPU
    gpu_batch_sim = 0  # GPU 分批大小 (沿 sim 维度)
    if use_gpu_mask:
        import cupy as cp
        try:
            free_mem = cp.cuda.Device(0).mem_info[0]
            # 每次上传的 sim 数: 2 arrays × batch × N × 2 bytes
            bytes_per_sim_row = len(sun_lons) * 2 * 2 + 256 * 8  # uint16 × 2 + mask overhead
            gpu_batch_sim = max(1000, min(n_sim, int(free_mem * 0.4) // max(bytes_per_sim_row, 1)))
            print(f"    [子集掩码计数] GPU 分批 ({all_sim_c.shape}, batch={gpu_batch_sim})")
        except Exception as e:
            print(f"    [子集掩码计数 GPU 初始化失败, 退回CPU] {e}")
            use_gpu_mask = False

    if not use_gpu_mask:
        print(f"    [子集掩码计数] CPU 向量化 ({all_sim_c.shape})")

    # 预计算所有 255 个 mask 的 sim_kc/sim_ko
    if use_gpu_mask:
        # GPU 分批: 沿 sim 维度分批上传, 每批计算所有 255 个 mask
        all_sim_kc = np.zeros((255, n_sim), dtype=np.int64)
        all_sim_ko = np.zeros((255, n_sim), dtype=np.int64)
        d_bits_c_obs = cp.asarray(bits_c)
        d_bits_o_obs = cp.asarray(bits_o)

        for s_start in range(0, n_sim, gpu_batch_sim):
            s_end = min(s_start + gpu_batch_sim, n_sim)
            d_sc = cp.asarray(all_sim_c[s_start:s_end])  # (batch, N) uint16
            d_so = cp.asarray(all_sim_o[s_start:s_end])
            for mi, m in enumerate(range(1, 256)):
                mb = np.uint16(m)
                all_sim_kc[mi, s_start:s_end] = cp.asnumpy(cp.sum((d_sc & mb) != 0, axis=1))
                all_sim_ko[mi, s_start:s_end] = cp.asnumpy(cp.sum((d_so & mb) != 0, axis=1))
            del d_sc, d_so
            cp.get_default_memory_pool().free_all_blocks()

        # 观测值也用 GPU
        all_koc = np.zeros(255, dtype=np.int64)
        all_koo = np.zeros(255, dtype=np.int64)
        for mi, m in enumerate(range(1, 256)):
            mb = np.uint16(m)
            all_koc[mi] = int(cp.sum((d_bits_c_obs & mb) != 0))
            all_koo[mi] = int(cp.sum((d_bits_o_obs & mb) != 0))
        del d_bits_c_obs, d_bits_o_obs
        cp.get_default_memory_pool().free_all_blocks()

    rows = []
    for mi, m in enumerate(range(1, 256)):  # 1-255 非空子集
        mask_bits = np.uint16(m)

        if use_gpu_mask:
            koc = int(all_koc[mi])
            koo = int(all_koo[mi])
            sim_kc = all_sim_kc[mi]
            sim_ko = all_sim_ko[mi]
        else:
            koc = int(np.sum((bits_c & mask_bits) != 0))
            koo = int(np.sum((bits_o & mask_bits) != 0))
            sim_kc = np.sum((all_sim_c & mask_bits) != 0, axis=1)
            sim_ko = np.sum((all_sim_o & mask_bits) != 0, axis=1)

        # Kepler 解析期望 (精确 At-Least-One 公式, 供交叉检查)
        p_none_conj = np.ones(len(sun_lons))  # 无任何行星合的概率
        p_none_opp = np.ones(len(sun_lons))
        for i in range(8):
            if m & (1 << i):
                # 多 bin 积分: 对 [-w, w) 范围内 2w 个 bin 求和
                p_conj_i = np.zeros(len(sun_lons))
                p_opp_i = np.zeros(len(sun_lons))
                for d in range(-w, w):
                    p_conj_i += hist_probs[i][(idx_conj + d) % 360]
                    p_opp_i += hist_probs[i][(idx_opp + d) % 360]
                p_none_conj *= (1 - p_conj_i)
                p_none_opp *= (1 - p_opp_i)
        kec_kepler = float(np.sum(1 - p_none_conj))
        keo_kepler = float(np.sum(1 - p_none_opp))

        # CTS 模拟均值作为主要 k_exp
        kec = float(sim_kc.mean())
        keo = float(sim_ko.mean())

        rc = koc / kec * 100 if kec > 0 else 0
        ro = koo / keo * 100 if keo > 0 else 0

        def _pval(k_obs, k_sims):
            pl = (np.sum(k_sims <= k_obs) + 1) / (n_sim + 1)
            pr = (np.sum(k_sims >= k_obs) + 1) / (n_sim + 1)
            return min(2 * min(pl, pr), 1.0)

        pc = _pval(koc, sim_kc)
        po = _pval(koo, sim_ko)

        has_earth = bool(m & EARTH_BIT)
        rows.append({
            'Mask': m, 'Label': _subset_label(m), 'N_Planets': _subset_np(m),
            'Has_Earth': has_earth,
            'Conj_k_obs': koc, 'Conj_k_exp': round(kec, 2), 'Conj_k_exp_kepler': round(kec_kepler, 2),
            'Conj_Ratio': round(rc, 2), 'Conj_p': round(pc, 4),
            'Opp_k_obs': koo, 'Opp_k_exp': round(keo, 2), 'Opp_k_exp_kepler': round(keo_kepler, 2),
            'Opp_Ratio': round(ro, 2), 'Opp_p': round(po, 4),
            'Asym_Amp': round(rc - ro, 2),
            'Window': w,
        })

    # 清理 GPU 显存和中间数组
    if use_gpu_mask:
        del all_sim_kc, all_sim_ko, all_koc, all_koo
        cp.get_default_memory_pool().free_all_blocks()

    return rows

def run_subset_scan(sun_lons, sun_idxs, planet_matrix, ephem_8p,
                    output_dir, prefix, thresholds_subset=[1,2,3,4,5],
                    n_sim=50000, rng=None, scope_tag='full'):
    """
    运行全 255 子集扫描，按含/不含地球分两个 CSV 保存。

    v3 修订(D 项):每个 (scope_tag, w) 调用独立派生 seed,跨条目 Monte Carlo
    误差独立。`rng` 保留为可选参数仅为向后兼容(若传入会被忽略警告)。
    """
    if rng is not None:
        import warnings
        warnings.warn("run_subset_scan: 传入的 rng 在 v3 已被忽略;"
                      "seed 现按 (prefix, scope_tag, w) 派生", stacklevel=2)
    import algo_workers
    all_rows = []
    for w in thresholds_subset:
        print(f"    子集扫描 w={w} ...")
        seed = algo_workers.derive_seed('subset_scan', prefix, scope_tag, w)
        rng_w = np.random.default_rng(seed)
        rows = scan_all_subsets(sun_lons, sun_idxs, planet_matrix, ephem_8p,
                                w, n_sim, rng_w)
        all_rows.extend(rows)

    df_all = pd.DataFrame(all_rows)

    # 分组保存
    df_earth = df_all[df_all['Has_Earth'] == True]
    df_no_earth = df_all[df_all['Has_Earth'] == False]

    f1 = os.path.join(output_dir, f'{prefix}_subset_scan_with_earth.csv')
    f2 = os.path.join(output_dir, f'{prefix}_subset_scan_no_earth.csv')

    df_earth.to_csv(f1, index=False)
    df_no_earth.to_csv(f2, index=False)

    print(f"    含地球: {len(df_earth)} 行 → {f1}")
    print(f"    不含地球: {len(df_no_earth)} 行 → {f2}")
    return f1, f2

# ============================================================
# 太阳周分段
# ============================================================
def run_solar_cycle_analysis(df, ephem_daily, cycles, output_dir, prefix,
                              thresholds_sc=[1,2,3,4,5], planet_cols=None,
                              n_sim=5000):
    """
    太阳周分段分析: 每个周期对每颗行星单独计算 Ratio + CTS p 值。

    参数:
        n_sim: CTS 模拟次数 (默认 5000, 精度 ±0.02)
    """
    if planet_cols is None:
        planet_cols = PLANET_COLS_7P
    output_file = os.path.join(output_dir, f'{prefix}_solar_cycle_segment.csv')

    import algo_workers
    results = []
    for sc_label, start, end in cycles:
        if 'date' in df.columns:
            mask = (df['date'] >= start) & (df['date'] <= end)
        else:
            continue
        df_sc = df[mask]
        if len(df_sc) < 50:
            continue

        print(f"    {sc_label}: {len(df_sc)} 条记录 ({start.strftime('%Y-%m')} ~ {end.strftime('%Y-%m')})")

        sun_lons = df_sc['hme_lon'].values.astype(np.float64)
        sun_idxs = df_sc['ephem_idx_daily'].values.astype(int)

        for w in thresholds_sc:
            for col in planet_cols:
                planet_name = PLANET_MAP[col]
                planet_single = df_sc[col].values.astype(np.float64)
                # 使用全时段的星历作为 CTS 基线
                j = PLANET_COLS.index(col)  # 在 8P 矩阵中的索引
                planet_daily = ephem_daily[:, j]

                # CTS 模拟计算 Ratio + p 值
                # v3 (D 项):每个 (prefix, SC, planet, w) 派生独立 seed
                seed = algo_workers.derive_seed(
                    'sc_segment', prefix, sc_label, planet_name, w)
                rng_pw = np.random.default_rng(seed)
                res = compute_single_planet_ceos(
                    sun_lons, planet_single, planet_daily,
                    sun_idxs, w, n_sim=n_sim, rng=rng_pw)

                for etype in ['Conjunction', 'Opposition']:
                    pfx = 'Conj' if etype == 'Conjunction' else 'Opp'
                    results.append({
                        'SC': sc_label, 'Start': start.strftime('%Y-%m'),
                        'End': end.strftime('%Y-%m'), 'N_Records': len(df_sc),
                        'Planet': planet_name, 'Window': w, 'Type': etype,
                        'k_obs': res[f'{pfx}_k_obs'],
                        'k_exp': res[f'{pfx}_k_exp'],
                        'Ratio': res[f'{pfx}_Ratio'],
                        'p_val': res[f'{pfx}_p'],
                        'Z_score': res[f'{pfx}_Z'],
                    })

    if results:
        df_sc_results = pd.DataFrame(results)
        df_sc_results.to_csv(output_file, index=False)
        print(f"  太阳周分段结果: {output_file}")
    return output_file


def run_solar_cycle_subset_scan(df, ephem_daily, cycles, output_dir, prefix,
                                 thresholds_subset=[1,2,3], n_sim=5000):
    """
    太阳周分段 × 255 子集扫描。

    对每个太阳周分别调用 run_subset_scan()，CTS 模拟只做 1 次，
    255 子集用位掩码瞬间完成。

    输出文件命名: {prefix}_{SC}_subset_scan_with_earth.csv
                  {prefix}_{SC}_subset_scan_no_earth.csv
    """
    for sc_label, start, end in cycles:
        if 'date' in df.columns:
            mask = (df['date'] >= start) & (df['date'] <= end)
        else:
            continue
        df_sc = df[mask]
        if len(df_sc) < 100:
            print(f"    [跳过] {sc_label}: 仅 {len(df_sc)} 条, 不足 100")
            continue

        print(f"    {sc_label}: {len(df_sc)} 条记录")

        sun_lons = df_sc['hme_lon'].values.astype(np.float64)
        sun_idxs = df_sc['ephem_idx_daily'].values.astype(int)
        planet_matrix = df_sc[PLANET_COLS].values.astype(np.float64)

        sc_prefix = f'{prefix}_{sc_label}'
        # v3 (D 项):scope_tag 含 SC 标签 → 不同 SC × w 派生独立 seed
        run_subset_scan(sun_lons, sun_idxs, planet_matrix, ephem_daily,
                        output_dir, sc_prefix, thresholds_subset, n_sim,
                        scope_tag=f'sc_{sc_label}')


# ============================================================
# 联合相位检验
# ============================================================
def compute_joint_phase(sun_lons, lons_A, lons_B, daily_A, daily_B,
                        phase_A, phase_B, w):
    """计算两颗行星联合相位检验"""
    N = len(sun_lons)
    dA = angular_diff(sun_lons, lons_A)
    dB = angular_diff(sun_lons, lons_B)
    mask = _is_phase(dA, phase_A, w) & _is_phase(dB, phase_B, w)
    k_obs = int(np.sum(mask))

    hA, _ = np.histogram(daily_A, bins=360, range=(0, 360), density=True)
    hB, _ = np.histogram(daily_B, bins=360, range=(0, 360), density=True)
    si = np.floor(sun_lons).astype(int) % 360
    # 多 bin 积分 (2w 个 bin)
    pA = np.zeros(len(sun_lons))
    pB = np.zeros(len(sun_lons))
    idx_A = _kepler_idx(si, phase_A)
    idx_B = _kepler_idx(si, phase_B)
    for d in range(-w, w):
        pA += hA[(idx_A + d) % 360]
        pB += hB[(idx_B + d) % 360]
    k_exp = float(np.sum(pA * pB))

    ratio = (k_obs / k_exp * 100) if k_exp > 0 else 0.0
    pa = k_exp / N if N > 0 else 0
    pv = binom_2tail(k_obs, N, pa) if 0 < pa < 1 else 1.0
    ci_lo, ci_hi = poisson_ci(k_obs)
    r_lo = ci_lo / k_exp * 100 if k_exp > 0 else 0
    r_hi = ci_hi / k_exp * 100 if k_exp > 0 else 0

    return {
        'k_obs': k_obs, 'k_exp': round(k_exp, 2),
        'Ratio%': round(ratio, 2), 'p_value': round(pv, 4),
        'CI_lo': round(r_lo, 1), 'CI_hi': round(r_hi, 1),
    }

# ============================================================
# 统计工具
# ============================================================
def binom_2tail(k, n, p):
    """双侧二项检验"""
    if n <= 0 or p <= 0 or p >= 1:
        return 1.0
    pl = stats.binom.cdf(k, n, p)
    pr = stats.binom.sf(k - 1, n, p)
    return min(2 * min(pl, pr), 1.0)

def poisson_ci(k, alpha=0.05):
    """Poisson 置信区间"""
    lo = 0.0 if k == 0 else stats.chi2.ppf(alpha / 2, 2 * k) / 2
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / 2
    return lo, hi

def fdr_correction(p_values):
    """Benjamini-Hochberg FDR 校正"""
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(p_arr)
    sorted_p = p_arr[sorted_idx]
    adjusted = np.empty(n)
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(sorted_p[i] * n / (i + 1),
                                       adjusted[sorted_idx[i + 1]])
    return np.clip(adjusted, 0, 1)

def sign_test_p(n_positive, n_total, p0=0.5):
    """跨太阳周符号一致性检验（双侧符号检验）"""
    pl = stats.binom.cdf(n_positive, n_total, p0)
    pr = stats.binom.sf(n_positive - 1, n_total, p0)
    return min(2.0 * min(pl, pr), 1.0)

def sig_stars(p):
    """显著性星号标记"""
    if p < 0.001: return '***'
    elif p < 0.01: return '** '
    elif p < 0.05: return '*  '
    return '   '

def cohens_h(p_obs, p_exp):
    """Cohen's h 效应量 (弧反正弦变换)"""
    return 2 * (np.arcsin(np.sqrt(p_obs)) - np.arcsin(np.sqrt(p_exp)))

def bootstrap_ratio_ci(k_obs, k_exp, n_bootstrap=10000, alpha=0.05, rng=None):
    """
    Bootstrap 估计 Ratio 的置信区间。
    使用 Poisson 重采样。
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if k_exp <= 0:
        return 0, 0
    samples = rng.poisson(k_obs, size=n_bootstrap)
    ratios = samples / k_exp * 100
    lo = np.percentile(ratios, 100 * alpha / 2)
    hi = np.percentile(ratios, 100 * (1 - alpha / 2))
    return round(lo, 2), round(hi, 2)

# ============================================================
# FDR 校正 (Algo 3 输出)
# ============================================================
def apply_fdr_to_algo3(csv_file):
    """对 Algo 3 输出应用 BH-FDR 校正"""
    if multipletests is None:
        print("  [跳过] FDR 校正: statsmodels 未安装")
        return

    df = pd.read_csv(csv_file)
    group_cols = ['Stage', 'Window', 'Type']
    actual_cols = [c for c in group_cols if c in df.columns]

    df['p_adj_bh'] = np.nan
    df['sig_fdr'] = False
    total_sig = 0

    for name, group_data in df.groupby(actual_cols):
        p_vals = group_data['p_val'].values
        idx = group_data.index
        reject, p_adj, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
        df.loc[idx, 'p_adj_bh'] = p_adj
        df.loc[idx, 'sig_fdr'] = reject
        total_sig += reject.sum()

    df.to_csv(csv_file, index=False)
    print(f"  FDR 校正完成: {total_sig} 个显著记录")
