#!/usr/bin/env python3
"""
01_compute_sg.py — 黑子 CEOS 全量计算
=====================================
使用 results/04_asymmetric/sg/cache_data/ 中的预处理数据，计算:
  1. Algo 1 (Total Pairs): 5阶段 × 分组 × w=1-5 × 冲合
  2. Algo 2 (At Least One): 同上
  3. Algo 3 (单体 781):    5阶段 × 781天体 × w=1-5 × 冲合
  4. Kuiper 检验
  5. 255 子集扫描 (含/不含地球分开保存)
  6. 太阳周分段 (SC12-SC25)

用法:
  ~/miniforge3/envs/py313_tian_env/bin/python 01_compute_sg.py
"""

import sys
import os
import time
import numpy as np

# 确保能导入同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ceos_engine as ce

# =====================================================================
# 配置参数
# =====================================================================

# 正式版参数
N_SIM_ALGO12 = 50000     # Algo 1/2 CTS 模拟次数 (提高精度, 解决临界 p 值)
N_SIM_ALGO3 = 100        # ⚠️ 仅用于信息展示，Algo 3 使用解析二项检验，此值不影响结果
N_SIM_SUBSET = 50000     # 子集扫描 CTS 模拟次数 (与 Algo1/2 统一)

# 窗口范围 (黑子无 CEOS 信号, w=1-5 足够)
THRESHOLDS_ALGO12 = list(range(1, 6))    # w=1-5
THRESHOLDS_ALGO3  = list(range(1, 6))    # w=1-5 (781天体)
THRESHOLDS_SUBSET = list(range(1, 6))    # w=1-5 (子集扫描)
THRESHOLDS_CYCLE  = list(range(1, 6))    # w=1-5 (太阳周)

# 并行
N_WORKERS = max(1, os.cpu_count() - 2)  # 留 2 核给系统

# 路径
CACHE_DIR  = ce.DEFAULT_CACHE_SG
OUTPUT_DIR = ce.DEFAULT_OUTPUT_SG

# =====================================================================
# 主流程
# =====================================================================
def main():
    print("=" * 70)
    print("黑子 CEOS 全量计算")
    print(f"模拟次数: Algo1/2={N_SIM_ALGO12}, Algo3={N_SIM_ALGO3}, Subset={N_SIM_SUBSET}")
    print(f"窗口: Algo1/2=1-5, Algo3/Subset/Cycle=1-5")
    print(f"并行核心数: {N_WORKERS}")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    overall_start = time.time()
    
    # 黑子面积分组排序键
    area_sort = lambda x: ('SMLX'.find(x[0]) if x[0] in 'SMLX' else 99, x)
    
    # 发现缓存文件
    stage_files = sorted([f for f in os.listdir(CACHE_DIR) 
                          if f.startswith('ready_') and f.endswith('.parquet')])
    print(f"\n找到 {len(stage_files)} 个阶段文件: {[f.replace('ready_','').replace('.parquet','') for f in stage_files]}")
    
    # --- Step 1: Algo 1 (Total Pairs) ---
    print("\n" + "=" * 60)
    print("Step 1: Algo 1 (Total Pairs) w=1-5")
    print("=" * 60)
    ce.run_algo12(CACHE_DIR, OUTPUT_DIR, THRESHOLDS_ALGO12, N_SIM_ALGO12,
                  N_WORKERS, stage_files, area_sort, algo_type='algo1')
    
    # --- Step 2: Algo 2 (At Least One) ---
    print("\n" + "=" * 60)
    print("Step 2: Algo 2 (At Least One) w=1-5")
    print("=" * 60)
    ce.run_algo12(CACHE_DIR, OUTPUT_DIR, THRESHOLDS_ALGO12, N_SIM_ALGO12,
                  N_WORKERS, stage_files, area_sort, algo_type='algo2')
    
    # --- Step 3: Algo 3 (单体 781) ---
    print("\n" + "=" * 60)
    print("Step 3: Algo 3 (单体 781 天体) w=1-5")
    print("=" * 60)
    algo3_file = ce.run_algo3(CACHE_DIR, OUTPUT_DIR, THRESHOLDS_ALGO3,
                               stage_files, prefix='sg')
    
    # FDR 校正
    if algo3_file and os.path.exists(algo3_file):
        print("\n  应用 FDR 校正...")
        ce.apply_fdr_to_algo3(algo3_file)
    
    # --- Step 4: Kuiper 检验 ---
    print("\n" + "=" * 60)
    print("Step 4: Kuiper 检验")
    print("=" * 60)
    ce.run_kuiper_test(CACHE_DIR, OUTPUT_DIR, stage_files, prefix='sg')
    
    # --- Step 5: 255 子集扫描 (分组) ---
    print("\n" + "=" * 60)
    print("Step 5: 255 子集扫描 (含/不含地球)")
    print("=" * 60)
    
    # 5a. All 阶段 Total (原始)
    df_all, ephem_daily = ce.load_sunspot_data(CACHE_DIR)
    sun_lons = df_all['hme_lon'].values.astype(np.float64)
    sun_idxs = df_all['ephem_idx_daily'].values.astype(int)
    planet_matrix = df_all[ce.PLANET_COLS].values.astype(np.float64)
    
    print("  [5a] All Total ...")
    ce.run_subset_scan(sun_lons, sun_idxs, planet_matrix, ephem_daily,
                       OUTPUT_DIR, 'sg', THRESHOLDS_SUBSET, N_SIM_SUBSET)
    
    # 5b. 按阶段分别扫描 (Daily, Onset, Duration, Dissipation)
    for stage_name in ['Daily', 'Onset', 'Duration', 'Dissipation']:
        print(f"  [5b] {stage_name} Total ...")
        try:
            df_stage = ce.load_sunspot_stage(CACHE_DIR, stage=stage_name)
            s_lons = df_stage['hme_lon'].values.astype(np.float64)
            s_idxs = df_stage['ephem_idx_daily'].values.astype(int)
            s_planets = df_stage[ce.PLANET_COLS].values.astype(np.float64)
            ce.run_subset_scan(s_lons, s_idxs, s_planets, ephem_daily,
                               OUTPUT_DIR, f'sg_{stage_name.lower()}',
                               THRESHOLDS_SUBSET, N_SIM_SUBSET)
        except Exception as e:
            print(f"    [跳过] {stage_name}: {e}")
    
    # 5c. 按面积分组扫描 (XLarge 样本少可能不稳定, 但值得尝试)
    for group_name in ['XLarge >2000', 'Large 500-2000']:
        print(f"  [5c] All/{group_name} ...")
        mask = df_all['Group'] == group_name
        if mask.sum() < 100:
            print(f"    [跳过] {group_name}: 仅 {mask.sum()} 条, 不足 100")
            continue
        g_lons = df_all.loc[mask, 'hme_lon'].values.astype(np.float64)
        g_idxs = df_all.loc[mask, 'ephem_idx_daily'].values.astype(int)
        g_planets = df_all.loc[mask, ce.PLANET_COLS].values.astype(np.float64)
        safe_name = group_name.replace(' ', '_').replace('<', 'lt').replace('>', 'gt')
        ce.run_subset_scan(g_lons, g_idxs, g_planets, ephem_daily,
                           OUTPUT_DIR, f'sg_{safe_name}',
                           THRESHOLDS_SUBSET, N_SIM_SUBSET)
    
    # --- Step 6: 太阳周分段 (分组) ---
    print("\n" + "=" * 60)
    print("Step 6: 太阳周分段 (SC12-SC25)")
    print("=" * 60)
    cycles = ce.load_solar_cycles()
    # 筛选 SC12 以后（黑子数据从 1878 年开始）
    cycles_sg = [(sc, s, e) for sc, s, e in cycles if int(sc[2:]) >= 12]
    print(f"  覆盖太阳周: {[c[0] for c in cycles_sg]}")
    
    # 6a. All Total (原始)
    print("  [6a] All Total ...")
    ce.run_solar_cycle_analysis(df_all, ephem_daily, cycles_sg,
                                 OUTPUT_DIR, 'sg', THRESHOLDS_CYCLE)
    
    # 6b. 按阶段分段
    for stage_name in ['Daily', 'Onset', 'Duration', 'Dissipation']:
        print(f"  [6b] {stage_name} Total ...")
        try:
            df_stage = ce.load_sunspot_stage(CACHE_DIR, stage=stage_name)
            ce.run_solar_cycle_analysis(df_stage, ephem_daily, cycles_sg,
                                         OUTPUT_DIR, f'sg_{stage_name.lower()}',
                                         THRESHOLDS_CYCLE)
        except Exception as e:
            print(f"    [跳过] {stage_name}: {e}")
    
    # 6c. 按面积分组分段
    for group_name in ['XLarge >2000', 'Large 500-2000']:
        print(f"  [6c] All/{group_name} ...")
        mask = df_all['Group'] == group_name
        if mask.sum() < 100:
            print(f"    [跳过] {group_name}: 仅 {mask.sum()} 条")
            continue
        df_group = df_all[mask].copy()
        safe_name = group_name.replace(' ', '_').replace('<', 'lt').replace('>', 'gt')
        ce.run_solar_cycle_analysis(df_group, ephem_daily, cycles_sg,
                                     OUTPUT_DIR, f'sg_{safe_name}',
                                     THRESHOLDS_CYCLE)
    
    # --- Step 7: 太阳周 × 255 子集扫描 ---
    print("\n" + "=" * 60)
    print("Step 7: 太阳周 × 255 子集扫描 (CTS)")
    print("=" * 60)
    
    # 7a. All Total
    print("  [7a] All Total ...")
    ce.run_solar_cycle_subset_scan(df_all, ephem_daily, cycles_sg,
                                    OUTPUT_DIR, 'sg',
                                    thresholds_subset=[1,2,3], n_sim=5000)
    
    # 完成
    total_elapsed = time.time() - overall_start
    print("\n" + "=" * 70)
    print(f"全部计算完成! 总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print("=" * 70)
    
    # 列出输出文件
    print("\n输出文件清单:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.csv'):
            fpath = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(fpath) / 1024
            print(f"  {f} ({size:.1f} KB)")


if __name__ == '__main__':
    main()
