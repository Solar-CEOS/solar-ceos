#!/usr/bin/env python3
"""
03_analyze_results.py — 分析全部 CEOS 计算结果
==============================================
读取 results/04_asymmetric/{sf,sg}/ 下的 CSV 文件，
输出一般性分析报告和可视化图表。

用法:
  python 03_analyze_results.py
"""

import sys, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 路径配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SG_DIR = os.path.join(PROJECT_ROOT, 'results', '04_asymmetric', 'sg')
SF_DIR = os.path.join(PROJECT_ROOT, 'results', '04_asymmetric', 'sf')
FIG_DIR = os.path.join(PROJECT_ROOT, 'results', '04_asymmetric', 'analysis')
os.makedirs(FIG_DIR, exist_ok=True)

# 行星映射
PLANET_MAP = {
    '199_lon': 'Mercury', '299_lon': 'Venus', '399_lon': 'Earth', '499_lon': 'Mars',
    '599_lon': 'Jupiter', '699_lon': 'Saturn', '799_lon': 'Uranus', '899_lon': 'Neptune',
}


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ============================================================
# 1. Algo 1/2 分析: Ratio 衰减曲线 (w=1-20)
# ============================================================
def analyze_algo12(csv_path, algo_name, dataset_name):
    """分析 Algo 1/2 结果: 打印汇总表 + 绘制衰减曲线"""
    if not os.path.exists(csv_path):
        print(f"  [跳过] {csv_path} 不存在")
        return

    df = pd.read_csv(csv_path)
    print_header(f"{dataset_name} — {algo_name}")

    # --- 汇总表: w=1,2,3,5,10,20 的合/冲 Ratio ---
    show_windows = [1, 2, 3, 5, 10, 15, 20]
    available_windows = sorted(df['Window'].unique())
    show_windows = [w for w in show_windows if w in available_windows]

    # 只看 Total 分组
    df_total = df[df['Group'] == 'Total']
    if len(df_total) == 0:
        print("  [警告] 没有 Total 分组")
        return

    # 透视表
    for stage in sorted(df_total['Stage'].unique()):
        df_s = df_total[df_total['Stage'] == stage]
        print(f"\n  【{stage}】 Total:")
        print(f"  {'Window':>6}  {'Conj_Ratio':>10}  {'Conj_p':>8}  {'Conj_Effect':>12}  │  {'Opp_Ratio':>10}  {'Opp_p':>8}  {'Opp_Effect':>12}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*12}  │  {'─'*10}  {'─'*8}  {'─'*12}")

        for w in show_windows:
            conj = df_s[(df_s['Window'] == w) & (df_s['Type'] == 'Conjunction')]
            opp = df_s[(df_s['Window'] == w) & (df_s['Type'] == 'Opposition')]

            if len(conj) > 0 and len(opp) > 0:
                cr = conj.iloc[0]
                orr = opp.iloc[0]
                c_sig = '***' if cr['p_val'] < 0.001 else ('**' if cr['p_val'] < 0.01 else ('*' if cr['p_val'] < 0.05 else ''))
                o_sig = '***' if orr['p_val'] < 0.001 else ('**' if orr['p_val'] < 0.01 else ('*' if orr['p_val'] < 0.05 else ''))
                print(f"  {w:>6}  {cr['Ratio']:>8.1f}%  {cr['p_val']:>8.4f}{c_sig:>1}  {cr['Effect']:>12}  │  {orr['Ratio']:>8.1f}%  {orr['p_val']:>8.4f}{o_sig:>1}  {orr['Effect']:>12}")

    # --- 绘制衰减曲线 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{dataset_name} {algo_name}: Ratio vs Window (Total)', fontsize=14, fontweight='bold')

    for ax_idx, (stage_filter, stage_label) in enumerate([
        ('Total', 'Total Group') if 'Flare' in df['Stage'].iloc[0] else ('All', 'All Stage'),
        ('Total', 'Total Group')
    ]):
        ax = axes[ax_idx]

        if ax_idx == 0:
            # 合 Conjunction
            stage_name = df['Stage'].unique()[0] if len(df['Stage'].unique()) == 1 else 'All'
            df_plot = df_total[(df_total['Stage'] == stage_name) & (df_total['Type'] == 'Conjunction')]
            if len(df_plot) == 0:
                df_plot = df_total[df_total['Type'] == 'Conjunction'].groupby('Window').first().reset_index()
            ax.plot(df_plot['Window'], df_plot['Ratio'], 'b-o', markersize=4, label='Conjunction')
            ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
            ax.set_title('Conjunction')
            ax.set_ylabel('Ratio (%)')
            ax.set_xlabel('Window w (°)')
        else:
            # 冲 Opposition
            stage_name = df['Stage'].unique()[0] if len(df['Stage'].unique()) == 1 else 'All'
            df_plot = df_total[(df_total['Stage'] == stage_name) & (df_total['Type'] == 'Opposition')]
            if len(df_plot) == 0:
                df_plot = df_total[df_total['Type'] == 'Opposition'].groupby('Window').first().reset_index()
            ax.plot(df_plot['Window'], df_plot['Ratio'], 'r-o', markersize=4, label='Opposition')
            ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
            ax.set_title('Opposition')
            ax.set_ylabel('Ratio (%)')
            ax.set_xlabel('Window w (°)')

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ds_en = 'flare' if 'flare' in dataset_name.lower() else 'sunspot'
    fig_name = f'{ds_en}_{algo_name.lower().replace(" ", "_")}_decay.png'
    fig_path = os.path.join(FIG_DIR, fig_name)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 图表已保存: {fig_path}")


# ============================================================
# 2. 各分组对比: Conj vs Opp, w=1-5
# ============================================================
def analyze_groups(csv_path, dataset_name, algo_name):
    """各分组的 CEOS 对比"""
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    print_header(f"{dataset_name} — {algo_name} 各分组对比 (w=1-5)")

    stage = df['Stage'].unique()[0] if len(df['Stage'].unique()) == 1 else 'All'
    df_s = df[df['Stage'] == stage] if stage in df['Stage'].values else df

    groups = [g for g in df_s['Group'].unique() if g != 'Total']
    groups.append('Total')

    for w in [1, 2, 3, 5]:
        print(f"\n  w={w}°:")
        print(f"  {'Group':>20}  {'Conj_Ratio':>10}  {'Opp_Ratio':>10}  {'Asym':>7}  {'Conj_p':>8}  {'Opp_p':>8}")
        print(f"  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*8}  {'─'*8}")

        for g in groups:
            conj = df_s[(df_s['Group'] == g) & (df_s['Window'] == w) & (df_s['Type'] == 'Conjunction')]
            opp = df_s[(df_s['Group'] == g) & (df_s['Window'] == w) & (df_s['Type'] == 'Opposition')]
            if len(conj) > 0 and len(opp) > 0:
                cr = conj.iloc[0]['Ratio']
                orr = opp.iloc[0]['Ratio']
                cp = conj.iloc[0]['p_val']
                op = opp.iloc[0]['p_val']
                asym = cr - orr
                c_s = '*' if cp < 0.05 else ''
                o_s = '*' if op < 0.05 else ''
                print(f"  {g:>20}  {cr:>8.1f}%  {orr:>8.1f}%  {asym:>+6.1f}  {cp:>7.4f}{c_s}  {op:>7.4f}{o_s}")


# ============================================================
# 3. 子集扫描分析
# ============================================================
def analyze_subsets(dir_path, prefix, dataset_name):
    """分析 255 子集扫描结果"""
    f_earth = os.path.join(dir_path, f'{prefix}_subset_scan_with_earth.csv')
    f_no_earth = os.path.join(dir_path, f'{prefix}_subset_scan_no_earth.csv')

    if not os.path.exists(f_earth) or not os.path.exists(f_no_earth):
        return

    df_e = pd.read_csv(f_earth)
    df_ne = pd.read_csv(f_no_earth)

    print_header(f"{dataset_name} — 255 子集扫描分析")

    for w in [1, 2, 3]:
        de = df_e[df_e['Window'] == w]
        dne = df_ne[df_ne['Window'] == w]

        print(f"\n  w={w}°:")

        # 含地球 Top 5 by Asym_Amp
        top_e = de.nlargest(5, 'Asym_Amp')
        print(f"  含地球 Top-5 (by Conj-Opp 不对称):")
        for _, r in top_e.iterrows():
            print(f"    {r['Label']:>30}  Conj={r['Conj_Ratio']:>6.1f}%  Opp={r['Opp_Ratio']:>6.1f}%  Asym={r['Asym_Amp']:>+6.1f}")

        # 不含地球 Top 5
        top_ne = dne.nlargest(5, 'Asym_Amp')
        print(f"  不含地球 Top-5:")
        for _, r in top_ne.iterrows():
            print(f"    {r['Label']:>30}  Conj={r['Conj_Ratio']:>6.1f}%  Opp={r['Opp_Ratio']:>6.1f}%  Asym={r['Asym_Amp']:>+6.1f}")

        # 正向比例
        pos_e = (de['Asym_Amp'] > 0).sum() / len(de) * 100
        pos_ne = (dne['Asym_Amp'] > 0).sum() / len(dne) * 100
        print(f"  正向比例: 含地球 {pos_e:.1f}% ({(de['Asym_Amp']>0).sum()}/{len(de)}) | 不含地球 {pos_ne:.1f}% ({(dne['Asym_Amp']>0).sum()}/{len(dne)})")

    # 绘图: 含/不含地球的 Asym_Amp 分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{dataset_name}: Subset CEOS Asymmetry Distribution (w=2)', fontsize=13, fontweight='bold')

    for ax, df_sub, title in zip(axes, [df_e, df_ne], ['With Earth (128 subsets)', 'Without Earth (127 subsets)']):
        d = df_sub[df_sub['Window'] == 2]['Asym_Amp']
        ax.hist(d, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        ax.axvline(x=d.median(), color='orange', linestyle='-', alpha=0.8, label=f'Median={d.median():.1f}')
        ax.set_title(title)
        ax.set_xlabel('Asym (Conj_Ratio - Opp_Ratio)')
        ax.set_ylabel('Count')
        ax.legend()

    plt.tight_layout()
    ds_en = 'flare' if 'flare' in dataset_name.lower() else 'sunspot'
    fig_path = os.path.join(FIG_DIR, f'{ds_en}_subset_asym_dist.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 图表已保存: {fig_path}")


# ============================================================
# 4. 太阳周分段分析
# ============================================================
def analyze_solar_cycles(csv_path, dataset_name):
    """太阳周分段结果分析"""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    print_header(f"{dataset_name} — 太阳周分段分析")

    # 重点看 Venus 和 Mars (w=2, Conjunction)
    for planet in ['Venus', 'Mars', 'Jupiter', 'Saturn']:
        dp = df[(df['Planet'] == planet) & (df['Window'] == 2) & (df['Type'] == 'Conjunction')]
        if len(dp) == 0: continue

        n_pos = (dp['Ratio'] > 100).sum()
        n_total = len(dp)
        print(f"\n  {planet} (w=2, Conjunction):")
        print(f"  {'SC':>6}  {'N':>7}  {'Ratio':>8}  {'k_obs':>6}  {'k_exp':>8}")
        for _, r in dp.iterrows():
            marker = '✓' if r['Ratio'] > 100 else '✗'
            print(f"  {r['SC']:>6}  {r['N_Records']:>7}  {r['Ratio']:>7.1f}%  {r['k_obs']:>6}  {r['k_exp']:>8.1f}  {marker}")
        print(f"  正向一致: {n_pos}/{n_total} ({n_pos/n_total*100:.0f}%)")

    # 绘图: Venus/Mars 各太阳周的 Ratio
    planets_to_plot = ['Venus', 'Mars', 'Jupiter']
    fig, axes = plt.subplots(len(planets_to_plot), 1, figsize=(12, 4*len(planets_to_plot)))
    if len(planets_to_plot) == 1:
        axes = [axes]
    fig.suptitle(f'{dataset_name}: Solar Cycle Conjunction Ratio (w=2)', fontsize=14, fontweight='bold')

    for ax, planet in zip(axes, planets_to_plot):
        dp = df[(df['Planet'] == planet) & (df['Window'] == 2) & (df['Type'] == 'Conjunction')]
        if len(dp) == 0: continue

        x = range(len(dp))
        colors = ['green' if r > 100 else 'red' for r in dp['Ratio']]
        ax.bar(x, dp['Ratio'] - 100, color=colors, alpha=0.7, edgecolor='gray')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(dp['SC'], rotation=45, fontsize=8)
        ax.set_ylabel('Ratio - 100%')
        ax.set_title(f'{planet}')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    ds_en = 'flare' if 'flare' in dataset_name.lower() else 'sunspot'
    fig_path = os.path.join(FIG_DIR, f'{ds_en}_solar_cycle_bars.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 图表已保存: {fig_path}")


# ============================================================
# 5. Kuiper 检验汇总
# ============================================================
def analyze_kuiper(csv_path, dataset_name):
    """Kuiper 检验结果汇总"""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    print_header(f"{dataset_name} — Kuiper 检验 Top-10")

    # 只看 Total 分组
    df_t = df[df['Group'] == 'Total'].sort_values('V_statistic', ascending=False)

    print(f"  {'Stage':>15}  {'Planet':>10}  {'N':>7}  {'V':>8}  {'p':>12}  {'Sig':>4}")
    for _, r in df_t.head(10).iterrows():
        planet = PLANET_MAP.get(r['Planet'], r['Planet'])
        print(f"  {r['Stage']:>15}  {planet:>10}  {r['N']:>7}  {r['V_statistic']:>8.4f}  {r['p_value']:>12.2e}  {r['Sig']:>4}")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("  CEOS 计算结果分析报告")
    print("=" * 70)

    # === Flare (All Total) ===
    analyze_algo12(os.path.join(SF_DIR, 'sf_algo1_total_pairs.csv'), 'Algo 1 (Total Pairs)', 'Flare')
    analyze_algo12(os.path.join(SF_DIR, 'sf_algo2_at_least_one.csv'), 'Algo 2 (At Least One)', 'Flare')
    analyze_groups(os.path.join(SF_DIR, 'sf_algo1_total_pairs.csv'), 'Flare', 'Algo 1')
    analyze_subsets(SF_DIR, 'sf', 'Flare')
    analyze_solar_cycles(os.path.join(SF_DIR, 'sf_solar_cycle_segment.csv'), 'Flare')
    analyze_kuiper(os.path.join(SF_DIR, 'sf_algo_kuiper_test.csv'), 'Flare')

    # === Flare (per X-ray Class) ===
    for class_name in ['B-Class', 'C-Class', 'M-Class', 'X-Class']:
        safe = class_name.replace('-', '_')
        prefix = f'sf_{safe}'
        label = f'Flare {class_name}'
        # 子集扫描 (per class)
        f_earth = os.path.join(SF_DIR, f'{prefix}_subset_scan_with_earth.csv')
        if os.path.exists(f_earth):
            analyze_subsets(SF_DIR, prefix, label)
        # 太阳周分段 (per class)
        f_cycle = os.path.join(SF_DIR, f'{prefix}_solar_cycle_segment.csv')
        if os.path.exists(f_cycle):
            analyze_solar_cycles(f_cycle, label)

    # === Sunspot ===
    analyze_algo12(os.path.join(SG_DIR, 'sg_algo1_total_pairs.csv'), 'Algo 1 (Total Pairs)', 'Sunspot')
    analyze_algo12(os.path.join(SG_DIR, 'sg_algo2_at_least_one.csv'), 'Algo 2 (At Least One)', 'Sunspot')
    analyze_groups(os.path.join(SG_DIR, 'sg_algo1_total_pairs.csv'), 'Sunspot', 'Algo 1')
    analyze_subsets(SG_DIR, 'sg', 'Sunspot')
    analyze_solar_cycles(os.path.join(SG_DIR, 'sg_solar_cycle_segment.csv'), 'Sunspot')
    analyze_kuiper(os.path.join(SG_DIR, 'sg_algo_kuiper_test.csv'), 'Sunspot')

    # === 总结 ===
    print_header("输出的图表文件")
    if os.path.exists(FIG_DIR):
        for f in sorted(os.listdir(FIG_DIR)):
            if f.endswith('.png'):
                print(f"  📊 {os.path.join(FIG_DIR, f)}")

    print("\n✅ 分析完成!")


if __name__ == '__main__':
    main()
