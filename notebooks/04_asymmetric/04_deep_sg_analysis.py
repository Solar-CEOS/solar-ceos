#!/usr/bin/env python3
"""
04_deep_sg_analysis.py — 黑子分组深度分析
找出稀释证据和分组亮点
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SG_DIR = os.path.join(PROJECT_ROOT, 'results', '04_asymmetric', 'sg')
SF_DIR = os.path.join(PROJECT_ROOT, 'results', '04_asymmetric', 'sf')
FIG_DIR = os.path.join(PROJECT_ROOT, 'results', '04_asymmetric', 'analysis')
os.makedirs(FIG_DIR, exist_ok=True)

def hr():
    print("=" * 75)

def main():

    # ============================================================
    # 1. 黑子面积分组梯度分析 (稀释证据)
    # ============================================================
    print("\n"); hr()
    print("  1. 黑子面积分组 Ratio 梯度 (稀释证据)")
    hr()

    df = pd.read_csv(os.path.join(SG_DIR, 'sg_algo1_total_pairs.csv'))

    # 选 All 阶段
    df_all = df[df['Stage'] == 'All']

    area_order = ['Small <100', 'Medium 100-500', 'Large 500-2000', 'XLarge >2000', 'Total']
    area_n = {}

    print("\n  Algo 1 - All Stage - Conjunction Ratio by Area:")
    print(f"  {'Group':>20}  {'N':>7}  {'w=1':>8}  {'w=2':>8}  {'w=3':>8}  {'w=5':>8}")
    print(f"  {'─'*20}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for g in area_order:
        dg = df_all[(df_all['Group'] == g) & (df_all['Type'] == 'Conjunction')]
        if len(dg) == 0: continue
        n = dg.iloc[0]['N_Records']
        area_n[g] = n
        vals = []
        for w in [1, 2, 3, 5]:
            row = dg[dg['Window'] == w]
            if len(row) > 0:
                vals.append(f"{row.iloc[0]['Ratio']:>7.1f}%")
            else:
                vals.append(f"{'N/A':>8}")
        print(f"  {g:>20}  {n:>7}  {'  '.join(vals)}")

    print("\n  Opposition Ratio by Area:")
    print(f"  {'Group':>20}  {'N':>7}  {'w=1':>8}  {'w=2':>8}  {'w=3':>8}  {'w=5':>8}")
    print(f"  {'─'*20}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for g in area_order:
        dg = df_all[(df_all['Group'] == g) & (df_all['Type'] == 'Opposition')]
        if len(dg) == 0: continue
        n = dg.iloc[0]['N_Records']
        vals = []
        for w in [1, 2, 3, 5]:
            row = dg[dg['Window'] == w]
            if len(row) > 0:
                vals.append(f"{row.iloc[0]['Ratio']:>7.1f}%")
            else:
                vals.append(f"{'N/A':>8}")
        print(f"  {g:>20}  {n:>7}  {'  '.join(vals)}")

    # ============================================================
    # 2. 黑子历程分组分析
    # ============================================================
    print("\n"); hr()
    print("  2. 黑子历程分组 (Onset/Duration/Dissipation/Daily) Ratio")
    hr()

    stages = ['Onset', 'Duration', 'Dissipation', 'Daily', 'All']
    print("\n  Algo 1 - Total group - Conjunction Ratio:")
    print(f"  {'Stage':>15}  {'N':>7}  {'w=1':>8}  {'w=2':>8}  {'w=3':>8}  {'w=5':>8}")
    print(f"  {'─'*15}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for stage in stages:
        ds = df[(df['Stage'] == stage) & (df['Group'] == 'Total') & (df['Type'] == 'Conjunction')]
        if len(ds) == 0: continue
        n = ds.iloc[0]['N_Records']
        vals = []
        for w in [1, 2, 3, 5]:
            row = ds[ds['Window'] == w]
            if len(row) > 0:
                vals.append(f"{row.iloc[0]['Ratio']:>7.1f}%")
            else:
                vals.append(f"{'N/A':>8}")
        print(f"  {stage:>15}  {n:>7}  {'  '.join(vals)}")

    print("\n  Algo 1 - Total group - Opposition Ratio:")
    print(f"  {'Stage':>15}  {'N':>7}  {'w=1':>8}  {'w=2':>8}  {'w=3':>8}  {'w=5':>8}")
    print(f"  {'─'*15}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for stage in stages:
        ds = df[(df['Stage'] == stage) & (df['Group'] == 'Total') & (df['Type'] == 'Opposition')]
        if len(ds) == 0: continue
        n = ds.iloc[0]['N_Records']
        vals = []
        for w in [1, 2, 3, 5]:
            row = ds[ds['Window'] == w]
            if len(row) > 0:
                vals.append(f"{row.iloc[0]['Ratio']:>7.1f}%")
            else:
                vals.append(f"{'N/A':>8}")
        print(f"  {stage:>15}  {n:>7}  {'  '.join(vals)}")

    # ============================================================
    # 3. 黑子 XLarge + 各阶段交叉分析
    # ============================================================
    print("\n"); hr()
    print("  3. 黑子大面积分组 × 各阶段 (交叉分析)")
    hr()

    for target_group in ['Large 500-2000', 'XLarge >2000', 'Medium 100-500']:
        print(f"\n  {target_group} - Conjunction Ratio:")
        print(f"  {'Stage':>15}  {'N':>7}  {'w=1':>8}  {'w=2':>8}  {'w=3':>8}  {'w=5':>8}")
    
        for stage in stages:
            ds = df[(df['Stage'] == stage) & (df['Group'] == target_group) & (df['Type'] == 'Conjunction')]
            if len(ds) == 0: continue
            n = ds.iloc[0]['N_Records']
            vals = []
            for w in [1, 2, 3, 5]:
                row = ds[ds['Window'] == w]
                if len(row) > 0:
                    r = row.iloc[0]['Ratio']
                    p = row.iloc[0]['p_val']
                    sig = '*' if p < 0.05 else ''
                    vals.append(f"{r:>6.0f}%{sig}")
                else:
                    vals.append(f"{'N/A':>8}")
            print(f"  {stage:>15}  {n:>7}  {'  '.join(vals)}")

    # ============================================================
    # 4. 耀斑分级 vs 黑子面积: Asym (Conj-Opp) 对比
    # ============================================================
    print("\n"); hr()
    print("  4. 不对称信号强度 (Conj_Ratio - Opp_Ratio) 对比")
    hr()

    df_sf = pd.read_csv(os.path.join(SF_DIR, 'sf_algo1_total_pairs.csv'))

    print("\n  耀斑 (by class):")
    print(f"  {'Group':>15}  {'N':>7}  {'Asym_w1':>10}  {'Asym_w2':>10}  {'Asym_w3':>10}  {'Asym_w5':>10}")

    for g in ['B-Class', 'C-Class', 'M-Class', 'X-Class', 'Total']:
        vals = []
        n = 0
        for w in [1, 2, 3, 5]:
            conj = df_sf[(df_sf['Group'] == g) & (df_sf['Window'] == w) & (df_sf['Type'] == 'Conjunction')]
            opp = df_sf[(df_sf['Group'] == g) & (df_sf['Window'] == w) & (df_sf['Type'] == 'Opposition')]
            if len(conj) > 0 and len(opp) > 0:
                n = conj.iloc[0]['N_Records']
                asym = conj.iloc[0]['Ratio'] - opp.iloc[0]['Ratio']
                vals.append(f"{asym:>+9.1f}")
            else:
                vals.append(f"{'N/A':>10}")
        print(f"  {g:>15}  {n:>7}  {'  '.join(vals)}")

    print("\n  黑子 All stage (by area):")
    print(f"  {'Group':>20}  {'N':>7}  {'Asym_w1':>10}  {'Asym_w2':>10}  {'Asym_w3':>10}  {'Asym_w5':>10}")

    for g in area_order:
        vals = []
        n = 0
        for w in [1, 2, 3, 5]:
            conj = df_all[(df_all['Group'] == g) & (df_all['Window'] == w) & (df_all['Type'] == 'Conjunction')]
            opp = df_all[(df_all['Group'] == g) & (df_all['Window'] == w) & (df_all['Type'] == 'Opposition')]
            if len(conj) > 0 and len(opp) > 0:
                n = conj.iloc[0]['N_Records']
                asym = conj.iloc[0]['Ratio'] - opp.iloc[0]['Ratio']
                vals.append(f"{asym:>+9.1f}")
            else:
                vals.append(f"{'N/A':>10}")
        print(f"  {g:>20}  {n:>7}  {'  '.join(vals)}")

    # ============================================================
    # 5. 绘图: 面积梯度 vs CEOS Asym (稀释证据)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CEOS Asymmetry by Data Category', fontsize=14, fontweight='bold')

    # 左图: 耀斑分级
    ax = axes[0]
    flare_groups = ['B-Class', 'C-Class', 'M-Class', 'X-Class', 'Total']
    for g in flare_groups:
        ws = []
        asyms = []
        for w in sorted(df_sf['Window'].unique()):
            conj = df_sf[(df_sf['Group'] == g) & (df_sf['Window'] == w) & (df_sf['Type'] == 'Conjunction')]
            opp = df_sf[(df_sf['Group'] == g) & (df_sf['Window'] == w) & (df_sf['Type'] == 'Opposition')]
            if len(conj) > 0 and len(opp) > 0:
                ws.append(w)
                asyms.append(conj.iloc[0]['Ratio'] - opp.iloc[0]['Ratio'])
        if ws:
            ax.plot(ws, asyms, '-o', markersize=5, label=g)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Window w (deg)')
    ax.set_ylabel('Asym (Conj_Ratio - Opp_Ratio)')
    ax.set_title('Flare: by X-ray Class')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 右图: 黑子面积
    ax = axes[1]
    for g in area_order:
        ws = []
        asyms = []
        for w in sorted(df_all['Window'].unique()):
            conj = df_all[(df_all['Group'] == g) & (df_all['Window'] == w) & (df_all['Type'] == 'Conjunction')]
            opp = df_all[(df_all['Group'] == g) & (df_all['Window'] == w) & (df_all['Type'] == 'Opposition')]
            if len(conj) > 0 and len(opp) > 0:
                ws.append(w)
                asyms.append(conj.iloc[0]['Ratio'] - opp.iloc[0]['Ratio'])
        if ws:
            ax.plot(ws, asyms, '-o', markersize=5, label=g)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Window w (deg)')
    ax.set_ylabel('Asym (Conj_Ratio - Opp_Ratio)')
    ax.set_title('Sunspot: by Area Category (All Stage)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'ceos_asym_by_category.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [Figure A] {fig_path}")

    # ============================================================
    # 6. 绘图: w=1-20 衰减曲线 (耀斑 vs 黑子 对比)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CEOS Ratio Decay: Flare vs Sunspot (Total)', fontsize=14, fontweight='bold')

    for ax, etype, color_f, color_s in zip(axes, ['Conjunction', 'Opposition'], ['blue', 'blue'], ['red', 'red']):
        # 耀斑
        sf_t = df_sf[(df_sf['Group'] == 'Total') & (df_sf['Type'] == etype)]
        if len(sf_t) > 0:
            ax.plot(sf_t['Window'], sf_t['Ratio'], '-o', color='darkorange', markersize=4, label=f'Flare (N={sf_t.iloc[0]["N_Records"]})')
    
        # 黑子 All-Total
        sg_t = df_all[(df_all['Group'] == 'Total') & (df_all['Type'] == etype)]
        if len(sg_t) > 0:
            ax.plot(sg_t['Window'], sg_t['Ratio'], '-s', color='steelblue', markersize=4, label=f'Sunspot (N={sg_t.iloc[0]["N_Records"]})')
    
        # 黑子 XLarge
        sg_xl = df_all[(df_all['Group'] == 'XLarge >2000') & (df_all['Type'] == etype)]
        if len(sg_xl) > 0:
            ax.plot(sg_xl['Window'], sg_xl['Ratio'], '--^', color='green', markersize=4, label=f'Sunspot XLarge (N={sg_xl.iloc[0]["N_Records"]})')
    
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(etype)
        ax.set_xlabel('Window w (deg)')
        ax.set_ylabel('Ratio (%)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'flare_vs_sunspot_decay.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Figure B] {fig_path}")

    # ============================================================
    # 7. 绘图: 黑子阶段 × Asym 热图
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    stages_plot = ['Onset', 'Duration', 'Dissipation', 'Daily', 'All']
    groups_plot = ['Small <100', 'Medium 100-500', 'Large 500-2000', 'XLarge >2000']
    w_plot = 3  # 固定一个窗口

    matrix = np.full((len(stages_plot), len(groups_plot)), np.nan)

    for i, stage in enumerate(stages_plot):
        for j, group in enumerate(groups_plot):
            conj = df[(df['Stage'] == stage) & (df['Group'] == group) & (df['Window'] == w_plot) & (df['Type'] == 'Conjunction')]
            opp = df[(df['Stage'] == stage) & (df['Group'] == group) & (df['Window'] == w_plot) & (df['Type'] == 'Opposition')]
            if len(conj) > 0 and len(opp) > 0:
                matrix[i, j] = conj.iloc[0]['Ratio'] - opp.iloc[0]['Ratio']

    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-60, vmax=60)
    ax.set_xticks(range(len(groups_plot)))
    ax.set_xticklabels(groups_plot, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(stages_plot)))
    ax.set_yticklabels(stages_plot)
    ax.set_title(f'Sunspot CEOS Asymmetry Heatmap (w={w_plot})', fontsize=13, fontweight='bold')

    # 在每个格子中标注数值
    for i in range(len(stages_plot)):
        for j in range(len(groups_plot)):
            v = matrix[i, j]
            if not np.isnan(v):
                color = 'white' if abs(v) > 30 else 'black'
                ax.text(j, i, f'{v:+.0f}', ha='center', va='center', color=color, fontsize=10, fontweight='bold')

    plt.colorbar(im, label='Asym (Conj% - Opp%)')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'sunspot_stage_area_heatmap.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Figure C] {fig_path}")

    # ============================================================
    # 8. 总结: 论文图表建议
    # ============================================================
    print("\n"); hr()
    print("  论文图表建议")
    hr()
    print("""
      [核心证据图]
      Fig.A  CEOS Asym by Category     — 耀斑/黑子分组 Asym 对比, 证明稀释效应
      Fig.B  Flare vs Sunspot Decay    — w=1-20 衰减曲线, 耀斑信号远强于黑子
      Fig.C  Sunspot Heatmap           — 阶段 × 面积交叉热图, 直观展示 XLarge 信号
  
      [支撑图]
      Fig.D  Subset Asym Distribution  — 含/不含地球子集不对称分布对比
      Fig.E  Solar Cycle Bars          — Venus 4 周期全部正向 (耀斑)
  
      [附录/补充]
      Fig.F  Algo 1 vs Algo 2 Decay    — Total Pairs vs At Least One 对比
    """)

    print("\n✅ 深度分析完成!")


if __name__ == '__main__':
    main()
