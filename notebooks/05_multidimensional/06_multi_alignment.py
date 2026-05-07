#!/usr/bin/env python3
"""
06_multi_alignment.py — 多星连珠 CEOS 分析
============================================
回应审稿人意见 #2:
  "当多星连珠时是否合增冲减的百分比显著提高，尤其是木、土连珠时是否如此；
   另外，如果木合土冲时是否部分抵消。"

Phase 1  可行性普查 (无 CTS, 秒级)
  ─ 统计各 w 下事件同时落入 0/1/2/…/7 颗行星合/冲窗口的数量分布
  ─ 快速判断哪些 (n_planets, w) 组合有足够样本量

Phase 2  连珠 CEOS + CTS 检验 (GPU 加速)
  ─ 2 星: C(7,2)=21 对 × 4 场景(均合/均冲/A合B冲/A冲B合) = 84 测试
  ─ 3 星: C(7,3)=35 组 × 8 场景 = 280 测试
  ─ 对通过小规模 CTS 预筛的组合做 N_SIM 次 CTS 模拟
  ─ 核心区别: 既有 subset scan 用 OR (bits & M ≠ 0, "至少一颗命中")
              本脚本用 AND (bits & M == M, "全部同时命中 = 连珠叠加")

7 行星排除地球 (与论文修订版一致)。

运行时间估算 (RTX 4090):
  Phase 1: < 10 s
  Phase 2: 5-30 min (取决于活跃测试数; 小 w 多数跳过, 大 w 测试多)
  无 GPU:  Phase 2 极慢 (SG 256K 事件 ~50h+; 仅 SF ~2h; 建议降 N_SIM=5000)

输出:
  results/05_multidimensional/phase1_alignment_census.csv
  results/05_multidimensional/phase2_pair_alignment_ceos.csv
  results/05_multidimensional/phase2_triple_alignment_ceos.csv

用法:
  ~/miniforge3/envs/ceos/bin/python 06_multi_alignment.py
"""

import sys, os, time, gc, warnings
import numpy as np
import pandas as pd
from itertools import combinations

# ── 路径 ──────────────────────────────────────────────────────────
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_ASYM_DIR  = os.path.abspath(os.path.join(_THIS_DIR, '..', '04_asymmetric'))
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
sys.path.insert(0, _ASYM_DIR)

import ceos_engine as ce
import algo_workers

OUTPUT_DIR = os.path.join(_PROJ_ROOT, 'results', '05_multidimensional')

# ── 7P 常量 (排除地球 399) ───────────────────────────────────────
P7_COLS      = ce.PLANET_COLS_7P   # ['199_lon','299_lon','499_lon',...]
P7_NAMES     = [ce.PLANET_MAP[c] for c in P7_COLS]
P7_SHORT     = ['Mer', 'Ven', 'Mar', 'Jup', 'Sat', 'Ura', 'Nep']
N_P          = 7
P7_IDX_IN_8P = [0, 1, 3, 4, 5, 6, 7]  # 在 8P 矩阵中的列索引

# ── 运行参数 ─────────────────────────────────────────────────────
W_RANGE      = list(range(1, 31))   # w = 1-30
N_SIM        = 50_000               # CTS 模拟次数 (与主分析一致)
N_SIM_SCREEN = 256                  # 活跃测试预筛: 小规模 CTS 次数
MIN_EVENTS   = 30                   # 观测或 CTS 期望过低则跳过完整 CTS


# ═══════════════════════════════════════════════════════════════════
# 位图计算
# ═══════════════════════════════════════════════════════════════════
def compute_bits_7p(sun_lons, planet_7p, w):
    """
    7P 合/冲命中位图.
    返回 (bits_c, bits_o), 各为 (N,) uint16 数组.
    bit i = 1 表示第 i 颗行星命中.
    """
    delta = np.mod(sun_lons[:, None] - planet_7p + 180, 360) - 180
    conj = np.abs(delta) <= w
    opp  = np.abs(np.abs(delta) - 180) <= w
    bc = np.zeros(len(sun_lons), dtype=np.uint16)
    bo = np.zeros(len(sun_lons), dtype=np.uint16)
    for i in range(planet_7p.shape[1]):
        bc |= conj[:, i].astype(np.uint16) << i
        bo |= opp[:, i].astype(np.uint16) << i
    return bc, bo


_POPCOUNT_TBL = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

def popcount(arr):
    """uint16 逐元素 popcount (查表法)"""
    return (_POPCOUNT_TBL[arr & 0xFF].astype(int)
          + _POPCOUNT_TBL[(arr >> 8) & 0xFF].astype(int))


# ═══════════════════════════════════════════════════════════════════
# 测试组合生成
# ═══════════════════════════════════════════════════════════════════
def _scenario_label(combo, phases):
    """可读的场景标签: 均合/均冲/Jup合Sat冲..."""
    if all(p == 'C' for p in phases):
        return '均合'
    if all(p == 'O' for p in phases):
        return '均冲'
    return ''.join(f"{P7_SHORT[c]}{'合' if p == 'C' else '冲'}"
                   for c, p in zip(combo, phases))


def generate_tests(max_n=3):
    """
    枚举 2~max_n 星连珠测试组合.
    2 星: 21 对 × 4 场景 = 84
    3 星: 35 组 × 8 场景 = 280
    返回 list[dict] with keys: name, scenario, n_planets, mask_c, mask_o
    """
    tests = []
    for n in range(2, max_n + 1):
        for combo in combinations(range(N_P), n):
            name = '+'.join(P7_SHORT[i] for i in combo)
            for pc in range(1 << n):
                mc = mo = np.uint16(0)
                phases = []
                for k, pi in enumerate(combo):
                    if pc & (1 << k):
                        mo |= np.uint16(1 << pi)
                        phases.append('O')
                    else:
                        mc |= np.uint16(1 << pi)
                        phases.append('C')
                tests.append(dict(
                    name=name, scenario=_scenario_label(combo, phases),
                    n_planets=n, mask_c=mc, mask_o=mo))
    return tests


# ═══════════════════════════════════════════════════════════════════
# Phase 1: 可行性普查
# ═══════════════════════════════════════════════════════════════════
def run_phase1(datasets):
    """统计各 (数据集, w) 下同时命中行星数的分布, 无 CTS"""
    print("\n" + "=" * 70)
    print("Phase 1: 多星连珠可行性普查")
    print("=" * 70)

    rows = []
    for ds, sl, p7 in datasets:
        N = len(sl)
        print(f"  {ds:22s} N={N:>9,}")
        for w in W_RANGE:
            bc, bo = compute_bits_7p(sl, p7, w)
            nc, no_ = popcount(bc), popcount(bo)
            for n in range(N_P + 1):
                cc = int(np.sum(nc == n))
                co = int(np.sum(no_ == n))
                rows.append(dict(Dataset=ds, Window=w, Phase='Conjunction',
                                 N_Simul=n, Count=cc,
                                 Pct=round(cc / N * 100, 4), N_Total=N))
                rows.append(dict(Dataset=ds, Window=w, Phase='Opposition',
                                 N_Simul=n, Count=co,
                                 Pct=round(co / N * 100, 4), N_Total=N))

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, 'phase1_alignment_census.csv')
    df.to_csv(path, index=False)
    print(f"\n  保存: {path} ({len(df):,} 行)")

    # ── 汇总表 ──
    print("\n  ≥2 星同时命中事件数 (选取 w = 1, 2, 5, 10, 20, 30):")
    hdr = f"  {'w':>3} {'Phase':>5} {'Dataset':>22} {'≥2':>8} {'n=2':>7} {'n=3':>7} {'n≥4':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for w in [1, 2, 5, 10, 20, 30]:
        if w not in W_RANGE:
            continue
        for ph in ['Conjunction', 'Opposition']:
            sub = df[(df['Window'] == w) & (df['Phase'] == ph) & (df['N_Simul'] >= 2)]
            for ds in sub['Dataset'].unique():
                sd = sub[sub['Dataset'] == ds]
                n2  = int(sd.loc[sd['N_Simul'] == 2, 'Count'].sum())
                n3  = int(sd.loc[sd['N_Simul'] == 3, 'Count'].sum())
                n4p = int(sd.loc[sd['N_Simul'] >= 4, 'Count'].sum())
                tot = n2 + n3 + n4p
                if tot > 0:
                    print(f"  {w:3d} {ph[:5]:>5} {ds:>22} "
                          f"{tot:8,} {n2:7,} {n3:7,} {n4p:7,}")
    return df


# ═══════════════════════════════════════════════════════════════════
# Phase 2: CTS 连珠检验
# ═══════════════════════════════════════════════════════════════════

# GPU 融合匹配核函数 (延迟初始化)
_GPU_MATCH = None

def _get_gpu_match():
    """CuPy ElementwiseKernel: 将 4 步位运算融合为 1 次 kernel launch"""
    global _GPU_MATCH
    if _GPU_MATCH is None:
        import cupy as cp
        _GPU_MATCH = cp.ElementwiseKernel(
            'uint16 bc, uint16 bo, uint16 mc, uint16 mo',
            'bool out',
            'out = ((bc & mc) == mc) && ((bo & mo) == mo)',
            'alignment_match')
    return _GPU_MATCH


def _cts_gpu(sun_lons, sun_idxs, ephem_7p, w, shifts, mc_arr, mo_arr):
    """
    GPU 批量 CTS 模拟.
    对每次循环时移:
      1. 计算时移后的行星位置 → 位图
      2. 用融合 kernel 对所有活跃测试计数
    返回 (n_active, n_sim) int64 数组.
    """
    import cupy as cp

    N = len(sun_lons)
    T, P = ephem_7p.shape
    n_sim = len(shifts)
    n_act = len(mc_arr)
    kern = _get_gpu_match()

    # 上传持久数据
    d_sun = cp.asarray(sun_lons, dtype=cp.float32)
    d_eph = cp.asarray(ephem_7p, dtype=cp.float32)
    d_idx = cp.asarray(sun_idxs, dtype=cp.int32)

    # 自动计算 batch 大小 (GPU 显存的 35%)
    free = cp.cuda.Device(0).mem_info[0]
    bytes_per_sim = N * P * 16 + N * 8   # 粗估: planets + bitmaps + 中间变量
    batch = max(50, min(n_sim, int(free * 0.35) // max(bytes_per_sim, 1)))

    k_sims = np.zeros((n_act, n_sim), dtype=np.int64)

    for s0 in range(0, n_sim, batch):
        s1 = min(s0 + batch, n_sim)
        b = s1 - s0

        # 1. 循环时移 → 行星位置
        bsh = cp.asarray(shifts[s0:s1], dtype=cp.int32)
        si  = (d_idx[None, :] + bsh[:, None]) % T
        sp  = d_eph[si]       # (b, N, P)
        del si

        # 2. 角度差 → 合/冲判定 → 位图
        dlt = cp.mod(d_sun[None, :, None] - sp + 180, 360) - 180
        del sp
        cj = cp.abs(dlt) <= w
        op = cp.abs(cp.abs(dlt) - 180) <= w
        del dlt
        bc = cp.zeros((b, N), dtype=cp.uint16)
        bo = cp.zeros((b, N), dtype=cp.uint16)
        for i in range(P):
            bc |= cj[:, :, i].astype(cp.uint16) << i
            bo |= op[:, :, i].astype(cp.uint16) << i
        del cj, op

        # 3. 对每个活跃测试: 融合 kernel 计数
        for ai in range(n_act):
            m = kern(bc, bo,
                     cp.uint16(int(mc_arr[ai])),
                     cp.uint16(int(mo_arr[ai])))
            k_sims[ai, s0:s1] = cp.asnumpy(cp.sum(m, axis=1))
            del m

        del bc, bo
        cp.get_default_memory_pool().free_all_blocks()

    del d_sun, d_eph, d_idx
    cp.get_default_memory_pool().free_all_blocks()
    return k_sims


def _cts_cpu(sun_lons, sun_idxs, ephem_7p, w, shifts, mc_arr, mo_arr):
    """CPU CTS (退回模式), 返回 (n_active, n_sim)"""
    N = len(sun_lons)
    T = ephem_7p.shape[0]
    n_sim = len(shifts)
    n_act = len(mc_arr)
    k_sims = np.zeros((n_act, n_sim), dtype=np.int64)

    for s, sh in enumerate(shifts):
        sp = ephem_7p[(sun_idxs + sh) % T]
        bc, bo = compute_bits_7p(sun_lons, sp, w)
        for ai in range(n_act):
            mc, mo = mc_arr[ai], mo_arr[ai]
            m_c = (bc & mc) == mc if mc else np.ones(N, dtype=bool)
            m_o = (bo & mo) == mo if mo else np.ones(N, dtype=bool)
            k_sims[ai, s] = np.sum(m_c & m_o)
        if s > 0 and s % 5000 == 0:
            print(f"        CTS {s}/{n_sim} ...", flush=True)

    return k_sims


def _run_cts_batch(sun_lons, sun_idxs, ephem_7p, w, shifts, mc_arr, mo_arr):
    """统一 CTS 调度: 优先 GPU, 失败退回 CPU."""
    if len(mc_arr) == 0 or len(shifts) == 0:
        return np.zeros((len(mc_arr), len(shifts)), dtype=np.int64)

    if algo_workers.HAS_GPU:
        try:
            return _cts_gpu(sun_lons, sun_idxs, ephem_7p, w, shifts, mc_arr, mo_arr)
        except Exception as e:
            print(f"    [GPU→CPU] {e}")
    return _cts_cpu(sun_lons, sun_idxs, ephem_7p, w, shifts, mc_arr, mo_arr)


def _apply_fdr(df):
    """按 (Dataset, N_Planets) family 应用 BH-FDR."""
    if df.empty:
        df = df.copy()
        df['p_adj_bh'] = np.nan
        df['sig_fdr'] = False
        return df

    out = df.copy()
    out['p_adj_bh'] = np.nan
    out['sig_fdr'] = False
    cts_mask = out['CTS_done'] == True
    for _, grp in out[cts_mask].groupby(['Dataset', 'N_Planets'], sort=False):
        idx = grp.index
        p_adj = ce.fdr_correction(grp['p_val'].values)
        out.loc[idx, 'p_adj_bh'] = np.round(p_adj, 6)
        out.loc[idx, 'sig_fdr'] = p_adj < 0.05
    return out


def run_phase2(ds_list, ephem_8p, tests):
    """Phase 2 主体: 对所有 (数据集, w, 测试) 进行连珠 CEOS 分析"""
    ephem_7p = ephem_8p[:, P7_IDX_IN_8P].astype(np.float64)
    T = ephem_7p.shape[0]
    nt = len(tests)
    mc_all = np.array([t['mask_c'] for t in tests], dtype=np.uint16)
    mo_all = np.array([t['mask_o'] for t in tests], dtype=np.uint16)

    rows = []

    for ds, sl, p7, si in ds_list:
        N = len(sl)
        print(f"\n  [{ds}] N={N:,}")

        for w in W_RANGE:
            t0 = time.time()

            # ── 观测位图 & 计数 ──
            bc, bo = compute_bits_7p(sl, p7, w)
            kobs = np.empty(nt, dtype=np.int64)
            for ti in range(nt):
                mc, mo = mc_all[ti], mo_all[ti]
                m_c = (bc & mc) == mc if mc else np.ones(N, dtype=bool)
                m_o = (bo & mo) == mo if mo else np.ones(N, dtype=bool)
                kobs[ti] = np.sum(m_c & m_o)

            # ── 活跃测试预筛 ──
            # 用小规模 CTS 估计 k_exp_screen, 避免漏掉 k_obs 很低但 k_exp 较高的抑制型结果.
            seed = algo_workers.derive_seed('multi_alignment', ds, w)
            rng = np.random.default_rng(seed)
            shifts = rng.integers(0, T, size=N_SIM)
            n_screen = min(N_SIM_SCREEN, N_SIM)
            k_screen = _run_cts_batch(sl, si, ephem_7p, w,
                                      shifts[:n_screen], mc_all, mo_all)
            kexp_screen = (k_screen.mean(axis=1)
                           if n_screen > 0 else np.zeros(nt, dtype=np.float64))
            active = np.maximum(kobs, kexp_screen) >= MIN_EVENTS
            na      = int(active.sum())
            act_idx = np.where(active)[0]
            act_map = {int(v): i for i, v in enumerate(act_idx)}

            # ── CTS 模拟 (仅对活跃测试) ──
            k_sims = None
            if na > 0:
                act_mc = mc_all[act_idx]
                act_mo = mo_all[act_idx]
                if n_screen >= N_SIM:
                    k_sims = k_screen[act_idx]
                else:
                    k_tail = _run_cts_batch(sl, si, ephem_7p, w,
                                            shifts[n_screen:], act_mc, act_mo)
                    if n_screen > 0:
                        k_sims = np.concatenate([k_screen[act_idx], k_tail], axis=1)
                    else:
                        k_sims = k_tail

            # ── 统计 ──
            n_sig = 0
            for ti in range(nt):
                ko = int(kobs[ti])
                if ti in act_map and k_sims is not None:
                    ks = k_sims[act_map[ti]]
                    ke = float(ks.mean())
                    sd = float(ks.std())
                    ratio = ko / ke * 100 if ke > 0 else 0.0
                    pl = (np.sum(ks <= ko) + 1) / (N_SIM + 1)
                    pr = (np.sum(ks >= ko) + 1) / (N_SIM + 1)
                    pv = min(2 * min(pl, pr), 1.0)
                    z  = (ko - ke) / sd if sd > 0 else 0.0
                    cts = True
                    if pv < 0.05:
                        n_sig += 1
                else:
                    ke = ratio = pv = z = np.nan
                    cts = False

                rows.append(dict(
                    Dataset=ds, Window=w,
                    Combo=tests[ti]['name'],
                    Scenario=tests[ti]['scenario'],
                    N_Planets=tests[ti]['n_planets'],
                    k_obs=ko,
                    k_exp=round(ke, 2) if cts else np.nan,
                    Ratio=round(ratio, 2) if cts else np.nan,
                    p_val=round(pv, 6) if cts else np.nan,
                    Z_score=round(z, 2) if cts else np.nan,
                    N_Records=N, CTS_done=cts))

            dt = time.time() - t0
            print(f"    w={w:2d}  active={na:3d}/{nt}  screen={n_screen:3d}  "
                  f"sig(p<.05)={n_sig:2d}  {dt:.1f}s")

        # 每个数据集完成后增量保存 (防崩溃丢失)
        _save_incremental(rows, tests)

    return pd.DataFrame(rows)


def _save_incremental(rows, tests):
    """增量保存已有结果"""
    if not rows:
        return
    df = _apply_fdr(pd.DataFrame(rows))
    df2 = df[df['N_Planets'] == 2]
    df3 = df[df['N_Planets'] == 3]
    f2 = os.path.join(OUTPUT_DIR, 'phase2_pair_alignment_ceos.csv')
    f3 = os.path.join(OUTPUT_DIR, 'phase2_triple_alignment_ceos.csv')
    if not df2.empty:
        df2.to_csv(f2, index=False)
    if not df3.empty:
        df3.to_csv(f3, index=False)


# ═══════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════
def load_all():
    """加载 SG (All 阶段) + SF 数据, 按 7P 列提取"""
    datasets = []   # Phase 1: (name, sun_lons, planet_7p)
    ds_idx   = []   # Phase 2: (name, sun_lons, planet_7p, sun_idxs)
    ephem_8p = None

    # ── 黑子 ──
    try:
        df, ephem_8p = ce.load_sunspot_data()
        sl = df['hme_lon'].values.astype(np.float64)
        p7 = df[P7_COLS].values.astype(np.float64)
        si = df['ephem_idx_daily'].values.astype(int)

        datasets.append(('SG_Total', sl, p7))
        ds_idx.append(('SG_Total', sl, p7, si))

        area_tags = {'Small <100': 'Small',
                     'Medium 100-500': 'Medium',
                     'Large 500-2000': 'Large',
                     'XLarge >2000': 'XLarge'}
        for grp, tag in area_tags.items():
            m = (df['Group'] == grp).values
            if m.sum() >= 100:
                datasets.append((f'SG_{tag}', sl[m], p7[m]))
                ds_idx.append((f'SG_{tag}', sl[m], p7[m], si[m]))

        print(f"  黑子: {len(df):,} 条 (All 阶段)")
    except Exception as e:
        print(f"  [SG 加载失败] {e}")

    # ── 耀斑 ──
    try:
        df, ephem_sf = ce.load_flare_data()
        if ephem_8p is None:
            ephem_8p = ephem_sf
        sl = df['hme_lon'].values.astype(np.float64)
        p7 = df[P7_COLS].values.astype(np.float64)
        si = df['ephem_idx_daily'].values.astype(int)

        datasets.append(('SF_Total', sl, p7))
        ds_idx.append(('SF_Total', sl, p7, si))

        for cls in ['B-Class', 'C-Class', 'M-Class', 'X-Class']:
            m = (df['Group'] == cls).values
            if m.sum() >= MIN_EVENTS:
                datasets.append((f'SF_{cls}', sl[m], p7[m]))
                ds_idx.append((f'SF_{cls}', sl[m], p7[m], si[m]))

        print(f"  耀斑: {len(df):,} 条")
    except Exception as e:
        print(f"  [SF 加载失败] {e}")

    return datasets, ds_idx, ephem_8p


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  多星连珠 CEOS 分析  (审稿人意见 #2)")
    print(f"  7P (排除地球), w=1-{W_RANGE[-1]}, CTS N_SIM={N_SIM:,}")
    print(f"  MIN_EVENTS={MIN_EVENTS}")
    print(f"  GPU: {'已启用' if algo_workers.HAS_GPU else '未检测到 (CPU 模式, 预计较慢)'}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_global = time.time()

    # ── 加载 ──
    print("\n--- 加载数据 ---")
    datasets, ds_idx, ephem_8p = load_all()
    if not datasets:
        print("  无数据可用, 退出")
        return

    # ── Phase 1 ──
    df_census = run_phase1(datasets)

    # ── 生成测试组合 ──
    tests = generate_tests(max_n=3)
    n2 = sum(1 for t in tests if t['n_planets'] == 2)
    n3 = sum(1 for t in tests if t['n_planets'] == 3)
    print(f"\n  测试组合: 双星={n2}, 三星={n3}, 合计={len(tests)}")

    # ── Phase 2 ──
    print("\n" + "=" * 70)
    print("Phase 2: 连珠 CEOS + CTS 检验")
    print("  连珠判定: (bits_c & mask_c == mask_c) AND (bits_o & mask_o == mask_o)")
    print(f"  max(k_obs, k_exp_screen@{N_SIM_SCREEN}) < {MIN_EVENTS} 的测试跳过完整 CTS")
    print(f"  BH-FDR 多重比较校正按 (Dataset, N_Planets) 分组")
    print("=" * 70)

    df_ceos = run_phase2(ds_idx, ephem_8p, tests)

    # ── BH-FDR 多重比较校正 ──
    df_ceos = _apply_fdr(df_ceos)
    cts_mask = df_ceos['CTS_done'] == True
    n_sig_raw = int((df_ceos[cts_mask]['p_val'] < 0.05).sum())
    n_sig_fdr = int(df_ceos['sig_fdr'].sum())
    print(f"\n  FDR 校正: raw p<0.05={n_sig_raw:,}, BH-FDR q<0.05={n_sig_fdr:,}")

    # ── 保存最终结果 ──
    df2 = df_ceos[df_ceos['N_Planets'] == 2]
    df3 = df_ceos[df_ceos['N_Planets'] == 3]
    f2 = os.path.join(OUTPUT_DIR, 'phase2_pair_alignment_ceos.csv')
    f3 = os.path.join(OUTPUT_DIR, 'phase2_triple_alignment_ceos.csv')
    df2.to_csv(f2, index=False)
    df3.to_csv(f3, index=False)
    print(f"\n  双星结果: {f2}")
    print(f"           ({len(df2):,} 行, 其中 CTS 完成: "
          f"{int(df2['CTS_done'].sum()):,})")
    print(f"  三星结果: {f3}")
    print(f"           ({len(df3):,} 行, 其中 CTS 完成: "
          f"{int(df3['CTS_done'].sum()):,})")

    # ── 木星+土星 高亮 (按 FDR 显著结果展示) ──
    print("\n" + "─" * 70)
    print("  木星+土星 (Jup+Sat) 连珠结果")
    print("─" * 70)
    js = df_ceos[(df_ceos['Combo'] == 'Jup+Sat') & (df_ceos['CTS_done'] == True)]
    if js.empty:
        print("  (无 CTS 完成的结果)")
    else:
        sig = js[js['sig_fdr'] == True].sort_values('p_adj_bh')
        if sig.empty:
            print("  (无 BH-FDR q<0.05 的结果)")
        else:
            print(f"  {'Dataset':>20s} {'w':>3} {'Scenario':>14s} "
                  f"{'k_obs':>6} {'Ratio%':>8} {'p_raw':>8} {'p_bh':>8} {'FDR':>4}")
            for _, r in sig.head(30).iterrows():
                fdr_tag = ('*' if r.get('sig_fdr', False) else '')
                print(f"  {r['Dataset']:>20s} {r['Window']:3.0f} "
                      f"{r['Scenario']:>14s} {r['k_obs']:6.0f} "
                      f"{r['Ratio']:8.2f} {r['p_val']:8.4f} "
                      f"{r['p_adj_bh']:8.4f} {fdr_tag:>4}")

    # ── 全局 Top 20 (FDR 显著) ──
    print("\n" + "─" * 70)
    print("  全局 FDR 显著结果 Top 20 (BH q < 0.05)")
    print("─" * 70)
    sig_all = df_ceos[df_ceos['sig_fdr'] == True]
    if sig_all.empty:
        print("  (无 BH-FDR q<0.05 的结果)")
        # 回退: 显示 raw p 最小的 20 个
        print("\n  (回退) raw p 最小 Top 20:")
        fallback = df_ceos[df_ceos['CTS_done'] == True].nsmallest(20, 'p_val')
        for _, r in fallback.iterrows():
            print(f"    {r['Dataset']:>20s} w={r['Window']:2.0f} "
                  f"{r['Combo']:>15s} {r['Scenario']:>14s} "
                  f"k={r['k_obs']:5.0f} Ratio={r['Ratio']:7.2f}% "
                  f"p={r['p_val']:.6f} q={r['p_adj_bh']:.6f}")
    else:
        print(f"  {'Dataset':>20s} {'w':>3} {'Combo':>15s} "
              f"{'Scenario':>14s} {'k_obs':>6} {'Ratio%':>8} "
              f"{'p_raw':>10} {'q_bh':>10}")
        for _, r in sig_all.sort_values('p_adj_bh').head(20).iterrows():
            print(f"  {r['Dataset']:>20s} {r['Window']:3.0f} "
                  f"{r['Combo']:>15s} {r['Scenario']:>14s} "
                  f"{r['k_obs']:6.0f} {r['Ratio']:8.2f} "
                  f"{r['p_val']:10.6f} {r['p_adj_bh']:10.6f}")

    elapsed = time.time() - t_global
    print(f"\n{'=' * 70}")
    print(f"  全部完成. 耗时: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
