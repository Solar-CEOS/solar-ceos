# algo_workers.py
# ================
# CTS 并行计算 worker 函数 — 支持 GPU (CuPy) 加速，自动退回 CPU (NumPy)
# 注意: 这些函数必须是 top-level 函数，不能是闭包或 lambda

import os, sys
import numpy as np

# =============================================================================
# GPU 检测 (自动退回 CPU)
# =============================================================================
# 自动设置 CUDA_PATH 为 conda 环境目录 (如果未设置)
if 'CUDA_PATH' not in os.environ:
    _conda_prefix = sys.prefix  # e.g. /home/bml/miniforge3/envs/ceos
    if os.path.isdir(os.path.join(_conda_prefix, 'lib')):
        os.environ['CUDA_PATH'] = _conda_prefix

try:
    import cupy as cp
    _test = cp.zeros(1)  # 验证 CUDA 可用
    del _test
    HAS_GPU = True
    print("[algo_workers] ✅ CuPy GPU 加速已启用 (4090)")
except Exception:
    cp = None
    HAS_GPU = False
    print("[algo_workers] ⚠️ CuPy 不可用, 使用 CPU 多进程模式")

# =============================================================================
# 1. 核心几何计算函数 (Algo 1 & 2) — CPU 版
# =============================================================================

def count_events_vectorized(sunspot_lons, planet_lons, w, event_type):
    """
    Algo 1: Total Pairs — 统计所有 (事件, 行星) 对中满足冲/合条件的总数
    sunspot_lons: (N,) 太阳活动事件经度
    planet_lons:  (N, P) 各行星经度 (P=行星数)
    w: 容差窗口 (度)
    event_type: 'Conjunction' 或 'Opposition'
    """
    if len(sunspot_lons) == 0:
        return 0
    # 角度差归一化到 [-180, 180]
    delta = np.mod(sunspot_lons[:, np.newaxis] - planet_lons + 180, 360) - 180
    if event_type == 'Conjunction':
        is_event = np.abs(delta) <= w
    elif event_type == 'Opposition':
        is_event = np.abs(np.abs(delta) - 180) <= w
    else:
        return 0
    return int(np.sum(is_event))


def count_events_at_least_once(sunspot_lons, planet_lons, w, event_type):
    """
    Algo 2: At Least One — 统计至少有一颗行星满足条件的事件数
    """
    if len(sunspot_lons) == 0:
        return 0
    delta = np.mod(sunspot_lons[:, np.newaxis] - planet_lons + 180, 360) - 180
    if event_type == 'Conjunction':
        is_evt_mat = np.abs(delta) <= w
    elif event_type == 'Opposition':
        is_evt_mat = np.abs(np.abs(delta) - 180) <= w
    else:
        return 0
    return int(np.sum(np.any(is_evt_mat, axis=1)))


# =============================================================================
# 2. CTS Worker 函数 (供 multiprocessing.Pool 调用, CPU 退回模式)
# =============================================================================

def cts_worker_algo1(seed, sunspot_lons, ephem_matrix, sunspot_indices, w, event_type):
    """Algo 1 CTS Worker: 循环时移后计算 Total Pairs"""
    np.random.seed(seed)
    T = ephem_matrix.shape[0]
    shift = np.random.randint(0, T)
    shifted_indices = (sunspot_indices + shift) % T
    shifted_planets = ephem_matrix[shifted_indices]
    return count_events_vectorized(sunspot_lons, shifted_planets, w, event_type)


def cts_worker_algo2(seed, sunspot_lons, ephem_matrix, sunspot_indices, w, event_type):
    """Algo 2 CTS Worker: 循环时移后计算 At Least One"""
    np.random.seed(seed)
    T = ephem_matrix.shape[0]
    shift = np.random.randint(0, T)
    shifted_indices = (sunspot_indices + shift) % T
    shifted_planets = ephem_matrix[shifted_indices]
    return count_events_at_least_once(sunspot_lons, shifted_planets, w, event_type)


# =============================================================================
# 3. GPU 批量 CTS 模拟 — 一次完成全部 N_SIM 次模拟
# =============================================================================

def cts_batch_gpu(sun_lons, ephem_matrix, sun_idxs, w, event_type,
                  n_sim, algo_type='algo1', seed=42, batch_size=None):
    """
    GPU 批量 CTS: 在 GPU 上一次完成 n_sim 次循环时移模拟。

    参数:
        sun_lons:     (N,)   事件经度 (numpy)
        ephem_matrix: (T, P) 星历矩阵 (numpy)
        sun_idxs:     (N,)   事件对应的星历索引 (numpy)
        w:            int    窗口半宽 (度)
        event_type:   str    'Conjunction' 或 'Opposition'
        n_sim:        int    模拟次数
        algo_type:    str    'algo1' 或 'algo2'
        seed:         int    随机种子
        batch_size:   int    GPU 分批大小 (None=自动计算)

    返回:
        k_sims: (n_sim,) numpy 数组, 每次模拟的事件计数
    """
    if not HAS_GPU:
        raise RuntimeError("CuPy 不可用")

    N = len(sun_lons)
    T, P = ephem_matrix.shape

    # 自动计算 batch_size: 每个 batch 需要 ~batch×N×P×12 bytes (idx+planets+delta)
    if batch_size is None:
        free_mem = cp.cuda.Device(0).mem_info[0]  # 可用显存 (bytes)
        safe_mem = int(free_mem * 0.6)  # 使用 60% 的可用显存
        bytes_per_sim = N * P * 12 + N * 4  # float32×3 + int32 索引
        batch_size = max(100, min(5000, safe_mem // max(bytes_per_sim, 1)))

    # 数据上传到 GPU
    d_sun = cp.asarray(sun_lons, dtype=cp.float32)
    d_ephem = cp.asarray(ephem_matrix, dtype=cp.float32)
    d_idx = cp.asarray(sun_idxs, dtype=cp.int32)

    # 生成全部随机 shift
    rng = np.random.default_rng(seed)
    all_shifts = rng.integers(0, T, size=n_sim)

    k_sims = np.empty(n_sim, dtype=np.int64)

    # 分批处理避免显存溢出
    for start in range(0, n_sim, batch_size):
        end = min(start + batch_size, n_sim)
        b_shifts = cp.asarray(all_shifts[start:end], dtype=cp.int32)

        # 批量索引: (batch, N) → shifted_indices
        shifted_idx = (d_idx[cp.newaxis, :] + b_shifts[:, cp.newaxis]) % T

        # 批量行星位置: (batch, N, P)
        shifted_planets = d_ephem[shifted_idx]
        del shifted_idx

        # 批量角度差: (batch, N, P)
        delta = cp.mod(d_sun[cp.newaxis, :, cp.newaxis] - shifted_planets + 180, 360) - 180
        del shifted_planets

        if event_type == 'Conjunction':
            is_event = cp.abs(delta) <= w
        else:
            is_event = cp.abs(cp.abs(delta) - 180) <= w
        del delta

        if algo_type == 'algo1':
            counts = cp.sum(is_event, axis=(1, 2))
        else:
            counts = cp.sum(cp.any(is_event, axis=2), axis=1)
        del is_event

        k_sims[start:end] = cp.asnumpy(counts)
        del counts

    # 清理 GPU 内存
    del d_sun, d_ephem, d_idx
    cp.get_default_memory_pool().free_all_blocks()

    return k_sims


def cts_batch_gpu_decay(sun_lons, ephem_7p, sun_idxs, w, event_type,
                        n_sim, seed=42, batch_size=5000):
    """
    Decay boundary 专用 GPU 批量 CTS (Total Pairs, 7P)。
    与 cts_batch_gpu 相同逻辑，algo_type='algo1'。
    """
    return cts_batch_gpu(sun_lons, ephem_7p, sun_idxs, w, event_type,
                         n_sim, algo_type='algo1', seed=seed, batch_size=batch_size)


# =============================================================================
# 4. 统一 CTS 调度: 优先 GPU, 失败退回 CPU multiprocessing
# =============================================================================

def derive_seed(*parts) -> int:
    """Stable hash-derived seed from arbitrary identifying parts.

    Uses hashlib.blake2s (stable across Python processes; Python's built-in
    hash() is salted per-process and unsuitable for reproducibility).

    Returns a 32-bit int compatible with numpy RandomState/default_rng.
    """
    import hashlib
    msg = '|'.join(str(p) for p in parts).encode('utf-8')
    return int.from_bytes(hashlib.blake2s(msg, digest_size=4).digest(), 'big')


def run_cts_simulation(sun_lons, ephem_matrix, sun_idxs, w, event_type,
                        n_sim, algo_type='algo1', n_workers=1, seed=None):
    """
    统一 CTS 调度: 自动选择 GPU 或 CPU。

    v3 修订(D 项):seed 由调用方派生(`derive_seed(stage, group, w, etype)`),
    不再使用全局默认 42 → 不同 (stage, window, type) 条目使用独立的随机序列,
    Monte Carlo 误差跨条目独立。
    若 seed=None,fallback 到 0 但发出警告(用于临时调试)。

    返回: k_sims (n_sim,) numpy 数组
    """
    if seed is None:
        import warnings
        warnings.warn("run_cts_simulation: seed=None — 使用 fallback seed=0 (可复现但不独立);"
                      "生产代码应传入 derive_seed(...) 派生的 seed",
                      stacklevel=2)
        seed = 0
    # 尝试 GPU
    if HAS_GPU:
        try:
            k_sims = cts_batch_gpu(sun_lons, ephem_matrix, sun_idxs, w, event_type,
                                    n_sim, algo_type=algo_type, seed=seed)
            return k_sims
        except Exception as e:
            print(f"    [GPU 失败, 退回 CPU] {e}")

    # CPU 退回: multiprocessing
    from multiprocessing import Pool
    worker_fn = cts_worker_algo1 if algo_type == 'algo1' else cts_worker_algo2
    seeds = np.random.RandomState(seed).randint(0, 1_000_000_000, n_sim)
    args = [(s, sun_lons, ephem_matrix, sun_idxs, w, event_type) for s in seeds]

    if n_workers > 1:
        with Pool(n_workers) as pool:
            k_sims = pool.starmap(worker_fn, args)
    else:
        k_sims = [worker_fn(*a) for a in args]

    return np.array(k_sims)
