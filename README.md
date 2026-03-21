# Solar CEOS

太阳活动的非对称统计效应（原CEOS）研究的代码、结果图表与部分数据。

本仓库包含：

- `notebooks/` — 数据预处理、统计分析与绘图脚本 / Notebook
- `results/` — 论文图表及对应统计输出
- `data/` — 大文件数据目录占位，已包含新增数据 `sfi_1975-2024.csv`

## 项目结构

```text
.
├── data/
│   ├── 00_raw/                # 原始数据（占位，需下载）
│   ├── interm/                # 中间缓存（占位，需下载）
│   └── ready/                 # 就绪数据（需下载，仅含 sfi_1975-2024.csv）
├── notebooks/
│   ├── 02_data_prep/          # 数据下载与清洗（9 notebooks）
│   ├── 03_coord_baseline/     # 坐标基准与基础图表（3 notebooks）
│   ├── 04_asymmetric/         # CEOS 非对称效应（8 py + 6 notebooks）
│   └── 05_multidimensional/   # 多行星扫描与稳健性（4 py + 1 notebook）
│       └── 10_robustness_tests/   # bootstrap / null planet / tidal
└── results/
    ├── 03_coord_baseline/     # Fig01–Fig03 (eps + xlsx)
    ├── 04_asymmetric/         # Fig04–Fig06 (eps + csv/xlsx)
    │   ├── analysis/          # 补充分析图 (png)
    │   ├── sf/                # 耀斑统计结果 (40 csv)
    │   │   └── cache_data/    # 缓存目录占位
    │   └── sg/                # 黑子群统计结果 (58 csv)
    │       └── cache_data/    # 缓存目录占位
    └── 05_multidimensional/   # Fig07–Fig09 (eps + png + csv)
        └── 10_robustness_tests/   # 稳健性测试输出 (csv + figures/*.png)
```

## 模块说明

### `02_data_prep/`

数据下载、清洗、融合与生命周期标记（9 个 notebook，按编号顺序执行）。

| 编号 | 文件 | 用途 |
|------|------|------|
| 01 | `01_download_ssn_sg.ipynb` | 下载黑子数与黑子群数据 |
| 02 | `02_download_flare.ipynb` | 下载耀斑数据 |
| 03 | `03_download_ephemeris.ipynb` | 下载星历数据 |
| 04 | `04_merge_planets_satellites_lonlat.ipynb` | 合并行星/卫星位置（经纬度） |
| 05 | `05_merge_781_planets_dwarfs_asteroids_parquet.ipynb` | 合并 781 个行星/矮行星/小天体 |
| 06 | `06_validate_sg_coords.ipynb` | 黑子群坐标校验 |
| 07 | `07_classify_sg_lifecycle.ipynb` | 黑子群生命周期分类 |
| 08 | `08_table1_sg_lifecycle_stats.ipynb` | Table 1: 生命周期统计 |
| 09 | `09_clean.ipynb` | 数据清洗收尾 |

### `03_coord_baseline/`

坐标系基准、几何伪影控制与基础图表。

- `01_fig01_table2_lat.ipynb` — 纬度分布与基准统计（Fig 1 / Table 2）
- `02_fig02_sg_sf_lon.ipynb` — 黑子群/耀斑经度相位分析（Fig 2）
- `03_fig03_urian_wing.ipynb` — 天王星“翼状”几何伪影分析（Fig 3）

### `04_asymmetric/`

CEOS 非对称效应、衰退阶段分析与分类对比。

**核心引擎：**

- `ceos_engine.py` — 高频置换统计引擎
- `algo_workers.py` — 算法并行工作模块

**计算流程（按顺序执行）：**

| 编号 | 文件 | 用途 |
|------|------|------|
| 00 | `00_prepare_cache.py` | 预处理缓存 |
| 00b | `00b_find_decay_boundary.py` | 衰退边界搜索 |
| 01 | `01_compute_sg.py` | 黑子群 CEOS 计算 |
| 02 | `02_compute_sf.py` | 耀斑 CEOS 计算 |
| 03 | `03_analyze_results.py` | 结果汇总分析 |
| 04 | `04_deep_sg_analysis.py` | 黑子群深度分析 |

**分析与图表 Notebook：**

- `05_run_all.ipynb` — 批量运行入口
- `06_fig4_flare_decay.ipynb` — Fig 4: 耀斑衰退分析
- `07_subset_asymmetry.ipynb` — 子集非对称分布分析
- `08_run_classification_contrast.ipynb` — 分类对比（ROC / 分类结果）
- `09_fig5_sunspot_dilution.ipynb` — Fig 5: 黑子稀释效应
- `10_fig6_phase_rose.ipynb` — Fig 6: 相位玫瑰图

### `05_multidimensional/`

多行星组合扫描、太阳周期稳定性与稳健性检验。

- `01_fig07_subset_scan_viz.py` — Fig 7: 子集扫描可视化
- `02_fig08_solar_cycle_summary.py` — Fig 8 数据汇总
- `03_fig08_solar_cycle_viz.py` — Fig 8: 太阳周期稳定性可视化
- `04_fig09_robustness_viz.py` — Fig 9: 稳健性检验可视化
- `05_fig07_fig09_showcase.ipynb` — Fig 7–9 展示 Notebook
- `10_robustness_tests/` — 附加测试脚本：`bootstrap_ci.py`、`null_planet_test.py`、`tidal_correlation.py`

## 结果内容

`results/` 目录包含论文全部可复现图表及中间统计数据：

| 子目录 | 内容 |
|--------|------|
| `03_coord_baseline/` | Fig 01–03 主图（eps）及源数据（xlsx） |
| `04_asymmetric/` | Fig 04–06 主图（eps）及源数据（csv/xlsx） |
| `04_asymmetric/sg/` | 黑子群统计结果（58 个 csv）；`cache_data/` 当前仅保留占位文件 |
| `04_asymmetric/sf/` | 耀斑统计结果（40 个 csv）；`cache_data/` 当前仅保留占位文件 |
| `04_asymmetric/analysis/` | 补充分析图（11 个 png） |
| `04_asymmetric/08_*` | 分类对比结果（ROC json + csv） |
| `05_multidimensional/` | Fig 07–09 主图（eps/png）及汇总数据（csv） |
| `05_multidimensional/10_robustness_tests/` | 稳健性测试输出（csv）及 `figures/` 下的 png 图 |

## 数据说明

由于 GitHub 文件大小限制，完整数据未全部纳入此仓库。`data/ready/` 目录仅包含本次新增的 `sfi_1975-2024.csv`，其余 15 个就绪数据文件（parquet/csv，共约 6 GB）需另行下载。以下目录保留为占位符：

- `data/00_raw/` — 原始数据
- `data/interm/` — 中间缓存
- `data/ready/` — 就绪数据（需下载补全）
- `results/04_asymmetric/sf/cache_data/` — 耀斑计算缓存
- `results/04_asymmetric/sg/cache_data/` — 黑子群计算缓存

- 审稿读者：请使用投稿系统或补充材料中提供的匿名下载链接
- 公开发布后：请从 Zenodo（DOI: _待补充_）下载完整数据

下载后将压缩包解压到仓库根目录，数据将自动合并到 `data/` 和 `results/` 目录。

## 运行环境

建议使用 Python 3.13，推荐通过 [Miniforge](https://github.com/conda-forge/miniforge) (Mamba) 管理环境：

```bash
# 创建并激活环境
mamba create -n ceos python=3.13 -y
conda activate ceos

# 安装依赖（全部来自 conda-forge，避免 conda/pip 混用冲突）
mamba install numpy pandas scipy astropy astroquery sunpy \
    matplotlib seaborn scikit-learn lightgbm \
    requests beautifulsoup4 ephem xlsxwriter openpyxl \
    pyarrow tqdm jupyterlab ipykernel -y

# 注册 Jupyter 内核
python -m ipykernel install --user --name ceos --display-name "Python (ceos)"
```

`04_asymmetric/` 中的高频置换与批量统计任务支持多核 CPU 并行，也支持通过 CuPy 进行 GPU 加速（自动检测，无 GPU 则退回 CPU）。如需启用 GPU 加速：

```bash
mamba install cupy -y
```

常规图表复现与结果检查不要求高性能服务器。

## License

当前目录尚未附带 `LICENSE` 文件。

- 若准备公开发布，请在仓库根目录补充正式许可证文本（例如 MIT）
- 在补充许可证前，不建议将本仓库表述为已完成开源授权
