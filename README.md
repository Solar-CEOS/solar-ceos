# Solar CEOS

Solar CEOS（Celestial Event Occurrence Statistics，天体相位事件发生统计）是一个可复现统计框架，用于检验太阳活动事件发生率是否随行星相位几何发生变化。本仓库对应论文三修版，包含数据处理脚本、统计检验程序、论文图表和可公开的小型结果数据。

本研究的主要对象包括黑子群、太阳耀斑、太阳周期分段、多行星组合扫描，以及稳健性和多重比较校正检验。

## 仓库内容

- `notebooks/`：数据预处理、统计分析、稳健性检验与绘图脚本
- `notebooks/_style/figstyle.py`：统一绘图风格模块
- `results/`：论文图表、源数据表和统计输出
- `data/`：数据目录占位和少量可直接公开的小型数据

## 项目结构

```text
.
├── data/
│   ├── 00_raw/                # 原始数据目录占位
│   ├── interm/                # 中间数据目录占位
│   └── ready/                 # 就绪数据目录；仓库仅保留小文件
├── notebooks/
│   ├── _style/                # 统一绘图风格
│   ├── 02_data_prep/          # 数据下载、清洗与合并
│   ├── 03_coord_baseline/     # 坐标基准与几何控制，Fig01-Fig03
│   ├── 04_asymmetric/         # 相位窗口统计与非对称检验，Fig04-Fig06
│   └── 05_multidimensional/   # 多行星扫描、FDR 审计与稳健性检验，Fig07-Fig09
└── results/
    ├── 02_data/               # Table 1 源数据
    ├── 03_coord_baseline/     # Fig01-Fig03，EPS/PNG 与源数据表
    ├── 04_asymmetric/         # Fig04-Fig06、分类对比与单事件统计结果
    └── 05_multidimensional/   # Fig07-Fig09、多行星扫描、FDR 与稳健性结果
```

## 模块说明

### `02_data_prep/`

数据下载、清洗、合并与黑子群生命周期标记。该部分保留为 notebook，便于审阅下载来源、人工检查和中间数据状态。

| 编号 | 文件 | 用途 |
|------|------|------|
| 01 | `01_download_ssn_sg.ipynb` | 下载黑子数与黑子群数据 |
| 02 | `02_download_flare.ipynb` | 下载太阳耀斑数据 |
| 03 | `03_download_ephemeris.ipynb` | 下载 JPL Horizons 星历 |
| 04 | `04_merge_planets_satellites_lonlat.ipynb` | 合并行星/卫星日心或视位置经纬度表 |
| 05 | `05_merge_781_planets_dwarfs_asteroids_parquet.ipynb` | 合并 781 个行星、矮行星和小天体数据 |
| 06 | `06_validate_sg_coords.ipynb` | 黑子群坐标校验 |
| 07 | `07_classify_sg_lifecycle.ipynb` | 黑子群生命周期阶段分类 |
| 08 | `08_table1_sg_lifecycle_stats.ipynb` | 生成 Table 1 生命周期统计 |
| 09 | `09_clean.ipynb` | 数据清洗收尾 |

### `03_coord_baseline/`

坐标基准、基础相位分布和几何伪影控制。

- `01_fig01_table2_lat.py`：纬度分布与基准统计，生成 Fig 1 和 Table 2 相关结果
- `01_fig01_plot_from_source.py`：从 `Fig01_Latitude_Source.xlsx` 快速重绘 Fig 1
- `02_fig02_sg_sf_lon.py`：黑子群/耀斑经度相位分析，生成 Fig 2 源数据
- `02_fig02_plot_from_source.py`：从 `Fig02_Spot_Flare_Longitude_Source.xlsx` 快速重绘 Fig 2
- `03_fig03_urian_wing.py`：天王星翼状几何伪影分析，生成 Fig 3
- `03_fig03_plot_from_source.py`：从源数据快速重绘 Fig 3

### `04_asymmetric/`

单事件相位窗口统计、合相/冲相非对称检验、衰退边界搜索，以及耀斑/黑子分类对比。

核心计算模块：

- `ceos_engine.py`：高频置换与循环时间平移统计引擎
- `algo_workers.py`：CPU/GPU 工作函数，支持批量模拟和并行计算

主要计算脚本：

| 编号 | 文件 | 用途 |
|------|------|------|
| 00 | `00_prepare_cache.py` | 构建预计算缓存 |
| 00b | `00b_find_decay_boundary.py` | 搜索不同窗口宽度下的衰退边界 |
| 01 | `01_compute_sg.py` | 黑子群相位窗口统计 |
| 02 | `02_compute_sf.py` | 太阳耀斑相位窗口统计 |
| 03 | `03_analyze_results.py` | 汇总和检查统计输出 |
| 04 | `04_deep_sg_analysis.py` | 黑子群补充诊断 |

图表与分析入口：

- `05_run_all.ipynb`：批量运行入口
- `06_fig4_flare_decay.py`：Fig 4 耀斑衰退阶段分析
- `07_subset_asymmetry.ipynb`：子集非对称分布探索
- `08_run_classification_contrast.py`：分类对比与 ROC 结果
- `09_fig5_sunspot_dilution.py`：Fig 5 黑子稀释效应分析
- `10_fig6_phase_rose.py`：Fig 6 全相位玫瑰图
- `10_fig6_phase_rose_plot_from_csv.py`：从 `Fig06_phase_rose.csv` 快速重绘 Fig 6

### `05_multidimensional/`

多行星组合扫描、太阳周期稳定性、多重比较校正和稳健性检验。

- `01_fig07_subset_scan_viz.py`：Fig 7 子集扫描可视化
- `02_fig08_solar_cycle_summary.py`：Fig 8 太阳周期分段数据汇总
- `03_fig08_solar_cycle_viz.py`：Fig 8 太阳周期稳定性图
- `04_fig09_robustness_viz.py`：Fig 9 稳健性检验图
- `05_fig07_fig09_showcase.ipynb`：Fig 7-Fig 9 展示 notebook
- `06_multi_alignment.py`：多行星 AND 取交连珠检验
- `07_fdr_audit.py`：Benjamini-Hochberg FDR 审计
- `10_robustness_tests/bootstrap_ci.py`：block bootstrap 置信区间
- `10_robustness_tests/null_planet_test.py`：虚拟行星阴性对照
- `10_robustness_tests/tidal_correlation.py`：潮汐相关性检验

## 结果目录

`results/` 中包含论文图表和复现实验所需的主要统计输出。

| 目录 | 内容 |
|------|------|
| `results/02_data/` | Table 1 源数据 |
| `results/03_coord_baseline/` | Fig01-Fig03 的 EPS/PNG 图和源数据表 |
| `results/04_asymmetric/` | Fig04-Fig06、ROC 分类对比和相位窗口统计输出 |
| `results/04_asymmetric/analysis/` | 补充诊断图 |
| `results/04_asymmetric/sf/` | 太阳耀斑统计结果 |
| `results/04_asymmetric/sg/` | 黑子群统计结果 |
| `results/05_multidimensional/` | Fig07-Fig09、多行星组合扫描和太阳周期汇总 |
| `results/05_multidimensional/fdr_audit/` | BH-FDR 校正结果，用于图中 q 值标注 |
| `results/05_multidimensional/10_robustness_tests/` | bootstrap、虚拟行星和潮汐检验输出 |

图表通常同时保存为 `.eps` 和 `.png`。EPS 用于投稿和排版，PNG 用于快速预览。

## 数据说明

由于 GitHub 文件大小限制，完整原始数据、中间数据和大型缓存不纳入本仓库。仓库中保留目录结构和少量小型数据文件，例如 `data/ready/sfi_1975-2024.csv`。

以下目录需要根据审稿阶段匿名下载链接或后续 Zenodo 数据归档补全：

- `data/00_raw/`：原始数据
- `data/interm/`：中间缓存
- `data/ready/`：可直接用于分析的就绪数据
- `results/04_asymmetric/sf/cache_data/`：耀斑计算缓存
- `results/04_asymmetric/sg/cache_data/`：黑子群计算缓存

`cache_data/` 下的 `.parquet`、`.npy`、`.pkl` 文件是运行缓存，体积较大，不建议提交到 GitHub。若只复现图表，可优先使用 `results/` 中已经保存的源数据表和 plot-only 脚本。

审稿阶段请使用投稿材料中提供的匿名下载链接。论文和数据正式发布后，请从 Zenodo 下载完整数据归档，并以届时公布的 DOI 为准。

## 运行环境

推荐使用 Python 3.13，并通过 Miniforge/Mamba 管理依赖。

```bash
mamba create -n ceos python=3.13 -y
conda activate ceos

mamba install numpy pandas scipy astropy astroquery sunpy \
    matplotlib seaborn scikit-learn lightgbm \
    requests beautifulsoup4 ephem xlsxwriter openpyxl \
    pyarrow tqdm jupyterlab ipykernel -y

python -m ipykernel install --user --name ceos --display-name "Python (ceos)"
```

`04_asymmetric/` 中的高频循环时间平移模拟支持多核 CPU，也支持可选的 CuPy GPU 加速。普通图表复现和结果检查不要求 GPU。

```bash
mamba install cupy -y
```

## 绘图规范

所有正式绘图脚本共用 `notebooks/_style/figstyle.py`。该模块统一设置中文核心期刊图件要求和国际天文期刊常用风格，包括四边框、内向刻度、单栏/双栏宽度、8-10 pt 字号、色盲友好配色，以及 EPS/PNG 双格式保存。

典型用法：

```python
from _style.figstyle import apply_acta_style, figsize_double, save_dual

apply_acta_style("double")
fig, axes = plt.subplots(..., figsize=figsize_double(aspect=...))
save_dual(fig, "results/<dir>/FigNN_<name>.eps")
```

`*_plot_from_source.py` 脚本用于从已保存源数据快速重绘图件，适合只调整版式或期刊图件规范时使用。

## 复现建议

完整复现可按模块编号顺序执行：

1. 补全 `data/` 中的大型数据文件。
2. 运行 `notebooks/02_data_prep/` 中的数据准备 notebook。
3. 运行 `notebooks/03_coord_baseline/` 生成 Fig01-Fig03。
4. 运行 `notebooks/04_asymmetric/` 的缓存、统计和图表脚本。
5. 运行 `notebooks/05_multidimensional/` 的多行星扫描、FDR 审计和稳健性检验。

若只检查论文图表，可直接使用 `results/` 中的源数据和 plot-only 脚本，无需重新执行全部高耗时统计。

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1 | 2025-12 | 初投稿版本，建立基础数据流程、CEOS 统计框架和初版论文图表 |
| v2 | 2026-03 | 二修版，重构 notebook 与结果目录，新增多行星连珠检验和 BH-FDR 审计，补充 FDR q 值标注 |
| v3 | 2026-05 | 三修版，重跑主要统计，使用稳定随机种子派生、14 日 block bootstrap、Fisher 合并 p 值、训练集内分位阈值，并统一图件风格和投稿版图表输出 |

历史版本已用 Git 标签固定：`v1-submitted`、`v2-revision`、`v3-revision`。可在 GitHub 的 Tags 页面查看或下载对应版本源码包。当前 `main` 分支对应三修版整理状态。

## 许可

代码采用 [MIT License](LICENSE) 开源发布。

数据产品和结果图表采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 协议。使用本仓库、图表或数据产品时，请注明来源并引用对应论文。
