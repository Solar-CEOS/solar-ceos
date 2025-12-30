# 太阳活动的行星调制：合增冲减效应与 P-M-A 模型

> [English Version (英文版)](README.md)

本代码库包含了论文《太阳活动的合增冲减效应与行星调制的P-M-A框架》的完整复现代码。其中包括原始数据处理、计算过程、统计分析以及论文图表的生成脚本。

## 1. 项目结构

项目主要包含以下三个目录：
- **`data/`**：原始数据来源说明及参考文献。
- **`notebooks/`**：源代码（Jupyter Notebook），按分析步骤组织。
- **`results/`**：生成的图表、表格及中间数据文件。

> [!IMPORTANT]
> **数据下载 (Zenodo)**
>
> 由于 GitHub 文件大小限制，数据集已拆分为以下三个压缩包托管至 Zenodo：
> - **`data_raw_interm.zip`**: 原始数据源及中间缓存文件。
> - **`data_ready.zip`**: 预处理完毕、可直接用于分析的就绪数据。
> - **`results.zip`**: 项目生成的图表及统计数据。
>
> **安装说明**: 请下载这三个文件至项目根目录，并选择 **"解压到当前文件夹" (Extract Here)**。它们将自动合并至 `data/` 和 `results/` 目录中。
>
> - **Zenodo 访问地址**: *(遵循双盲评审政策，匿名访问链接已包含在投稿系统附件中。文章发表后将公开正式 DOI 链接。)*

## 2. 原始数据来源 (Data Sources)

本研究使用的原始数据初次下载于 2025年5月。为保证复现性，建议优先使用网盘提供的快照数据。

| 数据内容 | 来源 / 参考文献 | 时间跨度 | 官方链接 |
| :--- | :--- | :--- | :--- |
| **黑子群统计** | Hathaway {hathaway_solar_2015} | 1874-2024 | [Active Regions](http://solarcyclescience.com/activeregions.html) |
| **黑子数 (SSN)** | WDC-SILSO {clette_silso_2015} | 1749-2025 | [SILSO Data](https://www.sidc.be/SILSO/datafiles) |
| **耀斑 (Flares)** | NOAA GOES / NGDC | 1975-2017 | [NOAA XRS](https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/) |
| **CME** | NASA CDAW {Yashiro2004} | 1996-2025 | [SOHO/LASCO](https://cdaw.gsfc.nasa.gov/CME_list/halo/halo.html) |
| **行星星历** | NASA JPL Horizons {giorgini1996} | 1849-2050 | [JPL SSD](https://ssd.jpl.nasa.gov/sb/) |

## 3. 代码说明 (Notebooks)

代码按论文章节被划分为5个主要模块 (Notebooks)。
`.ipynb` 文件包含了完整的分析逻辑和输出结果。GitHub 能够直接渲染 Notebook，但对于较大的文件或复杂的交互式图表，建议下载后在本地运行查看。

### 3.1 `notebooks/01_data_prep` (数据源与预处理)
- **对应论文章节**: §2 (数据来源与预处理) & §3 (统计挑战)
- **核心功能**:
  - 下载原始科学数据 (SSN, Flare, Ephemeris)。
  - 执行数据清洗与标准化：
    - **黑子群生命周期标记**: 区分起始 (Onset), 消散 (Diss), 全程 (Dur), 单日 (Daily) 阶段 (表 1)。
    - **有效样本量 ($N_{eff}$) 计算**: 评估时间序列自相关性 ($\rho_1$) 并校正自由度。

### 3.2 `notebooks/02_carr_vs_eclip_dist` (坐标系几何基准)
- **对应论文章节**: §4 (坐标系选择对太阳活动空间分布统计特征的影响)
- **核心功能**:
  - 对比 **日面坐标系 (Heliographic)** 与 **日心黄道坐标系 (Heliocentric Ecliptic)**。
  - **相位锁定分析 (Fig01)**: 揭示消散阶段黑子在日面坐标系下的非物理聚集，验证了黄道坐标系下的各向同性基准。
  - **南北不对称性 (Fig02)**: 利用 CUSUM 曲线展示黄道坐标系如何消除由 $B_0$ 变化引入的一年周期几何系统偏差 (Rosenberg-Coleman Effect)。
  - **统计检验 (表 2)**: 包含 Wilcoxon 符号秩检验与 Bootstrap 稳健性验证。

### 3.3 `notebooks/03_planet_artifact` (行星伪影与零假设构建)
- **对应论文章节**: §3 (统计挑战及校正策略) & §5 (周期巧合实例)
- **核心功能**:
  - **混叠效应复现 (Fig03)**: 重现了由于采样窗口长度（如88年与土星/施瓦贝周期的共振）导致的虚假统计信号，展示红噪声模型的重要性。
  - **几何伪影解析 (Fig04)**: 解析天王星“天蝠图”的几何起源，证明其为 $F_{geo}$ 几何对齐因子导致的采样偏差。

### 3.4 `notebooks/04_conj_enh_opp_sup` (合增冲减效应 CEOS)
- **对应论文章节**: §5 (日心黄道视角下的相位关联)
- **核心功能**:
  - **CEOS 效应量化 (Fig05)**: 统计验证行星合相 ($0^\circ$) 的显著增强与冲相 ($180^\circ$) 的抑制效应。
  - **概率密度校正**: 构建 $P_{Kep}(\lambda) \propto r^2$ 开普勒驻留时间概率模型，消除“远日点效应”。
  - **贝叶斯推断 (Fig06)**: 计算后验比率 $\lambda = k_{obs}/k_{exp}$，确证该效应的单峰极性不对称特征（区别于线性潮汐的双峰对称）。
  - **循环置换检验 (CTS)**: 在保留时序自相关结构的前提下，通过 $N=10,000$ 次随机相位平移验证统计显著性。
  - **FDR 校正**: 针对 781 个天体进行全参数空间扫描，基于 Benjamini-Hochberg 过程控制错误发现率，证明除八大行星外小天体无显著频次效应。

### 3.5 `notebooks/05_p_m_a_model` (P-M-A 混合模型)
- **对应论文章节**: §6 (P-M-A 混合分析框架)
- **核心功能**:
  - **P (Planetary-Baseline)**: 使用 Ridge 回归从行星星历提取 11 年基线趋势 ($R^2 \approx 0.6$) (表 3)。
  - **M (Modulation)**: 使用 LightGBM 捕捉行星对太阳自转频带能量包络的非线性振幅调制 (Fig07)。
  - **A (Autocorrelation)**: 使用 LSTM 网络拟合包含混沌动态的高频残差。
  - **模型评估 (Fig08 & 表 5)**: 对比不同配置模型的 $R^2$ 与 DM 统计量，验证混合架构的优越性。
  - **影子测试 (表 4)**: 验证物理模型的预测力显著优于维度一致但时间平移的随机对照组。

## 4. 结果文件

- **`results/`** 目录下的结构与 code 目录一一对应。
- 这里存放了所有生成的 CSV 统计表、Excel 数据透视表以及论文中使用的矢量图 (EPS/PDF)。

## 5. 运行环境

本代码在以下环境中开发并测试通过：

- **操作系统**: Ubuntu 24.04 LTS
- **硬件**: AMD Ryzen 9950X, 192GB RAM, NVIDIA RTX 4090

> [!TIP]
> **Python 环境配置建议**
>
> 为了确保依赖库的兼容性（特别是 TensorFlow 和 SunPy），推荐使用 **Python 3.12**。
> 建议使用 `Miniforge` (Mamba) 进行环境管理，以下是 Linux 下的安装与配置示例：

```bash
# 1. 下载并安装 Miniforge (Linux-x86_64)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
chmod +x Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh

# 2. 创建新环境 (指定 Python 3.12)
mamba create -n sunspot_env python=3.12 -y

# 3. 激活环境
conda activate sunspot_env

# 4. 安装业务库 (科学计算 + 机器学习 + 绘图)
mamba install pandas numpy scipy astropy astroquery sunpy scikit-learn lightgbm \
    statsmodels pygam scikit-optimize joblib pybaselines matplotlib seaborn \
    requests beautifulsoup4 openpyxl tqdm pyarrow lxml ipywidgets bottleneck tabulate \
    tensorflow numexpr xlsxwriter ipykernel jupyter jupyterlab-language-pack-zh-CN -y

# 5. 将该环境添加为 Jupyter 的内核
python -m ipykernel install --user --name sunspot_env --display-name "Python (py312sunspot)"
```