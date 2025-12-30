# Conjunctional Enhancement and Oppositional Suppression in Solar Activity and the Planetary-Modulated P-M-A Framework

> [中文说明 (Chinese Version)](README_zh.md)

This repository contains the complete reproduction code for the paper **"Conjunctional Enhancement and Oppositional Suppression in Solar Activity and the Planetary-Modulated P-M-A Framework"**. It includes raw data processing, computational procedures, statistical analysis, and scripts for generating the figures and tables presented in the manuscript.

## 1. Project Structure

The project is organized into three main directories:
- **`data/`**: Metadata and references for raw data sources.
- **`notebooks/`**: Source code (Jupyter Notebooks) organized by analysis steps.
- **`results/`**: Generated figures, tables, and intermediate data files.

> [!IMPORTANT]
> **Data Access (Zenodo)**
>
> Due to GitHub file size limitations, the dataset has been deposited on Zenodo in three archives:
> - **`data_raw_interm.zip`**: Raw data sources and intermediate cache files.
> - **`data_ready.zip`**: Preprocessed data ready for analysis.
> - **`results.zip`**: Generated figures and statistical tables.
>
> **Installation**: Download all three files to your project root and **"Extract Here"** (unzip to current folder). They will automatically merge into the `data/` and `results/` directories.
>
> - **Zenodo Repository**: *(The anonymous access link has been provided in the manuscript submission system for double-blind peer review. A public DOI will be available upon publication.)*

## 2. Raw Data Sources

The raw data used in this study was initially downloaded in May 2025. For reproducibility, it is recommended to use the snapshot data provided via the cloud link above.

| Content | Source / Reference | Time Span | Official Link |
| :--- | :--- | :--- | :--- |
| **Sunspot Groups** | Hathaway {hathaway_solar_2015} | 1874-2024 | [Active Regions](http://solarcyclescience.com/activeregions.html) |
| **Sunspot Number (SSN)** | WDC-SILSO {clette_silso_2015} | 1749-2025 | [SILSO Data](https://www.sidc.be/SILSO/datafiles) |
| **Solar Flares** | NOAA GOES / NGDC | 1975-2017 | [NOAA XRS](https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/) |
| **CMEs** | NASA CDAW {Yashiro2004} | 1996-2025 | [SOHO/LASCO](https://cdaw.gsfc.nasa.gov/CME_list/halo/halo.html) |
| **Planetary Ephemerides** | NASA JPL Horizons {giorgini1996} | 1849-2050 | [JPL SSD](https://ssd.jpl.nasa.gov/sb/) |

## 3. Code Description (Notebooks)

The codebase is split into 5 modules corresponding to the sections of the paper.
Each `.ipynb` notebook contains the complete analysis logic and outputs. While GitHub can render notebooks directly, it is recommended to download and run them locally for large files or complex interactive figures.

### 3.1 `notebooks/01_data_prep` (Data Source & Preprocessing)
- **Paper Section**: §2 (Data Sources & Preprocessing) & §3 (Statistical Challenges)
- **Core Functions**:
  - Download raw scientific data (SSN, Flare, Ephemeris).
  - Perform data cleaning and standardization:
    - **Sunspot Group Lifecycle Marking**: Distinguish between Onset, Diss (Dissipation, Phase-Locked), Dur (Duration), and Daily stages (Table 1).
    - **Effective Sample Size ($N_{eff}$) Calculation**: Evaluate time series autocorrelation ($\rho_1$) and correct degrees of freedom.

### 3.2 `notebooks/02_carr_vs_eclip_dist` (Coordinate System Geometric Baseline)
- **Paper Section**: §4 (Impact of Coordinate System Choice on Spatial Statistics)
- **Core Functions**:
  - Compare **Heliographic** vs. **Heliocentric Ecliptic** coordinate systems.
  - **Phase Locking Analysis (Fig01)**: Reveals non-physical clustering of decaying spots in the Heliographic frame, verifying the isotropic baseline of the Ecliptic frame.
  - **North-South Asymmetry (Fig02)**: Uses CUSUM curves to demonstrate how the Ecliptic frame eliminates the 1-year periodic geometric systemic bias (Rosenberg-Coleman Effect) introduced by $B_0$ variation.
  - **Statistical Tests (Table 2)**: Includes Wilcoxon signed-rank test and Bootstrap robustness verification.

### 3.3 `notebooks/03_planet_artifact` (Planetary Artifacts & Null Hypothesis)
- **Paper Section**: §3 (Statistical Challenges & Correction Strategies) & §5 (Coincidence Examples)
- **Core Functions**:
  - **Aliasing Reproduction (Fig03)**: Reproduces spurious statistical signals caused by sampling window lengths (e.g., 88-year resonance with Saturn/Schwabe cycles), demonstrating the importance of Red Noise models.
  - **Geometric Artifact Analysis (Fig04)**: Resolves the geometric origin of the "Uranian Wing Diagram," proving it is a sampling bias caused by the geometric alignment factor $F_{geo}$.
  - **FDR Correction**: Applies Benjamini-Hochberg procedure to control the False Discovery Rate for a full parameter scan of 781 bodies, proving no significant frequency effects beyond the 8 major planets.

### 3.4 `notebooks/04_conj_enh_opp_sup` (CEOS Effect)
- **Paper Section**: §5 (Phase Association in Heliocentric Ecliptic Perspective)
- **Core Functions**:
  - **CEOS Quantification (Fig05)**: Statistically verifies the significant enhancement at planetary Conjunction ($0^\circ$) and suppression at Opposition ($180^\circ$).
  - **Probability Density Correction**: Constructs $P_{Kep}(\lambda) \propto r^2$ Keplerian dwell-time probability model to eliminate the "Aphelion Effect."
  - **Bayesian Inference (Fig06)**: Calculates the posterior ratio $\lambda = k_{obs}/k_{exp}$, confirming the unimodal polarity asymmetry (distinguishing it from the bimodal symmetry of linear tides).
  - **Cyclic Time-Shift (CTS) Test**: Verifies statistical significance via $N=10,000$ random phase shifts while preserving time series autocorrelation structure.

### 3.5 `notebooks/05_p_m_a_model` (P-M-A Hybrid Model)
- **Paper Section**: §6 (P-M-A Hybrid Analysis Framework)
- **Core Functions**:
  - **P (Planetary-Baseline)**: Uses Ridge Regression to extract the 11-year baseline trend from planetary ephemerides ($R^2 \approx 0.6$) (Table 3).
  - **M (Modulation)**: Uses LightGBM to capture the non-linear amplitude modulation of the solar rotation band energy envelope by planets (Fig07).
  - **A (Autocorrelation)**: Uses LSTM networks to fit the high-frequency residuals containing chaotic dynamics.
  - **Model Evaluation (Fig08 & Table 5)**: Compares $R^2$ and DM statistics across different configurations, validating the superiority of the hybrid architecture.
  - **Shadow Test (Table 4)**: Verifies that the physical model's predictive power is significantly better than randomized controls with consistent dimensions but time shifts.

## 4. Results

- **`results/`**: The directory structure mirrors the code directory.
- It contains all generated CSV statistical tables, Excel pivot tables, and vector graphics (EPS/PDF) used in the manuscript.

## 5. Environment

The code was developed and tested in the following environment:

- **OS**: Ubuntu 24.04 LTS
- **Hardware**: AMD Ryzen 9950X, 192GB RAM, NVIDIA RTX 4090

> [!TIP]
> **Python Environment Setup**
>
> To ensure dependency compatibility (especially for TensorFlow and SunPy), **Python 3.12** is recommended.
> We suggest using `Miniforge` (Mamba) for environment management. Below is an example installation and configuration for Linux:

```bash
# 1. Download and Install Miniforge (Linux-x86_64)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
chmod +x Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh

# 2. Create New Environment (Specify Python 3.12)
mamba create -n sunspot_env python=3.12 -y

# 3. Activate Environment
conda activate sunspot_env

# 4. Install Business Logic Libraries (SciComp + ML + Plotting)
mamba install pandas numpy scipy astropy astroquery sunpy scikit-learn lightgbm \
    statsmodels pygam scikit-optimize joblib pybaselines matplotlib seaborn \
    requests beautifulsoup4 openpyxl tqdm pyarrow lxml ipywidgets bottleneck tabulate \
    tensorflow numexpr xlsxwriter ipykernel jupyter jupyterlab-language-pack-zh-CN -y

# 5. Add Environment as Jupyter Kernel
python -m ipykernel install --user --name sunspot_env --display-name "Python (py312sunspot)"
```
