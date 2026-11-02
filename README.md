# A Deep Learning Framework for Functional Eutrophication Assessment: Mapping the Trophic Transfer Index

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![R Version](https://img.shields.io/badge/R-4.4.2-276DC3.svg)](https://www.r-project.org/)
[![PyTorch](https://img.shields.io/badge/torch-0.16.3-EE4C2C.svg)](https://torch.mlverse.org/)

**Authors:** Adolf Stips*, Ove Pärn, Luca Polimene, Diego Macias-Moy*, Svetla Miladinova, Elisa Garcia-Gorriz, Olaf Duteil  
**Journal:** *Environmental Modelling & Software* (2026)  
**Contact:** `adolf.stips@ext.ec.europa.eu` | `adolf.stips@gmail.com`

---

## Overview

This repository contains the complete data-processing, neural network training, Explainable AI (SHAP), and visualization pipeline for the manuscript **"A Deep Learning Framework for Functional Eutrophication Assessment: Mapping the Trophic Transfer Index"**.

As a synthetic proof-of-concept, this framework applies Multi-Layer Perceptron (`MLP`) and Long Short-Term Memory (`LSTM`) neural networks to emulate the continuous **Trophic Transfer Index (TTI)** and classify **Good Environmental Status (GES; TTI > 0.71)** across the Baltic Sea (29,502 marine grid cells) using 70 years (1953–2022; 14 pentads) of simulated physical-biogeochemical data from the coupled **GETM-ERGOM** ecosystem model.

The repository evaluates four core experimental configurations across three environmental variable suites (`Comprehensive`, `MSFD_Baseline`, and `Remote_Sensing`):
1. **`MLP_SPAT`**: Feedforward spatial gap-filling via non-overlapping 10-Fold Spatial Block Cross-Validation ($10 \times 10$ pixel blocks; 80% Train / 10% Validation / 10% Test rotation).
2. **`LSTM_SPAT`**: Recurrent spatial gap-filling ingesting 35-year (7-pentad) chronological sequences (`[batch, sequence_length = 7, features]`) via 10-Fold Spatial Block Cross-Validation.
3. **`MLP_TIME`**: Feedforward rolling-origin temporal hindcast across 6 unseen target pentads (1995–2020) partitioned across non-overlapping spatial blocks (60% Train / 20% Val / 20% Test).
4. **`LSTM_TIME`**: Recurrent rolling-origin temporal hindcast ingesting 35-year lookback sequences across 6 unseen target pentads (1995–2020).

---

## Repository Structure

```text
TTI_ANN_FIG/
├── README.md                                  # Workflow documentation and instructions
├── LICENSE                                    # GNU General Public License v3.0 (GPL-3.0)
├── GES_Baltic_5year_ALLE_Combined_4D.nc.zip   # Zipped NetCDF input dataset (14 pentads, 1955-2020)
│
├── 00_Run_All_Workflow.R                      # Master script to execute the full pipeline
├── 01_Kfold_MLP_SPAT.R                        # 10-Fold Spatial Block CV (MLP_SPAT)
├── 02_Kfold_LSTM_SPAT.R                       # 10-Fold Spatial Block CV (LSTM_SPAT)
├── 03_Rolling_MLP_TIME.R                      # Rolling-Origin Temporal Forecast (MLP_TIME)
├── 04_Rolling_LSTM_TIME.R                     # Rolling-Origin Temporal Forecast (LSTM_TIME)
├── 05_Spatial_Ensemble_MLP_SPAT.R             # 10-Fold Spatial Bootstrap Ensemble Reconstruction
├── 06_Generate_Manuscript_Tables.R            # Generates Manuscript Tables 2, 3, and 4
├── 07_Plot_Fig1_Spatial_Block_Map.R           # Generates Figure 1
├── 08_Plot_Figs2_3_4_SHAP_Analysis.R          # Generates Figures 2, 3, and 4
├── 09_Plot_Figs5_6_S3_Spatial_Ensemble.R      # Generates Figures 5, 6, and Supplementary Fig S3
├── 10_Plot_FigsS1_S2_Confusion_Matrices.R     # Generates Supplementary Figures S1 and S2
│
├── Output/                                    # Pre-computed & generated .rds model outputs
│   ├── Summary_MLP_SPAT.rds
│   ├── Summary_LSTM_SPAT.rds
│   ├── Summary_MLP_TIME.rds
│   ├── Summary_LSTM_TIME.rds
│   ├── SHAP_RAW_MLP_SPAT.rds
│   ├── SHAP_RAW_LSTM_SPAT.rds
│   ├── SHAP_RAW_MLP_TIME.rds
│   ├── SHAP_RAW_LSTM_TIME.rds
│   ├── CF_Matrix_MLP_SPAT.rds
│   ├── CF_Matrix_LSTM_SPAT.rds
│   ├── CF_Matrix_MLP_TIME.rds
│   ├── CF_Matrix_LSTM_TIME.rds
│   ├── Ensemble_Map_Data_MLP_SPAT.rds
│   └── Ensemble_TimeSeries_MLP_SPAT.rds
│
├── Tables/                                    # Generated Manuscript Tables (.docx and .csv)
└── Figures/                                   # Generated Publication Figures (.png, 500 DPI)