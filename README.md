# bacterial-growth-model
Python code for estimating physiological state (Q = exp(-h0)) in bacterial growth using the Baranyi-Roberts framework. Analyzes 40 ComBase datasets for E. coli and L. monocytogenes, including model fitting, parameter estimation and figure generation. Supports manuscript submitted to International Journal of Food Microbiology.
# bacterial-growth-model

Python code for estimating physiological state parameters in bacterial growth using the Baranyi-Roberts framework.

## Overview

This repository contains the complete code used in the manuscript:

**"Independence of Physiological State and Growth Rate in Bacteria: A Mathematical Analysis of the Baranyi-Roberts Framework Using 40 ComBase Datasets for E. coli and L. monocytogenes"**

*Submitted to International Journal of Food Microbiology*

## Features

- **Baranyi-Roberts Model Fitting**: Estimates μ_max, h0, K, and Q = exp(-h0) from growth curves
- **40 ComBase Datasets**: Includes analysis for E. coli O157:H7 (20 records) and L. monocytogenes (20 records)
- **Comprehensive Statistical Analysis**: RMSE, R², AIC, BIC, paired t-tests, linear regression
- **Mathematical Analysis**: Stability analysis, transcritical and Hopf bifurcation analysis
- **Figure Generation**: All figures from the manuscript

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn

## Installation

```bash
git clone https://github.com/abadiabraha20/bacterial-growth-model.git
cd bacterial-growth-model
pip install -r requirements.txt
