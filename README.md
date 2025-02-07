# ABoVE_NEE_seasonality

## Overview

This repository contains code for analyzing the seasonality of Net Ecosystem Exchange (NEE) in the ABoVE (Arctic-Boreal Vulnerability Experiment) study area. We evaluated NEE estimates derived from three distinct approaches: atmospheric inversions, upscaled flux measurements, and terrestrial biosphere models (TBMs), using atmospheric CO2 observations collected during airborne campaigns organized as part of the Carbon in Arctic Reservoirs Vulnerability Experiment (CARVE) and the Arctic Carbon Atmospheric Profiles (Arctic-CAP) campaigns during 2012-2014 and 2017, respectively.

## Contents

- `src/`: Directory containing analysis scripts.
- `data/`: Directory containing input data.
- `results/`: Directory containing output results.
- `README.md`: This file.

## Data
- Atmospheric CO2 concentration measurements and footprints: CARVE and Arctic-CAP
- Atmospheric inversions: GCB 2023
- Upscaled flux tower measurements: X-BASE, Upscaled ABCflux
- TBM outputs: TRENDY v11
- Remote sensing data: APAR (calculated from MODIS FPAR and CERES PAR), GOME-2 SIF, GOSIF GPP

## Analysis Steps

1. **Data Preprocessing for atmospheric observations**: 
    - Match observations with footprint files
        - match_arctic_cap_airborne_footprint.py
        - match_carve_footprint.py
    - Select observations above 2,000 m above ground level and calculate background signals
    - Calculate CO2 enhancement/drawdown

2. **Data Preprocessing for carbon flux and remote sensing datasets**: 
    - Regriding all datasets to 0.5 degree, monthly resolution
        - regrid_abcflux.py
        - regrid_ceres_par.py
        - regrid_gfed_v4.py
        - regrid_gosif_gpp.py
        - regrid_inversion_GCP2023.py
        - regrid_odiac_FF.py
        - regrid_trendy_v9.py
        - regrid_trendy_v11.py

3. **Model Evaluation**:
    - Convert NEE surface flux fields remote sensing datasets to CO2 enhancement/drawdown
    - Calculate correlation between modeled and observed CO2 enhancement/drawdown

4. **Seasonality Analysis**:
    - Extract seasonal variations of NEE and remote sensing datasets
    - Calculate multiyear average of seasonality

5. **Other scripts**:
    - Generate plots to visualize seasonal patterns in NEE.
    - Create comparative graphs to highlight differences between the three approaches.

6. **Figures**:

## Contact

For any questions or issues, please contact jwen@carnegiescience.edu.