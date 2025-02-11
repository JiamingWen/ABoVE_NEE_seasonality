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
- TBM outputs: TRENDY (v11, v9)
- Remote sensing data: APAR (calculated from MODIS FPAR and CERES PAR), GOME-2 SIF, GOSIF GPP

## Analysis Steps

1. **Data Preprocessing for atmospheric observations**: 
    - Match observations with footprint files
        - `match_arctic_cap_airborne_footprint.py`
        - `match_carve_footprint.py`
    - Select observations above 2,000 m above ground level and calculate background signals
        - `calculate_arctic_cap_airborne_background.py`
        - `calculate_carve_airborne_background.py`
    - Calculate CO2 enhancement/drawdown relative to background signals
        - `calculate_arctic_cap_airborne_change.py`
        - `calculate_carve_airborne_change.py`
    - Create H matrix
        - `derive_h_matrix_arctic_cap.py`
        - `derive_h_matrix_carve.py`
        - ancillary scripts: `utils.py`; `config_carve2012.ini`; `config_carve2013.ini`; `config_carve2014.ini`; `config_arctic_cap2017.ini`
    - Summarize/Aggregate footprint sensitivity: 
    (1) for each atmospheric observation, summarize footprint sensitivity from different regions (e.g., land vs ocean; within vs out of ABoVE; forests vs shrubs vs tundra)
    (2) Aggregate footprint sensitivity over all atmospheric observations for each month/year
        - `summarize_footprint_regional_influence.py`
        - `summarize_footprint_regional_influence_selected.py` (for selected atmospheric observations)

2. **Data Preprocessing for other datasets**: 
    - Generate a table with information of id, lat, lon, land cover, whether in the ABoVE region
        - `create_cellid_table.py`
    - Determine the dominant land cover of each 0.5 degree pixel based on land cover area fraction
        - `select_esa_cci_dominant_landcover.R`
    - Create a bbox shapefile that contains the ABoVE region, used to download MODIS FPAR from AppEEARS
        -  `create_ABoVE_bbox.R`
    - Download MODIS data from AppEEARS
        - `download_AppEEARS.R`
    - Aggregate MODIS FPAR/LAI from 8-day to monthly
        - `aggregate_monthly_modis_fpar_lai.py`
    - Regriding all datasets to 0.5 degree, monthly resolution
        - `regrid_abcflux.py`
        - `regrid_ceres_par.py`
        - `regrid_gfed_v4.py`
        - `regrid_gosif_gpp.py`
        - `regrid_inversion_GCP2023.py`
        - `regrid_modis_fpar_lai.py`
        - `regrid_odiac_FF.py`
        - `regrid_trendy_v9.py`
        - `regrid_trendy_v11.py`

3. **Model Evaluation**:
    - Convert NEE surface flux or remote sensing fields to concentration space
        - `convert_flux_to_concentration.py` (recommended memory: 100-150 GB)
        - `convert_variable_to_concentration_bymonth.py`: sometimes require large memory; calculate at monthly basis; if added together for each year, the results are equal to `convert_flux_to_concentration.py`
        - `convert_flux_to_concentration_only_seasonal.py`: replacing the original fields with mean seasonal cycle
    - Calculate correlation between modeled and observed CO2 enhancement/drawdown
        - `evaluate_stat.py`
        - `evaluate_stat_multiyear.py`
    - Fit regression (Month, Month x LC)
        - `fit_regression_Month.py`
        - `fit_regression_Month_LC.py`

4. **Seasonality Analysis**:
    - Extract seasonal variations of NEE and remote sensing datasets
        - `extract_seasonal.py`
        - `extract_seasonal_multiyear.py`
    - Modify NEE season cycle and examine its consistency with atmospheric observations

5. **Other scripts for additional analyses**:

        - calculate_seasonality_uncertainty.py
        - compare_inv_insitu_satellite.py

6. **Figures in the manuscript**:
    - Figure 1: `Fig1.py`
    - Figure 2: `Fig2.py`
    - Figure 3: `Fig3.py`
    - Figure 4: `Fig4.py`
    - Figure 5: `Fig5.py`

    - Figure S2: `Fig2_different_year.py`
    - Figure S4: `Fig2.py_full.py` 
    - Figure S5: `Fig2_different_lc`

## Contact

For any questions or issues, please contact jwen@carnegiescience.edu.