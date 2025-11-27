# ABoVE_NEE_seasonality

## Overview

This repository contains code for analyzing the seasonality of Net Ecosystem Exchange (NEE) in the ABoVE (Arctic-Boreal Vulnerability Experiment) study area. We evaluated NEE estimates derived from three distinct approaches: atmospheric inversions, upscaled eddy-covariance (EC) flux tower measurements, and terrestrial biosphere models (TBMs), using atmospheric CO2 observations collected during airborne campaigns organized as part of the Carbon in Arctic Reservoirs Vulnerability Experiment (CARVE) and the Arctic Carbon Atmospheric Profiles (Arctic-CAP) campaigns during 2012-2014 and 2017, respectively.

## Contents

- `src/`: Directory containing analysis scripts.
- `data/`: Directory containing input data.
- `results/`: Directory containing output results.
- `README.md`: This file.

## Data
- Atmospheric CO2 concentration measurements and footprints: CARVE and Arctic-CAP
- Atmospheric inversions: GCB 2023
- Upscaled EC datasets: X-BASE, Upscaled ABCflux
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
        - `derive_h_matrix_arctic_cap_monthly.py`
        - `derive_h_matrix_carve_monthly.py`
        - ancillary scripts: `utils.py`
        - additional scripts for finer temporal resolutions (3-hourly and daily): `derive_h_matrix_3hourly.py`, `aggregate_footprint_3hourly_daily_monthly.py`
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
    - Aggregate MODIS FPAR/LAI from 8-day to monthly
        - `aggregate_monthly_modis_fpar_lai.py`
    - Regriding all datasets to 0.5 degree, monthly resolution
        - `regrid_abcflux.py`
        - `regrid_ceres_par.py`
        - `regrid_gfed_v41.py`
        - `regrid_gfed_v5.py`
        - `regrid_gosif_gpp.py`
        - `regrid_inversion_GCP2023.py`
        - `regrid_inversions_highres.py`
        - `regrid_modis_fpar_lai.py`
        - `regrid_odiac_FF.py`
        - `regrid_trendy_v9.py`
        - `regrid_trendy_v11.py`
        - `regrid_x_base_highres.py`

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
    - Plot seasonal variations of NEE
        - `plot_seasonal.py`
        - `plot_seasonal_different_years.py`
    - Plot seasonal variations of scaled remtoe sensing variables
        - `plot_seasonal_scaled_remote_sensing.py`
    - Modify NEE season cycle and examine its consistency with atmospheric observations
        - `modify_TRENDY_component_seasonal_example.py`
        - `modify_TRENDY_component_seasonal_groupH.py`
        - `modify_X_BASE_component_seasonal_groupH.py`
    - Compute Mean Absolute Deviation of seasonality compared to reference data
        - `Fig3.py`
    - Analyze the relationship between NEE seasonlity bias and consistency with atmospheric observations
        - `examine_cor_mad.py`
    - Analyze the relationship between NEE seasonlity bias and GPP/Reco seasonality bias
        - `Fig5.py`
    - Analyze the relationship of NEE seasonality bias between the whole region and individual land cover
        - `examine_mad_across_lc.py`

5. **Other scripts for additional analyses**:
    - Sensitivity tests for model evaluation
        - Use afternoon-only data: `evaluate_stat_multiyear_afternoon.py`, `Fig2_afternoon.py`
        - Use finer temporal resolution NEE: `aggregate_inversions_3hourly_daily_monthly.py`, `convert_flux_to_concentration_3hourly.py`, `convert_flux_to_concentration_daily.py`, `convert_flux_to_concentration_monthly.py`, `convert_flux_to_concentration_diurnal_cycle_x_base.py`, `evaluate_stat_multiyear_highres.py`, `evaluate_stat_multiyear_highres_plot.py`, `evaluate_stat_multiyear_diurnal_x_base.py`, `Fig2_diurnal_x_base.py`
    - Compare relative proportion of carbon component (Ra, Rh to GPP)
        - `compare_component_flux_ratio.py`
    - Evaluate seasonality difference between each pair of TRENDY models
        - `calculate_seasonality_uncertainty.py`
    - Evaluate consistency between atmospheric observations and inversions based on different CO2 input sources
        - `compare_inv_insitu_satellite.py`
    

6. **Figures in the manuscript**:
    - Figure 1: `Fig1.py`
    - Figure 2: `Fig2.py`
    - Figure 3: `Fig3.py`
    - Figure 4: `Fig4.py`
    - Figure 5: `Fig5.py`

    - Figure S1: `modify_TRENDY_component_seasonal_example.py`
    - Figure S2: `Fig2_different_year.py`
    - Figure S3: `compare_inv_insitu_satellite.py`
    - Figure S4: `Fig2_scaled.py`
    - Figure S5: `Fig2_different_lc`
    - Figure S6: `Fig2.py_full.py` 
    - Figure S7: `extract_seasonal_multiyear.py`
    - Figure S8: `examine_cor_mad.py`
    - Figure S9: `examine_mad_across_lc.py`
    - Figure S10: `plot_seasonal_scaled_remote_sensing.py`
    - Figure S11: `compare_component_flux_ratio.py`
    - Figure S12-S16, S19-20: `Fig3.py`
    - Figure S18: `Fig4.py`
    - Figure S19: `Fig5.py`
    - Figure S21: `calculate_seasonality_uncertainty.py`

## Contact

For any questions or issues, please contact jwen@carnegiescience.edu.