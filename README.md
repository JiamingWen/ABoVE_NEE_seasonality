# ABoVE_NEE_seasonality

## Overview

This repository contains code for analyzing the seasonality of Net Ecosystem Exchange (NEE) in the ABoVE (Arctic-Boreal Vulnerability Experiment) study area. We evaluated NEE estimates derived from three distinct approaches: atmospheric inversions, upscaled eddy-covariance (EC) flux tower measurements, and terrestrial biosphere models (TBMs), using atmospheric CO2 observations collected during airborne campaigns organized as part of the Carbon in Arctic Reservoirs Vulnerability Experiment (CARVE) and the Arctic Carbon Atmospheric Profiles (Arctic-CAP) campaigns during 2012-2014 and 2017, respectively.

## Contents

- `src/`: Directory containing analysis scripts.
- `README.md`: This file.

The `data/` and `result/` directories are not included in this repository due to file size limitations. The datasets used in this study are all publicly available. Please feel free to contact me if you need access to any results or intermediate files.

## Data
- Atmospheric CO2 concentration measurements and footprints: CARVE and Arctic-CAP
- Atmospheric inversions: GCB 2023
- Upscaled EC datasets: X-BASE, Upscaled ABCflux
- TBM outputs: TRENDY (v11, v9)
- Remote sensing data: APAR (calculated from MODIS FPAR and CERES PAR), GOME-2 SIF, GOSIF GPP
- Fossil fuel emissions: ODIAC 2022 (and GridFED v2024.0 for sensitivity tests)
- Fire emissions: GFED v4.1 (and GFED v5 for sensitivity tests)
- Ocean fluxes: ensemble of nine data products from GCB 2024
- Land cover: ESA-CCI

## Analysis Steps and Overview of Scripts

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
    - Summarize/Aggregate footprint sensitivity: 
    (1) for each atmospheric observation, summarize footprint sensitivity from different regions (e.g., land vs ocean; within vs out of ABoVE; forests vs shrubs vs tundra)
    (2) Aggregate footprint sensitivity over all atmospheric observations for each month/year
    (3) Make plots for footprint sensitivity
        - `summarize_footprint_regional_influence.py`
        - `summarize_footprint_regional_influence_selected.py` (for selected atmospheric observations)

2. **Data Preprocessing for other datasets**: 
    - Generate a table with information of id, lat, lon, land cover, whether in the ABoVE region
        - `create_cellid_table.py`
    - Determine the dominant land cover of each 0.5 degree pixel based on land cover area fraction
        - `select_esa_cci_dominant_landcover.R`
    - Aggregate MODIS FPAR/LAI from 8-day to monthly
        - `aggregate_monthly_modis_fpar_lai.py`
    - Regriding all datasets to 0.5 degree, monthly resolution (in `others/regrid`)
        - `regrid_abcflux.py`
        - `regrid_ceres_par.py`
        - `regrid_gfed_v5.py`
        - `regrid_gfed_v41.py`
        - `regrid_gosif_gpp.py`
        - `regrid_gridfed.py`
        - `regrid_inversion_GCP2023.py`
        - `regrid_inversions_highres.py`
        - `regrid_modis_fpar_lai.py`
        - `regrid_ocean_fluxes.py`
        - `regrid_odiac_FF.py`
        - `regrid_trendy_v9.py`
        - `regrid_trendy_v11.py`
        - `regrid_x_base_highres.py`

3. **Model Evaluation**:
    - Convert NEE surface flux or remote sensing fields to the concentration space
        - `convert_flux_to_concentration.py` (recommended memory: 100-150 GB)
        - `convert_variable_to_concentration_bymonth.py`: sometimes require large memory; calculate at monthly basis; if added together for each year, the results are equal to `convert_flux_to_concentration.py`
        - `convert_flux_to_concentration_only_seasonal.py`: replacing the original fields with the mean seasonal cycle
    - Calculate correlation between modeled and observed CO2 enhancement/drawdown
        - `evaluate_stat.py`: use single year's data
        - `evaluate_stat_multiyear.py`: use all years' data
    - Fit regression (Month, Month x LC)
        - `fit_regression_Month.py`
        - `fit_regression_Month_LC.py`
    - Evaluate consistency between atmospheric observations and inversions based on different CO2 input sources
        - `compare_inv_insitu_satellite.py`

4. **Seasonality Analysis**:
    - Extract seasonal variations of NEE and remote sensing datasets
        - `extract_seasonal.py`
        - `extract_seasonal_multiyear.py`
    - Plot seasonal variations of NEE
        - `plot_seasonal.py`
        - `plot_seasonal_different_years.py`
    - Plot seasonal variations of scaled remote sensing variables
        - `plot_seasonal_scaled_remote_sensing.py`
    - Modify NEE season cycle and examine its consistency with atmospheric observations
        - `modify_TRENDY_component_seasonal_example.py`
        - `modify_TRENDY_component_seasonal_groupH.py`
        - `modify_X_BASE_component_seasonal_groupH.py`
    - Compute Mean Absolute Deviation of seasonality compared to reference data
        - `Fig3.py`
    - Analyze the relationship between NEE seasonality bias and consistency with atmospheric observations
        - `examine_cor_mad.py`
    - Analyze the relationship between NEE seasonality bias and GPP/Reco seasonality bias
        - `Fig5.py`
    - Analyze the relationship of NEE seasonality bias between the whole region and individual land cover
        - `examine_mad_across_lc.py`
    - Compare relative proportion of carbon component (Ra, Rh to GPP)
        - `compare_component_flux_ratio.py`
    - Evaluate seasonality difference between each pair of TRENDY models
        - `calculate_seasonality_uncertainty.py`

5. **Other scripts for additional analyses**:
    - Examine quality filtering criteria for atmospheric observations.
        - `other_tests/data_filter/data_filter_summary.py`
    - Sensitivity tests for model evaluation
        - Use atmospheric observations from different time of day: (1) all daytime observations (default); (2) afternoon-only data. (in `others/sensitivity_afternoon`)
            - `evaluate_stat_multiyear_afternoon.py`
            - `Fig2_afternoon.py`
        - Use model outputs at different temporal resolutions to examine the influence of NEE diurnal and sub-monthly variations: (1) monthly NEE (default); (2) hourly and daily NEE for CT-NOAA, CTE, and X-BASE. (in `others/other_tests/sensitivity_high_res_nee`)
            - `derive_h_matrix_3hourly.py`
            - `aggregate_footprint_3hourly_daily_monthly.py`
            - `compare_day_night_footprint_hist_heatmap.py`
            - `compare_day_night_footprint_map.py`
            <br>
            - `aggregate_inversions_3hourly_daily_monthly.py`
            - `convert_flux_to_concentration_3hourly.py`
            - `convert_flux_to_concentration_daily.py`
            - `convert_flux_to_concentration_monthly.py`
            - `convert_flux_to_concentration_diurnal_cycle_x_base.py`
            - `decompose_var_monthly_diurnal.py`
            - `examine_enhancement_diff_diurnal.py`
            <br>
            - `evaluate_stat_multiyear_highres.py`
            - `evaluate_stat_multiyear_highres_plot.py`
            - `evaluate_stat_multiyear_highres_afternoon.py`
            - `evaluate_stat_multiyear_highres_afternoon_plot.py`
            - `evaluate_stat_multiyear_diurnal_x_base.py`
            - `Fig2_diurnal_x_base.py`
            - `Fig2_diurnal_x_base_other_metrics.py`
            - `Fig2_diurnal_x_base_mean_seasonal.py`
            <br>
            - `modify_TRENDY_component_seasonal_groupH_x_base_diurnal.py`
            - `Fig4_diurnal_x_base.py`
        - Use different background calculations: (1) calculate the average CO2
concentration measured beyond 2,000 m above ground level during each flight date (default) (2) use 3,000 m as height cutoff; (3) extract background values from Carbon Tracker CO2 fields; (4) extract background values from empirical background fields. (in `others/other_tests/sensitivity_backgrounds`) 
            * `extract_background_3k.py`
            * `extract_background_ct_ebg.py`
            * `compare_background.py`
            * &nbsp;
            * `calculate_airborne_change_background.py`
            * `evaluate_stat_multiyear_background.py`
            * `evaluate_stat_multiyear_background_diurnal_x_base.py`
            * &nbsp;
            * `Fig2_background.py`
            * `Fig2_other_metrics_background.py`
        - Use different fossil fuel emission datasets: (1) ODIAC 2022 (default); 
        (2) GridFED v2024.0. (in `others/other_tests/sensitivity_fossil`)
            - `compare_enhancement_fossil.py`
            - `evaluate_stat_multiyear_fossil.py`
            - `Fig2_fossil.py`
            - `Fig2_fossil_other_metrics.py`
        - Use different fire emission datasets: (1) GFEDv4.1 (default); (2) GFED v5.  (in `others/other_tests/sensitivity_fire`)
            - `compare_enhancement_gfed_versions.py`
            - `evaluate_stat_multiyear_gfed5.py`
            - `Fig2_gfed5.py`
            - `Fig2_gfed5_other_metrics.py`
        - Account for ocean fluxes: (1) do not account for ocean fluxes (default); (2) account for ocean fluxes, using the ensemble of nine data products from GCB 2024
            - `convert_flux_to_concentration_ocean.py`
            - `compare_enhancement_ocean.py`
            - `evaluate_stat_multiyear_ocean_flux.py`
            - `evaluate_stat_multiyear_ocean_criterion.py`
            - `Fig2_ocean_flux.py`
            - `Fig2_ocean_flux_other_metrics.py`
            - `Fig2_ocean_criterion.py`
            - `Fig2_ocean_criterion_other_metrics.py`
            - `examine_decreased_correlation.py`
    - Comparing agreement with atmospheric observations for inversion prior and posterior fluxes (in `others/other_tests/inversion_prior`)

6. **Figures and Tabels in the manuscript**:
    - Figure 1: `Fig1.py`
    - Figure 2: `Fig2.py`
    - Figure 3: `Fig3.py`
    - Figure 4: `Fig4.py`
    - Figure 5: `Fig5.py`

    - Figure S1: `modify_TRENDY_component_seasonal_example.py`
    - Figure S2-S4: `plot_scatterplot_obs_model_co2_enhancement.py`
    - Figure S5: `others/sensitivity_afternoon/Fig2_afternoon.py`
    - Figure S6: `evaluate_stat_multiyear_highres_plot.py`
    - Figure S7: `Fig2_diurnal_x_base.py`
    - Figure S8: `Fig2_different_year.py`
    - Figure S9: `compare_inv_insitu_satellite.py`
    - Figure S10: `Fig2_scaled.py`
    - Figure S11: `Fig2_different_lc`
    - Figure S12: `Fig2_full.py` 
    - Figure S13: `examine_cor_mad.py`
    - Figure S14: `examine_mad_across_lc.py`
    - Figure S15: `plot_seasonal_scaled_remote_sensing.py`
    - Figure S16-S20, S24-S25: `Fig3.py`
    - Figure S21: `compare_component_flux_ratio.py`
    - Figure S22: `Fig5.py`
    - Figure S23: `Fig4.py`
    - Figure S26: `calculate_seasonality_uncertainty.py`

    - Table S2: `Fig2_other_metrics.py`
    - Table S3: `Fig4.py`

## Python and package version

Python: 3.11.10.
Package versions are provided in `src/requirements.txt`.

    
## Contact

For any questions or issues, please contact jwen@carnegiescience.edu.