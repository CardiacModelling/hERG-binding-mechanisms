# Data description

Raw data for FDA Milnes' protocol from Li et al. 2017.

- `[drug_name].csv`: Raw data with multiple sweeps, repeats, and concentrations.
- `save_FDA_csv.sh`: Runs `save_FDA_csv.py` for all compounds to create `../Milnes-data`.
- `save_FDA_data.sh`: Runs `save_FDA_data.py` and `save_FDA_data_all.py` for all compounds to save as `*.npy` data.
- `plot_FDA_data.sh`: Runs `plot_FDA_data.py`, `plot_FDA_data_all.py` and `plot_FDA_saturation.py` for all compounds to make simple plots and save as `*.png`.
- `calculate_RMSE_expt_data.sh`: Runs `calculate_RMSE_expt_data.py` for all compounds to compute the bootstrap RMSE for the data and save as `[drug_name]-bootstrap-*-samples.txt`.
- `compare-boxplots.py`: Make a simple boxplot comparison for `[drug_name]-bootstrap-*-samples.txt` for all compounds.
