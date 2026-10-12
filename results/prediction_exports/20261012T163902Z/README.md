# Saved predictions versus EIA demand

These CSVs contain original saved point forecasts and non-imputed observed EIA demand. All times are UTC hour-ending labels; issue time assumes current-hour demand is already known. Signed error is predicted minus observed, in MW.

The good, typical and poor seven-day windows are selected after evaluation by Ridge 24-hour MAE among complete common windows. The figures show the 24-hour horizon; CSVs include all requested horizons. This illustrates variation and is not an unbiased measure of aggregate skill.

Full-data MAE and RMSE for all five models and four horizons were independently reproduced before selecting examples. See independently_reproduced_metrics.csv, selected_periods.csv and manifest.json. No model was refitted and no original artifact was overwritten.
