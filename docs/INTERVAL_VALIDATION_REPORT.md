# Validation-only rolling interval calibration

## Conclusion

A development-only search supports different rolling windows at horizons
1/24/48 hours, but does not justify replacing static intervals universally.
No 168-hour candidate meets the development coverage gate. All such failures
are retained: static intervals remain the frozen policy there. Selected
48-hour policies improve aggregate coverage and Winkler score at the cost of
width, while seasonal undercoverage and interval-issuance failures remain.
These are exploratory comparisons on previously inspected test folds, not
independent confirmation or conditional coverage guarantees.

## Protocol, chronology and information set

The unchanged EIA-930 TEX snapshot supplies retrospective hourly demand.
No data rebuilding, downloading, GPU/MPS training or cloud operation occurred.
Point-model information is the existing no-weather demand/calendar design.
Training and calibration use the unchanged target-imputation masks.

Development blocks, all half-open UTC intervals:

- Initial fitting: 2015-08-01 to 2023-01-24.
- Earlier Ridge penalty-selection block: 2023-01-24 to 2023-04-25.
- Final point-model fitting combines those two earlier blocks, with every
  supervised target strictly before 2023-04-25.
- Disjoint calibration warm-up origins: 2023-04-25 to 2023-07-01.
- Interval-policy validation origins: 2023-07-01 to 2024-04-01. Only targets
  strictly before 2024-04-01 are scored.

The 67-day warm-up supplies 60 days of arrived feedback after the longest
168-hour horizon. Gap forecasts crossing July 1 enter rolling state only when
their targets arrive. They cannot enter the static pre-validation pool.
Development input is truncated before April 2024; primary test labels and
fold calibration labels do not select the interval policy.

The fixed Ridge training recipe retains its five previously defined penalties
and its earlier internal hold-out/refitting procedure; this is not new point-
model tuning against interval-validation outcomes. LightGBM retains 600 trees,
63 leaves, learning rate 0.05, seed 0 and the existing L2 settings, with quantile
training disabled because only point predictions are needed. Two CPU threads
are used. New fitted development estimators and predictions are saved locally.
Original four-fold point predictions are reused without refitting or alteration.

Predetermined candidate windows are 7/14/28/60 elapsed days, separately for
each model/horizon, with at least 168 finite nonimputed residual scores.
At origin t, eligible feedback timestamps fall in `(t - window, t]`.
Feedback is target time plus observation delay, not an issue-row shift.
The run assumes zero delay, consistent with the existing benchmark origin
knowing demand through its hour ending. Actual EIA telemetry publication and
revision times are not available; this assumption cannot establish operational
causality. Tests also cover a positive three-hour feedback delay.

The selected window minimizes mean validation 90% Winkler score among
candidates with coverage in [88%, 92%] and complete eligible validation keys;
exact ties prefer the shorter window. Selection is separate per model/horizon,
not per nominal level. The same selected window serves 80/90/95% intervals.
If none qualifies, static remains selected and the failure is recorded.
Static intervals preserve the existing implementation and calibration pool;
rolling margins use the actual finite-score order-statistic rank. The legacy
static NumPy `higher` interpolation convention is retained rather than altering
published evidence to match a new implementation convention.

## Freezing and disclosed application correction

Protocol was written before development outcomes. `selection.json` was frozen
at 2026-10-13 07:57:35.485933 UTC before reading the primary prediction archives
for test application. Its public copy is
`configs/interval_validation_selection_2026-10.json`.

The first test application stopped when selected 7-day windows lacked 168
eligible scores after excluded imputed targets. No test artifact had been
written at that point. Neither the window choices, coverage gate, minimum
score count nor forecasts were changed. Application handling was corrected
**after this failure was seen**: retain unissued intervals as NaN, report their
counts, and compute width/Winkler/paired comparisons on exact common issued
keys. Therefore this missing-interval reporting rule was not fully preregistered.
It is a transparent implementation correction, not an independently confirmed
policy or a test-selected fallback. Missing intervals are additionally counted
as failures in an availability-adjusted coverage diagnostic. They are never
filled with fabricated bounds or excluded without a denominator disclosure.

Static results on common keys can differ from earlier all-key static summaries.
All-key static intervals remain stored in the new local interval archive.
No historical tables or prediction archives were overwritten.

## Metrics and interpretation

Coverage is the fraction with observed demand inside inclusive bounds. Width
is mean upper minus lower bound in MW. Winkler is width plus `2/alpha` times
out-of-interval distance. Tables pair static and selected intervals on identical
fold/origin/target/horizon keys, excluding imputed targets. Target seasons use
Chicago interval-start time, preserving the hour-ending/DST convention.

Paired differences use 500 moving-block replicates within each fold, seed 0,
with 7/14/28-day sensitivity (168/336/672 retained hourly rows). Missing-key
gaps can extend the elapsed span of a bootstrap block; these day labels are
approximate for the common-key subset. They condition on the fitted models and selected
policy, do not include tuning uncertainty, and do not adjust for the multiple
model/horizon/level comparisons. Overlapping target paths, serial dependence
and small effective seasonal sample counts limit interpretation. Interval
selection at 90% does not ensure nominal coverage at 80% or 95%.

## Development results

| model | h | days | n | coverage_percent | width_mw | winkler_mw | complete |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 1 | 7 | 6599 | 90.2409 | 1354.4855 | 1794.0768 | True |
| lightgbm | 24 | 7 | 6576 | 86.9982 | 9987.2640 | 14751.5608 | True |
| lightgbm | 48 | 7 | 6552 | 86.5232 | 14579.7707 | 21661.3337 | True |
| lightgbm | 168 | 7 | 6432 | 85.7743 | 17620.6606 | 26237.8288 | True |
| ridge | 1 | 7 | 6599 | 90.4834 | 1262.3514 | 1698.9210 | True |
| ridge | 24 | 7 | 6576 | 88.2603 | 9144.7042 | 13475.0794 | True |
| ridge | 48 | 7 | 6552 | 86.2332 | 12420.4097 | 20007.4414 | True |
| ridge | 168 | 7 | 6432 | 84.7637 | 15354.2301 | 27023.0798 | True |
| lightgbm | 1 | 14 | 6599 | 90.4228 | 1335.0650 | 1811.5876 | True |
| lightgbm | 24 | 14 | 6576 | 88.6709 | 10002.9749 | 14365.4774 | True |
| lightgbm | 48 | 14 | 6552 | 88.1105 | 14797.6527 | 21181.4629 | True |
| lightgbm | 168 | 14 | 6432 | 86.1007 | 18381.5736 | 26798.2308 | True |
| ridge | 1 | 14 | 6599 | 90.4683 | 1245.4827 | 1700.7853 | True |
| ridge | 24 | 14 | 6576 | 89.3552 | 9189.7477 | 13619.0799 | True |
| ridge | 48 | 14 | 6552 | 89.0110 | 12968.6641 | 19356.4125 | True |
| ridge | 168 | 14 | 6432 | 87.0025 | 16348.9440 | 26075.8655 | True |
| lightgbm | 1 | 28 | 6599 | 90.1349 | 1332.8273 | 1839.8674 | True |
| lightgbm | 24 | 28 | 6576 | 89.2184 | 10155.2579 | 14738.9596 | True |
| lightgbm | 48 | 28 | 6552 | 89.1484 | 14899.6562 | 21116.7782 | True |
| lightgbm | 168 | 28 | 6432 | 87.6399 | 18555.5427 | 27404.1235 | True |
| ridge | 1 | 28 | 6599 | 90.1046 | 1243.7300 | 1700.4489 | True |
| ridge | 24 | 28 | 6576 | 89.4009 | 9180.1508 | 13849.8438 | True |
| ridge | 48 | 28 | 6552 | 89.9725 | 13156.2149 | 19368.7703 | True |
| ridge | 168 | 28 | 6432 | 87.7954 | 16785.0295 | 26402.1408 | True |
| lightgbm | 1 | 60 | 6599 | 90.3016 | 1335.6099 | 1848.8029 | True |
| lightgbm | 24 | 60 | 6576 | 89.1575 | 10342.5312 | 15425.3851 | True |
| lightgbm | 48 | 60 | 6552 | 89.5757 | 15242.7551 | 21993.7196 | True |
| lightgbm | 168 | 60 | 6432 | 86.8159 | 18069.7614 | 27608.7513 | True |
| ridge | 1 | 60 | 6599 | 90.2258 | 1239.6393 | 1700.5330 | True |
| ridge | 24 | 60 | 6576 | 89.3096 | 9443.5135 | 14393.5989 | True |
| ridge | 48 | 60 | 6552 | 89.3315 | 13254.4864 | 20453.8604 | True |
| ridge | 168 | 60 | 6432 | 87.8420 | 16903.0011 | 27700.2917 | True |

Frozen windows: Ridge 7/7/14 days at 1/24/48 hours; LightGBM 7/14/28 days.
Both 168-hour policies retain static. Ridge's best 168-hour development
coverage is 87.842%, narrowly below 88%; LightGBM's is 87.640%. The tolerance
is not relaxed. Development counts per model/horizon are 6599/6576/6552/6432.

## Four-fold test results at all requested levels

| model | method | h | nominal_percent | n | coverage_percent | width_mw | winkler_mw |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | conformal_static | 1 | 95.0000 | 17324 | 95.2898 | 1864.2103 | 2420.6061 |
| lightgbm | conformal_static | 1 | 90.0000 | 17324 | 90.1466 | 1415.1607 | 1995.7275 |
| lightgbm | conformal_static | 1 | 80.0000 | 17324 | 80.5357 | 1037.5927 | 1595.6720 |
| lightgbm | conformal_static | 24 | 95.0000 | 17400 | 93.5402 | 12386.6141 | 19419.8239 |
| lightgbm | conformal_static | 24 | 90.0000 | 17400 | 88.2241 | 9506.7330 | 15338.5375 |
| lightgbm | conformal_static | 24 | 80.0000 | 17400 | 78.2759 | 6928.0054 | 11833.1509 |
| lightgbm | conformal_static | 48 | 95.0000 | 17304 | 92.5566 | 16345.1733 | 26622.2120 |
| lightgbm | conformal_static | 48 | 90.0000 | 17304 | 86.7256 | 12868.6611 | 21300.5908 |
| lightgbm | conformal_static | 48 | 80.0000 | 17304 | 76.9360 | 9634.3474 | 16499.5763 |
| lightgbm | conformal_static | 168 | 95.0000 | 16824 | 91.8985 | 19198.2867 | 31477.3188 |
| lightgbm | conformal_static | 168 | 90.0000 | 16824 | 86.4123 | 15586.8420 | 25329.1106 |
| lightgbm | conformal_static | 168 | 80.0000 | 16824 | 76.6049 | 11760.1134 | 19955.6885 |
| lightgbm | selected_policy | 1 | 95.0000 | 17324 | 95.1570 | 1842.5522 | 2391.5640 |
| lightgbm | selected_policy | 1 | 90.0000 | 17324 | 90.3660 | 1446.0533 | 1979.8879 |
| lightgbm | selected_policy | 1 | 80.0000 | 17324 | 80.5068 | 1054.0648 | 1584.1241 |
| lightgbm | selected_policy | 24 | 95.0000 | 17400 | 93.5345 | 13041.2752 | 19094.2371 |
| lightgbm | selected_policy | 24 | 90.0000 | 17400 | 88.5402 | 10095.7394 | 15210.9827 |
| lightgbm | selected_policy | 24 | 80.0000 | 17400 | 78.8103 | 7316.0491 | 11788.7952 |
| lightgbm | selected_policy | 48 | 95.0000 | 17304 | 93.3830 | 18007.7202 | 26083.7060 |
| lightgbm | selected_policy | 48 | 90.0000 | 17304 | 88.7656 | 14195.7743 | 21059.5507 |
| lightgbm | selected_policy | 48 | 80.0000 | 17304 | 78.7795 | 10442.4227 | 16451.7086 |
| lightgbm | selected_policy | 168 | 95.0000 | 16824 | 91.8985 | 19198.2867 | 31477.3188 |
| lightgbm | selected_policy | 168 | 90.0000 | 16824 | 86.4123 | 15586.8420 | 25329.1106 |
| lightgbm | selected_policy | 168 | 80.0000 | 16824 | 76.6049 | 11760.1134 | 19955.6885 |
| ridge | conformal_static | 1 | 95.0000 | 17324 | 95.4341 | 1802.8748 | 2342.9622 |
| ridge | conformal_static | 1 | 90.0000 | 17324 | 90.5622 | 1384.0212 | 1928.1931 |
| ridge | conformal_static | 1 | 80.0000 | 17324 | 80.1547 | 970.4658 | 1523.6398 |
| ridge | conformal_static | 24 | 95.0000 | 17209 | 93.1954 | 12143.6609 | 19484.4874 |
| ridge | conformal_static | 24 | 90.0000 | 17209 | 87.5821 | 9292.0903 | 15377.8986 |
| ridge | conformal_static | 24 | 80.0000 | 17209 | 77.9302 | 6575.0273 | 11676.0291 |
| ridge | conformal_static | 48 | 95.0000 | 17304 | 91.4933 | 14925.8355 | 26742.8116 |
| ridge | conformal_static | 48 | 90.0000 | 17304 | 86.2575 | 11907.0124 | 20861.2464 |
| ridge | conformal_static | 48 | 80.0000 | 17304 | 76.1385 | 8908.3531 | 15947.2961 |
| ridge | conformal_static | 168 | 95.0000 | 16824 | 89.4912 | 17916.5430 | 33404.2290 |
| ridge | conformal_static | 168 | 90.0000 | 16824 | 82.8103 | 14351.8698 | 26788.5257 |
| ridge | conformal_static | 168 | 80.0000 | 16824 | 71.4693 | 10715.2722 | 20923.6948 |
| ridge | selected_policy | 1 | 95.0000 | 17324 | 95.1801 | 1776.6348 | 2336.5928 |
| ridge | selected_policy | 1 | 90.0000 | 17324 | 90.6084 | 1377.6846 | 1912.5899 |
| ridge | selected_policy | 1 | 80.0000 | 17324 | 80.4837 | 995.6769 | 1516.7367 |
| ridge | selected_policy | 24 | 95.0000 | 17209 | 92.3819 | 12457.8328 | 19391.5071 |
| ridge | selected_policy | 24 | 90.0000 | 17209 | 87.9772 | 9834.9192 | 14975.2599 |
| ridge | selected_policy | 24 | 80.0000 | 17209 | 78.9819 | 7163.3443 | 11438.0040 |
| ridge | selected_policy | 48 | 95.0000 | 17304 | 93.2559 | 17047.0561 | 25528.9936 |
| ridge | selected_policy | 48 | 90.0000 | 17304 | 88.4824 | 13553.6873 | 20318.5241 |
| ridge | selected_policy | 48 | 80.0000 | 17304 | 78.9991 | 9987.2346 | 15761.8633 |
| ridge | selected_policy | 168 | 95.0000 | 16824 | 89.4912 | 17916.5430 | 33404.2290 |
| ridge | selected_policy | 168 | 90.0000 | 16824 | 82.8103 | 14351.8698 | 26788.5257 |
| ridge | selected_policy | 168 | 80.0000 | 16824 | 71.4693 | 10715.2722 | 20923.6948 |

## Interval availability

Each count applies at all three nominal levels. Missing bounds cannot receive
a finite Winkler score; common-key scores do not measure this failure.

| model | h | n_targets | n_issued | n_unavailable | issuance_fraction | coverage_missing_as_failure_percent |
| --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 1 | 17492 | 17324 | 168 | 0.9904 | 89.4981 |
| lightgbm | 24 | 17400 | 17400 | 0 | 1.0000 | 88.5402 |
| lightgbm | 48 | 17304 | 17304 | 0 | 1.0000 | 88.7656 |
| lightgbm | 168 | 16824 | 16824 | 0 | 1.0000 | 86.4123 |
| ridge | 1 | 17492 | 17324 | 168 | 0.9904 | 89.7382 |
| ridge | 24 | 17400 | 17209 | 191 | 0.9890 | 87.0115 |
| ridge | 48 | 17304 | 17304 | 0 | 1.0000 | 88.4824 |
| ridge | 168 | 16824 | 16824 | 0 | 1.0000 | 82.8103 |

## Seasonal 90% coverage and interval scores

| model | method | h | season | n | coverage_percent | width_mw | winkler_mw |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | conformal_static | 1 | autumn | 4370 | 91.8764 | 1466.3089 | 1977.8076 |
| lightgbm | conformal_static | 1 | spring | 4414 | 90.4622 | 1363.3592 | 1891.5133 |
| lightgbm | conformal_static | 1 | summer | 4414 | 88.3779 | 1433.0031 | 2097.4460 |
| lightgbm | conformal_static | 1 | winter | 4126 | 89.8691 | 1397.3170 | 2017.3772 |
| lightgbm | conformal_static | 24 | autumn | 4370 | 94.3021 | 10240.6815 | 12356.1243 |
| lightgbm | conformal_static | 24 | spring | 4414 | 84.9343 | 8763.7202 | 15410.5065 |
| lightgbm | conformal_static | 24 | summer | 4368 | 91.3462 | 9757.8288 | 13720.2194 |
| lightgbm | conformal_static | 24 | winter | 4248 | 82.1798 | 9265.5648 | 19995.8557 |
| lightgbm | conformal_static | 48 | autumn | 4370 | 93.1579 | 13990.7060 | 17340.6021 |
| lightgbm | conformal_static | 48 | spring | 4414 | 83.1219 | 11732.4757 | 20079.1636 |
| lightgbm | conformal_static | 48 | summer | 4320 | 90.6713 | 13244.2354 | 18559.0466 |
| lightgbm | conformal_static | 48 | winter | 4200 | 79.7619 | 12508.9718 | 29524.4007 |
| lightgbm | conformal_static | 168 | autumn | 4370 | 89.7483 | 16637.0951 | 21233.5639 |
| lightgbm | conformal_static | 168 | spring | 4414 | 86.9053 | 14524.6482 | 21071.2593 |
| lightgbm | conformal_static | 168 | summer | 4080 | 90.4902 | 15897.7387 | 24746.1967 |
| lightgbm | conformal_static | 168 | winter | 3960 | 77.9798 | 15291.5036 | 35195.2677 |
| lightgbm | selected_policy | 1 | autumn | 4370 | 90.5034 | 1410.5984 | 1980.0307 |
| lightgbm | selected_policy | 1 | spring | 4414 | 89.8731 | 1362.6114 | 1883.1107 |
| lightgbm | selected_policy | 1 | summer | 4414 | 90.4395 | 1583.9370 | 2109.9008 |
| lightgbm | selected_policy | 1 | winter | 4126 | 90.6689 | 1425.3631 | 1944.1812 |
| lightgbm | selected_policy | 24 | autumn | 4370 | 88.7643 | 8561.0490 | 12579.1909 |
| lightgbm | selected_policy | 24 | spring | 4414 | 87.8568 | 10165.1674 | 14886.0282 |
| lightgbm | selected_policy | 24 | summer | 4368 | 89.8581 | 9909.6327 | 14352.2457 |
| lightgbm | selected_policy | 24 | winter | 4248 | 87.6648 | 11793.7282 | 19139.0058 |
| lightgbm | selected_policy | 48 | autumn | 4370 | 89.2220 | 12185.5514 | 18031.4181 |
| lightgbm | selected_policy | 48 | spring | 4414 | 88.3099 | 13939.4509 | 20127.4743 |
| lightgbm | selected_policy | 48 | summer | 4320 | 90.1852 | 13559.6827 | 18618.3575 |
| lightgbm | selected_policy | 48 | winter | 4200 | 87.3095 | 17211.0129 | 27700.7602 |
| lightgbm | selected_policy | 168 | autumn | 4370 | 89.7483 | 16637.0951 | 21233.5639 |
| lightgbm | selected_policy | 168 | spring | 4414 | 86.9053 | 14524.6482 | 21071.2593 |
| lightgbm | selected_policy | 168 | summer | 4080 | 90.4902 | 15897.7387 | 24746.1967 |
| lightgbm | selected_policy | 168 | winter | 3960 | 77.9798 | 15291.5036 | 35195.2677 |
| ridge | conformal_static | 1 | autumn | 4370 | 92.0137 | 1433.6628 | 1894.0844 |
| ridge | conformal_static | 1 | spring | 4414 | 90.7567 | 1334.4291 | 1879.6173 |
| ridge | conformal_static | 1 | summer | 4414 | 88.8536 | 1401.5591 | 2008.6288 |
| ridge | conformal_static | 1 | winter | 4126 | 90.6447 | 1365.7357 | 1930.2349 |
| ridge | conformal_static | 24 | autumn | 4370 | 95.1487 | 10486.9615 | 12250.6141 |
| ridge | conformal_static | 24 | spring | 4414 | 82.9860 | 8098.2740 | 14870.5827 |
| ridge | conformal_static | 24 | summer | 4368 | 91.3690 | 9706.0444 | 13665.8046 |
| ridge | conformal_static | 24 | winter | 4057 | 80.3549 | 8858.2145 | 21141.7517 |
| ridge | conformal_static | 48 | autumn | 4370 | 93.3867 | 13196.2128 | 16337.1056 |
| ridge | conformal_static | 48 | spring | 4414 | 82.2157 | 10600.3163 | 18807.9892 |
| ridge | conformal_static | 48 | summer | 4320 | 90.1157 | 12338.1248 | 17435.3401 |
| ridge | conformal_static | 48 | winter | 4200 | 79.1190 | 11495.4756 | 31250.1721 |
| ridge | conformal_static | 168 | autumn | 4370 | 87.2540 | 15310.9005 | 22312.7716 |
| ridge | conformal_static | 168 | spring | 4414 | 82.0571 | 13382.8043 | 21689.4420 |
| ridge | conformal_static | 168 | summer | 4080 | 87.9412 | 14636.0668 | 23160.7302 |
| ridge | conformal_static | 168 | winter | 3960 | 73.4596 | 14080.9017 | 41149.0829 |
| ridge | selected_policy | 1 | autumn | 4370 | 90.5950 | 1342.4694 | 1877.5716 |
| ridge | selected_policy | 1 | spring | 4414 | 90.3942 | 1317.8630 | 1876.8027 |
| ridge | selected_policy | 1 | summer | 4414 | 90.5301 | 1496.3991 | 2001.4654 |
| ridge | selected_policy | 1 | winter | 4126 | 90.9355 | 1351.9786 | 1892.8850 |
| ridge | selected_policy | 24 | autumn | 4370 | 87.5057 | 8113.1420 | 12174.2313 |
| ridge | selected_policy | 24 | spring | 4414 | 87.9474 | 10033.4263 | 14740.3564 |
| ridge | selected_policy | 24 | summer | 4368 | 88.7592 | 9385.5299 | 14039.0895 |
| ridge | selected_policy | 24 | winter | 4057 | 87.6756 | 11957.3960 | 19255.8986 |
| ridge | selected_policy | 48 | autumn | 4370 | 90.1831 | 11630.5918 | 16672.8540 |
| ridge | selected_policy | 48 | spring | 4414 | 87.9248 | 13595.7744 | 18939.1585 |
| ridge | selected_policy | 48 | summer | 4320 | 88.3796 | 12181.4319 | 18259.6681 |
| ridge | selected_policy | 48 | winter | 4200 | 87.4048 | 16921.8533 | 27679.0851 |
| ridge | selected_policy | 168 | autumn | 4370 | 87.2540 | 15310.9005 | 22312.7716 |
| ridge | selected_policy | 168 | spring | 4414 | 82.0571 | 13382.8043 | 21689.4420 |
| ridge | selected_policy | 168 | summer | 4080 | 87.9412 | 14636.0668 | 23160.7302 |
| ridge | selected_policy | 168 | winter | 3960 | 73.4596 | 14080.9017 | 41149.0829 |

## Paired 90% comparisons and block-length sensitivity

Negative Winkler differences favor selected intervals. Coverage differences
are proportions, not percentage points. These exploratory 95% intervals
have no multiple-comparison correction.

| model | h | block_days | n | effective_blocks_approx | winkler_difference_mw | winkler_lo | winkler_hi | coverage_difference | coverage_lo | coverage_hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 1 | 7 | 17324 | 103.1190 | -15.8396 | -35.7813 | 3.7498 | 0.0022 | -0.0058 | 0.0120 |
| lightgbm | 1 | 14 | 17324 | 51.5595 | -15.8396 | -34.6548 | 2.8650 | 0.0022 | -0.0083 | 0.0152 |
| lightgbm | 1 | 28 | 17324 | 25.7798 | -15.8396 | -33.2882 | 2.1940 | 0.0022 | -0.0108 | 0.0134 |
| lightgbm | 24 | 7 | 17400 | 103.5714 | -127.5548 | -869.6586 | 484.8639 | 0.0032 | -0.0117 | 0.0238 |
| lightgbm | 24 | 14 | 17400 | 51.7857 | -127.5548 | -866.2124 | 388.4684 | 0.0032 | -0.0159 | 0.0247 |
| lightgbm | 24 | 28 | 17400 | 25.8929 | -127.5548 | -906.2776 | 360.5811 | 0.0032 | -0.0205 | 0.0255 |
| lightgbm | 48 | 7 | 17304 | 103.0000 | -241.0401 | -1387.8060 | 735.4822 | 0.0204 | 0.0048 | 0.0427 |
| lightgbm | 48 | 14 | 17304 | 51.5000 | -241.0401 | -1486.0088 | 794.6158 | 0.0204 | -0.0005 | 0.0495 |
| lightgbm | 48 | 28 | 17304 | 25.7500 | -241.0401 | -1241.7723 | 944.3326 | 0.0204 | -0.0052 | 0.0496 |
| lightgbm | 168 | 7 | 16824 | 100.1429 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lightgbm | 168 | 14 | 16824 | 50.0714 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lightgbm | 168 | 28 | 16824 | 25.0357 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ridge | 1 | 7 | 17324 | 103.1190 | -15.6032 | -31.3848 | -0.7913 | 0.0005 | -0.0056 | 0.0076 |
| ridge | 1 | 14 | 17324 | 51.5595 | -15.6032 | -35.2523 | -0.6788 | 0.0005 | -0.0059 | 0.0089 |
| ridge | 1 | 28 | 17324 | 25.7798 | -15.6032 | -29.6955 | 0.1307 | 0.0005 | -0.0084 | 0.0077 |
| ridge | 24 | 7 | 17209 | 102.4345 | -402.6387 | -1273.0863 | 315.7992 | 0.0040 | -0.0129 | 0.0240 |
| ridge | 24 | 14 | 17209 | 51.2173 | -402.6387 | -1217.9516 | 181.0579 | 0.0040 | -0.0160 | 0.0301 |
| ridge | 24 | 28 | 17209 | 25.6086 | -402.6387 | -1031.6100 | 188.3274 | 0.0040 | -0.0187 | 0.0247 |
| ridge | 48 | 7 | 17304 | 103.0000 | -542.7223 | -1957.2175 | 638.1147 | 0.0222 | 0.0025 | 0.0439 |
| ridge | 48 | 14 | 17304 | 51.5000 | -542.7223 | -2172.6762 | 601.3182 | 0.0222 | 0.0003 | 0.0505 |
| ridge | 48 | 28 | 17304 | 25.7500 | -542.7223 | -1556.0787 | 369.7500 | 0.0222 | -0.0006 | 0.0515 |
| ridge | 168 | 7 | 16824 | 100.1429 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ridge | 168 | 14 | 16824 | 50.0714 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ridge | 168 | 28 | 16824 | 25.0357 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## What improved, what worsened, and recommendation

At 90%, Ridge's 48-hour coverage increases from 86.2575% to 88.4824%,
width from 11,907.01 to 13,553.69 MW (+13.83%), and Winkler falls from
20,861.25 to 20,318.52 MW (-2.60%). LightGBM's coverage increases from
86.7256% to 88.7656%, width from 12,868.66 to 14,195.77 MW, and Winkler
falls by 241.04 MW (-1.13%). Both coverage improvements are descriptively
useful, but their seven-day bootstrap Winkler intervals include zero:
Ridge [-1,957.22, 638.11] MW; LightGBM [-1,387.81, 735.48] MW.
Do not infer a reliably positive net utility solely from aggregate coverage.

One-hour gains are small; Ridge's seven-day Winkler interval barely excludes
zero and does not by itself survive a claim across multiple block lengths,
models, levels and horizons. At 24 hours, selected Ridge 90% coverage is
87.9772% on common issued keys, and only 87.0115% when unavailable intervals
count as failures. Selection does not repair nominal coverage there.
At Ridge 24 hours/95%, coverage **worsens** from 93.1954% to 92.3819%,
even though Winkler decreases. Tuning at one level did not ensure all-level
coverage. Wider intervals are a cost, not an unqualified benefit.

Selected winter 90% coverage at 48 hours improves to 87.4048% for Ridge and
87.3095% for LightGBM, from 79.1190% and 79.7619%. Those seasonal failures
persist below nominal. At 168 hours, no development candidate qualifies;
static remains unchanged at 82.8103%/86.4123% overall and 73.4596%/77.9798%
winter coverage for Ridge/LightGBM. Earlier favorable test-only 60-day rolling
results cannot override this preregistered development failure.

The 7-day/168-score constraint prevents interval issuance on 168 one-hour
rows per model and 191 Ridge 24-hour rows after missing/imputed feedback.
This operational failure is not repaired by changing the window on test data.
Accordingly, **do not adopt this selection as a universal replacement for static
intervals or call it calibrated operational uncertainty**. The 48-hour arms
justify further controlled study and may be useful descriptively where their
wider bounds are acceptable. Production adoption requires an independently
locked later holdout, verified observation latency, explicit predeclared
missing-feedback behavior, and an externally chosen cost/coverage threshold.
A new availability-tolerant policy would be a separately registered experiment,
not a retrospective correction to these frozen choices. No ACI retuning occurred.

## Resources and retained failed attempt

The pilot took 8.218 s and peaked at 814,481,408 bytes (776.75 MiB). Full
development took 33.501 s and peaked at 951,664,640 bytes (907.58 MiB).
Both passed the original 1 GiB budget. The initial test application produced
numerical outputs but failed its final memory guard at 1,537,900,544 bytes
(1.432 GiB). Its failure metadata and all artifacts remain untouched in the
development directory. No successful test metadata was fabricated for it.

The user explicitly approved a 2 GiB CPU cap on October 13, 2026. The
continuation added one-second peak-memory/time checks, failure metadata,
a separate new evaluation directory and array-based bootstrap index assembly.
These affect resource handling, not draws, keys, policy or calculations.
The final test application completed in 186.639 s with 1,299,103,744-byte
peak RSS (1.210 GiB), below 2 GiB and the unchanged 1,200-second limit.
All five metric/availability/bootstrap CSVs are byte-identical to the failed
attempt; selection hash also matches exactly. Signal checks may be delayed
inside native calls, so they are monitoring guards rather than a hard OS
memory reservation. The failure is retained as provenance, not omitted.

## Verification and provenance

124 deterministic CPU tests passed: 119 non-neural plus five neural fixtures
in separate processes. Neural fixtures are not neural benchmark
experiments. New tests cover elapsed eligibility and gaps at all horizons,
positive feedback delay, score counts, sparse-score expiry, legacy hourly
regression equivalence, disjoint split chronology, fail-closed selection,
periodic memory/time guards, overwrite refusal, positional API compatibility and paired reproducibility.
Ruff lint, touched-file formatting, bytecode compilation, offline uv lock,
whitespace checks and the four existing evidence-package checks passed.
Existing pandas deprecation warnings remain. No numerical result was changed
to make tests pass.

`check_interval_validation` independently checks all 432 aggregate/fold/season
rows by direct coverage/width/Winkler formulas, all issuance denominators,
144 sampled causal score pools, exact equality of observed demand and point
predictions to saved baseline rows, frozen selection reproduced solely from
development metrics, input hashes, boundaries and measured resource limits.
It does not independently rerun every bootstrap draw or point-model fit.
Five original baseline prediction hashes match before/after execution; the
processed dataset, immutable four test fold definitions and existing evidence
packages remain unchanged. Only source, tests, a frozen selection configuration
and Markdown documentation are published. Pilot (~3.8 MiB), development
(~38 MiB) and completed evaluation (~23 MiB) evidence, including models and
parquet archives, stay ignored.

Dataset version: `eia930-TEX+openmeteo@2026-10-10`. Processed SHA-256:
`1e3acfa6c4070d0dc69d6a2d854964f09afc42b3291221fda646c606249ca5ab`.
Execution source base commit: `91652b01f78a87292b622b1c31e21c08c955b2fe`,
with dirty source hashes and lock/data/version provenance recorded. Executed
test runner and online calibration sources are archived in the evaluation
directory. Development sources have their contemporaneous hashes; subsequent
application corrections are disclosed above. Packages were not installed or
upgraded. CPU only; no dataset download, GPU/MPS job, paid API or cloud change.

Frozen selection SHA-256: `e5ed07fb6f39e2bcf4763174a73b464ad4fbe16f3d8c3d9e695a86f08071e97a`.
Protocol SHA-256: `41caafce6aca61cc1eb088198679bec38d2aa9e6b65498cba06e11a262ee4f7a`.

## Reproducibility and local evidence

- Pilot: `results/interval_validation/20261013T075606Z/`.
- Development and retained failed application: `results/interval_validation/20261013T075702Z/`.
- Completed test evaluation: `results/interval_validation/20261013T151258Z/`.

Full 80/90/95% seasonal and fold metrics are in `test_season.csv` and
`test_fold.csv`; all levels are available in `test_overall.csv`.
`test_availability.csv` reports target/issued counts and missing-as-failure
coverage. `paired_block_comparisons.csv` holds all level/block sensitivities.
`test_intervals.parquet` preserves every eligible target and missing interval.
Saved development point models, predictions, protocol, selection and run
metadata are in the development directory. `test_run.json` records final
resource use, source/lock/data hashes, versions, baseline hashes and all folds.

Run from the repository root using its existing project environment. Keep the
failed attempt intact and allow the runner to create a new evaluation directory:

```sh
PYTHONPATH=src GRIDCAST_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MPLCONFIGDIR=results/scratch/matplotlib \
  .venv/bin/python -m gridcast.experiments.interval_validation test \
  --development-source results/interval_validation/20261013T075702Z \
  --memory-limit-mib 2048
PYTHONPATH=src .venv/bin/python -m gridcast.experiments.check_interval_validation \
  results/interval_validation/20261013T151258Z
```

For a fresh development replication, first run the same module with `pilot`,
then `development` (default 1 GiB cap), and supply its new output directory to
the test stage. Model recipes and protocol are defined in the source, not
inferred from test results. Selecting another window, changing observation
latency, or adding a test-dependent fallback constitutes a new protocol.
The four previously inspected test folds remain exploratory; this run does
not validate an independent holdout or guarantee exchangeable residuals.
