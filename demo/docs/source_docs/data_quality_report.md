# Data Quality Report

## Dataset: dataset_final.csv (Train set, before 2023-01-01)

| Check | Result |
|-------|--------|
| Missing values | 0 |
| Duplicate rows | 0 |
| Duplicate dates | 0 |
| INF values | 0 |
| Lag1 consistency | OK |

## Top features by IQR outlier count
| feature                |   n_outliers_IQR |   pct_outliers |
|:-----------------------|-----------------:|---------------:|
| net_imports_change_pct |              419 |           20.1 |
| real_rate              |              392 |           18.8 |
| conflict_event_count   |              329 |           15.8 |
| production_change_pct  |              225 |           10.8 |
| fatalities             |              183 |            8.8 |
| sp500_return           |              182 |            8.7 |
| cpi_lag                |              153 |            7.3 |
| unemployment_lag       |              152 |            7.3 |
| oil_return_lag2        |              136 |            6.5 |
| oil_return             |              135 |            6.5 |

> Note: Financial returns naturally have fat tails (high kurtosis).
> Outliers here are expected market events, NOT data errors.
