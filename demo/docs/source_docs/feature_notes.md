# Feature Notes

| Feature | Skew | Kurt | %Outlier | Notes |
|---------|------|------|----------|-------|
| yield_spread | -0.247 | -0.483 | 0.0% | OK |
| cpi_lag | 1.018 | 0.25 | 7.3% | OK |
| unemployment_lag | 3.048 | 10.774 | 7.3% | Highly skewed — consider log-transform. Fat tails (financial).  |
| inventory_change_pct | 0.259 | 0.501 | 1.4% | OK |
| gdelt_goldstein | -0.298 | 1.169 | 2.8% | OK |
| gdelt_events | 0.544 | 0.436 | 0.7% | OK |
| gdelt_tone_7d | 3.676 | 21.318 | 2.5% | Highly skewed — consider log-transform. Fat tails (financial).  |
| gdelt_tone_30d | 4.234 | 22.745 | 2.6% | Highly skewed — consider log-transform. Fat tails (financial).  |
| gdelt_goldstein_7d | -0.082 | 0.892 | 2.9% | OK |
| conflict_event_count | 1.902 | 2.956 | 15.8% | OK |
| fatalities | 2.822 | 9.684 | 8.8% | Highly skewed — consider log-transform.  |
| oil_return | -0.389 | 12.095 | 6.5% | Fat tails (financial).  |
| usd_return | -0.079 | 1.978 | 3.6% | OK |
| sp500_return | -0.532 | 14.988 | 8.7% | Fat tails (financial).  |
| vix_return | 2.575 | 20.559 | 5.5% | Highly skewed — consider log-transform. Fat tails (financial).  |
| oil_volatility_7d | 1.785 | 3.891 | 4.9% | OK |
| fed_rate_change | 2.575 | 217.109 | 2.7% | Highly skewed — consider log-transform. Fat tails (financial).  |
| fed_rate_regime | 0.122 | -1.965 | 0.0% | OK |
| real_rate | -1.456 | 0.864 | 18.8% | OK |
| inventory_zscore | 0.307 | -1.041 | 0.0% | OK |
| production_change_pct | -0.222 | 18.561 | 10.8% | Fat tails (financial).  |
| net_imports_change_pct | 4.903 | 114.934 | 20.1% | Highly skewed — consider log-transform. Fat tails (financial). Many IQR outliers (expected for returns).  |
| conflict_intensity_7d | -0.413 | 0.003 | 0.4% | OK |
| fatalities_7d | 1.031 | 0.578 | 1.8% | OK |
| geopolitical_stress_index | -0.164 | 0.734 | 1.7% | OK |
| oil_return_lag1 | -0.388 | 12.101 | 6.5% | Fat tails (financial).  |
| oil_return_lag2 | -0.389 | 12.042 | 6.5% | Fat tails (financial).  |
| vix_lag1 | 2.354 | 10.66 | 2.4% | Highly skewed — consider log-transform. Fat tails (financial).  |
| gdelt_tone_lag1 | 2.962 | 16.362 | 3.6% | Highly skewed — consider log-transform. Fat tails (financial).  |
| gdelt_volume_lag1 | -0.416 | -0.228 | 0.3% | OK |
| day_of_week | -0.002 | -1.3 | 0.0% | OK |
| month | -0.01 | -1.206 | 0.0% | OK |