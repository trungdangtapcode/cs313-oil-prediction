# EDA Summary

## Run
- run label: step5_upgraded
- dataset: `/home/vund/.svn/data/processed/dataset_final_noleak_processed.csv`
- split column: `oil_return_fwd1_date`
- train rows: 2082
- test rows: 840

## Class Balance
- train: UP=51.3%, DOWN=48.7%
- test: UP=49.8%, DOWN=50.2%

## Signal Summary
- KS significant features: 10/27
- Mann-Whitney significant features: 11/27
- Point-biserial significant features: 7/27
- Cohen's d > 0.2: 0/27

## Time-Series / Collinearity
- stationary series in ADF sample: 0/6
- VIF > 10: 3
- VIF 5-10: 6

## Leakage Risk
- low risk features: 23
- medium risk features: 4
- high risk features: 0

## Top 10 Features By Research Score
                    feature    group  signal_score  stability_score  research_score   risk
            day_of_week_sin Calendar      0.885886         0.999615        0.920005    low
       inventory_change_pct   Supply      0.572458         0.695673        0.609423    low
 conflict_event_count_log1p Conflict      0.694442         0.342445        0.588843    low
           fatalities_log1p Conflict      0.563132         0.648695        0.588801    low
    gdelt_volume_lag1_log1p    GDELT      0.533999         0.664876        0.573262    low
           inventory_zscore   Supply      0.564377         0.527953        0.553450    low
                 oil_return   Market      0.390333         0.905577        0.544906 medium
            oil_return_lag2   Market      0.361212         0.903654        0.523944    low
                  month_sin Calendar      0.341725         0.941319        0.521603    low
conflict_intensity_7d_log1p Conflict      0.675657         0.156497        0.519909    low

## Recommended Keep
                    feature    group  research_score  train_test_ks
            day_of_week_sin Calendar        0.920005       0.000192
       inventory_change_pct   Supply        0.609423       0.152017
 conflict_event_count_log1p Conflict        0.588843       0.328462
           fatalities_log1p Conflict        0.588801       0.175484
    gdelt_volume_lag1_log1p    GDELT        0.573262       0.167401
           inventory_zscore   Supply        0.553450       0.235797
            oil_return_lag2   Market        0.523944       0.048127
                  month_sin Calendar        0.521603       0.029312
conflict_intensity_7d_log1p Conflict        0.519909       0.421346
        fatalities_7d_log1p Conflict        0.489552       0.227378

## Use With Caution
                    feature    group  research_score  train_test_ks   risk
 conflict_event_count_log1p Conflict        0.588843       0.328462    low
           fatalities_log1p Conflict        0.588801       0.175484    low
    gdelt_volume_lag1_log1p    GDELT        0.573262       0.167401    low
           inventory_zscore   Supply        0.553450       0.235797    low
                 oil_return   Market        0.544906       0.047166 medium
conflict_intensity_7d_log1p Conflict        0.519909       0.421346    low
        fatalities_7d_log1p Conflict        0.489552       0.227378    low
            gdelt_goldstein    GDELT        0.482760       0.299451    low
          vix_return_slog1p   Market        0.456385       0.044518 medium
         gdelt_goldstein_7d    GDELT        0.449563       0.330112    low

## High-Risk Features
None

## Notes
- `signal_score` measures class separability on the train period.
- `stability_score` penalizes features that shift strongly from train to test.
- `research_score = 0.7 * signal_score + 0.3 * stability_score`.
- This EDA is forward-target aware: the split uses `oil_return_fwd1_date` when available.
- Same-day market returns are marked medium risk because they are only valid for end-of-day T -> T+1 forecasting.