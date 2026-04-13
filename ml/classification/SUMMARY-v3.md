# Classification Pipeline Summary v3

Snapshot time: `2026-04-13 23:31:28 +07`

## 1. Những gì đã sửa trước khi rerun

- Đổi bài toán sang `end-of-day T -> predict oil_return_fwd1 của T+1`.
- `ml/config.py` dùng `TARGET=oil_return_fwd1` và split theo `oil_return_fwd1_date`.
- Các step classification đã chuyển sang quy trình `train -> validation -> final holdout test`, không còn chọn model/feature/threshold trực tiếp trên `test`.
- Thêm `set_global_seed()` và gọi ở đầu các step để reproducible với `Seed: 42`.
- Chuẩn hóa tên output theo đúng số bước (`step1_*`, `step4_*`, `step6_*`, `step7_*`...).
- `step2` đã vá stacking để không dùng `cross_val_predict` sai với `TimeSeriesSplit`.
- `step5` đã vá lỗi correlation clustering bị `NaN` trong distance matrix.
- `step4` đã đổi sang so sánh `28` case:
  - `Spearman` riêng
  - `MI` riêng
  - `MI + Spearman`
  - các subset `TOP_10/15/20/25/30/40/50/60/70`
  - cộng thêm `ALL`

## 2. Các step đã rerun xong và artifact hiện đáng dùng

### Step 1

File chính:
- `step1_validation_results.csv`
- `step1_holdout_result.csv`

Kết quả:
- Model được chọn trên validation: `GradientBoosting`
- Holdout:
  - `Accuracy = 0.5250`
  - `F1_macro = 0.5019`
  - `AUC = 0.5449`

### Step 3

File chính:
- `step3_validation_results.csv`
- `step3_results.csv`

Các target/model tốt nhất đã lưu:
- `1d_t05 / XGB`
  - `Val_Accuracy = 0.5814`
  - `Val_F1_macro = 0.5640`
  - `Holdout_Accuracy = 0.5041`
  - `Holdout_F1_macro = 0.4811`
  - `Holdout_AUC = 0.5228`
- `1d_raw / XGB`
  - `Val_Accuracy = 0.5769`
  - `Val_F1_macro = 0.5749`
  - `Holdout_Accuracy = 0.5214`
  - `Holdout_F1_macro = 0.4869`
  - `Holdout_AUC = 0.5258`
- `1d_t03 / GBM`
  - `Val_Accuracy = 0.5507`
  - `Holdout_Accuracy = 0.5349`

### Step 4

File chính:
- `step4_feature_ranking.csv`
- `step4_subset_comparison.csv`
- `step4_best_by_ranking.csv`
- `step4_results.csv`

Best case trong 28 case:
- `MI_TOP_60`
- `Val_Accuracy = 0.5769`
- `Val_F1_macro = 0.5607`

Best theo từng family:
- `Spearman`: `SPEARMAN_TOP_70`, `Val_Accuracy = 0.5577`
- `MI`: `MI_TOP_60`, `Val_Accuracy = 0.5769`
- `MI + Spearman`: `MI_SPEARMAN_TOP_70`, `Val_Accuracy = 0.5577`
- `ALL_82`: `Val_Accuracy = 0.5538`

Final model train trên case thắng:
- `GBM`
  - `Holdout_Accuracy = 0.5179`
  - `Holdout_F1_macro = 0.4860`
  - `Holdout_AUC = 0.5100`

### Step 6

File chính:
- `step6_validation_results.csv`
- `step6_results.csv`

Best overall:
- `LGBM + step_50pct_3x`
  - `Val_Accuracy = 0.6000`
  - `Val_F1_macro = 0.5981`
  - `Holdout_Accuracy = 0.5286`
  - `Holdout_F1_macro = 0.4896`
  - `Holdout_AUC = 0.5251`

Các best-per-model:
- `LGBM + step_50pct_3x`
- `XGB + linear_01`
- `GBM + step_50pct_3x`

### Step 7

File chính:
- `step7_validation_results.csv`
- `step7_results.csv`

Winner trên validation:
- `GBM`

Holdout:
- `Accuracy = 0.5119`
- `F1_macro = 0.4677`
- `AUC = 0.5282`

## 3. Step đang chạy tại thời điểm snapshot

### Step 2

Trạng thái:
- Đang chạy nền
- PID: `1524568`
- Log: `logs/step2_rerun_20260413_232951.log`

Cấu hình:
- `Seed: 42`
- `SEARCH_N_JOBS = 8`
- `MODEL_N_JOBS = 12`

Tiến độ hiện tại:
- Đã load data và chọn `TOP_25`
- Đang ở block `GBM_v2`
- Chưa có artifact mới hoàn chỉnh tại thời điểm snapshot

Ghi chú:
- `step2` là bước nặng do có `RandomizedSearchCV(n_iter=30)` cho nhiều model cộng thêm ensemble
- Stacking đã được vá sang `TimeSeriesStackingClassifier`

### Step 5

Trạng thái:
- Đang chạy nền
- PID: `1524572`
- Log: `logs/step5_rerun_20260413_232951.log`

Cấu hình:
- `Seed: 42`
- `SEARCH_N_JOBS = 8`
- `MODEL_N_JOBS = 12`

Tiến độ hiện tại:
- Correlation clustering: xong
- Permutation importance: xong
- Compare feature sets: xong
- Đang ở `FINAL TRAIN ON ALL_82`

Interim validation comparison đã in ra log:

| Set | N | LGBM_Val | XGB_Val | GBM_Val |
|---|---:|---:|---:|---:|
| ALL_82 | 82 | 0.5423 | 0.5654 | 0.5923 |
| CLUSTER_43 | 43 | 0.5692 | 0.5923 | 0.5308 |
| CLUSTER_POS_21 | 21 | 0.5500 | 0.5462 | 0.5000 |
| PERM_TOP_10 | 10 | 0.4962 | 0.5115 | 0.5077 |
| PERM_TOP_15 | 15 | 0.5346 | 0.5308 | 0.5038 |
| PERM_TOP_20 | 20 | 0.5615 | 0.5385 | 0.4962 |
| PERM_TOP_25 | 25 | 0.5308 | 0.5423 | 0.5308 |

Lưu ý:
- Theo logic hiện tại, `ALL_82` đang là feature set được chọn vào final training vì là dòng đầu tiên đạt `best_acc = 0.5923`.
- `step5_results.csv` đang có trong thư mục nhưng chưa chắc là artifact mới của run hiện tại; chỉ coi log `step5_rerun_20260413_232951.log` là nguồn trạng thái đúng cho tới khi run xong.

## 4. Trạng thái file và script hỗ trợ

- Script chạy song song đã có: `ml/classification/run_parallel_steps.sh`
- Log hiện tại của rerun:
  - `ml/classification/results/logs/step2_rerun_20260413_232951.log`
  - `ml/classification/results/logs/step5_rerun_20260413_232951.log`

## 5. Nhận xét ngắn ở thời điểm hiện tại

- Trong các step đã xong, kết quả holdout tốt nhất hiện tại là:
  - `step6`: `Accuracy = 0.5286`
  - `step1`: `AUC = 0.5449`
- `step4` cho thấy ranking `MI` đang mạnh hơn `Spearman` và `MI + Spearman` trong bộ 28 case.
- `step5` hiện cho tín hiệu rằng `clustering + permutation` chưa chắc thắng `ALL_82`, vì tại validation snapshot hiện tại `ALL_82` và `CLUSTER_43` đang ngang đỉnh ở mức `0.5923` nhưng do tie-break theo thứ tự, `ALL_82` được chọn để final-train.
- `step2` và `step5` vẫn cần hoàn tất để có báo cáo cuối cùng đầy đủ cho toàn bộ `step1 -> step7`.
