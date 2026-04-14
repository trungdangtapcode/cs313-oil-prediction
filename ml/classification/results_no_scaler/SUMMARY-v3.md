# Classification Pipeline Summary v3

Updated time: `2026-04-13 23:56:14 +07`

## 1. Các chỉnh sửa đã áp dụng trước khi rerun

- Chuyển bài toán sang `end-of-day T -> predict oil_return_fwd1 của T+1`.
- `ml/config.py` dùng `TARGET=oil_return_fwd1` và split theo `oil_return_fwd1_date`.
- Các step classification đã chuyển sang quy trình `train -> validation -> holdout test`.
- Thêm `set_global_seed()` và gọi ở đầu các step để reproducible với `Seed: 42`.
- Chuẩn hóa tên output theo đúng step.
- `step2` đã vá stacking sang `TimeSeriesStackingClassifier` để tránh lỗi với `TimeSeriesSplit`.
- `step5` đã vá lỗi `NaN` trong correlation clustering và thêm cấu hình chạy nhanh hơn qua `SEARCH_N_JOBS` / `MODEL_N_JOBS`.
- `step4` đã mở rộng so sánh thành `28` case:
  - `Spearman`
  - `MI`
  - `MI + Spearman`
  - các subset `TOP_10/15/20/25/30/40/50/60/70`
  - cộng thêm `ALL`

## 2. Kết quả rerun hoàn chỉnh theo step

### Step 1

File:
- `step1_validation_results.csv`
- `step1_holdout_result.csv`

Kết quả:
- Chọn trên validation: `GradientBoosting`
- Holdout:
  - `Accuracy = 0.5250`
  - `F1_macro = 0.5019`
  - `AUC = 0.5449`

### Step 2

File:
- `step2_validation_results.csv`
- `step2_holdout_result.csv`

Log:
- `results/logs/step2_rerun_20260413_232951.log`

Validation leaderboard:
- `Voting`: `Acc = 0.5538`, `F1_macro = 0.5522`, `AUC = 0.5655`
- `GBM_v2`: `Acc = 0.5577`, `F1_macro = 0.5485`, `AUC = 0.5688`
- `LGBM_v2`: `Acc = 0.5423`, `F1_macro = 0.5422`, `AUC = 0.5743`
- `XGB_v2`: `Acc = 0.5269`, `F1_macro = 0.4814`, `AUC = 0.5719`
- `Stacking`: `Acc = 0.5192`, `F1_macro = 0.3747`, `AUC = 0.5960`
- `SVM_RBF_v2`: `Acc = 0.5077`, `F1_macro = 0.3566`, `AUC = 0.5374`

Ứng viên được chọn:
- `Voting`

Holdout:
- `Accuracy = 0.5214`
- `F1_macro = 0.4914`
- `AUC = 0.5344`

Ghi chú:
- `Voting` thắng theo tiêu chí sort hiện tại là `F1_macro` trước `Accuracy`.
- `GBM_v2` có `Accuracy` validation nhỉnh hơn một chút, nhưng `Voting` có `F1_macro` cao hơn.

### Step 3

File:
- `step3_validation_results.csv`
- `step3_results.csv`

Các target/model tốt nhất:
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

File:
- `step4_feature_ranking.csv`
- `step4_subset_comparison.csv`
- `step4_best_by_ranking.csv`
- `step4_results.csv`

Best case trong `28` case:
- `MI_TOP_60`
- `Val_Accuracy = 0.5769`
- `Val_F1_macro = 0.5607`

Best theo từng family:
- `Spearman`: `SPEARMAN_TOP_70`, `Val_Accuracy = 0.5577`
- `MI`: `MI_TOP_60`, `Val_Accuracy = 0.5769`
- `MI + Spearman`: `MI_SPEARMAN_TOP_70`, `Val_Accuracy = 0.5577`
- `ALL_82`: `Val_Accuracy = 0.5538`

Final model trên case thắng:
- `GBM`

Holdout:
- `Accuracy = 0.5179`
- `F1_macro = 0.4860`
- `AUC = 0.5100`

### Step 5

File:
- `step5_selected_features.csv`
- `step5_perm_importance.csv`
- `step5_set_comparison.csv`
- `step5_results.csv`

Log:
- `results/logs/step5_rerun_20260413_232951.log`

So sánh feature set:
- `ALL_82`: best validation `0.5923`
- `CLUSTER_43`: best validation `0.5923`
- `CLUSTER_POS_21`: best validation `0.5500`
- `PERM_TOP_20`: best validation `0.5615`

Kết quả cuối:
- Feature set được chọn để final train: `ALL_82`
- Lý do: `ALL_82` và `CLUSTER_43` hòa `best_acc = 0.5923`, và logic hiện tại chọn dòng xuất hiện trước

Final model:
- `GBM`
  - `Val_Accuracy = 0.5500`
  - `Val_F1_macro = 0.5417`
  - `Val_AUC = 0.5719`
  - `Holdout_Accuracy = 0.5429`
  - `Holdout_F1_macro = 0.5391`
  - `Holdout_AUC = 0.5502`

Model còn lại:
- `XGB`
  - `Val_Accuracy = 0.5385`
  - `Val_F1_macro = 0.5304`
  - `Val_AUC = 0.5987`
- `LGBM`
  - `Val_Accuracy = 0.5154`
  - `Val_F1_macro = 0.4775`
  - `Val_AUC = 0.5359`

Ghi chú quan trọng:
- Step này tên là “smart feature selection”, nhưng ở run hiện tại feature set thắng cuối cùng lại là `ALL_82`.
- Nghĩa là clustering + permutation importance không tạo ra reduced set tốt hơn rõ ràng trên validation/holdout ở cấu hình hiện tại.

### Step 6

File:
- `step6_validation_results.csv`
- `step6_results.csv`

Best overall:
- `LGBM + step_50pct_3x`
  - `Val_Accuracy = 0.6000`
  - `Val_F1_macro = 0.5981`
  - `Holdout_Accuracy = 0.5286`
  - `Holdout_F1_macro = 0.4896`
  - `Holdout_AUC = 0.5251`

### Step 7

File:
- `step7_validation_results.csv`
- `step7_results.csv`

Winner trên validation:
- `GBM`

Holdout:
- `Accuracy = 0.5119`
- `F1_macro = 0.4677`
- `AUC = 0.5282`

## 3. Tóm tắt so sánh nhanh

Nếu chỉ so các step cùng bài toán chuẩn `oil_return_fwd1 > 0` và cùng kiểu holdout cuối:

- `Step 5` hiện là kết quả mạnh nhất trên holdout:
  - `Accuracy = 0.5429`
  - `F1_macro = 0.5391`
  - `AUC = 0.5502`
- `Step 1` đứng tốt ở baseline:
  - `Accuracy = 0.5250`
  - `AUC = 0.5449`
- `Step 6` có validation rất cao nhưng holdout không vượt `Step 5`:
  - `Accuracy = 0.5286`
  - `AUC = 0.5251`
- `Step 2` cải thiện validation nhưng holdout chưa vượt baseline mạnh:
  - `Accuracy = 0.5214`
  - `AUC = 0.5344`
- `Step 4` thấp hơn `Step 5`, cho thấy MI ranking đơn giản có ích nhưng chưa thắng cách giữ full set:
  - `Accuracy = 0.5179`
  - `AUC = 0.5100`
- `Step 7` tuning rộng hơn cho tree model không vượt `Step 5`:
  - `Accuracy = 0.5119`
  - `AUC = 0.5282`

## 4. Kết luận 

- Trong batch kết quả này, ứng viên tốt nhất đang là:
  - `Step 5 / GBM / ALL_82`
  - `Holdout_Accuracy = 0.5429`
  - `Holdout_F1_macro = 0.5391`
  - `Holdout_AUC = 0.5502`
- Điểm đáng chú ý là:
  - `MI` là ranking mạnh nhất trong `step4`
  - `step2` ensemble không thắng rõ holdout
  - `step5` cho thấy smart selection không thắng reduced-set; full set `ALL_82` lại tốt nhất
  - `step6` cải thiện validation mạnh nhưng chưa chuyển hóa thành holdout tốt hơn `step5`

## 5. Artifact chính để xem tiếp

- `results/step1_holdout_result.csv`
- `results/step2_validation_results.csv`
- `results/step2_holdout_result.csv`
- `results/step3_results.csv`
- `results/step4_best_by_ranking.csv`
- `results/step4_results.csv`
- `results/step5_set_comparison.csv`
- `results/step5_results.csv`
- `results/step6_results.csv`
- `results/step7_results.csv`
- `results/logs/step2_rerun_20260413_232951.log`
- `results/logs/step5_rerun_20260413_232951.log`
