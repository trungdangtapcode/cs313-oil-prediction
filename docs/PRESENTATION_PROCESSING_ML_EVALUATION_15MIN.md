# Processing + Machine Learning + Evaluation

Phan nay duoc viet de dung lam script cho khoang `15 phut` thuyet trinh.  
Muc tieu la giu slide gon, de nhin, nhung van du thong tin de tra loi cau hoi ky thuat.

## Tong quan cach dung

- So slide de xuat: `8 slides`
- Tong thoi gian: `14-16 phut`
- Nhip noi:
  - `Processing`: 5-6 phut
  - `Machine Learning`: 4-5 phut
  - `Evaluation + ket luan`: 4-5 phut
- Visual:
  - moi slide toi da `1-2 visual`
  - uu tien `1 chart + 3-4 bullet`
  - tranh nhoi qua nhieu bang so tren cung 1 slide

---

## Slide 1. Muc tieu va dong chay tong the

**Muc tieu slide**
- Dat context cho nguoi nghe.
- Noi ro phan nay tap trung vao 3 cau hoi:
  - du lieu da duoc xu ly nhu the nao
  - da train model ra sao
  - danh gia ket qua nhu the nao

**Noi dung tren slide**
- Bai toan: du bao huong gia dau `T -> T+1`
- Input: du lieu tong hop tu `market + FRED + EIA + GDELT + ACLED`
- Pipeline hien tai:
  - `step4`: feature engineering
  - `step4b`: leakage cleanup
  - `step5`: reduction
  - `step5b`: deterministic processing
  - `step5c`: scaled export cho training
  - `ML step1 -> step7`: baseline den feature selection / tuning

**Visual de xuat**
- Dung mot flow ngang don gian:
  - `Raw / Integrated Data -> Step4 -> Step4b -> Step5 -> Step5b -> Step5c -> ML Step1-7`
- Khong can hinh co san; ve bang SmartArt hoac 6 box ngang la du.

**Script noi**
“Phan nay em trinh bay 3 phan lien nhau: processing, machine learning, va evaluation.  
Muc tieu cua bai toan la du doan huong bien dong gia dau o ngay `T+1`, dua tren thong tin co duoc den cuoi ngay `T`.  
Pipeline hien tai khong chi dung feature engineering ban dau, ma con them cac buoc cleanup va processing de phu hop hon voi forecasting. Sau do em chay tu `step1` den `step7` de so sanh baseline, ensemble, technical features, feature selection va weighted training.”

**Thoi gian goi y**
- `1.0 - 1.5 phut`

---

## Slide 2. Tu EDA baseline den modeling pipeline

**Muc tieu slide**
- Giai quyet xung dot “tai sao khong train thang tren pipeline cu”.
- Framing day la `cai tien`, khong phai `phu dinh`.

**Noi dung tren slide**
- `other_eda_preprocess` la `EDA-first baseline`
  - lam tot feature engineering va reduction ban dau
  - phu hop cho EDA, presentation, baseline train
- Pipeline hien tai la `forecasting/modeling refinement`
  - giu backbone cu
  - bo sung buoc audit leakage
  - doi target sang forward target `oil_return_fwd1`
  - tach ro du lieu dung cho train model

**Visual de xuat**
- Layout 2 cot:
  - trai: `EDA Baseline`
  - phai: `Forecasting Refinement`
- Moi cot 3 bullet ngan.

**Script noi**
“Em khong xem pipeline cu la sai. Phan cua nhom truoc la mot baseline rat tot cho EDA va cho viec tao feature.  
Phan em lam la buoc refine de dua no sang dung cho forecasting nghiem tuc hon. Nghia la em giu lai backbone feature engineering, nhung bo sung them cac buoc cleanup va doi target sang `T -> T+1`.  
Vi vay day la ban nang cap de train model, khong phai mot nhanh xung dot voi cong viec truoc do.”

**Thoi gian goi y**
- `1.5 phut`

---

## Slide 3. Processing pipeline hien tai lam gi

**Muc tieu slide**
- Cho thay tung step co vai tro ro rang.
- Giai thich ngan gon `step4b`, `step5`, `step5b`, `step5c`.

**Noi dung tren slide**
- `Step4`: tao feature tu market, FRED, EIA, GDELT, ACLED
- `Step4b`: loai cac cot co timing / preprocessing contamination
- `Step5`: reduction, bo cot trung gian va cot du thua
- `Step5b`: deterministic processing
  - `log1p`
  - `signed_log1p`
  - `sin/cos` cho calendar
- `Step5c`: them
  - `SimpleImputer`
  - `StandardScaler`
  - `RobustScaler`
  - `PowerTransformer (Yeo-Johnson)`

**Visual de xuat**
- Su dung 4 box dung doc:
  - `Step4b`
  - `Step5`
  - `Step5b`
  - `Step5c`
- O moi box ghi 1 dong “muc dich chinh”.

**Script noi**
“Sau `step4`, em chen them `step4b` de scrub cac feature ma em audit ra la de gay contamination cho forecasting.  
Sau do van giu `step5` reduction, tuc la bo cac cot trung gian va cot du thua.  
`step5b` la lop processing an toan, chi dung nhung transform khong can fit tren full data, vi du `log1p`, `signed_log1p`, va `sin/cos` cho calendar.  
`step5c` la ban tien dung de train nhanh, trong do du lieu da duoc impute va scale san.”

**Thoi gian goi y**
- `1.5 - 2.0 phut`

---

## Slide 4. Vi sao can `step4b`: leakage cleanup

**Muc tieu slide**
- Noi ro ly do ky thuat cho refinement.
- Day la slide quan trong nhat cua phan processing.

**Noi dung tren slide**
- Nhom cot bi loai trong `step4b`:
  - macro/FRED timing risk:
    - `cpi_lag`
    - `unemployment_lag`
    - `real_rate`
    - `fed_rate_change`
    - `fed_rate_regime`
  - preprocessing-based risk:
    - `geopolitical_stress_index`
    - `oil_volatility_7d`
- Tu duy chinh:
  - EDA dataset != final forecasting dataset
  - can tach dataset “de phan tich” va dataset “de train”

**Visual de xuat**
- Neu chi dung 1 visual, uu tien 1 hinh flow “keep / drop”.
- Co the chen them 1 note nho:
  - `other_eda_preprocess = baseline`
  - `current pipeline = refined training dataset`

**Script noi**
“Ly do em khong train truc tiep tren branch EDA cu nam o cho mot so feature van chua du chat cho forecasting.  
Cu the, block macro/FRED van con cac cot nhu `cpi_lag`, `unemployment_lag`, `real_rate`, `fed_rate_change`, `fed_rate_regime`, trong khi timing cua monthly macro tren daily timeline chua phan anh chat release timing.  
Ngoai ra, `geopolitical_stress_index` va `oil_volatility_7d` co preprocessing risk do cach fit transform.  
Nen `step4b` duoc them vao de tach bo du lieu modeling ra khoi bo du lieu EDA.”

**Thoi gian goi y**
- `2.0 phut`

---

## Slide 5. Feature profile sau processing

**Muc tieu slide**
- Cho thay sau khi process, feature nao noi bat.
- Noi ngan gon ve tinh hieu dung duoc cho modeling.

**Noi dung tren slide**
- EDA tren `step5b` cho thay:
  - target train/test kha can bang
  - `10/27` feature co y nghia theo KS
  - `11/27` co y nghia theo Mann-Whitney
  - `7/27` co point-biserial significant
- Top feature theo research score:
  - `day_of_week_sin`
  - `inventory_change_pct`
  - `conflict_event_count_log1p`
  - `fatalities_log1p`
  - `gdelt_volume_lag1_log1p`

**Visual de xuat**
- Dung 1 trong 2 hinh sau:
  - [step5_upgraded_06_signal_scores.png](/home/vund/.svn/eda_classification/step5_upgraded/step5_upgraded_06_signal_scores.png)
  - [step5_upgraded_07_ranking_and_shift.png](/home/vund/.svn/eda_classification/step5_upgraded/step5_upgraded_07_ranking_and_shift.png)

**Script noi**
“Sau khi cleanup va process, em chay lai EDA de kiem tra xem bo du lieu moi con tin hieu gi.  
Ket qua cho thay signal van co, nhung khong phai kieu qua manh hay qua dep. Do nay la dieu tot vi no thuc te hon.  
Nhom feature noi bat nhat sau processing la calendar, supply, conflict va GDELT attention.  
Day cung la co so de em dua sang phan machine learning va feature selection.”

**Thoi gian goi y**
- `1.5 phut`

---

## Slide 6. Machine learning workflow: Step 1 -> Step 7

**Muc tieu slide**
- Cho hoi dong thay quy trinh training co he thong, khong phai chay 1 model don le.

**Noi dung tren slide**
- `Step1`: baseline models
  - Logistic Regression, SVM, RF, GBM, XGB, LGBM, MLP
- `Step2`: ensemble / stacking / voting
- `Step3`: them technical features
- `Step4`: filter-based feature selection
- `Step5`: smart selection
  - cluster + permutation importance
- `Step6`: recency weighting
- `Step7`: focused comparison XGBoost vs GBM

**Visual de xuat**
- Dung 1 timeline ngang 7 nut.
- Moi nut chi ghi `baseline`, `ensemble`, `technical`, `selection`, `smart selection`, `weighting`, `final compare`.

**Script noi**
“Phan machine learning khong dung lai o mot baseline. Em chay mot loat 7 buoc de mo rong tu de den kho.  
`Step1` la baseline de nhin mat bang chung. `Step2` la ensemble. `Step3` them technical features. `Step4` va `Step5` la hai tang feature selection, trong do `Step5` ket hop cluster va permutation importance. `Step6` kiem tra weighting theo tinh gan day cua du lieu, va `Step7` la so gang cuoi giua XGBoost va GBM.  
Voi `step5c`, du lieu da duoc scale san nen khong scale lai trong train pipeline.”

**Thoi gian goi y**
- `1.5 phut`

---

## Slide 7. Evaluation setup

**Muc tieu slide**
- Giai thich ro split va cach doc metric.
- Day la slide de tranh bi hoi “co dung test de tune khong”.

**Noi dung tren slide**
- Dataset train:
  - target date `2015-01-08 -> 2022-12-30`
- Dataset test:
  - target date `2023-01-02 -> 2026-03-20`
- `SearchCV` chi chay trong train
- Metric chinh:
  - `Accuracy`
  - `F1_macro`
  - `AUC`
- Luu y:
  - theo workflow hien tai, model duoc so sanh bang test metrics
  - vi vay test o day la `comparative evaluation set`

**Visual de xuat**
- Dung 1 timeline thoi gian:
  - Train ben trai
  - Test ben phai
- Neu can hinh co san:
  - [step5_upgraded_03_target_over_time.png](/home/vund/.svn/eda_classification/step5_upgraded/step5_upgraded_03_target_over_time.png)

**Script noi**
“Setup danh gia hien tai rat ro rang: train den cuoi 2022, test tu dau 2023 den 2026.  
Tat ca SearchCV van chi nam trong tap train. Sau do model duoc fit lai va danh gia tren tap test.  
Trong workflow hien tai, chung em dung test metrics de so sanh model, nen cach goi chinh xac hon la evaluation set so sanh mo hinh. Em noi ro diem nay de minh minh bach ve cach doc ket qua.”

**Thoi gian goi y**
- `1.5 phut`

---

## Slide 8. Ket qua va model duoc de xuat

**Muc tieu slide**
- Chot ket qua tot nhat va thong diep cuoi.
- Day la slide tong hop quan trong nhat cua phan ML + evaluation.

**Noi dung tren slide**
- Ket qua noi bat:
  - Best `Accuracy`: `Step1 / LightGBM = 0.5405`
  - Best `F1_macro`: `Step1 / LogisticRegression = 0.5166`
  - Best `AUC`: `Step3 / 1d_raw XGB = 0.5623`
  - Best trong nhom `Step4 -> Step7`: `Step5 / XGB`
    - `Accuracy = 0.5333`
    - `F1_macro = 0.5029`
    - `AUC = 0.5446`
- Feature set cua `Step5 / XGB` bao gom:
  - market + technical
  - supply
  - GDELT
  - conflict
  - calendar
- Thong diep chot:
  - baseline manh nhat: `Step1 LightGBM`
  - mo hinh refined can bang nhat: `Step5 XGB`

**Visual de xuat**
- Visual 1:
  - [step6_comparison.png](/home/vund/.svn/ml/classification/results_step5c/step6_comparison.png)
- Visual 2 nho o goc:
  - [step1_roc.png](/home/vund/.svn/ml/classification/results_step5c/step1_roc.png)

**Script noi**
“Neu chi nhin ket qua tot nhat theo tung metric, thi `Step1 / LightGBM` cho accuracy cao nhat, `Step1 / LogisticRegression` cho F1 macro cao nhat, va `Step3 / XGB` cho AUC cao nhat.  
Tuy nhien, neu em uu tien mot mo hinh da qua feature selection va van giu duoc ket qua on, thi `Step5 / XGB` la lua chon can bang nhat trong nhom refinement steps.  
Feature set duoc chon o buoc nay cung hop ly ve mat domain, vi no ket hop market, technical, supply, GDELT, conflict va calendar, thay vi phu thuoc qua manh vao mot nhom duy nhat.”

**Thoi gian goi y**
- `2.0 phut`

---

## Slide 9. Ket luan va thong diep de bao ve phan processing + ML

**Muc tieu slide**
- Chot narrative de bao ve phan viec cua minh.
- Rat huu ich neu co tranh luan ve “vi sao khong dung thang pipeline cu”.

**Noi dung tren slide**
- Processing hien tai la `cai tien` tren cung backbone
- `step4b` giup tach EDA dataset va training dataset
- `step5b` la ban an toan hon de evaluate
- `step5c` la ban tien dung de train nhanh
- ML da duoc verify end-to-end tren `step1 -> step7`
- Ket qua cuoi cung:
  - pipeline refined hop ly hon cho forecasting `T -> T+1`

**Visual de xuat**
- 1 slide text-only rat sach, them 1 dong takeaway dam mau:
  - `EDA baseline -> Forecasting-safe refinement -> Verified ML pipeline`

**Script noi**
“Thong diep cuoi em muon chot la: em khong thay the cong viec cu, ma em refine no de phu hop hon voi forecasting.  
`other_eda_preprocess` la EDA baseline tot. Pipeline hien tai giu lai backbone do, nhung tach bo du lieu train model ra ro rang hon bang `step4b`, `step5b`, va `step5c`.  
Sau do em da verify bang mot quy trinh ML day du tu `step1` den `step7`.  
Vi vay, neu muc tieu la trinh bay EDA, branch cu van co gia tri. Con neu muc tieu la train va danh gia model du bao `T+1`, thi nen dung pipeline refined hien tai.”

**Thoi gian goi y**
- `1.0 - 1.5 phut`

---

## Appendix nho de tra loi luc hoi dap

### Neu bi hoi: “Tai sao khong train truc tiep tren `other_eda_preprocess`?”

Tra loi ngan:

“Vi branch do phu hop hon cho EDA baseline. De train forecasting nghiem tuc hon, em bo sung `step4b` de loai cac feature macro va preprocessing co contamination risk, sau do process tiep bang `step5b` va `step5c`. Em xem day la ban refine, khong phai xung dot voi pipeline cu.”

### Neu bi hoi: “Step5b va Step5c khac nhau gi?”

Tra loi ngan:

- `step5b`: deterministic processing, sach hon cho evaluation
- `step5c`: `step5b` + impute + scale san, tien cho training nhanh

### Neu bi hoi: “Ket qua nao la mo hinh de xuat?”

Tra loi ngan:

- neu uu tien `Accuracy`: `Step1 / LightGBM`
- neu uu tien `F1_macro` va da qua refinement steps: `Step5 / XGB`
- neu uu tien `AUC`: `Step3 / XGB`

---

## File hinh nen uu tien dung

EDA:
- [step5_upgraded_06_signal_scores.png](/home/vund/.svn/eda_classification/step5_upgraded/step5_upgraded_06_signal_scores.png)
- [step5_upgraded_07_ranking_and_shift.png](/home/vund/.svn/eda_classification/step5_upgraded/step5_upgraded_07_ranking_and_shift.png)
- [step5_upgraded_03_target_over_time.png](/home/vund/.svn/eda_classification/step5_upgraded/step5_upgraded_03_target_over_time.png)

ML / Evaluation:
- [step1_roc.png](/home/vund/.svn/ml/classification/results_step5c/step1_roc.png)
- [step1_confusion.png](/home/vund/.svn/ml/classification/results_step5c/step1_confusion.png)
- [step6_comparison.png](/home/vund/.svn/ml/classification/results_step5c/step6_comparison.png)
- [step6_weight_schemes.png](/home/vund/.svn/ml/classification/results_step5c/step6_weight_schemes.png)

Bao cao tong hop:
- [results_step5c/REPORT.md](/home/vund/.svn/ml/classification/results_step5c/REPORT.md)
- [step5_upgraded_eda_summary.md](/home/vund/.svn/eda_classification/step5_upgraded/reports/step5_upgraded_eda_summary.md)
