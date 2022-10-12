# KMA-wheather_contest

## Final 2nd(기상청장상) | Ensemble DNN + Important Features(SHAP)

+ 주최 및 주관: 기상청 
+ 링크: https://bd.kma.go.kr/contest/main.do

----

## Summary
+ ### Data      
    + 최근 성층권 오존은 감소추세이며, 오존이 감소함에 따라 대기 중의 UV-B 흡수가 감소하고 지표에서의 UV-B량이 증가하여 인간에게 피부암, 백내장, 면역체계 악화를 촉진시킴
    + 기상자료를 활용하여 미래시점에 대한 보다 정확한 자외선 산출 기술의 개발 필요성 증가
    + 천리안위성 2A호의 관측 데이터를 사용하며, 총 16개의 채널이 존재(가시채널 4개, 적외채널 12개)
    + 총 2년 치의 데이터가 존재하며 전처리 시 변수 형태 균일화, 결측값 대체 및 이상치 처리, 범주형 데이터의 0~1 범위 수치화, 관측 시간에 따른 cyclical encoding, min-max 정규화 등을 수행
    + 결측값에 대한 보간 처리 시 여러 방법 중 ‘time’ 방식을 선택, ‘linear’ 또는 ‘poly’ 방식으로 보간 시 데이터의 주기적 특성이 반영되지 않는 문제 발생
    
         ![image](https://user-images.githubusercontent.com/30611947/187855804-76c31c95-3ecb-4c4c-a5ac-a1f542b2ca74.png)
        
    + 주기적 특성이 보존되는 보간을 지점마다 수행하여 15개 모든 지점에 대해서 각각 적용
    
    + 보간 시, ‘uv’ 변수의 값에서 직전 시점과의 차이 값을 계산한 ‘diff’ 변수를 생성 후 일정 값(3 표준편차) 이상으로 급격히 변하는 구간은 nan처리 후 재보간 수행
       
         ![image](https://user-images.githubusercontent.com/30611947/187855338-edd2e8de-c308-458f-b85d-51a0f0d6fbc3.png)

       
    +  165번 지점의 보간을 5회 반복 수행한 경우 값의 변화를 나타내는 그림으로 보간을 한 번 수행했을 때보다 여러번 반복 수행한 경우 ‘UV’, ‘Diff’ 변수의 분포가 완만해지는 것을 확인할 수 있음 
    
    + 보간을 통해 이상치 및 결측치를 처리한 후, 15개의 범주를 가지는 ‘sateza‘, ’height‘ 변수 및 4개의 값을 가지는 ’landtype‘ 변수는 더미화를 수행하였으며, ’stn‘ 변수의 경우 0~1 사이의 값을 가지도록 수치화
    
    + 데이터의 시간적 정보를 반영하기 위해 ‘Date_time’ 변수에 대해 sine 및 cosine 함수를 이용한cyclical encoding을 수행
         
         ![image](https://user-images.githubusercontent.com/30611947/187855556-a5fb2d77-cb60-48cf-8b06-e198ca141365.png)
         
    + cyclical encoding을 통해 모델이 잘 학습할 수 있는 변수로 변환. x는 변수를 의미하며, 시간(hour) 기준 주기는 24시 이므로 max는 24, 일(day) 기준 주기는 31일이며 max는 31을 의미함
    
</br>

+ ### Model     
    + 전처리된 데이터에 시간정보를 반영한 변수를 생성하였기 때문에, 입력데이터가 시간 순으로 들어가야 하는 시계열 학습 방식이 아닌 tabular data로써 다양한 모델 적용 가능하며
tabular data에 우수한 성능을 보이는 머신러닝 및 딥러닝 모델 적용

    + 시도 모델: 머신러닝 모델(LightGBM) 딥러닝 모델(DNN, TabNet, LSTM), 앙상블 모델(Soft voting ensemble) 
    + 후보 모델: LightGBM, DNN, TabNet 
    + 모델 제외 사유: LSTM의 경우 window size 이용하여 데이터를 구성하는데, 처음의 window size 만큼은 예측이 불가
    + 5-fold 진행

</br>

+ ### Train     
    + CosineAnnealingLR - scheduler
    + MSELoss - Loss  
    + AdamW - optimizer
    + amp.GradScaler        
    + EarlyStopping(min_delta: 1e-6) 
    + Baseline 실험을 통해 후보 모델별 성능을 도출하였으며, 실험 결과 DNN 모델이 LightGBM과 TabNet보다 우수한 결과를 보임을 알 수 있고 후처리의 여부가 유의미한 성능 변화를 주지 않는 것을 볼 수 있음
    

      | Model                | RMSE post-processing(O) | RMSE post-processing(X) |
      |----------------------|-------------------------|-------------------------|
      | LightGBM             |         0.629798        |       **0.629760**      |
      | TabNet               |       **0.613479**      |         0.613494        |
      | DNN                  |         0.604392        |       **0.604293**      |



    + 따라서, 후처리를 하지 않은 DNN 모델을 최종 모델로 선정하였으며, 해당 최종 모델로 중요 변수 순위화 및 중요 변수 선택에 따른 성능 변화 등의 추가 실험을 진행
    + SHAP value를 통해 변수 중요도를 구하였으며, 구한 중요변수 중 상위 50개만을 선택하여 학습을 진행함

         ![image](https://user-images.githubusercontent.com/30611947/187861436-740868f4-f564-441b-bbbc-0b0943240f7b.png)

----
## Directory
        .
        ├── WEATHER_CONTEST_INFERENCE_ENSEMBLE.py
        ├── 전처리, 학습, 추론.ipynb
        ├── WEATHER_CONTEST_MAIN.py
        ├── WEATHER_CONTEST_MODEL.py
        ├── WEATHER_CONTEST_UTILS.py
        ├── RESULTS
        └── Data

        2 directories, 5 files
---- 
## Environment 
+ (cuda10.2, cudnn7, ubuntu18.04)
+ 사용한 Docker image는 Docker Hub에 업로드되어 환경을 제공합니다.
  + https://hub.docker.com/r/lsy2026/weather_contest/tags
  
  
## Libraries
+ (cuda10.2, cudnn7, ubuntu18.04) 기준
  + python==3.9.7
  + pandas==1.3.4
  + numpy==1.20.3
  + tqdm==4.62.3
  + sklearn==0.24.2
  + torch==1.10.2+cu102

---- 

### Terminal Command Example for train
```
!python3 WEATHER_CONTEST_MAIN.py \
--model 'DNN' \
--batch_size 2048 \
--hidden_dim 300 \
--depth 3 \
--drop_out 0.120456273696527 \
\
\
--lr 0.0004884905424561053 \
--lr_t 6 \
--lr_scheduler 'CosineAnnealingLR' \
--gamma 0.3885689024409893 \
--patience 30 \
--min_delta 1e-6 \
--weight_decay 0.0003755587070945709 \
\
\
--train_dataset 'TEST_TRAIN_Preprocessed.csv' \
--epochs 300 \
--top_features 50 \
--augmentation True \
--grad_scale False \
--n_fold 5 \
--num_workers 16 \
--device '0,1,2,3' \
--text 'FINAL_top50_AUG'
```

Result: 
```
model_save_name: FINAL_top50_AUG_220803_0010(DNN_2048_300_3_0.120456273696527__0.0004884905424561053_6_CosineAnnealingLR_0.3885689024409893_30_5_50_True_False_0.0003755587070945709)_fold_
Device: cuda
GPU activate --> Count of using GPUs: 4
100%|█████████████████████████████████████████| 617/617 [01:10<00:00,  8.78it/s]
100%|█████████████████████████████████████████| 155/155 [00:17<00:00,  8.93it/s]
FOLD : 1/5    EPOCH : 1/300
TRAIN_Loss : 0.010292   VALID_Loss : 0.001340   BEST_LOSS : 0.001340
==> best model saved 1 epoch / loss: 0.001340  /  difference -99.998660 

...
3Fold - VALID Loss:  0.0008387710148048017
4Fold - VALID Loss:  0.0008319829016052667
5Fold - VALID Loss:  0.0008290791977947999
k-fold Valid Loss:  0.0008324413673741922
```

### Terminal Command Example for inference
```
!python3 WEATHER_CONTEST_INFERENCE_ENSEMBLE.py \
--model 'DNN' \
--model_save_name \
'FINAL_top50_220801_2131(DNN_2048_300_3_0.120456273696527__0.0004884905424561053_6_CosineAnnealingLR_0.3885689024409893_20_5_50_False_False_0.0003755587070945709)_fold_1' \
'FINAL_top50_220801_2131(DNN_2048_300_3_0.120456273696527__0.0004884905424561053_6_CosineAnnealingLR_0.3885689024409893_20_5_50_False_False_0.0003755587070945709)_fold_4' \
'FINAL_top50_220802_0311(DNN_2048_300_3_0.120456273696527__0.0004884905424561053_6_CosineAnnealingLR_0.3885689024409893_30_5_50_False_False_0.0003755587070945709)_fold_1' \
'FINAL_top50_AUG_220803_0010(DNN_2048_300_3_0.120456273696527__0.0004884905424561053_6_CosineAnnealingLR_0.3885689024409893_30_5_50_True_False_0.0003755587070945709)_fold_3' \
\
\
--top_features 50 \
--hidden_dim '300' \
--depth '3' \
--drop_out '0.120456273696527' \
\
\
--test_path 'TEST_TEST_Preprocessed.csv' \
--submit_path '자외선_검증데이터셋.csv' \
--scaler_path 'TEST_weather_min_max.pickle' \
--n_fold 1 \
--text 'FINAL_4_MODEL'
```
----
+ ### Result
    
    + 중요 변수만을 선택하여 학습 시, 상위 50개의 변수를 선택하여 학습한 경우 RMSE는 0.596673이며, 50개의 변수를 이용하여 앙상블 시 RMSE는 0.592461으로 RMSE 성능이 가장 좋은 것을 볼 수 있음
    

      | Model                | RMSE post-processing(O) | RMSE post-processing(X) |
      |----------------------|-------------------------|-------------------------|
      | TOP 30               |         0.609335        |       **0.609270**      |
      | TOP 40               |         0.599597        |       **0.599517**      |
      | TOP 50               |         0.599045        |       **0.596673**      |
      | Ensemble             |         0.592531        |       **0.592461**      |


