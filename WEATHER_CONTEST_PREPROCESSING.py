import os 
import pickle
import argparse
import numpy as np
import pandas as pd 
from tqdm import tqdm 
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from WEATHER_CONTEST_UTILS import (load_data, trans_func, ranges, dummy_and_add_feature)

import warnings
warnings.filterwarnings(action='ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch Preprocessing', add_help=False)

    # Model parameters
    parser.add_argument('--folder_path', default='./Data/', type=str)
    parser.add_argument('--mode', default='TRAIN', type=str)
    parser.add_argument('--iter_len', default=3, type=int)
    parser.add_argument('--save_path', default='./', type=str)
    parser.add_argument('--text', default='Hi', type=str)

    return parser


def main(args):
        
    ####################
    ### 파라미터 설정 ###
    ####################

    folder_path = args.folder_path
    mode = args.mode
    iter_len = args.iter_len
    save_path = args.save_path
    text = args.text
    scaler_path = save_path + "{}_weather_min_max.pickle".format(text)
    
    
    bin_df = load_data(folder_path, mode)
    format_ = '%Y%m%d%H%M'
    
    if mode == 'TRAIN':    
            
        # 두 개의 시간 변수를 활용하여 datetime 'Date_Time' 변수 생성
        bin_df['Date_Time'] = bin_df['yyyymmdd'].astype(str) + bin_df['hhnn'].apply(trans_func)
        bin_df['Date_Time'] = bin_df['Date_Time'].apply(lambda x: datetime.strftime(datetime.strptime(x, format_),'%Y-%m-%d %H:%M'))

        bin_df.drop(['yyyymmdd', 'hhnn'], axis=1, inplace=True)

        # 'stn', 'Date_Time' 기준으로 정렬
        bin_df = bin_df.sort_values(["stn", "Date_Time"]).reset_index(drop=True)

        # 'Date_Time' 변수를 datetime 변수로 지정 후 dataframe의 index로 설정
        bin_df['Date_Time'] = pd.to_datetime(bin_df['Date_Time'])
        bin_df.set_index(keys=['Date_Time'], inplace=True)

        # 결측값을 nan으로 변환 후 interpolate method의 'time'을 기준으로 보간 수행
        bin_df.replace(-999.0, np.nan, inplace=True)
        bin_df = bin_df.interpolate(method='time')

        ###########################
        ### stn별 uv 전처리 수행 ###
        ###########################

        final_df = bin_df
        del bin_df

        # 'time' 기준 보간을 수행할 때 값이 튀는 경우가 존재함 -> 튀는 값에 대해 반복 보간 수행
        for i in range(iter_len):
            stn_bin_df = pd.DataFrame()

            # stn을 나누지 않으면 stn이 붙어있는 경우 보간이 겹침 -> stn별로 나눠서 보간을 수행
            for stn_num in final_df.stn.unique():
                
                stn_df = final_df[final_df.stn == stn_num]
                
                # 한 스텝 이동할 때 uv의 차이(양수영역 및 음수영역)가 3std diff 이상인 경우 해당 영역 선택
                region_positive = ranges(list(np.where(stn_df['uv'].diff(periods=1) >= (stn_df['uv'].diff(periods=1).std()*3))[0]))
                region_negative = ranges(list(np.where(abs(stn_df['uv'].diff(periods=1)) >= (abs(stn_df['uv'].diff(periods=1).std()*3)))[0]))
                
                # 3std diff 이상인 영역 리스트화
                stn_nan_list_positive = list(set(list(sum(region_positive,()))))
                stn_nan_list_negative = list(set(list(sum(region_negative,()))))

                # 해당 영역 보간 수행 양수 음수 영역으로 나눠서 하지 않을 경우 보간 값이 뭉쳐짐 -> 양수 보간 후 음수 보간 수행
                stn_df['uv'].iloc[stn_nan_list_positive] = np.nan
                stn_df = stn_df.interpolate(limit_direction ='both', method='time')
                
                stn_df['uv'].iloc[stn_nan_list_negative] = np.nan
                stn_df = stn_df.interpolate(limit_direction ='both', method='time')
                
                # stn에 대한 보간이 수행된 후 stn_bin_df에 concat하여 아래로 합침
                stn_bin_df = pd.concat([stn_bin_df, stn_df], axis=0)
                
            # 최종적으로 다 합쳐진 stn_bin_df에 있을지 모르는 nan에 대한 보간 수행
            final_df = stn_bin_df.interpolate(limit_direction ='both', method='time')
        
    else:        
        # 두 개의 시간 변수를 활용하여 datetime 'Date_Time' 변수 생성
        bin_df = bin_df.reset_index()
        bin_df['Date_Time'] = bin_df['YearMonthDayHourMinute'].apply(lambda x: datetime.strftime(datetime.strptime(str(x), format_),'%Y-%m-%d %H:%M'))
        bin_df.drop(['YearMonthDayHourMinute'], axis=1, inplace=True)

        # 'stn', 'Date_Time' 기준으로 정렬
        bin_df = bin_df.sort_values(["stn", "Date_Time"]).reset_index(drop=True)

        # 'Date_Time' 변수를 datetime 변수로 지정 후 dataframe의 index로 설정
        bin_df['Date_Time'] = pd.to_datetime(bin_df['Date_Time'])
        bin_df.set_index(keys=['Date_Time'], inplace=True)

        # 결측값을 nan으로 변환 후 interpolate method의 'time'을 기준으로 보간 수행
        bin_df.replace(-999.0, np.nan, inplace=True)
        bin_df = bin_df.interpolate(method='time')

        final_df = bin_df
        del bin_df

    
    # min-max scaling을 하지 않을 변수 선택
    not_scale_col = ['stn', 'lat', 'lon', 'landtype', 'height', 'sateza']
    scale_col = [i for i in list(final_df.columns) if i not in not_scale_col]

    bin_df = pd.DataFrame()
    
    if mode == 'TRAIN':
        # stn마다 변수별 최대, 최소값 구하기
        max_arr = final_df[scale_col].max().values
        min_arr = final_df[scale_col].min().values
            
        # 사전에 변수별 최대, 최소값 저장
        min_max_dict = {scale_col[i]:[min_arr[i], max_arr[i]] for i in range(len(scale_col))}
        
        # min-max scaling
        for col, (col_min, col_max) in min_max_dict.items():
            final_df[col] = final_df[col] - col_min
            final_df[col] = final_df[col] / (col_max-col_min)

        final_df['lat'] = final_df['lat']/90
        final_df['lon'] = final_df['lon']/360
        
    else:
        with open(scaler_path, 'rb') as fr:
            min_max_dict = pickle.load(fr)
    
        # min-max scaling
        for col, (col_min, col_max) in min_max_dict.items():
            final_df[col] = final_df[col] - col_min
            final_df[col] = final_df[col] / (col_max-col_min)

        final_df['lat'] = final_df['lat']/90
        final_df['lon'] = final_df['lon']/360
        
    bin_df = final_df
    del final_df
        
    # station encoding 
    station_encoder = LabelEncoder()
    station_encoder.fit(bin_df['stn'].unique())

    # station이 15개 일 때, 0~14 이므로 0~1사이로 embedding 하기위해 len-1을 나눔 
    bin_df['stn'] = station_encoder.transform(bin_df['stn']) / (len(bin_df['stn'].unique())-1)
    
    # 'landtype', 'sateza', 'height' 변수 더미화
    landtype_dummy = pd.get_dummies(bin_df['landtype'], prefix = 'landtype')
    sateza_dummy = pd.get_dummies(bin_df['sateza'], prefix = 'sateza')
    height_dummy = pd.get_dummies(bin_df['height'], prefix = 'height')

    bin_df = pd.concat([bin_df, landtype_dummy, sateza_dummy, height_dummy], axis=1)
    bin_df.drop(['landtype', 'sateza', 'height'], axis=1, inplace=True)

    # 'Date_Time' 변수를 활용하여 cyclical embedding을 통한 'sin_minute', 'cos_minute', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month' 변수 생성
    bin_df.reset_index(inplace=True)
    bin_df[['sin_minute', 'cos_minute', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']] = bin_df['Date_Time'].astype(str).apply(dummy_and_add_feature).tolist()
    bin_df.set_index(keys=['Date_Time'], inplace=True)

    # Prepocessed data save 
    if mode == 'TRAIN':
        with open(scaler_path, 'wb') as fw:
            pickle.dump(min_max_dict, fw)
        print("scaler is saved at {}".format(scaler_path))

    bin_df.to_csv(save_path+"{}_{}_Preprocessed.csv".format(text, mode), index=True)
    print(save_path+"{}_{}_Preprocessed.csv is saved!".format(text, mode))


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Preprocessing script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)