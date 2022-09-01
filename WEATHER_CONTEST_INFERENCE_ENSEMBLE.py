import os 
import copy 
import pickle
import random
import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from WEATHER_CONTEST_UTILS import str2bool, Custom_Dataset
from WEATHER_CONTEST_MODEL import Network

import warnings
warnings.filterwarnings(action='ignore')



def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch Inference', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='DNN', type=str)
    parser.add_argument('--model_save_name', nargs='+', default='load_models', type=str)
    parser.add_argument('--scaler_path', default='TEST_weather_min_max.pickle', type=str)
    
    parser.add_argument('--hidden_dim', nargs='+', type=str)
    parser.add_argument('--depth', nargs='+', type=str)
    parser.add_argument('--drop_out', nargs='+', type=str)
    
    # Test parameters
    parser.add_argument('--test_path', default='TEST_TEST_Preprocessed.csv', type=str)
    parser.add_argument('--submit_path', default='자외선_검증데이터셋.csv', type=str)
    parser.add_argument('--top_features', default=10, type=int)
    parser.add_argument('--augmentation', default=False, type=str2bool)
    parser.add_argument('--n_fold', default=5, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--device', default='0,1,2,3', type=str)
    parser.add_argument('--text', default='TEXT', type=str)

    return parser


def main(args):
    
    seed=10
        
    config = {
        # Model parameters
        'model': args.model,
        'model_save_name': args.model_save_name,
        'scaler_path': args.scaler_path,
        
        'hidden_dim': int(args.hidden_dim[0]),
        'depth': int(args.depth[0]),
        'drop_out': float(args.drop_out[0]),

        # Training parameters
        'batch_size': 2048,
        'test_path': args.test_path,
        'submit_path': args.submit_path,
        'n_fold': args.n_fold,
        'num_workers': args.num_workers,
        'device': args.device,
        'augmentation': args.augmentation,
        'top_features': args.top_features,
        
        'text':args.text
        }

    # -------------------------------------------------------------------------------------------
    # Dataload
    test_df = pd.read_csv(config['test_path'])
    submit_df = pd.read_csv(config['submit_path'])
    test_df['Date_Time'] = pd.to_datetime(test_df['Date_Time'])
    test_df.set_index(keys=['Date_Time'], inplace=True)

    # -------------------------------------------------------------------------------------------

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print('Device: %s' % device)
    if (device.type == 'cuda') or (torch.cuda.device_count() > 1):
        print('GPU activate --> Count of using GPUs: %s' % torch.cuda.device_count())
    config['device'] = device

    # -------------------------------------------------------------------------------------------
                                    
    top_columns = ['solarza', 'band1', 'band4', 'band2', 'band12', 'cos_hour', 'esr', 'cos_month', 'band13',
                    'landtype_3', 'band11', 'band3', 'landtype_0', 'band5', 'stn', 'band7', 'height_69.56',
                    'sateza_43.74521', 'sateza_43.67046', 'height_38.0', 'landtype_2', 'band14', 'sateza_43.77999',
                    'sin_hour', 'sateza_42.69599', 'sateza_41.82617', 'sateza_40.96396', 'height_72.38', 'height_68.99',
                    'sateza_40.67788', 'band15', 'sateza_42.62037', 'band6', 'sateza_41.52495', 'sin_month',
                    'landtype_4', 'height_82.0', 'sateza_42.40541', 'height_68.94', 'height_71.0', 'height_2.28',
                    'height_58.7', 'height_85.5', 'sateza_38.96359', 'height_47.0', 'height_26.04', 'band10',
                    'sateza_43.95463', 'sateza_42.02608', 'height_53.5', 'sin_day', 'band16', 'height_222.8',
                    'sateza_41.04849', 'band8', 'band9', 'cos_day', 'lat', 'lon',
                    'sateza_41.81018', 'cos_minute', 'sin_minute', 'height_62.9']
    
    test_df = test_df[top_columns[:config['top_features']]+['uv']]
                                                                    
    config['input_dim'] = test_df.shape[1]-1
    config['mode'] = 'TEST'

    Test_dataset = Custom_Dataset(X=test_df.drop(['uv'], axis=1), Y=test_df['uv'], config=config)
    Test_loader = DataLoader(Test_dataset, batch_size=config['batch_size'], 
                            num_workers=config['num_workers'], prefetch_factor=config['batch_size']*2,
                            shuffle=False, drop_last=False, pin_memory=True)

    # Model
    models = []
    for model_name in config['model_save_name']:
        model_dict = torch.load('./RESULTS/'+ model_name + ".pt")
        
        model = Network(config).to(config['device']) 
        model = nn.DataParallel(model).to(config['device'])
        model.module.load_state_dict(model_dict) if torch.cuda.device_count() > 1 else model.load_state_dict(model_dict)
        
        models.append(model)
        
    # Inference
    SOLAR_results = []
    relu = torch.nn.ReLU()
    
    for batch_id, batch in tqdm(enumerate(Test_loader), total=len(Test_loader)):
        x = batch['x'].to(config['device'])
        
        for fold, model in enumerate(models):
            model.eval()
            with torch.no_grad():
                if fold == 0:
                    output = model(x)
                else:
                    output = output+model(x)
                    
        output = output / len(models)
        output = relu(output)
        output = output.cpu().numpy()
        SOLAR_results.extend(output)

    SOLAR_results = [item for sublist in SOLAR_results for item in sublist]
    test_df['uv'] = SOLAR_results

    # Post-processing
    with open(config['scaler_path'], 'rb') as fr:
        reverse_scaling = pickle.load(fr)
        
    reverse_scaling['uv']

    test_df['uv'] = test_df['uv'] * (reverse_scaling['uv'][1]-reverse_scaling['uv'][0])
    test_df['uv'] = test_df['uv'] + reverse_scaling['uv'][0]

    station_encoder = LabelEncoder()
    station_encoder.fit(submit_df['STN'].unique())
    test_df['stn'] = station_encoder.inverse_transform((test_df['stn']*(len(test_df['stn'].unique())-1)).astype(np.float16).astype(int))
    test_df = test_df.sort_values(["Date_Time","stn"]).reset_index()

    submit_df['UV'] = test_df['uv']
    submit_df.to_csv('./RESULTS/'+config['text']+".csv", index=False)
    print('./RESULTS/'+config['text']+".csv is saved!")
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Inference script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
