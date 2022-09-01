import os
import copy 
import random
import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from WEATHER_CONTEST_UTILS import str2bool, Custom_Dataset, CosineAnnealingWarmUpRestarts, EarlyStopping
from WEATHER_CONTEST_MODEL import Network

import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch Training', add_help=False)
    
    
    # Model parameters
    parser.add_argument('--model', default='DNN', type=str)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--drop_out', default=0.1, type=float)


    # Optimizer parameters
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_t', default=10, type=int)
    parser.add_argument('--lr_scheduler', default='CosineAnnealingLR', type=str)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--min_delta', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)


    # Training parameters
    parser.add_argument('--train_dataset', default='TEST_TRAIN_Preprocessed.csv', type=str)
    parser.add_argument('--top_features', default=10, type=int)
    parser.add_argument('--grad_scale', default=True, type=str2bool)
    parser.add_argument('--augmentation', default=True, type=str2bool)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--n_fold', default=5, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--text', default='default', type=str)
    parser.add_argument('--device', default='0,1,2,3', type=str)

    return parser


def main(args):
    
    seed = 10
    suffix = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%y%m%d_%H%M")

    config = {
        # Model parameters
        'model': args.model,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'depth': args.depth,
        'drop_out': args.drop_out,
        'dest_range': {
                    'train1' : ['2020-01', '2021-05'],
                    'valid' : ['2021-06', '2021-06'],
                    'train2' : ['2021-07', '2021-12'],
                    },
        
        # Optimizer parameters
        'lr': args.lr,
        'lr_t': args.lr_t,
        'lr_scheduler': args.lr_scheduler,
        'gamma': args.gamma,
        'patience': args.patience,
        'min_delta': args.min_delta,
        'weight_decay': args.weight_decay,
        
        # Training parameters
        'train_dataset': args.train_dataset,
        'top_features': args.top_features,
        'grad_scale': args.grad_scale,
        'augmentation': args.augmentation,
        'epochs': args.epochs,
        'n_fold': args.n_fold,
        'num_workers': args.num_workers,
        'text': args.text,
        'device': args.device,
        }
    
        
    model_save_name='./RESULTS/'+config['text']+"_"+suffix+"("+ str(config['model'])+"_"+\
                                                                str(config['batch_size'])+"_"+\
                                                                str(config['hidden_dim'])+"_"+\
                                                                str(config['depth'])+"_"+\
                                                                str(config['drop_out'])+"__"+\
                                                                str(config['lr'])+"_"+\
                                                                str(config['lr_t'])+"_"+\
                                                                str(config['lr_scheduler'])+"_"+\
                                                                str(config['gamma'])+"_"+\
                                                                str(config['patience'])+"_"+\
                                                                str(config['n_fold'])+"_"+\
                                                                str(config['top_features'])+"_"+\
                                                                str(config['augmentation'])+"_"+\
                                                                str(config['grad_scale'])+"_"+\
                                                                str(config['weight_decay'])+")_fold_"
                                                            
    config['model_save_name'] = model_save_name
    print('model_save_name: '+config['model_save_name'].split("/")[-1])
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

    # Dataload
    train_data = pd.read_csv(config['train_dataset'], index_col=0)
    train_data.reset_index(drop=True, inplace=True)
    
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
    
    train_data = train_data[top_columns[:config['top_features']]+['uv']]

    # -------------------------------------------------------------------------------------------
    
    config['input_dim'] = train_data.shape[1]-1
    config['mode'] = 'TRAIN'
    n_fold = config['n_fold']
        
    # KFold
    k_valid_loss = []     

    kf = KFold(n_splits=config['n_fold'], shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_data)):
        
        train_df = train_data.iloc[train_idx]
        valid_df = train_data.iloc[valid_idx]

        
        # Train
        Train_dataset = Custom_Dataset(X=train_df.drop(['uv'], axis=1), Y=train_df['uv'], config=config)
        Train_loader = DataLoader(Train_dataset, batch_size=config['batch_size'], 
                                num_workers=config['num_workers'], prefetch_factor=config['batch_size']*2,
                                shuffle=True, drop_last=False, pin_memory=True)

        # Valid
        Valid_dataset = Custom_Dataset(X=valid_df.drop(['uv'], axis=1), Y=valid_df['uv'], config=config)
        Valid_loader = DataLoader(Valid_dataset, batch_size=config['batch_size'], 
                                num_workers=config['num_workers'], prefetch_factor=config['batch_size']*2,
                                shuffle=True, drop_last=False, pin_memory=True)

        # Model
        model = Network(config).to(config['device'])
        model = nn.DataParallel(model).to(config['device'])

        if config['lr_scheduler'] == 'CosineAnnealingLR':
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['lr_t'], eta_min=0)
            
        elif config['lr_scheduler'] == 'CosineAnnealingWarmUpRestarts':
            optimizer = torch.optim.AdamW(model.parameters(), lr=0, weight_decay=config['weight_decay'])
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=config['lr_t'], eta_max=config['lr'], gamma=config['gamma'], T_mult=1, T_up=0)

        criterion = nn.MSELoss().cuda()
        scaler = torch.cuda.amp.GradScaler() 
        early_stopping = EarlyStopping(patience=config['patience'], mode='min', min_delta=config['min_delta'])
        
        best_loss=100
        epochs = config['epochs']

        train_losses, valid_losses = [], []

        for epoch in range(epochs):
            train_loss = 0
            valid_loss = 0

            model.train()
            tqdm_dataset = tqdm(enumerate(Train_loader), total=len(Train_loader))
            
            for batch_id, batch in tqdm_dataset:
                
                optimizer.zero_grad()
                x = batch['x'].to(config['device'])
                y = batch['y'].to(config['device'])

                if config['grad_scale'] == True:
                    with torch.cuda.amp.autocast():
                        pred = model(x)
                        loss = criterion(pred.squeeze(), y)
                        
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                        
                else:
                    pred = model(x)
                    loss = criterion(pred.squeeze(), y)
                    
                    loss.backward()
                    optimizer.step()
        
                train_loss += loss.item()

            train_loss = train_loss/len(Train_loader)
            train_losses.append(train_loss)

            scheduler.step()

            model.eval()
            tqdm_valid_dataset = tqdm(enumerate(Valid_loader), total=len(Valid_loader))
            
            for batch_id, val_batch in tqdm_valid_dataset:
                with torch.no_grad():
                    val_x = val_batch['x'].to(config['device'])
                    val_y = val_batch['y'].to(config['device'])

                    val_pred = model(val_x)
                    val_loss = criterion(val_pred.squeeze(), val_y)

                valid_loss += val_loss.item()

            valid_loss = valid_loss/len(Valid_loader)
            valid_losses.append(valid_loss)
                
            print_best = 0    
            if valid_losses[-1] <= best_loss:
                difference = valid_losses[-1] - best_loss
                best_loss = valid_losses[-1]
                        
                best_idx = epoch+1
                model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.module.state_dict()
                best_model_wts = copy.deepcopy(model_state_dict)
                
                # load and save best model weights
                model.module.load_state_dict(best_model_wts)
                torch.save(best_model_wts, config['model_save_name'] + str(fold+1) + ".pt")
                print_best = '==> best model saved %d epoch / loss: %.6f  /  difference %.6f'%(best_idx, best_loss, difference)

            print(f'FOLD : {fold+1}/{n_fold}    EPOCH : {epoch+1}/{epochs}')
            print(f'TRAIN_Loss : {train_loss:.6f}   VALID_Loss : {valid_loss:.6f}   BEST_LOSS : {best_loss:.6f}')
            print('\n') if type(print_best)==int else print(print_best,'\n')

            if early_stopping.step(torch.tensor(valid_losses[-1])):
                break
            
        print("\n")
        print("BEST epoch : {}    BEST Loss : {}".format(best_idx, best_loss))
        k_valid_loss.append(best_loss)
        
    print(config['model_save_name'] + ' model is saved!')
    
    print("1Fold - VALID Loss: ", k_valid_loss[0])
    print("2Fold - VALID Loss: ", k_valid_loss[1])
    print("3Fold - VALID Loss: ", k_valid_loss[2])
    print("4Fold - VALID Loss: ", k_valid_loss[3])
    print("5Fold - VALID Loss: ", k_valid_loss[4])
    
    print("k-fold Valid Loss: ",np.mean(k_valid_loss))        
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
