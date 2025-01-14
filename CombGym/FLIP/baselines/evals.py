from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr
from scipy import stats
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import os
import math

import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def concat_tensor(tensor_list, keep_tensor = False):
    """ converts a list of tensors to a numpy array for stats analysis """
    for i, item in enumerate(tensor_list):
        item.to('cpu')
        if i == 0:
            output_tensor = item
        if i > 0:
            output_tensor = torch.cat((output_tensor, item), 0)
    
    if keep_tensor:
        return output_tensor
    else:
        return np.array(output_tensor)

def regression_eval(predicted, labels, SAVE_PATH):
    """ 
    input: 1D tensor or array of predicted values and labels
    output: saves spearman, MSE, and graph of predicted vs actual 
    """
    #predicted = np.array(predicted)
    #labels = np.array(labels)
    predicted = np.array(predicted).flatten()  # 转换为一维
    labels = np.array(labels).flatten()        # 转换为一维


    rho, _ = stats.spearmanr(predicted, labels) # spearman
    mse = mean_squared_error(predicted, labels) # MSE

    
    results_df = pd.DataFrame({
    'Predicted': predicted,
    'Labels': labels}, index=np.arange(len(predicted)))
    if str(SAVE_PATH).endswith('test'):
        p = results_df['Predicted'].values
        l = results_df['Labels'].values
        k=math.floor(len(results_df['Predicted'])*0.01)
        predict_ndcg = ndcg_score([l], [p],k=k)
        print('ndcg:',round(predict_ndcg, 2))
        results_df.loc[0,'ndcg'] = round(predict_ndcg, 2)
        results_df.loc[0,'spearman'] = round(rho,2)

    # Ensure the SAVE_PATH is a directory, create if it does not exist
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    SAVE_PATH_str = str(SAVE_PATH)
    dirs = SAVE_PATH_str.strip('/').split('/')
    file_name = '_'.join(dirs[-4:-1])
    file_path = os.path.join(SAVE_PATH_str, f'{file_name}.csv')

    #dirs = SAVE_PATH_str.strip('/').split('/')
    #file_name = dirs[-4:-1]
    
    file_path = os.path.join(SAVE_PATH, f'{file_name}.csv')


    results_df.to_csv(file_path, index=True)

    if str(SAVE_PATH).endswith('test'):
        #parent_dir = os.path.dirname(os.path.dirname(SAVE_PATH))
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(SAVE_PATH)))
        test_result_dir = os.path.join(parent_dir, 'test_result')
    
        
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)
    
        
        test_result_file_path = os.path.join(test_result_dir, f'{file_name}.csv')
        results_df.to_csv(test_result_file_path, index=True)

    # remove graphing - causes segmentation fault
    #plt.figure()
    #plt.title('predicted (y) vs. labels (x)')
    #sns.scatterplot(x = labels, y = predicted, s = 2, alpha = 0.2)
    #plt.savefig(SAVE_PATH / 'preds_vs_labels.png', dpi = 300)

    return round(rho, 2), round(mse, 2)

def evaluate_esm(data_iterator, model, device, size, mean, mut_mean, SAVE_PATH):
    """ run data through model and print eval stats """
    
    # create a tensor to hold results
    out = np.empty([size])
    labels = np.empty([size])

    s = 0 
    
    model.eval()
    model.to(device)

    with torch.no_grad(): # evaluate validation loss here 
        for i, (inp, l) in enumerate(data_iterator):
            
            inp = inp.to(device)

            if mean or mut_mean: 
                o = model(inp).squeeze().cpu()
            else:
                m = (inp[:, :, 0] != 0).long().to(device)
                o = model(inp, m).squeeze().cpu()  # Forward prop without storing gradients

            b = inp.shape[0] 
            out[s: s + b:] = o
            labels[s: s + b:] = l

            s += b

    if mean:
        SAVE_PATH = SAVE_PATH / 'mean'
    if mut_mean:
        SAVE_PATH = SAVE_PATH / 'mut_mean'
        
    SAVE_PATH.mkdir(parents=True, exist_ok=True) # make directory if it doesn't exist already
    with open(SAVE_PATH / 'preds_labels_raw.pickle', 'wb') as f:
        pickle.dump((out, labels), f)
    
    rho, mse = regression_eval(predicted=out, labels=labels, SAVE_PATH=SAVE_PATH)

    return rho, mse


def evaluate_cnn(data_iterator, model, device, MODEL_PATH, SAVE_PATH):
    """ run data through model and print eval stats """

    model = model.to(device)
    bestmodel_save = MODEL_PATH / 'bestmodel.tar' 
    sd = torch.load(bestmodel_save)
    model.load_state_dict(sd['model_state_dict'])
    print('loaded the saved model')

    def test_step(model, batch):
        src, tgt, mask = batch
        src = src.to(device).float()
        tgt = tgt.to(device).float()
        mask = mask.to(device).float()
        output = model(src, mask)
        return output.detach().cpu(), tgt.detach().cpu()
    
    model = model.eval()

    outputs = []
    tgts = []
    n_seen = 0
    for i, batch in enumerate(data_iterator):
        output, tgt = test_step(model, batch)
        outputs.append(output)
        tgts.append(tgt)
        n_seen += len(batch[0])

    out = torch.cat(outputs).numpy()
    labels = torch.cat(tgts).cpu().numpy()
    #rho, mse = regression_eval(predicted=out, labels=labels, SAVE_PATH=SAVE_PATH)
    SAVE_PATH.mkdir(parents=True, exist_ok=True) # make directory if it doesn't exist already
    with open(SAVE_PATH / 'preds_labels_raw.pickle', 'wb') as f:
        pickle.dump((out, labels), f)
    rho, mse = regression_eval(predicted=out, labels=labels, SAVE_PATH=SAVE_PATH)
    
    return rho, mse

                    
def evaluate_ridge(X, y, model, SAVE_PATH):
    out = model.predict(X)
    rho, mse = regression_eval(predicted=out, labels=y, SAVE_PATH=SAVE_PATH)
    return rho, mse
