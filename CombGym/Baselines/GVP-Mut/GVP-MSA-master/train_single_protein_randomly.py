import os,math
import argparse
import numpy as np
import math
from omegaconf import OmegaConf

from utils import *
from data import *
from gvpmsa import *

def main(args):
    data_config = OmegaConf.load('data_config.yaml')
    dataset_name = args.train_dataset_names
    print(args.msa_in)
    #五折交叉验证，切分训练集验证集测试集
    fold_num=5
    for fold_idx in range(0,fold_num):
        datas = get_splited_data(dataset_name = dataset_name,
                                     data_split_method = 0,
                                     folder_num = fold_num,
                                     train_ratio=0.7,val_ratio=0.1,test_ratio=0.2,
                                     suffix = '')
        (train_dfs,val_dfs,test_dfs) = datas[fold_idx]
    
        train_df_dict = {dataset_name:train_dfs}
        val_df_dict = {dataset_name:val_dfs}
        test_df_dict = {dataset_name:test_dfs}
        #设置分类损失输出三维，不设置输出一维
        if args.classification_loss:
            data_category=True
            out_dim=3
        else: 
            data_category = False
            out_dim = 1
        gvp_msa = GVPMSA(
                output_dir=os.path.join(args.output_dir,'{}'.format(dataset_name)),
                dataset_names=[dataset_name],
                train_dfs_dict=train_df_dict,
                val_dfs_dict=val_df_dict,
                test_dfs_dict=test_df_dict,
                dataset_config=data_config,
                device = args.device,
                data_category=data_category,
                out_dim=out_dim,
                lr = args.lr,
                batch_size = args.batch_size,
                n_ensembles=args.n_ensembles,

                multi_train=args.multi_model,
                msa_in = args.msa_in,
                pdb_path_prefix = 'input_data',)
        #记录第几折训练
        gvp_msa.logger.write('training on fold {} \n'.format(fold_idx))
        #训练当前折，以及是否保存参数
        gvp_msa.train_onefold(fold_idx,epochs=args.epochs,patience=args.patience,
                       save_checkpoint=args.save_checkpoint, save_prediction=args.save_prediction)
    
if __name__ == "__main__":
    def str2bool(str):
        if type(str) == bool:
            return str
        else:
            return True if str.lower() == 'true' else False
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #指定训练数据集的名称
    parser.add_argument('--train_dataset_names', action='store', required=True)
    #运行在哪个设备上，默认 'cuda:0'
    parser.add_argument('--device',action='store', default='cuda:0', help='run on which device')
    #指定集成模型中的模型数量，默认为 3。
    parser.add_argument('--n_ensembles', action='store', type=int, default=3, help='number of models in ensemble')
    #线性层从 MSA Transformer 投影的隐藏维度，默认为 128。
    parser.add_argument('--esm_msa_linear_hidden', action='store', type=int, default=128, help='hidden dim of linear layer projected from MSA Transformer')
    #GVP 模型中 GVP 层的数量，默认为 2。
    parser.add_argument('--n_layers', action='store', type=int, default=2, help='number of GVP layers')
    #是否使用分类损失，默认为 False。
    parser.add_argument('--classification_loss', action='store',type=str2bool, default=False, help='penalize with classification loss')
    #是否训练多蛋白质模型，每个蛋白质都有自己的顶层参数，默认为 False。
    parser.add_argument('--multi_model', action='store',type=str2bool, default=False, help='train multi-protein, each protein have their own top parameters')
    #是否将 MSA 信息加入模型，默认为 True。
    parser.add_argument('--msa_in', action='store',type=str2bool, default=True, help='add msa information into to model')
    #pars最大训练轮数，默认为 1500。
    parser.add_argument('--epochs', action='store', type=int, default=1500, help='maximum epochs')
    #早停的耐心值，即连续多少轮没有改进就停止训练，默认为 200。
    parser.add_argument('--patience', action='store', type=int, default=200,help='patience for early stopping')
    #学习率，默认为 1e-4
    parser.add_argument('--lr', action='store', default=1e-4,help='learning rate')
    #批处理大小，默认为 50。
    parser.add_argument('--batch_size', action='store', type=int, default=50, help='batch size')
    parser.add_argument('--output_dir', action='store',default='results/single_protein_random_split', help='directory to save model, prediction, etc.')
    parser.add_argument('--save_checkpoint', action='store',type=str2bool, default=False, help='save pytorch model checkpoint')
    parser.add_argument('--save_prediction', action='store',type=str2bool, default=True, help='save prediction')
    
    args = parser.parse_args()
    main(args)
