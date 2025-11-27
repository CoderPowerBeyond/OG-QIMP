import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
import pandas as pd
import numpy as np
import networkx as nx
import os
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
import random
import dgl
import warnings
import time

warnings.filterwarnings("ignore", message=".*scatter.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
import statistics


from logzero import logger
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score,r2_score,mean_squared_error,mean_absolute_error
from sklearn.metrics import precision_recall_curve, auc

from sklearn import metrics

from model.ka_gat import KA_GAT
from model.sdn import EnhancedOG_PGAT
from model.mlp_gat import MLP_GAT
from model.kan_gat import KAN_GAT
from model.po_gat import PO_GAT
from torch.optim.lr_scheduler import StepLR
from ruamel.yaml import YAML
from utils.splitters import ScaffoldSplitter
from utils.graph_path import path_complex_mol
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class CustomDataset(Dataset):
    def __init__(self, label_list, graph_list):
        self.labels = label_list
        self.graphs = graph_list
        self.device = torch.device('cpu') 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index].to(self.device)
        
        graph = self.graphs[index].to(self.device)
        
        return label, graph
    


def collate_fn(batch):
    labels, graphs = zip(*batch) 

    labels = torch.stack(labels)

    batched_graph = dgl.batch(graphs)

    return labels, batched_graph



def has_node_with_zero_in_degree(graph):
    if (graph.in_degrees() == 0).any():
                return True
    return False






def is_file_in_directory(directory, target_file):
    file_path = os.path.join(directory, target_file)
    return os.path.isfile(file_path)


#others
def get_label():
    """Get that default sider task names and return the side results for the drug"""
    
    return ['label']


#tox21,12     
def get_tox():
    """Get that default sider task names and return the side results for the drug"""
    
    return ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
           'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

#clintox,2
def get_clintox():
    
    return ['FDA_APPROVED', 'CT_TOX']

#sider,27
def get_sider():

    return ['Hepatobiliary disorders',
           'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
           'Investigations', 'Musculoskeletal and connective tissue disorders',
           'Gastrointestinal disorders', 'Social circumstances',
           'Immune system disorders', 'Reproductive system and breast disorders',
           'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
           'General disorders and administration site conditions',
           'Endocrine disorders', 'Surgical and medical procedures',
           'Vascular disorders', 'Blood and lymphatic system disorders',
           'Skin and subcutaneous tissue disorders',
           'Congenital, familial and genetic disorders',
           'Infections and infestations',
           'Respiratory, thoracic and mediastinal disorders',
           'Psychiatric disorders', 'Renal and urinary disorders',
           'Pregnancy, puerperium and perinatal conditions',
           'Ear and labyrinth disorders', 'Cardiac disorders',
           'Nervous system disorders',
           'Injury, poisoning and procedural complications']

#muv
def get_muv():
    
    return ['MUV-466','MUV-548','MUV-600','MUV-644','MUV-652','MUV-689','MUV-692',
            'MUV-712','MUV-713','MUV-733','MUV-737','MUV-810','MUV-832','MUV-846',
            'MUV-852',	'MUV-858','MUV-859']



def creat_data(datafile, encoder_atom, encoder_bond,batch_size,train_ratio,vali_ratio,test_ratio):
    

    datasets = datafile

    directory_path = 'data/processed/'
    target_file_name = datafile +'.pth'

    if is_file_in_directory(directory_path, target_file_name):

        return True
    
    else:

        df = pd.read_csv('data/' + datasets + '.csv')#
        if datasets == 'tox21':
            smiles_list, labels = df['smiles'], df[get_tox()] 
            #labels = labels.replace(0, -1)
            labels = labels.fillna(0)

        if datasets == 'muv':
            smiles_list, labels = df['smiles'], df[get_muv()]  
            labels = labels.fillna(0)

        if datasets == 'sider':
            smiles_list, labels = df['smiles'], df[get_sider()]  

        if datasets == 'clintox':
            smiles_list, labels = df['smiles'], df[get_clintox()] 
        
        if datasets == 'qm9':
            # QM9数据集：homo, lumo, gap三个回归目标
            smiles_list = df['SMILES']
            labels = df[['HOMO', 'LUMO', 'Gap']]

        if datasets in ['hiv','bbbp','bace']:
            smiles_list, labels = df['smiles'], df[get_label()] 
            
        #labels = labels.replace(0, -1)
        #labels = labels.fillna(0)

        #smiles_list, labels = df['smiles'], df['label']        
        #labels = labels.replace(0, -1)
        
        #labels, min_val, max_val = min_max_normalize(labels)

        # 检查smiles_list是否被正确初始化
        if 'smiles_list' not in locals():
            raise ValueError(f"数据集 '{datasets}' 未在creat_data函数中定义处理逻辑，请添加相应的处理代码。")

        data_list = []
        feature_sets = ("atomic_number", "basic", "cfid", "cgcnn")
        for i in range(len(smiles_list)):
            if i % 10000 == 0:
                print(i)

            smiles = smiles_list[i]
            
            #if has_isolated_hydrogens(smiles) == False and conformers_is_zero(smiles) == True :

            Graph_list = path_complex_mol(smiles, encoder_atom, encoder_bond)
            if Graph_list == False:
                continue

            else:
                if has_node_with_zero_in_degree(Graph_list):
                    continue
                
                else:
                    data_list.append([smiles, torch.tensor(labels.iloc[i]),Graph_list])



        #data_list = [['occr',albel,[c_size, features, edge_indexs],[g,liearn_g]],[],...,[]]

        print('Graph list was done!')

        splitter = ScaffoldSplitter().split(data_list, frac_train=train_ratio, frac_valid=vali_ratio, frac_test=test_ratio)
        
        print('splitter was done!')
        
        # 用于保存归一化参数
        label_mean = None
        label_std = None
        
        # 如果是QM9数据集，基于训练集计算标准化参数
        if datasets == 'qm9':
            # 提取所有训练标签堆叠成 tensor
            train_structs = splitter[0]
            all_train_labels = torch.stack([item[1] for item in train_structs]).float()
            
            label_mean = all_train_labels.mean(dim=0).numpy()
            label_std = all_train_labels.std(dim=0).numpy()
            
            print(f"基于训练集计算的标准化参数: mean={label_mean}, std={label_std}")
            
            # 定义标准化函数
            def normalize_labels(dataset_list, mean, std):
                mean_t = torch.tensor(mean).float()
                std_t = torch.tensor(std).float()
                normalized_list = []
                for item in dataset_list:
                    normalized_label = (item[1].float() - mean_t) / (std_t + 1e-8)
                    normalized_list.append([item[0], normalized_label, item[2]])
                return normalized_list

            # 对三个集合应用标准化
            splitter = (
                normalize_labels(splitter[0], label_mean, label_std),
                normalize_labels(splitter[1], label_mean, label_std),
                normalize_labels(splitter[2], label_mean, label_std)
            )
        
        train_label = []
        train_graph_list = []
        for tmp_train_graph in splitter[0]:
            train_label.append(tmp_train_graph[1])
            train_graph_list.append(tmp_train_graph[2])

        valid_label = []
        valid_graph_list = []
        for tmp_valid_graph in splitter[1]:
            valid_label.append(tmp_valid_graph[1])
            valid_graph_list.append(tmp_valid_graph[2])

        test_label = []
        test_graph_list = []
        for tmp_test_graph in splitter[2]:
            test_label.append(tmp_test_graph[1])
            test_graph_list.append(tmp_test_graph[2])

        save_dict = {
            'train_label': train_label,
            'train_graph_list': train_graph_list,
            'valid_label': valid_label,
            'valid_graph_list': valid_graph_list,
            'test_label': test_label,
            'test_graph_list': test_graph_list,
            'batch_size': batch_size,
            'shuffle': True,
        }
        
        # 如果是QM9数据集，保存归一化参数
        if datasets == 'qm9' and label_mean is not None:
            save_dict['label_mean'] = label_mean
            save_dict['label_std'] = label_std
            print(f"保存归一化参数: mean={label_mean}, std={label_std}")
            
        torch.save(save_dict, 'data/processed/'+ datafile +'.pth')


def train(model, device, train_loader, valid_loader, optimizer, epoch, loss_select='bce', model_select='sdn'):
    epoch_start_time = time.time()
    model.train()

    total_train_loss = 0.0
    train_num = 0

    for batch_idx, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        train_label_value = []
        y = data[0]
        
        #train_label_value.append(torch.unsqueeze(y, dim=0))
        #graph_list = update_node_features(data[1]).to(device)
        graph_list = data[1].to(device)
        node_features = graph_list.ndata['feat'].to(device)
        edge_features = graph_list.edata['feat'].to(device)
        
        #output = model(batch_g_list = graph_list, device = device, resent = resent,pooling=pooling).cpu()
        if model_select == 'sdn':
            # Convert DGL to PyTorch Geometric for OG_PGAT_Complete
            from torch_geometric.data import Data, Batch
            
            pyg_data_list = []
            node_start = 0
            edge_start = 0
            
            for i in range(graph_list.batch_size):
                num_nodes = graph_list.batch_num_nodes()[i].item()
                num_edges = graph_list.batch_num_edges()[i].item()
                
                graph_node_features = node_features[node_start:node_start + num_nodes]
                graph_edge_features = edge_features[edge_start:edge_start + num_edges]
                
                # 获取当前图的边索引，并调整节点索引
                src, dst = graph_list.edges()
                graph_src = src[edge_start:edge_start + num_edges] - node_start
                graph_dst = dst[edge_start:edge_start + num_edges] - node_start
                graph_edges = torch.stack([graph_src, graph_dst], dim=0)
                
                pyg_data = Data(
                    x=graph_node_features,
                    edge_index=graph_edges,
                    edge_attr=graph_edge_features,
                    y=y[i].unsqueeze(0).float()
                )
                pyg_data_list.append(pyg_data)
                
                node_start += num_nodes
                edge_start += num_edges
            
            pyg_batch = Batch.from_data_list(pyg_data_list)
            model_output = model(pyg_batch, return_intermediate=True)
            
            # 处理模型返回字典的情况
            if isinstance(model_output, dict):
                output = model_output['prediction'].cpu()
            else:
                output = model_output.cpu()
        else:
            output = model(graph_list, node_features, edge_features).cpu()
       
        # 对于回归任务，直接使用y和output，不需要循环处理
        is_regression = loss_select in ['l1', 'l2', 'sml1']
        if is_regression:
            # 回归任务：直接flatten，保持样本顺序 [sample1_dim0, sample1_dim1, ..., sample1_dimN, sample2_dim0, ...]
            arr_label = y.float().cpu().flatten()
            arr_pred = output.float().cpu().flatten()
        else:
            # 分类任务：保持原有逻辑（处理-1值）
            arr_label = torch.Tensor().cpu()
            arr_pred = torch.Tensor().cpu()
            for j in range(y.shape[1]):
                c_valid = np.ones_like(y[:, j], dtype=bool)
                c_label, c_pred = y[c_valid, j], output[c_valid, j]
                zero = torch.zeros_like(c_label)
                c_label = torch.where(c_label == -1, zero, c_label)
                
                arr_label = torch.cat((arr_label,c_label),0)
                arr_pred = torch.cat((arr_pred,c_pred),0)
            
            arr_pred = arr_pred.float()
            arr_label = arr_label.float()
        
        # 根据损失函数类型决定是否应用sigmoid
        # 回归任务（l1, l2, sml1）不需要sigmoid，分类任务（bce）需要
        if not is_regression and model_select not in ['schnet', 'schnet_orbital']:
            arr_pred = torch.sigmoid(arr_pred)

        # 根据loss_select选择损失函数
        # 对于回归任务，使用mean reduction以保持损失值在合理范围
        if loss_select == 'l1':
            loss = nn.L1Loss(reduction='mean')(arr_pred, arr_label)
        elif loss_select == 'l2':
            loss = nn.MSELoss(reduction='mean')(arr_pred, arr_label)
        elif loss_select == 'sml1':
            loss = nn.SmoothL1Loss(reduction='mean')(arr_pred, arr_label)
        elif loss_select == 'bce':
            loss = nn.BCELoss(reduction='mean')(arr_pred, arr_label)
        else:
            loss = nn.BCELoss(reduction='mean')(arr_pred, arr_label)  # 默认使用BCE
        
        # 累加批次平均损失（用于计算整个epoch的平均损失）
        total_train_loss = total_train_loss + loss.item()
        train_num += 1
        loss.backward()
        optimizer.step()
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    total_loss_val = 0.0

    for batch_idx, valid_data in enumerate(valid_loader):

        y = valid_data[0]
        #label_value.append(torch.unsqueeze(y, dim=0))
        graph_list = valid_data[1].to(device)
        node_features = graph_list.ndata['feat'].to(device)
        edge_features = graph_list.edata['feat'].to(device)
        #output = model(batch_g_list = graph_list, device = device, resent = resent,pooling=pooling).cpu()
        if model_select == 'sdn':
            # Convert DGL to PyTorch Geometric for OG_PGAT_Complete
            from torch_geometric.data import Data, Batch
            
            pyg_data_list = []
            node_start = 0
            edge_start = 0
            
            for i in range(graph_list.batch_size):
                num_nodes = graph_list.batch_num_nodes()[i].item()
                num_edges = graph_list.batch_num_edges()[i].item()
                
                graph_node_features = node_features[node_start:node_start + num_nodes]
                graph_edge_features = edge_features[edge_start:edge_start + num_edges]
                
                # 获取当前图的边索引，并调整节点索引
                src, dst = graph_list.edges()
                graph_src = src[edge_start:edge_start + num_edges] - node_start
                graph_dst = dst[edge_start:edge_start + num_edges] - node_start
                graph_edges = torch.stack([graph_src, graph_dst], dim=0)
                
                pyg_data = Data(
                    x=graph_node_features,
                    edge_index=graph_edges,
                    edge_attr=graph_edge_features,
                    y=y[i].unsqueeze(0).float()
                )
                pyg_data_list.append(pyg_data)
                
                node_start += num_nodes
                edge_start += num_edges
            
            pyg_batch = Batch.from_data_list(pyg_data_list)
            model_output = model(pyg_batch, return_intermediate=True)
            
            # 处理模型返回字典的情况
            if isinstance(model_output, dict):
                output = model_output['prediction'].cpu()
            else:
                output = model_output.cpu()
        else:
            output = model(graph_list, node_features, edge_features).cpu()

        # 对于回归任务，直接使用y和output，不需要循环处理
        is_regression = loss_select in ['l1', 'l2', 'sml1']
        if is_regression:
            # 回归任务：直接flatten，保持样本顺序 [sample1_dim0, sample1_dim1, ..., sample1_dimN, sample2_dim0, ...]
            arr_label = y.float().cpu().flatten()
            arr_pred = output.float().cpu().flatten()
        else:
            # 分类任务：保持原有逻辑（处理-1值）
            arr_label = torch.Tensor().cpu()
            arr_pred = torch.Tensor().cpu()
            for j in range(y.shape[1]):
                c_valid = np.ones_like(y[:, j], dtype=bool)
                c_label, c_pred = y[c_valid, j], output[c_valid, j]
                zero = torch.zeros_like(c_label)
                c_label = torch.where(c_label == -1, zero, c_label)
                
                arr_label = torch.cat((arr_label,c_label),0)
                arr_pred = torch.cat((arr_pred,c_pred),0)
            
            arr_pred = arr_pred.float()
            arr_label = arr_label.float()
        
        # 根据损失函数类型决定是否应用sigmoid
        # 回归任务（l1, l2, sml1）不需要sigmoid，分类任务（bce）需要
        if not is_regression and model_select not in ['schnet', 'schnet_orbital']:
            arr_pred = torch.sigmoid(arr_pred)

        # 根据loss_select选择损失函数
        # 对于回归任务，使用mean reduction以保持损失值在合理范围
        if loss_select == 'l1':
            loss = nn.L1Loss(reduction='mean')(arr_pred, arr_label)
        elif loss_select == 'l2':
            loss = nn.MSELoss(reduction='mean')(arr_pred, arr_label)
        elif loss_select == 'sml1':
            loss = nn.SmoothL1Loss(reduction='mean')(arr_pred, arr_label)
        elif loss_select == 'bce':
            loss = nn.BCELoss(reduction='mean')(arr_pred, arr_label)
        else:
            loss = nn.BCELoss(reduction='mean')(arr_pred, arr_label)  # 默认使用BCE
        #loss = FocalLoss(arr_pred, arr_label)

        # 累加批次平均损失（用于计算整个epoch的平均损失）
        total_loss_val += loss.item()
    
    # 计算epoch时间
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    
    # 计算平均损失（而不是总和）
    avg_train_loss = total_train_loss / train_num if train_num > 0 else 0.0
    avg_val_loss = total_loss_val / len(valid_loader) if len(valid_loader) > 0 else 0.0
        
    print(f"Epoch {epoch}|Train Loss: {avg_train_loss:.4f}| Vali Loss:{avg_val_loss:.4f}| Time: {epoch_duration:.2f}s")

    return avg_train_loss, avg_val_loss


def predicting(model, device, data_loader, loss_select='bce', model_select='sdn', label_mean=None, label_std=None):
    model.eval()
    
    total_preds = torch.Tensor().cpu()
    total_labels = torch.Tensor().cpu()

    
    with torch.no_grad():
        
        for batch_idx, data in enumerate(data_loader):

            y = data[0]
            #true = inverse_min_max_normalize(y,min_val, max_val)
            
            #graph_list = update_node_features(data[1]).to(device)
            graph_list = data[1].to(device)
            node_features = graph_list.ndata['feat'].to(device)
            edege_features = graph_list.edata['feat'].to(device)
            #output = model(batch_g_list = graph_list, device = device, resent = resent,pooling=pooling).cpu() 
            if model_select == 'sdn':
                # Convert DGL to PyTorch Geometric for OG_PGAT_Complete
                from torch_geometric.data import Data, Batch
                
                pyg_data_list = []
                node_start = 0
                edge_start = 0
                
                for i in range(graph_list.batch_size):
                    num_nodes = graph_list.batch_num_nodes()[i].item()
                    num_edges = graph_list.batch_num_edges()[i].item()
                    
                    graph_node_features = node_features[node_start:node_start + num_nodes]
                    graph_edge_features = edege_features[edge_start:edge_start + num_edges]
                    
                    # 获取当前图的边索引，并调整节点索引
                    src, dst = graph_list.edges()
                    graph_src = src[edge_start:edge_start + num_edges] - node_start
                    graph_dst = dst[edge_start:edge_start + num_edges] - node_start
                    graph_edges = torch.stack([graph_src, graph_dst], dim=0)
                    
                    pyg_data = Data(
                        x=graph_node_features,
                        edge_index=graph_edges,
                        edge_attr=graph_edge_features,
                        y=y[i].unsqueeze(0).float()
                    )
                    pyg_data_list.append(pyg_data)
                    
                    node_start += num_nodes
                    edge_start += num_edges
                
                pyg_batch = Batch.from_data_list(pyg_data_list)
                model_output = model(pyg_batch, return_intermediate=True)
                
                # 处理模型返回字典的情况
                if isinstance(model_output, dict):
                    output = model_output['prediction'].cpu()
                else:
                    output = model_output.cpu()
            else:
                output = model(graph_list, node_features, edege_features).cpu()

            # 对于回归任务，直接使用y和output，不需要循环处理
            is_regression = loss_select in ['l1', 'l2', 'sml1']
            if is_regression:
                # 回归任务：直接flatten，保持样本顺序 [sample1_dim0, sample1_dim1, ..., sample1_dimN, sample2_dim0, ...]
                arr_label = y.float().cpu().flatten()
                arr_pred = output.float().cpu().flatten()
            else:
                # 分类任务：保持原有逻辑（处理-1值）
                arr_label = torch.Tensor().cpu()
                arr_pred = torch.Tensor().cpu()
                for j in range(y.shape[1]):
                    c_valid = np.ones_like(y[:, j], dtype=bool)
                    c_label, c_pred = y[c_valid, j], output[c_valid, j]
                    zero = torch.zeros_like(c_label)
                    c_label = torch.where(c_label == -1, zero, c_label)

                    arr_label = torch.cat((arr_label,c_label),0)
                    arr_pred = torch.cat((arr_pred,c_pred),0)
                
                arr_pred = arr_pred.float()
                arr_label = arr_label.float()
            
            # 根据任务类型决定是否应用sigmoid
            if not is_regression and model_select not in ['schnet', 'schnet_orbital']:
                arr_pred = torch.sigmoid(arr_pred)
                    
            total_preds = torch.cat((total_preds, arr_pred), 0)
            total_labels = torch.cat((total_labels, arr_label), 0)

    # 根据任务类型选择评估指标
    is_regression = loss_select in ['l1', 'l2', 'sml1']
    if is_regression:
        # 回归任务：只计算MAE
        from sklearn.metrics import mean_absolute_error
        
        # 如果标签被归一化了，需要反归一化
        if label_mean is not None and label_std is not None:
            # 反归一化：pred * std + mean
            # 处理多维标签的情况
            preds_flat = total_preds.numpy().flatten()
            labels_flat = total_labels.numpy().flatten()
            
            # 如果label_mean和label_std是数组，需要按维度处理
            if isinstance(label_mean, np.ndarray) and len(label_mean) > 1:
                # 多维标签：需要reshape后按维度反归一化
                num_samples = len(preds_flat) // len(label_mean)
                preds_reshaped = preds_flat.reshape(num_samples, len(label_mean))
                labels_reshaped = labels_flat.reshape(num_samples, len(label_mean))
                
                # 反归一化：标准化值 * std + mean
                preds_denorm = preds_reshaped * label_std + label_mean
                labels_denorm = labels_reshaped * label_std + label_mean
                
                preds_denorm = preds_denorm.flatten()
                labels_denorm = labels_denorm.flatten()
            else:
                # 单维标签或标量
                mean_val = label_mean[0] if isinstance(label_mean, np.ndarray) else label_mean
                std_val = label_std[0] if isinstance(label_std, np.ndarray) else label_std
                preds_denorm = preds_flat * std_val + mean_val
                labels_denorm = labels_flat * std_val + mean_val
            
            # 计算MAE
            mae = mean_absolute_error(labels_denorm, preds_denorm)
            print(f"MAE: {mae:.4f}")
        else:
            mae = mean_absolute_error(total_labels.numpy().flatten(), total_preds.numpy().flatten())
            print(f"MAE: {mae:.4f}")
        return -mae  # 返回负MAE以便与AUC的优化方向一致（越大越好）
    else:
        # 分类任务：计算AUC
        AUC = roc_auc_score(total_labels.numpy().flatten(), total_preds.numpy().flatten())
        return AUC




def parse_arguments():
    parser = argparse.ArgumentParser(description="help")

    parser.add_argument("--config", type=str, help="path")

    args = parser.parse_args()
    args.config = './config/gat_path.yaml'
    if args.config:
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)

        for key, value in config.items():
            setattr(args, key, value)

    return args


if __name__ == '__main__':
    
    #mp.set_start_method('spawn', force=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
        # 设置CUDA内存管理
        torch.cuda.empty_cache()
        # 设置内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print(f'GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    

    args = parse_arguments()
    for key, value in vars(args).items():
        if key != 'config':
            print(f"{key}: {value}")
    
    
    datafile = args.select_dataset
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    vali_ratio = args.vali_ratio
    test_ratio = args.test_ratio
    target_map = {'tox21':12,'muv':17,'sider':27,'clintox':2,'bace':1,'bbbp':1,'hiv':1,'qm9':3}
    target_dim = target_map[datafile]

    

    encoder_atom = args.encoder_atom
    encoder_bond = args.encoder_bond

    encode_dim = [0,0]
    encode_dim[0] = 92
    encode_dim[1] = 21
    

    
    creat_data(datafile, encoder_atom, encoder_bond, batch_size, train_ratio, vali_ratio, test_ratio)

    model_select = args.model_select
    loss_select = args.loss_select

    state = torch.load('data/processed/'+datafile+'.pth')

    # 读取归一化参数（如果存在）
    label_mean = state.get('label_mean', None)
    label_std = state.get('label_std', None)
    if label_mean is not None and label_std is not None:
        # 转换为numpy数组以便后续使用
        if isinstance(label_mean, torch.Tensor):
            label_mean = label_mean.numpy()
        if isinstance(label_std, torch.Tensor):
            label_std = label_std.numpy()
        # 如果是单个值，需要处理为数组
        if not isinstance(label_mean, np.ndarray):
            label_mean = np.array([label_mean] * target_dim) if target_dim > 1 else np.array([label_mean])
        if not isinstance(label_std, np.ndarray):
            label_std = np.array([label_std] * target_dim) if target_dim > 1 else np.array([label_std])

    loaded_train_dataset = CustomDataset(state['train_label'], state['train_graph_list'])
    loaded_valid_dataset = CustomDataset(state['valid_label'], state['valid_graph_list'])
    loaded_test_dataset = CustomDataset(state['test_label'], state['test_graph_list'])
    
   

    loaded_train_loader = DataLoader(loaded_train_dataset, batch_size=batch_size, shuffle=state['shuffle'],num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    if vali_ratio == 0.0:
        loaded_valid_loader = []
    else:
        loaded_valid_loader = DataLoader(loaded_valid_dataset, batch_size=batch_size, shuffle=state['shuffle'],num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    loaded_test_loader = DataLoader(loaded_test_dataset, batch_size=batch_size, shuffle=state['shuffle'],num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)


    print('dataset was loaded!')

    
    iter = args.iter
    head = args.head
    num_layers = args.num_layers
    LR = args.LR
    NUM_EPOCHS = args.NUM_EPOCHS
    grid = args.grid
    num_layers = args.num_layers
    pooling = args.pooling

    All_AUC = []
    seed = 42
    set_seed(seed)

    for i in range(iter):
        
        AUC_list = []
        if model_select == 'kagat':
            model = KA_GAT(in_node_dim=encode_dim[0], in_edge_dim=encode_dim[1], hidden_dim=64, out_1=32, out_2=target_dim, gride_size=grid, 
                              head=head,layer_num=num_layers, pooling = pooling)  
        
        elif model_select == 'sdn':
            model = EnhancedOG_PGAT(in_node_dim=encode_dim[0], in_edge_dim=encode_dim[1], hidden_dim=64, out_1=32, out_2=target_dim, num_layers=num_layers)  
            
        elif model_select =='kangat':
            model = KAN_GAT(in_node_dim=encode_dim[0], in_edge_dim=encode_dim[1], hidden_dim=64, out_1=32, out_2=target_dim, gride_size=grid, 
                              head=head,layer_num=num_layers, pooling = pooling)  
        
        elif model_select == 'mlpgat':
            model = MLP_GAT(in_node_dim=encode_dim[0], in_edge_dim=encode_dim[1], hidden_dim=64, out_1=32, out_2=target_dim, gride_size=grid, 
                              head=head,layer_num=num_layers, pooling = pooling)
        
        elif model_select == 'pogat':
            model = PO_GAT(in_node_dim=encode_dim[0], in_edge_dim=encode_dim[1], hidden_dim=64, out_1=32, out_2=target_dim, gride_size=grid, 
                              head=head,layer_num=num_layers, pooling = pooling)  
        
        elif model_select == 'schnet':
            from model.schnet import SchNet
            model = SchNet(num_atoms=100, hidden_channels=64, num_filters=64, 
                          num_interactions=6, cutoff=5.0, num_gaussians=50, 
                          out_dim=target_dim, pooling=pooling)
        
        elif model_select == 'schnet_orbital':
            from model.schnet import OrbitalAwareSchNet
            model = OrbitalAwareSchNet(num_atoms=100, hidden_channels=64, num_filters=64, 
                                     num_interactions=6, cutoff=5.0, num_gaussians=50, 
                                     out_dim=target_dim, pooling=pooling)

        else:
            print('No model can be run!')
        #print(model)head, layer_num, pooling
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        train_loss_dic = {}
        vali_loss_dic = {}

        #model = modeling().to(device)
        model = model.to(device)
        
        if loss_select == 'l1':
            #loss_fn = nn.L1Loss()
            loss_fn = nn.L1Loss(reduction='sum')#sum，mean,none

        elif loss_select == 'l2':
            loss_fn = nn.MSELoss(reduction='none')

        elif loss_select == 'sml1':
            loss_fn = nn.SmoothL1Loss(reduction='sum')#mean,none,sum

        elif loss_select == 'bce':
            loss_fn = nn.BCELoss(reduction='mean')
        

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        best_auc = 0
        
        for epoch in range(NUM_EPOCHS):
            train_loss,vali_loss = train(model, device, loaded_train_loader, loaded_valid_loader, optimizer, epoch + 1, loss_select, model_select)
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            AUC = predicting(model, device, loaded_test_loader, loss_select, model_select, label_mean, label_std)
            
            if AUC > best_auc:
                best_auc = AUC
                logger.info(f'AUC: {best_auc:.5f}')
                formatted_number = "{:.5f}".format(best_auc)
                best_auc = float(formatted_number)
                AUC_list.append(best_auc)

            if epoch % 10 == 0:
                #MAE_list.append(best_MAE)
                print("-------------------------------------------------------")
                print("epoch:",epoch)
                print('best_MAE:', best_auc)
            
            if epoch == NUM_EPOCHS-1:
                print(f"the best result up to {i+1}-loop is {best_auc:.4f}.")
                formatted_number = "{:.5f}".format(best_auc)
                All_AUC.append(best_auc)
    torch.save(model.state_dict(), 'model.pth')
    
    mean_value = statistics.mean(All_AUC)
    std_dev = statistics.stdev(All_AUC)
    print("mean:", mean_value)
    print("std:", std_dev)