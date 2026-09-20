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

# from model.ka_gat import KA_GAT
# from model.kagcn import KA_GCN
from model.gat import GAT
from model.sdn import EnhancedOG_PGAT
# from model.mlp_gat import MLP_GAT
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


def get_label():
    return ['label']


def get_tox():
    return ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
           'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']


def get_clintox():
    return ['FDA_APPROVED', 'CT_TOX']


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


def get_muv():
    return ['MUV-466','MUV-548','MUV-600','MUV-644','MUV-652','MUV-689','MUV-692',
            'MUV-712','MUV-713','MUV-733','MUV-737','MUV-810','MUV-832','MUV-846',
            'MUV-852',    'MUV-858','MUV-859']


def ensure_qm7b_graph_csv(
    out_path="data/qm7b_graph.csv",
    qm7_path="data/qm7.csv",
    qm7b_path="data/qm7b.csv",
    target_col="ae-pbe0",
):
    if os.path.isfile(out_path):
        return out_path
    if not os.path.isfile(qm7_path):
        raise FileNotFoundError(f"Need {qm7_path} for SMILES to build QM7b graphs.")
    if not os.path.isfile(qm7b_path):
        raise FileNotFoundError(f"Need {qm7b_path} for QM7b targets.")
    q7 = pd.read_csv(qm7_path)
    qb = pd.read_csv(qm7b_path)
    if target_col not in qb.columns:
        raise KeyError(f"Column {target_col!r} not in {qm7b_path}. Available tail: {list(qb.columns)[-8:]}")
    n = min(len(q7), len(qb))
    if not (qb["molecule_id"].values[:n] == np.arange(n)).all():
        qb = qb.sort_values("molecule_id").reset_index(drop=True)
        n = min(len(q7), len(qb))
    out = pd.DataFrame(
        {
            "smiles": q7["smiles"].values[:n],
            "y": qb[target_col].values[:n].astype(np.float64),
        }
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(
        f"Built {out_path}: {n} molecules (SMILES from qm7.csv, target={target_col} from qm7b.csv)."
    )
    return out_path


def creat_data(
    datafile,
    encoder_atom,
    encoder_bond,
    batch_size,
    train_ratio,
    vali_ratio,
    test_ratio,
    max_molecules=0,
    shuffle_seed=42,
    force_rebuild=False,
):

    datasets = datafile

    directory_path = 'data/processed/'
    target_file_name = datafile +'.pth'

    if is_file_in_directory(directory_path, target_file_name) and (not force_rebuild):

        return True

    else:

        if datasets == "qm7b":
            ensure_qm7b_graph_csv()
            df = pd.read_csv("data/qm7b_graph.csv")
        else:
            df = pd.read_csv("data/" + datasets + ".csv")
        total_before_subset = len(df)
        if max_molecules and int(max_molecules) > 0:
            max_n = min(int(max_molecules), len(df))
            df = df.sample(n=max_n, random_state=shuffle_seed).reset_index(drop=True)
            print(f"Using subset: {max_n}/{total_before_subset} molecules")
        if datasets == 'tox21':
            smiles_list, labels = df['smiles'], df[get_tox()]
            labels = labels.fillna(0)

        if datasets == 'muv':
            smiles_list, labels = df['smiles'], df[get_muv()]
            labels = labels.fillna(0)

        if datasets == 'sider':
            smiles_list, labels = df['smiles'], df[get_sider()]

        if datasets == 'clintox':
            smiles_list, labels = df['smiles'], df[get_clintox()]

        if datasets == 'qm9':
            smiles_list = df['SMILES']
            labels = df[['HOMO', 'LUMO', 'Gap']]

        if datasets == "qm7b":
            smiles_list = df["smiles"]
            labels = df[["y"]]

        if datasets in ['hiv','bbbp','bace']:
            smiles_list, labels = df['smiles'], df[get_label()]

        if 'smiles_list' not in locals():
            raise ValueError(f"数据集 '{datasets}' 未在creat_data函数中定义处理逻辑，请添加相应的处理代码。")

        data_list = []
        feature_sets = ("atomic_number", "basic", "cfid", "cgcnn")
        for i in range(len(smiles_list)):
            if i % 10000 == 0:
                print(i)

            smiles = smiles_list[i]

            Graph_list = path_complex_mol(smiles, encoder_atom, encoder_bond)
            if Graph_list == False:
                continue

            else:
                if has_node_with_zero_in_degree(Graph_list):
                    continue

                else:
                    data_list.append([smiles, torch.tensor(labels.iloc[i]),Graph_list])

        print('Graph list was done!')

        splitter = ScaffoldSplitter().split(data_list, frac_train=train_ratio, frac_valid=vali_ratio, frac_test=test_ratio)

        print('splitter was done!')

        label_mean = None
        label_std = None

        if datasets in ("qm9", "qm7b"):
            train_structs = splitter[0]
            all_train_labels = torch.stack([item[1] for item in train_structs]).float()

            label_mean = all_train_labels.mean(dim=0).numpy()
            label_std = all_train_labels.std(dim=0).numpy()

            print(f"基于训练集计算的标准化参数: mean={label_mean}, std={label_std}")

            def normalize_labels(dataset_list, mean, std):
                mean_t = torch.tensor(mean).float()
                std_t = torch.tensor(std).float()
                normalized_list = []
                for item in dataset_list:
                    normalized_label = (item[1].float() - mean_t) / (std_t + 1e-8)
                    normalized_list.append([item[0], normalized_label, item[2]])
                return normalized_list

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

        if datasets in ("qm9", "qm7b") and label_mean is not None:
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

        if model_select in ('sdn', 'schnet', 'schnet_orbital', 'dimenet_pp', 'orbitnet', 'mace', 'cmole'):
            graph_list = data[1]
        else:
            graph_list = data[1].to(device)
        node_features = graph_list.ndata['feat'].to(device)
        edge_features = graph_list.edata['feat'].to(device)

        if model_select in ('sdn', 'schnet', 'schnet_orbital', 'dimenet_pp', 'orbitnet', 'mace', 'cmole'):
            from torch_geometric.data import Data, Batch

            pyg_data_list = []
            node_start = 0
            edge_start = 0

            for i in range(graph_list.batch_size):
                num_nodes = graph_list.batch_num_nodes()[i].item()
                num_edges = graph_list.batch_num_edges()[i].item()

                graph_node_features = node_features[node_start:node_start + num_nodes]
                graph_edge_features = edge_features[edge_start:edge_start + num_edges]

                src, dst = graph_list.edges()
                graph_src = src[edge_start:edge_start + num_edges] - node_start
                graph_dst = dst[edge_start:edge_start + num_edges] - node_start
                graph_edges = torch.stack([graph_src, graph_dst], dim=0)

                pyg_kw = dict(
                    x=graph_node_features,
                    edge_index=graph_edges,
                    edge_attr=graph_edge_features,
                    y=y[i].unsqueeze(0).float()
                )
                if "coor" in graph_list.ndata:
                    pyg_kw["pos"] = graph_list.ndata["coor"][node_start:node_start + num_nodes].float()
                if "z" in graph_list.ndata:
                    pyg_kw["z"] = graph_list.ndata["z"][node_start:node_start + num_nodes].long()
                pyg_data = Data(**pyg_kw)
                pyg_data_list.append(pyg_data)

                node_start += num_nodes
                edge_start += num_edges

            pyg_batch = Batch.from_data_list(pyg_data_list).to(device)
            model_output = model(pyg_batch, return_intermediate=True)
            if isinstance(model_output, dict):
                output = model_output['prediction'].cpu()
            else:
                output = model_output.cpu()
        else:
            output = model(graph_list, node_features, edge_features).cpu()

        is_regression = loss_select in ['l1', 'l2', 'sml1']
        if is_regression:
            arr_label = y.float().cpu().flatten()
            arr_pred = output.float().cpu().flatten()
        else:
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

        if not is_regression:
            arr_pred = torch.sigmoid(arr_pred)

        if loss_select == 'l1':
            loss = nn.L1Loss(reduction='mean')(arr_pred, arr_label)
        elif loss_select == 'l2':
            loss = nn.MSELoss(reduction='mean')(arr_pred, arr_label)
        elif loss_select == 'sml1':
            loss = nn.SmoothL1Loss(reduction='mean')(arr_pred, arr_label)
        elif loss_select == 'bce':
            loss = nn.BCELoss(reduction='mean')(arr_pred.to(device), arr_label.to(device))
        else:
            loss = nn.BCELoss(reduction='mean')(arr_pred.to(device), arr_label.to(device))

        total_train_loss = total_train_loss + loss.item()
        train_num += 1
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_loss_val = 0.0

    for batch_idx, valid_data in enumerate(valid_loader):

        y = valid_data[0]
        if model_select in ('sdn', 'schnet', 'schnet_orbital', 'dimenet_pp', 'orbitnet', 'mace', 'cmole'):
            graph_list = valid_data[1]
        else:
            graph_list = valid_data[1].to(device)
        node_features = graph_list.ndata['feat'].to(device)
        edge_features = graph_list.edata['feat'].to(device)
        if model_select in ('sdn', 'schnet', 'schnet_orbital', 'dimenet_pp', 'orbitnet', 'mace', 'cmole'):
            from torch_geometric.data import Data, Batch
            pyg_data_list = []
            node_start = 0
            edge_start = 0
            for i in range(graph_list.batch_size):
                num_nodes = graph_list.batch_num_nodes()[i].item()
                num_edges = graph_list.batch_num_edges()[i].item()
                graph_node_features = node_features[node_start:node_start + num_nodes]
                graph_edge_features = edge_features[edge_start:edge_start + num_edges]
                src, dst = graph_list.edges()
                graph_src = src[edge_start:edge_start + num_edges] - node_start
                graph_dst = dst[edge_start:edge_start + num_edges] - node_start
                graph_edges = torch.stack([graph_src, graph_dst], dim=0)
                pyg_kw = dict(x=graph_node_features, edge_index=graph_edges, edge_attr=graph_edge_features, y=y[i].unsqueeze(0).float())
                if "coor" in graph_list.ndata:
                    pyg_kw["pos"] = graph_list.ndata["coor"][node_start:node_start + num_nodes].float()
                if "z" in graph_list.ndata:
                    pyg_kw["z"] = graph_list.ndata["z"][node_start:node_start + num_nodes].long()
                pyg_data_list.append(Data(**pyg_kw))
                node_start += num_nodes
                edge_start += num_edges
            pyg_batch = Batch.from_data_list(pyg_data_list).to(device)
            model_output = model(pyg_batch, return_intermediate=True)
            if isinstance(model_output, dict):
                output = model_output['prediction'].cpu()
            else:
                output = model_output.cpu()
        else:
            output = model(graph_list, node_features, edge_features).cpu()

        is_regression = loss_select in ['l1', 'l2', 'sml1']
        if is_regression:
            arr_label = y.float().cpu().flatten()
            arr_pred = output.float().cpu().flatten()
        else:
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
        if not is_regression:
            arr_pred = torch.sigmoid(arr_pred)

        if loss_select == 'l1':
            loss = nn.L1Loss(reduction='mean')(arr_pred, arr_label)
        elif loss_select == 'l2':
            loss = nn.MSELoss(reduction='mean')(arr_pred, arr_label)
        elif loss_select == 'sml1':
            loss = nn.SmoothL1Loss(reduction='mean')(arr_pred, arr_label)
        elif loss_select == 'bce':
            loss = nn.BCELoss(reduction='mean')(arr_pred.to(device), arr_label.to(device))
        else:
            loss = nn.BCELoss(reduction='mean')(arr_pred.to(device), arr_label.to(device))

        total_loss_val += loss.item()

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

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

            if model_select in ('sdn', 'schnet', 'schnet_orbital', 'dimenet_pp', 'orbitnet', 'mace', 'cmole'):
                graph_list = data[1]
            else:
                graph_list = data[1].to(device)
            node_features = graph_list.ndata['feat'].to(device)
            edege_features = graph_list.edata['feat'].to(device)
            if model_select in ('sdn', 'schnet', 'schnet_orbital', 'dimenet_pp', 'orbitnet', 'mace', 'cmole'):
                from torch_geometric.data import Data, Batch
                pyg_data_list = []
                node_start = 0
                edge_start = 0
                for i in range(graph_list.batch_size):
                    num_nodes = graph_list.batch_num_nodes()[i].item()
                    num_edges = graph_list.batch_num_edges()[i].item()
                    graph_node_features = node_features[node_start:node_start + num_nodes]
                    graph_edge_features = edege_features[edge_start:edge_start + num_edges]
                    src, dst = graph_list.edges()
                    graph_src = src[edge_start:edge_start + num_edges] - node_start
                    graph_dst = dst[edge_start:edge_start + num_edges] - node_start
                    graph_edges = torch.stack([graph_src, graph_dst], dim=0)
                    pyg_kw = dict(x=graph_node_features, edge_index=graph_edges, edge_attr=graph_edge_features, y=y[i].unsqueeze(0).float())
                    if "coor" in graph_list.ndata:
                        pyg_kw["pos"] = graph_list.ndata["coor"][node_start:node_start + num_nodes].float()
                    if "z" in graph_list.ndata:
                        pyg_kw["z"] = graph_list.ndata["z"][node_start:node_start + num_nodes].long()
                    pyg_data_list.append(Data(**pyg_kw))
                    node_start += num_nodes
                    edge_start += num_edges
                pyg_batch = Batch.from_data_list(pyg_data_list).to(device)
                model_output = model(pyg_batch, return_intermediate=True)
                if isinstance(model_output, dict):
                    output = model_output['prediction'].cpu()
                else:
                    output = model_output.cpu()
            else:
                output = model(graph_list, node_features, edege_features).cpu()

            is_regression = loss_select in ['l1', 'l2', 'sml1']
            if is_regression:
                arr_label = y.float().cpu().flatten()
                arr_pred = output.float().cpu().flatten()
            else:
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
            if not is_regression:
                arr_pred = torch.sigmoid(arr_pred)
            total_preds = torch.cat((total_preds, arr_pred), 0)
            total_labels = torch.cat((total_labels, arr_label), 0)

    is_regression = loss_select in ['l1', 'l2', 'sml1']
    if is_regression:
        from sklearn.metrics import mean_absolute_error

        if label_mean is not None and label_std is not None:
            preds_flat = total_preds.numpy().flatten()
            labels_flat = total_labels.numpy().flatten()

            if isinstance(label_mean, np.ndarray) and len(label_mean) > 1:
                num_samples = len(preds_flat) // len(label_mean)
                preds_reshaped = preds_flat.reshape(num_samples, len(label_mean))
                labels_reshaped = labels_flat.reshape(num_samples, len(label_mean))

                preds_denorm = preds_reshaped * label_std + label_mean
                labels_denorm = labels_reshaped * label_std + label_mean

                preds_denorm = preds_denorm.flatten()
                labels_denorm = labels_denorm.flatten()
            else:
                mean_val = label_mean[0] if isinstance(label_mean, np.ndarray) else label_mean
                std_val = label_std[0] if isinstance(label_std, np.ndarray) else label_std
                preds_denorm = preds_flat * std_val + mean_val
                labels_denorm = labels_flat * std_val + mean_val

            mae = mean_absolute_error(labels_denorm, preds_denorm)
            print(f"MAE: {mae:.4f}")
        else:
            mae = mean_absolute_error(total_labels.numpy().flatten(), total_preds.numpy().flatten())
            print(f"MAE: {mae:.4f}")
        return mae
    else:
        AUC = roc_auc_score(total_labels.numpy().flatten(), total_preds.numpy().flatten())
        return AUC


def parse_arguments():
    parser = argparse.ArgumentParser(description="help")

    parser.add_argument(
        "--config",
        type=str,
        default="./config/gat_path.yaml",
        help="YAML config (e.g. config/gat_path_qm7b.yaml for QM7b + OrbitNet)",
    )
    parser.add_argument(
        "--pretrain_ckpt",
        type=str,
        default=None,
        help="Optional checkpoint: SDN loads backbone only (classifier skipped); "
        "SchNet loads embedding+interactions, re-inits lin1/lin2 for current out_dim.",
    )

    args = parser.parse_args()
    config_path = args.config
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        for key, value in config.items():
            setattr(args, key, value)

    return args


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
        torch.cuda.empty_cache()
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
    target_map = {
        "tox21": 12,
        "muv": 17,
        "sider": 27,
        "clintox": 2,
        "bace": 1,
        "bbbp": 1,
        "hiv": 1,
        "qm9": 3,
        "qm7b": 1,
    }
    if datafile not in target_map:
        raise KeyError(f"Unknown select_dataset={datafile!r}. Add it to target_map or fix config.")
    target_dim = target_map[datafile]

    encoder_atom = args.encoder_atom
    encoder_bond = args.encoder_bond

    encode_dim = [0,0]
    encode_dim[0] = 92
    encode_dim[1] = 21

    max_molecules = int(getattr(args, "max_molecules", 0) or 0)
    force_rebuild_data = bool(getattr(args, "force_rebuild_data", False))
    creat_data(
        datafile,
        encoder_atom,
        encoder_bond,
        batch_size,
        train_ratio,
        vali_ratio,
        test_ratio,
        max_molecules=max_molecules,
        shuffle_seed=42,
        force_rebuild=force_rebuild_data,
    )

    model_select = str(args.model_select).lower().replace("-", "_")
    if model_select == "ka_gat":
        model_select = "kagat"
    if model_select in ("ka_gcn", "ka-gcn"):
        model_select = "kagcn"
    loss_select = args.loss_select

    state = torch.load('data/processed/'+datafile+'.pth')

    label_mean = state.get('label_mean', None)
    label_std = state.get('label_std', None)
    if label_mean is not None and label_std is not None:
        if isinstance(label_mean, torch.Tensor):
            label_mean = label_mean.numpy()
        if isinstance(label_std, torch.Tensor):
            label_std = label_std.numpy()
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
    if model_select in ('sdn', 'schnet', 'schnet_orbital', 'dimenet_pp', 'orbitnet', 'mace', 'cmole'):
        def count_with_coor(graph_list):
            n = len(graph_list)
            with_coor = sum(1 for g in graph_list if isinstance(g, dgl.DGLGraph) and 'coor' in g.ndata)
            return with_coor, n
        tr_c, tr_n = count_with_coor(state['train_graph_list'])
        va_c, va_n = count_with_coor(state['valid_graph_list'])
        te_c, te_n = count_with_coor(state['test_graph_list'])
        if tr_n > 0:
            print(f"3D check: train {tr_c}/{tr_n}, valid {va_c}/{va_n}, test {te_c}/{te_n} graphs have 'coor'.")
            if tr_c == tr_n and te_c == te_n:
                print("  -> Training and evaluation WILL use 3D (pos).")
            elif tr_c == 0:
                print("  -> WARNING: No graphs have 'coor'. Model will NOT use 3D; re-run creat_data to build graphs with 3D.")
            else:
                print("  -> WARNING: Some graphs missing 'coor'. Re-process data with path_complex_mol that embeds 3D.")

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
            _kagat_sigmoid = loss_select not in ['l1', 'l2', 'sml1']
            model = KA_GAT(
                in_node_dim=encode_dim[0],
                in_edge_dim=encode_dim[1],
                hidden_dim=64,
                out_1=32,
                out_2=target_dim,
                gride_size=grid,
                head=head,
                layer_num=num_layers,
                pooling=pooling,
                sigmoid_readout=_kagat_sigmoid,
            )
        elif model_select == 'kagcn':
            _kagcn_sigmoid = loss_select not in ['l1', 'l2', 'sml1']
            model = KA_GCN(
                in_node_dim=encode_dim[0],
                in_edge_dim=encode_dim[1],
                hidden_dim=64,
                out_1=32,
                out_2=target_dim,
                gride_size=grid,
                head=head,
                layer_num=num_layers,
                pooling=pooling,
                sigmoid_readout=_kagcn_sigmoid,
            )
        elif model_select == 'gat':
            model = GAT(in_node_dim=encode_dim[0], in_edge_dim=encode_dim[1], hidden_dim=64, out_1=32, out_2=target_dim,
                        gride_size=grid, head=head, layer_num=num_layers, pooling=pooling)

        elif model_select == 'sdn':
            model = EnhancedOG_PGAT(
                in_node_dim=encode_dim[0],
                in_edge_dim=encode_dim[1],
                hidden_dim=64,
                out_1=32,
                out_2=target_dim,
                num_layers=num_layers,
            )

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
                          out_dim=target_dim, pooling=pooling, in_node_dim=encode_dim[0])
        elif model_select == 'schnet_orbital':
            from model.schnet import OrbitalAwareSchNet
            model = OrbitalAwareSchNet(num_atoms=100, hidden_channels=64, num_filters=64,
                                      num_interactions=6, cutoff=5.0, num_gaussians=50,
                                      out_dim=target_dim, pooling=pooling, in_node_dim=encode_dim[0])
        elif model_select == 'dimenet_pp':
            from model.dimenet_pp import DimeNetPlusPlus
            model = DimeNetPlusPlus(in_node_dim=encode_dim[0], hidden_channels=128, out_channels=target_dim,
                                   num_blocks=4, num_radial=50, cutoff=5.0)
        elif model_select == 'orbitnet':
            from model.orbitnet import OrbitNet
            model = OrbitNet(in_node_dim=encode_dim[0], hidden_channels=64, num_layers=4,
                            num_rbf=50, cutoff=5.0, out_dim=target_dim, dropout=0.1)
        elif model_select == 'mace':
            from model.mace import build_mace_for_ogqimp

            model = build_mace_for_ogqimp(
                target_dim=target_dim,
                hidden=128,
                dropout=0.25,
            )
        elif model_select == 'cmole':
            from model.cmole import EnhancedOG_PGAT as CMOLE

            model = CMOLE(
                input_dim=encode_dim[0],
                hidden=128,
                n_layers=num_layers,
                dropout=0.25,
                num_tasks=target_dim,
                use_equivariant=True,
                use_se3=True,
            )

        else:
            raise RuntimeError(
                f"Unknown model_select={model_select!r}. Check config/gat_path.yaml (e.g. sdn, kagat, kagcn, schnet, mace, cmole)."
            )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        train_loss_dic = {}
        vali_loss_dic = {}

        model = model.to(device)

        pretrain_path = getattr(args, "pretrain_ckpt", None)
        if pretrain_path and str(pretrain_path).strip():
            pretrain_path = os.path.abspath(str(pretrain_path).strip())
            if model_select not in ("sdn", "schnet", "schnet_orbital"):
                print(
                    f"[pretrain_ckpt] ignored (supported for sdn / schnet / schnet_orbital, got {model_select})."
                )
            elif not os.path.isfile(pretrain_path):
                print(f"[pretrain_ckpt] file not found: {pretrain_path}")
            elif model_select == "sdn":
                from utils.pretrain_backbone import load_pretrained_backbone

                print(f"[pretrain_ckpt] loading SDN backbone from {pretrain_path}")
                load_pretrained_backbone(pretrain_path, model, map_location=device, verbose=True)
            else:
                from utils.pretrain_backbone import load_pretrained_schnet_backbone

                print(f"[pretrain_ckpt] loading SchNet backbone from {pretrain_path}")
                load_pretrained_schnet_backbone(pretrain_path, model, map_location=device, verbose=True)

        if loss_select == 'l1':
            loss_fn = nn.L1Loss(reduction='sum')

        elif loss_select == 'l2':
            loss_fn = nn.MSELoss(reduction='none')

        elif loss_select == 'sml1':
            loss_fn = nn.SmoothL1Loss(reduction='sum')

        elif loss_select == 'bce':
            loss_fn = nn.BCELoss(reduction='mean')

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        is_regression = loss_select in ['l1', 'l2', 'sml1']
        best_auc = float('inf') if is_regression else 0.0

        for epoch in range(NUM_EPOCHS):
            train_loss,vali_loss = train(model, device, loaded_train_loader, loaded_valid_loader, optimizer, epoch + 1, loss_select, model_select)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            AUC = predicting(model, device, loaded_test_loader, loss_select, model_select, label_mean, label_std)

            improved = (AUC < best_auc) if is_regression else (AUC > best_auc)
            if improved:
                best_auc = AUC
                metric_name = 'MAE' if is_regression else 'AUC'
                logger.info(f'{metric_name}: {best_auc:.5f}')
                formatted_number = "{:.5f}".format(best_auc)
                best_auc = float(formatted_number)
                AUC_list.append(best_auc)

            if epoch % 10 == 0:
                print("-------------------------------------------------------")
                print("epoch:",epoch)
                if is_regression:
                    print('best_MAE:', best_auc)
                else:
                    print('best_AUC:', best_auc)

            if epoch == NUM_EPOCHS-1:
                print(f"the best result up to {i+1}-loop is {best_auc:.4f}.")
                formatted_number = "{:.5f}".format(best_auc)
                All_AUC.append(best_auc)

        torch.save(model.state_dict(), 'model_{}_test.pth'.format(args.select_dataset))

    mean_value = statistics.mean(All_AUC)
    std_dev = statistics.stdev(All_AUC) if len(All_AUC) >= 2 else 0.0
    print("mean:", mean_value)
    print("std:", std_dev)
