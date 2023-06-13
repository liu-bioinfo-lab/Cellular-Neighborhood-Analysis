import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

import torch_geometric.nn as Gnn
import torch_geometric.data as Gdata
from torch.utils.data import DataLoader, Dataset, random_split

import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from sklearn.preprocessing import normalize

import community.community_louvain as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from networkx.algorithms.community.centrality import girvan_newman

import warnings
warnings.filterwarnings('ignore')


## Data Preprocess ##
endocrine_cols = ['α cell', 'β cell' ,'δ cell', 'γ cell', 'epsilon cell']
immune_cols = ['B cell', 'T cell', 'Macrophage', 'Other immune cell']
vascular_cols = ['EC','Pericyte']
miscellaneous = ['miscellaneous']
celltype_cols = endocrine_cols + immune_cols + vascular_cols + miscellaneous
loc_cols = ['XMin', 'XMax', 'YMin', 'YMax']
mid_cols = ['XMid', 'YMid']
label_col = ['T2D_label']

ND_DONORS = ['AFES372', 'ACHM315', 'AFFN281', 'AFG1440', 'AGBA390', 'ABI2259']
T2D_DONORS = ['AEDN413', 'AEJR177', 'ADLE098', 'ABHQ115', 'ABIC495', 'ABIQ254', 'ADIX484', 'AFCM451', 'ACIA085', 'ADBI307']

label_dict = {}
for donor in ND_DONORS:
    label_dict[donor]='ND'
for donor in T2D_DONORS:
    label_dict[donor]='T2D'

def get_type(x, feature_columns=celltype_cols):
    for i in feature_columns:
        if x[i] == 1:
            return i
    return 'NoneType'
 
def preprocess(trg_csv_path:str):
    raw_DF = pd.read_csv(trg_csv_path)
    raw_DF.rename(columns={'ɛ cell': 'epsilon cell'}, inplace=True)
    raw_DF['miscellaneous'] = raw_DF['Arg1 Positive Classification'] | raw_DF['Ki67 Positive Classification']
    raw_DF['T2D_label'] = raw_DF['Group']=='T2D'

    raw_DF['CellType'] = raw_DF.apply(lambda x: get_type(x), axis=1)
    raw_DF['IsletID'] = raw_DF.apply(lambda x: x['Donor'] + '-' + str(x['Islet']), axis=1)
    for islet, islet_indices in raw_DF.groupby('IsletID').groups.items():
        raw_DF.loc[islet_indices, ['XMin', 'XMax']] -= raw_DF.loc[islet_indices, 'XMin'].min()
        raw_DF.loc[islet_indices, ['YMin', 'YMax']] -= raw_DF.loc[islet_indices, 'YMin'].min()

    raw_DF['XMid'] = (raw_DF['XMin'] + raw_DF['XMax'])/2.0
    raw_DF['YMid'] = (raw_DF['YMin'] + raw_DF['YMax'])/2.0

    raw_DF[['IsletID', 'CellType']+mid_cols+celltype_cols]
    return raw_DF


## Read Islet-wise Info ##
def get_category(raw_DF, 
                 feature_cols = celltype_cols, 
                 save_root=None):
    ## Islet-wise info as a Single Graph.
    donors = set(raw_DF['Donor'].values)
    Islet_info = []

    for don in tqdm(donors):
        donor_DF = raw_DF[raw_DF['Donor']==don]
        
        # -------- Islet Level -------- #
        num_islets = 0
        islets = set(donor_DF['Islet'].values)
        for islet in islets:
            islet_DF = donor_DF[donor_DF['Islet']==islet]
            trg_islet_dict = OrderedDict()
            trg_islet_dict['IsletID'] = don+'-'+str(islet)
            trg_islet_dict['label'] = donor_DF['T2D_label'].values[0]

            XMID = np.mean(islet_DF['XMid'].values)
            YMID = np.mean(islet_DF['YMid'].values)

            trg_islet_dict['center_x'] = XMID
            trg_islet_dict['center_y'] = YMID

            DY = np.array(islet_DF['YMid'].values)-YMID
            DX = np.array(islet_DF['XMid'].values)-XMID
            # print(DX.shape, DY.shape)
            dists=np.linalg.norm(np.stack([DX, DY]).T, axis=1)
            dist_u = np.mean(dists)
            dist_sigma = np.std(dists)

            islet_DF['center_distance'] = (dists - dist_u)/(dist_sigma)

            dist = ['center_distance']
            trg_islet_dict['features'] = islet_DF[loc_cols+dist+feature_cols+['IsletID']].reset_index(drop=True)
            Islet_info.append(trg_islet_dict)
            
            if save_root is not None:
                if not os.path.exists(save_root):
                    os.makedirs(save_root, exist_ok=True)
                islet_save_file = don+'-islet-'+str(islet)+donor_DF['Group'].values[0]+'.csv'
                islet_save_path = os.path.join(save_root, islet_save_file)
                islet_DF.to_csv(islet_save_path)
    return Islet_info

## Construct Graph Structure ##
class GraphDataset(Gdata.Dataset):
    def __init__(self, 
                 data_info, 
                 feature_cols, 
                 loc_cols, 
                 weighted=False):
        super().__init__()
        self.weighted = weighted
        # self.num_classes = 2
        self.sizes = []        
        self.data = self._get_data(data_info, feature_cols, loc_cols)
    
    def _get_data(self, data_info, feature_cols, loc_cols, n_edge=None):
        data_entries = []
        
        for idx, data_entry in enumerate(tqdm(data_info)):
            label = torch.tensor(int(data_entry['label'])).long()
            features = torch.from_numpy(np.asarray(data_entry['features'][feature_cols].values)).float()
            
            loc = data_entry['features'][loc_cols].values
            x = np.mean(loc[:, 0:2], axis=-1)
            y = np.mean(loc[:, 2:4], axis=-1)
            
            center = np.stack((x, y)).T
            pos = center

            n = len(data_entry['features'])
            logits = np.linalg.norm(center.reshape(1, n, 2) - center.reshape(n, 1, 2), axis=2)

            if not self.weighted:
                n_edge = min(8, n) * n
                threshold = np.sort(logits.reshape(-1))[min(n_edge+n, len(logits.flatten())-1)]
                adj = (logits <= threshold).astype(float) - np.eye(n)
                edge_index = torch.tensor(np.array(list(adj.nonzero())))
                data = Gdata.Data(x=features, edge_index=edge_index, y=label, loc=pos)
                
            else:
                edge_weights_matrix = np.log2(1.0/(0.005+normalize(logits))) # 0.005: smoothing
                n_edge = min(20, n//5) * n
                threshold = np.sort(logits.reshape(-1))[min(n_edge+n, len(logits.flatten())-1)]
                adj = (logits <= threshold).astype(float) - np.eye(n)
                edge_index = torch.tensor(np.array(list(adj.nonzero())))

                # Edge Weights
                edge_weights = np.zeros(edge_index.shape[1])
                for idx, x in enumerate(edge_index.T):
                    edge_weights[idx] = logits[x[0].item(), x[1].item()]
                edge_weights = torch.tensor(edge_weights, dtype=torch.float)
                data = Gdata.Data(x=features, edge_index=edge_index, y=label, loc=pos, edge_weights=edge_weights)
            
            m = n // 3
            idx_list = list(range(n))
            random.shuffle(idx_list)
            train_ids = idx_list[:m]
            val_ids = idx_list[m:2*m]
            test_ids = idx_list[-m:]
            data.train_mask = torch.zeros(n).to(dtype=torch.bool)
            data.train_mask[train_ids] = True
            data.val_mask = torch.zeros(n).to(dtype=torch.bool)
            data.val_mask[val_ids] = True
            data.test_mask = torch.zeros(n).to(dtype=torch.bool)
            data.test_mask[test_ids] = True
            data.ID = data_entry['IsletID']
            data_entries.append(data)
            self.sizes.append(n)
        return data_entries
            
    def len(self):
        return len(self.data)
    
    def get(self, idx):
        return self.data[idx]
    

## Graph Community partition ##
# compute the best partition with Louvain Modularity Algorithm #
def louvain_community_detection(G, G_pos=None, title=None, weighted=True, fig_dir_path=None, cmap_string='Dark2', edge_coloring=True, show=True):
  plt.figure(figsize=(16, 12))
  partition = community_louvain.best_partition(G, weight='edge_weights', random_state=42, randomize=False)
  if show:
    print(f'#Communities: {len(set(partition.values()))}')

    # draw the graph
    if G_pos is None:
        pos = nx.spring_layout(G)
    else:
        pos = G_pos
    # color the nodes according to their partition
    cmap = cm.get_cmap(cmap_string, max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                            cmap=cmap, node_color=list(partition.values()))
  
    if edge_coloring:
        weights_labels = np.array(list(nx.get_edge_attributes(G,'edge_weights').values()), dtype=float)
        weights = 2*np.log2(weights_labels/np.max(weights_labels))
        nx.draw_networkx_edges(G, pos, alpha=0.7, width=weights, edge_cmap=cmap, edge_color=[list(partition.values())[u] for u, v in G.edges])
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.5)
    if title is not None:
        plt.title(title)
    ax = plt.gca()
    plt.text(0.1, 0.95, f'#Communities: {len(set(partition.values()))}', horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, fontsize=15)
    if fig_dir_path is not None:
        if not os.path.exists(fig_dir_path):
            os.makedirs(fig_dir_path)
        plt.savefig(os.path.join(fig_dir_path, f'{title}.png'))
  return partition

def Graphset_Partition(trg_csv_path, 
                       celltype_cols=celltype_cols, 
                       loc_cols=loc_cols, 
                       weighted=True, 
                       preview=3, 
                       data_dir=None, 
                       fig_dir=None,
                       edge_coloring=True, 
                       cmap_string='Dark2'):
    raw_DF = preprocess(trg_csv_path=trg_csv_path)
    islet_root = os.path.join(data_dir, 'islet_data')
    islet_info = get_category(raw_DF=raw_DF, feature_cols=celltype_cols, save_root=islet_root)
    print('==> Start Graph Construction... 🐳 ')
    graph_set = GraphDataset(islet_info, celltype_cols+['center_distance'], loc_cols, weighted=weighted)
    print('==> Start Graph Partition... 🐳 ')
    for i in tqdm(range(len(graph_set))):
        # print('\n=====================================')
        G = to_networkx(graph_set[i], to_undirected=True, remove_self_loops=True, edge_attrs=['edge_weights'])
        G_pos = graph_set[i].loc
        partition = louvain_community_detection(G, G_pos, title=graph_set[i].ID, fig_dir_path=fig_dir, weighted=True, edge_coloring=True, cmap_string='Dark2', show=(i < preview))
        graph_set[i].communities = np.array(partition.values())
        islet_info[i]['features']['community_idx']=list(partition.values())
    return raw_DF, islet_info, graph_set


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trg_csv_path', type=str, required=True, default='/mlodata1/sfan/CFIDF/data/Cell-ID_by-islet.csv')
parser.add_argument('--data_dir', type=str, required=True, default='/mlodata1/sfan/CFIDF/data')
parser.add_argument('--fig_dir', type=str, required=True, default='/mlodata1/sfan/CFIDF/figs')

if __name__ == '__main__' :
    args = parser.parse_args()
    raw_DF, islet_info, graph_set = Graphset_Partition(trg_csv_path=args.trg_csv_path, 
                                                        celltype_cols=celltype_cols,
                                                        weighted=True,
                                                        data_dir=args.data_dir,
                                                        fig_dir=args.fig_dir,
                                                        preview=3)
