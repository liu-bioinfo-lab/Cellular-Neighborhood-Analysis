import os
import json
import pickle
import numpy as np
import pandas as pd
from analysis import kmeans_centroids

def get_freq(x,y):
    if y==0:
        return list(np.zeros_like(x))
    return list(x/y)

def CFIDF(num_celltype_per_communiy, 
          num_cell_per_communiy, 
          sum_celltype_all_communiy, 
          sum_cell_all_communiy, ):
    CF = np.array([get_freq(x, y) for x, y in zip(num_celltype_per_communiy,num_cell_per_communiy)], dtype=float)
    IDF = np.log(np.array(sum_cell_all_communiy/sum_celltype_all_communiy, dtype=float))
    CFIDF_score = CF*IDF
    return CFIDF_score

def get_CFIDF(islet_info, type_cols, n_clusters=6):
    community_list = []

    for i in range(len(islet_info)):
        communities = islet_info[i]['features'].groupby(by='community_idx')
        for key in set(islet_info[i]['features']['community_idx'].values):
            community_name = islet_info[i]['IsletID']+'-'+str(key)
            community_celltype_series = communities.get_group(key)[type_cols].sum()
            community_celltype_series['community_name']=community_name
            community_celltype_series['Donor']=islet_info[i]['IsletID'].split('-')[0]
            community_list.append(community_celltype_series)
    
    community_DF = pd.concat(community_list, axis=1).transpose()
    community_DF['IsletID'] = ['-'.join(x.split('-')[:2]) for x in community_DF['community_name']]
    community_DF['community_ID'] = [x.split('-')[-1] for x in community_DF['community_name']]
    community_DF.rename(columns={'ɛ cell': 'epsilon cell'}, inplace=True)
    
    num_celltype_per_communiy = community_DF[type_cols].values
    num_cell_per_communiy = community_DF[type_cols].sum(axis=1).values
    sum_celltype_all_communiy = community_DF[type_cols].values.sum(axis=0)
    sum_cell_all_communiy = community_DF[type_cols].values.sum()
    cfidf_islet_score = CFIDF(num_celltype_per_communiy, 
                            num_cell_per_communiy, 
                            sum_celltype_all_communiy, 
                            sum_cell_all_communiy, )
    community_labels, niche_clusters = kmeans_centroids(n_clusters=n_clusters, scores=cfidf_islet_score)
    community_DF['Community_Labels'] = list(community_labels)
    return community_DF, cfidf_islet_score, niche_clusters


# type_cols = [x for x in raw_DF['CellType'].unique() if x !='NoneType']