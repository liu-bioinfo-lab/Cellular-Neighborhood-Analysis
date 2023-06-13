from gcd import *
from analysis import *
from metric import *
import warnings
warnings.filterwarnings('ignore')

# ISLET_INFO_PATH = '/content/drive/Shareddrives/2021-CodeX-personal/Voronoi_Analysis/islet_info.pkl'
# COMMUNITY_DF_PATH = '/content/drive/Shareddrives/2021-CodeX-personal/Voronoi_Analysis/communityDF-5.csv'
# K_CENTROIDS_PATH = '/content/drive/Shareddrives/2021-CodeX-personal/Voronoi_Analysis/K-cent.pkl'
# K_BEST = 5
# RESOLUTION = 1.0
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trg_csv_path', type=str, required=True, default='/mlodata1/sfan/CFIDF/data/Cell-ID_by-islet.csv')
parser.add_argument('--data_dir', type=str, required=True, default='/mlodata1/sfan/CFIDF/data')
parser.add_argument('--fig_dir', type=str, required=True, default='/mlodata1/sfan/CFIDF/figs')
parser.add_argument('--n_cluster', type=int, required=False, default=6)
parser.add_argument('--resolution', type=float, required=False, default=0.5)

parser.add_argument('--enrichment_analysis', action='store_true')
parser.add_argument('--diff_analysis', action='store_true')
parser.add_argument('--voronoi_analysis', action='store_true')
parser.add_argument('--cca_analysis', action='store_true')

args = parser.parse_args()
ISLET_INFO_PATH = os.path.join(args.data_dir, 'islet_info.pkl')
COMMUNITY_DF_PATH = os.path.join(args.data_dir, f'communityDF-{args.n_cluster}.csv')
K_CENTROIDS_PATH = os.path.join(args.data_dir, 'K-cent.pkl')

def get_celltype(idx, islet_DF, type_cols):
    if len(islet_DF[type_cols].iloc[idx].values.nonzero()[0])==0:
        return 'NoneType'
    return type_cols[islet_DF[type_cols].iloc[idx].values.nonzero()[0][0]]
  
def make_data():
    raw_DF, islet_info, _ = Graphset_Partition(trg_csv_path=args.trg_csv_path,
                                                        weighted=True,
                                                        data_dir=args.data_dir,
                                                        fig_dir=args.fig_dir,
                                                        preview=3)
    type_cols = [x for x in raw_DF['CellType'].unique() if x !='NoneType']
    community_DF, cfidf_islet_score, niche_clusters = get_CFIDF(islet_info, type_cols, n_clusters=args.n_cluster)

    Community_Reference = {}
    for _,row in community_DF[['IsletID','community_ID', 'Community_Labels']].iterrows():
        isletID, comID, comLabel = row.values
        if isletID not in Community_Reference.keys():
            Community_Reference[isletID] = OrderedDict()
        Community_Reference[isletID][comID] = comLabel 
    
    for islet in tqdm(islet_info):
        islet_id = islet['IsletID']
        islet_DF = islet['features']
        islet_DF.rename(columns={'ɛ cell': 'epsilon cell'}, inplace=True)
        community_label = [Community_Reference[islet_id][str(comID)] for comID in islet_DF['community_idx'].values]
        islet_DF['XMid'] = np.mean(islet_DF[['XMin', 'XMax']], axis=1)
        islet_DF['YMid'] = np.mean(islet_DF[['YMin', 'YMax']], axis=1)
        islet_DF['community_label'] = community_label
        islet_DF['Donor'] = [x.split('-')[0] for x in islet_DF['IsletID']]
        islet_DF['islet-index'] = [x.split('-')[-1] for x in islet_DF['IsletID']]
        islet_DF['celltype'] = [get_celltype(idx, islet_DF, type_cols=type_cols) for idx in range(len(islet_DF))]
    with open(ISLET_INFO_PATH, 'wb') as trg:
        pickle.dump(islet_info, trg)
    community_DF.to_csv(COMMUNITY_DF_PATH, index=False)
    with open(K_CENTROIDS_PATH, 'wb') as trg:
        pickle.dump(niche_clusters, trg)
    return islet_info, community_DF, niche_clusters, cfidf_islet_score
    
def read_data(islet_info_path=ISLET_INFO_PATH, community_DF_path=COMMUNITY_DF_PATH, k_centroids_path = K_CENTROIDS_PATH):
    if os.path.exists(islet_info_path) and os.path.exists(community_DF_path) and os.path.exists(k_centroids_path):
        with open(islet_info_path, 'rb') as trg:
            ISLET_INFO = pickle.load(trg)
        with open(k_centroids_path, 'rb') as trg:
            K_CENTROIDS = pickle.load(trg)
        COMMUNITY_DF = pd.read_csv(community_DF_path)
        type_cols = [x for x in raw_DF['CellType'].unique() if x !='NoneType']
        _, CFIDF_SCORES , _ = get_CFIDF(islet_info=ISLET_INFO, type_cols=type_cols, n_clusters=args.n_cluster)
    else:
        ISLET_INFO, COMMUNITY_DF, K_CENTROIDS, CFIDF_SCORES = make_data()
    return ISLET_INFO, COMMUNITY_DF, K_CENTROIDS, CFIDF_SCORES

if __name__ == '__main__' :
    print('========= Experiment Start =========')
    print('Loading data...')
    raw_DF = preprocess(trg_csv_path=args.trg_csv_path)
    type_cols = [x for x in raw_DF['CellType'].unique() if x !='NoneType']
    ISLET_INFO, COMMUNITY_DF, K_CENTROIDS, CFIDF_SCORES = read_data()
    ND_DONORS = ['AFES372', 'ACHM315', 'AFFN281', 'AFG1440', 'AGBA390', 'ABI2259']
    T2D_DONORS = ['AEDN413', 'AEJR177', 'ADLE098', 'ABHQ115', 'ABIC495', 'ABIQ254', 'ADIX484', 'AFCM451', 'ACIA085', 'ADBI307']
    
    if args.enrichment_analysis:
        enrichment_analysis(n_clusters=args.n_cluster, 
                            niche_clusters=K_CENTROIDS, 
                            type_cols=type_cols, 
                            fig_dir=args.fig_dir,
                            r=args.resolution)
    
    if args.diff_analysis:
        diff_analysis(islet_info=ISLET_INFO, 
                      nd_donors_list=ND_DONORS, 
                      t2d_donors_list=T2D_DONORS, 
                      type_cols=type_cols, 
                      n_cluster=args.n_cluster, 
                      fig_dir=args.fig_dir)
    
    if args.cca_analysis:
        cca_analysis(islet_info=ISLET_INFO, 
                    test_trg_cells=type_cols, 
                    n_cluster=args.n_cluster, 
                    fig_dir=args.fig_dir, 
                    all_cell_DF=None, all_freqs=None)
    
    if args.voronoi_analysis:
        vplot_all(islet_info=ISLET_INFO)
    
    print('Congratulations! 🥦🥬🥒🫑🌽🥕')
    