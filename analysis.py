import os
import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from shapely.geometry import MultiPoint, Point, Polygon

from scipy.spatial import Voronoi
from scipy.stats import pearsonr,spearmanr
import itertools

from sklearn.cross_decomposition import CCA
import networkx as nx

import statsmodels.api as sm

## Elbow filtering: for selecting the proper num_clusters (K) for k_means clustering
def Elbow_filter(X, y=None, k_start=1, k_end=20, metric='distortion'):
    tag_model = KMeans(init='k-means++')
    visualizer = KElbowVisualizer(tag_model, k=range(k_start, k_end+1), metric=metric, timings=False)

    visualizer.fit(X)
    visualizer.show()
    
def kmeans_elbow_analysis(CFIDF_score, k_end=10):
    Elbow_filter(X=CFIDF_score, k_start=1, k_end=k_end, metric='distortion')
    # Elbow_filter(X=CFIDF_score, k_start=2, k_end=k_end, metric='calinski_harabasz')
    Elbow_filter(X=CFIDF_score, k_start=2, k_end=k_end, metric = 'silhouette')
    
def kmeans_centroids(n_clusters, scores):
    km = KMeans(n_clusters=n_clusters, random_state=0)
    community_labels = km.fit_predict(scores)
    community_centers = list(km.cluster_centers_)
    return community_labels, community_centers


def enrichment_analysis(n_clusters, 
                        niche_clusters, 
                        type_cols, 
                        fig_dir=None,
                        r=0.5):
    fc = pd.DataFrame(niche_clusters, columns=type_cols)
    fc.rename(columns={'ɛ cell': 'epsilon cell'}, inplace=True)
    if 'epsilon cell' not in type_cols:
        type_cols[type_cols.index('ɛ cell')]='epsilon cell'
    mean_freq = np.mean(fc.loc[[i for i in range(n_clusters)], type_cols].mean().values)

    g = sns.clustermap(fc.loc[[i for i in range(n_clusters)], type_cols], center=mean_freq, cmap="vlag", row_cluster=False,
                    dendrogram_ratio=(.1, .2),
                    cbar_pos=(.02, .32, .03, .2),
                    linewidths=.75, figsize=(20, 10))

    g.ax_row_dendrogram.remove()
    plt.title(f'Louvian Community(k={n_clusters})', fontsize=15)
    if fig_dir is not None:
        save_path = os.path.join(fig_dir, f'enrichment_k={n_clusters}_r={r}.png')
        g.savefig(save_path)
        print(f'Fig(enrichment analysis) saved to {save_path}! 🍓')


## voronoi analysis
def get_color(i,scatter_palette = sns.color_palette('bright')):
    if int(i)<10:
        return scatter_palette[int(i)]
    return 'black'
        
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    adapted from https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647 3.18.2019
    
    
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def plot_voronoi(points,colors,invert_y = True,edge_color = 'facecolor',line_width = .1,alpha = 1,size_max=np.inf):

    if invert_y:
        points[:,1] = max(points[:,1])-points[:,1]
    vor = Voronoi(points)

    regions, vertices = voronoi_finite_polygons_2d(vor)

    pts = MultiPoint([Point(i) for i in points])
    mask = pts.convex_hull
    new_vertices = []
    if type(alpha)!=list:
        alpha = [alpha]*len(points)
    areas = []
    for i,(region,alph) in enumerate(zip(regions,alpha)):
        polygon = vertices[region]
        shape = list(polygon.shape)
        shape[0] += 1
        p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
        areas+=[p.area]
        if p.area <size_max:
            poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
            new_vertices.append(poly)
            if edge_color == 'facecolor':
                plt.fill(*zip(*poly), alpha=alph,edgecolor=  colors[i],linewidth = line_width , facecolor = colors[i])
            else:
                plt.fill(*zip(*poly), alpha=alph,edgecolor=  edge_color,linewidth = line_width, facecolor = colors[i])
        # else:

        #     plt.scatter(np.mean(p.boundary.xy[0]),np.mean(p.boundary.xy[1]),c = colors[i])
    return areas

def draw_voronoi_scatter(spot,c,voronoi_palette = sns.color_palette('rocket_r'),scatter_palette = sns.color_palette('bright'),X = 'X:X', Y = 'Y:Y',voronoi_hue = 'neighborhood10',scatter_hue = 'ClusterName',
        figsize = (8,8),
         voronoi_kwargs = {},
         scatter_kwargs = {}, fname='', title=None):

    '''
    plot voronoi of a region and overlay the location of specific cell types onto this
    
    spot:  cells that are used for voronoi diagram
    c:  cells that are plotted over voronoi
    palette:  color palette used for coloring neighborhoods
    X/Y:  column name used for X/Y locations
    hue:  column name used for neighborhood allocation
    figsize:  size of figure
    voronoi_kwargs:  arguments passed to plot_vornoi function
    scatter_kwargs:  arguments passed to plt.scatter()

    returns sizes of each voronoi to make it easier to pick a size_max argument if necessary
    '''
    if len(c)>0:
        neigh_alpha = .3
    else:
        neigh_alpha = .3
        
    voronoi_kwargs = {**{'alpha':neigh_alpha},**voronoi_kwargs}
    scatter_kwargs = {**{'s':50,'alpha':1,'marker':'.'},**scatter_kwargs}
    
    plt.figure(figsize = figsize)
    ax = plt.gca()
    colors  = [voronoi_palette[i] for i in spot[voronoi_hue]]
    a = plot_voronoi(spot[[X,Y]].values,
                 colors,#[{0:'white',1:'red',2:'purple'}[i] for i in spot['color']],
                     edge_color = 'black', line_width = .1,
                 **voronoi_kwargs)
    
    if len(c)>0:
        if 'c' not in scatter_kwargs:
            colors  = [get_color(i) for i in c[scatter_hue]]
            scatter_kwargs['c'] = colors
            
        plt.scatter(x = c[X],y = (max(spot[Y])-c[Y].values),
                  **scatter_kwargs
                   )
    plt.axis('on')
    
    legend1 = plt.legend(handles=[mpatches.Patch(color=voronoi_palette[0], alpha=0.3, label='0'),
                                  mpatches.Patch(color=voronoi_palette[1], alpha=0.3, label='1'),
                                  mpatches.Patch(color=voronoi_palette[2], alpha=0.3, label='2'),
                                  mpatches.Patch(color=voronoi_palette[3], alpha=0.3, label='3'),
                                  mpatches.Patch(color=voronoi_palette[4], alpha=0.3, label='4')
                                 ],
                         title='CN', bbox_to_anchor=(1.20, 0), loc='lower right', frameon=False)
    ax.add_artist(legend1)
    legend2 = plt.legend(handles=[plt.plot([],[], marker='o', ms=10, ls='', color=scatter_palette[0], label='α cell')[0],
                                  plt.plot([],[], marker='o', ms=10, ls='', color=scatter_palette[1], label='β cell')[0],
                                  plt.plot([],[], marker='o', ms=10, ls='', color=scatter_palette[2], label='δ cell')[0],
                                  plt.plot([],[], marker='o', ms=10, ls='', color=scatter_palette[3], label='γ cell')[0],
                                  plt.plot([],[], marker='o', ms=10, ls='', color=scatter_palette[4], label='epsilon cell')[0],
                                  plt.plot([],[], marker='o', ms=10, ls='', color=scatter_palette[5], label='T cell')[0],
                                  plt.plot([],[], marker='o', ms=10, ls='', color=scatter_palette[6], label='Macrophage')[0],
                                  plt.plot([],[], marker='o', ms=10, ls='', color=scatter_palette[7], label='miscellaneous')[0],
                                  plt.plot([],[], marker='o', ms=10, ls='', color=scatter_palette[8], label='Other immune cell')[0],
                                  plt.plot([],[], marker='o', ms=10, ls='', color=scatter_palette[9], label='EC')[0],
                                  plt.plot([],[], marker='o', ms=10, ls='', color='black', label='Pericyte')[0]],
                         title='CT', bbox_to_anchor=(1.35, 1), loc='upper right', frameon=False)
    if title is not None:
        plt.title(title)
    plt.savefig(fname, dpi=600, bbox_inches='tight')
    return a

def vplot_cell(islet_entry, cell_type, fig_dir):
    islet_DF = islet_entry['features']
    islet_name = islet_entry['IsletID']
    spot = islet_DF
    cd4s = spot[spot[cell_type]==1]
    if fig_dir is not None:
        save_dir = os.join(fig_dir, f'/Voronoi_Analysis/v_cell_plot/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.join(save_dir, f'{islet_name+"-"+cell_type}.png')
    _ = draw_voronoi_scatter(spot, cd4s, X='XMid', Y='YMid', voronoi_hue='community_label', scatter_hue='ColorIndex', fname=save_path)


def vplot_islet(islet_entry, fig_dir):
    islet_DF = islet_entry['features']
    islet_name = islet_entry['IsletID']
    spot = islet_DF
    if fig_dir is not None:
        save_dir = os.join(fig_dir, f'/Voronoi_Analysis/v_islet_plot/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.join(save_dir, f'{islet_name}.png')
    _ = draw_voronoi_scatter(spot, spot, X='XMid', Y='YMid', voronoi_hue='community_label', scatter_hue='ColorIndex', fname=save_path, title=islet_entry['IsletID'])

def vplot_all(islet_info, fig_dir=None):
    color_dict = {'α cell': 0, 'β cell': 1, 'δ cell': 2, 'γ cell': 3, 'epsilon cell': 4, 'T cell': 5, 'Macrophage': 6, 'miscellaneous': 7, 'Other immune cell': 8, 'EC': 9, 'Pericyte': 10}
    if fig_dir is not None:
        save_dir = os.join(fig_dir, f'/Voronoi_Analysis/v_islet_plot/')
    for islet_entry in tqdm(islet_info):
        islet_entry['features'] = islet_entry['features'][islet_entry['features']['celltype']!='NoneType']
        islet_entry['features']['ColorIndex'] = islet_entry['features']['celltype'].apply(lambda x: color_dict[x])
        print('Name:', islet_entry['IsletID'])
        try:
            vplot_islet(islet_entry, fig_dir=save_dir)
            print(f'Fig(voronoi analysis) saved to {save_dir}! 🥑')
        except:
            continue
    

## celltype differential analysis
def normalize(X):
    arr = np.array(X.fillna(0).values)
    return pd.DataFrame(np.log2(1e-3 + arr/arr.sum(axis =1, keepdims = True)), index = X.index.values, columns = X.columns).fillna(0)

def diff_analysis(islet_info, nd_donors_list, t2d_donors_list, type_cols, n_cluster, fig_dir=None):
    t2d_feats_list, nd_feats_list, all_feats_list = [], [], []
    for islet in islet_info:
        if islet['IsletID'].split('-')[0] in nd_donors_list:
            nd_feats_list.append(islet['features'])
        else:
            t2d_feats_list.append(islet['features'])
        all_feats_list.append(islet['features'])
    
    all_cell_DF = pd.concat(all_feats_list).reset_index(drop=True)
    # t2d_cell_DF = pd.concat(t2d_feats_list).reset_index(drop=True)
    # nd_cell_DF = pd.concat(nd_feats_list).reset_index(drop=True)
    ct_freq = all_cell_DF.groupby('Donor').apply(lambda x: x['celltype'].value_counts()).unstack().reset_index().fillna(0)
    all_freqs = all_cell_DF.groupby(['Donor', 'community_label']).apply(lambda x: x['celltype'].value_counts()).unstack().reset_index().fillna(0)
    all_donors = t2d_donors_list+nd_donors_list
    donor_labels = pd.Series([1]*len(t2d_donors_list)+[0]*len(nd_donors_list))
    X_cts = normalize(ct_freq.set_index('Donor').loc[all_donors,type_cols])

    # normalized neighborhood specific cell type frequencies
    df_list = []

    for nb in range(n_cluster):
        cond_nb = all_freqs.loc[all_freqs['community_label']==nb,['Donor'] + type_cols].rename({col: col+'_'+str(nb) for col in type_cols},axis = 1).set_index('Donor')
        df_list.append(normalize(cond_nb))

    X_cond_nb = pd.concat(df_list,axis = 1).loc[all_donors].fillna(0)
    
    #differential enrichment for all cell subsets
    changes = {}
    for col in type_cols:
        for nb in range(n_cluster):
            #build a design matrix with a constant, group 0 or 1 and the overall frequencies
            X = pd.concat([X_cts.reset_index()[col], donor_labels.astype('int'),pd.Series(np.ones(len(donor_labels)), index = donor_labels.index.values)],axis = 1).values
            if col+'_%d'%nb in X_cond_nb.columns:
                #set the neighborhood specific ct freqs as the outcome
                Y = X_cond_nb[col+'_%d'%nb].values
                X = X[~pd.isna(Y)]
                Y = Y[~pd.isna(Y)]
                #fit a linear regression model
                results = sm.OLS(Y,X).fit()
                #find the params and pvalues for the group coefficient
                changes[(col,nb)] = (results.pvalues[1], results.params[1])

    #make a dataframe with coeffs and pvalues
    dat = (pd.DataFrame(changes).loc[1].unstack())
    dat = pd.DataFrame(np.nan_to_num(dat.values),index = dat.index, columns = dat.columns).T.sort_index(ascending=True).loc[:,X_cts.columns]
    pvals = (pd.DataFrame(changes).loc[0].unstack()).T.sort_index(ascending=True).loc[:,X_cts.columns]

    #this is where you should correct pvalues for multiple testing 
    p_threshold = 0.15
    p_threshold_high = 0.2
    p_threshold_mid = 0.1
    p_threshold_low = 0.05

    #plot as heatmap
    dat_mean = dat.values.mean()
    f, ax = plt.subplots(figsize = (20,12))
    g = sns.heatmap(dat,cmap = 'vlag', center=dat_mean,cbar=True,ax = ax)
    for a,b in zip(*np.where (pvals<p_threshold_high)):
        plt.text(b+.5,a+.55,'*',fontsize = 30,ha = 'center',va = 'center')

    for a,b in zip(*np.where (pvals<p_threshold_mid)):
        plt.text(b+.6,a+.55,'*',fontsize = 30,ha = 'center',va = 'center')

    for a,b in zip(*np.where (pvals<p_threshold_low)):
        plt.text(b+.4,a+.55,'*',fontsize = 30,ha = 'center',va = 'center')
    plt.tight_layout()
    plt.title('T2D(R) V.S. ND(B)', fontdict={'fontsize':25})
    if fig_dir is not None:
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        save_path = os.path.join(fig_dir, f'T2D-ND-enrich_diff.png')
        plt.savefig(save_path)
        print(f'Fig(enrich-diff analysis) saved to {save_path}! 🫐')
        

## Canonical Correlation Analysis
def do_cca(X_density, 
           group_patients, 
           chks, nbs, n_perms,fun = 'pearson'):
    cca = CCA(n_components=1,max_iter = 1000)
    if fun == 'pearson':
        func = pearsonr
    if fun == 'spearman':
        func = spearmanr
    
    cols = [chk + '_' for chk in chks]
    stats = {}
    for gp in ['ND', 'T2D']:
        stats[gp] = {}
        for k,nb1 in enumerate(nbs):
            for l,nb2 in enumerate(nbs):
                if (k<l):
                    nb1_cols = [c +str(nb1) for c in cols]
                    nb2_cols = [c +str(nb2) for c in cols]
                    dat = X_density.loc[group_patients[gp],nb1_cols+nb2_cols].dropna(axis = 1, how = 'all').dropna(axis = 0, how = 'any')
                    
                    a = dat.loc[:,[n for n in nb1_cols if n in dat.columns]].values
                    b = dat.loc[:,[n for n in nb2_cols if n in dat.columns]].values
                    if (len(a)<2) or (a.shape[1]<1) or (b.shape[1]<1):
                        print('continuing',nb1,nb2)
                        continue

    
                    x,y = cca.fit_transform(a,b)
                    arr = np.zeros(n_perms)
                    #compute the canonical correlation
                    stats[gp][nb1,nb2] = (func(x.squeeze(),y.squeeze())[0],arr)
                    for i in range(n_perms):
                        idx = np.arange(len(a))
                        np.random.shuffle(idx)
                        #compute over n_perms permutations
                        xt,yt = cca.fit_transform(a[idx],b)
                        arr[i] = func(xt.squeeze(),yt.squeeze())[0]
        # print(gp, 'done')
    return stats

def draw_stats(stats_treg, trg_cell, nbs, cutoff = 0.2, fig_dir=None):
    g1 = nx.Graph()
    g2 = nx.Graph()
    for k,nb1 in enumerate(nbs):
        for l,nb2 in enumerate(nbs):
            if (k<l):
                p1 = np.mean(stats_treg['ND'][(nb1,nb2)][1]<stats_treg['ND'][(nb1,nb2)][0])
                p2 = np.mean(stats_treg['T2D'][(nb1,nb2)][1]<stats_treg['T2D'][(nb1,nb2)][0])
                if (1-p1)<cutoff:
                    g1.add_edge(nb1,nb2,weight = 1-p1)
                    print('gp1',nb1,nb2,p1,p2)
                if (1-p2)<cutoff:
                    g2.add_edge(nb1,nb2,weight = 1-p2)
                    print('gp2',nb1,nb2,p1,p2)
    
    pal = sns.color_palette('bright',10)
    dash = {True: '-', False: ':'}

    # to make graph with common node postiions  (in final paper, we use node positions of g1 however)
    g_comb = g1.copy()
    for source,dest in g2.edges:
        g_comb.add_edge(source,dest)
        
    pos = nx.drawing.nx_pydot.graphviz_layout(g_comb,'neato')
    f,ax = plt.subplots(ncols = 2,sharex=True,sharey=True,figsize = (20,10))
    # nx.draw_networkx(g1,pos = pos,ax = ax[0],node_color=[sns.color_palette('bright')[i] for i in g1.nodes])
    plt.subplot(121)
    for k,v in pos.items():
        x,y = v
        plt.scatter([x],[y],c = pal[k], s = 1000,zorder = 3)
        plt.text(x,y, k, fontsize = 30, zorder = 10,ha = 'center', va = 'center')


    atrs = nx.get_edge_attributes(g1, 'weight')    
    for e0,e1 in g1.edges():
        p = atrs[e0,e1]
        plt.plot([pos[e0][0],pos[e1][0]],[pos[e0][1],pos[e1][1]], c= 'black',alpha = np.clip(a=1-10*p, a_min=0.001, a_max=1.0),linewidth = np.clip(a=8-80*p, a_min=0.001, a_max=8.0))
        plt.text(0.5*(pos[e0][0]+pos[e1][0]),0.5*(pos[e0][1]+pos[e1][1]), 'p = %.3f'%p)
        
    plt.axis('off')
    plt.title(f'ND', fontsize=20)

    plt.subplot(122)
    # nx.draw_networkx(g2,pos = pos,ax = ax[1],node_color=[sns.color_palette('bright')[i] for i in g2.nodes])
    for k,v in pos.items():
        x,y = v
        plt.scatter([x],[y],c = pal[k], s = 1000,zorder = 3)
        plt.text(x,y, k, fontsize = 30, zorder = 10,ha = 'center', va = 'center')
    atrs = nx.get_edge_attributes(g2, 'weight')    
    for e0,e1 in g2.edges():
        p = atrs[e0,e1]
        plt.plot([pos[e0][0],pos[e1][0]],[pos[e0][1],pos[e1][1]], c= 'black',alpha = np.clip(a=1-10*p, a_min=0.001, a_max=1.0),linewidth = np.clip(a=8-80*p, a_min=0.001, a_max=8.0))
        # plt.plot([pos[e0][0],pos[e1][0]],[pos[e0][1],pos[e1][1]], c= 'black',alpha = 1-5*p,linewidth = 15-75*p)
        plt.text(0.5*(pos[e0][0]+pos[e1][0]),0.5*(pos[e0][1]+pos[e1][1]), 'p = %.3f'%p)
    #plt.ylim(-100,350)
    plt.axis('off')
    plt.title(f'T2D', fontsize=20)

    plt.suptitle(f'{trg_cell}-cell Inter-CN Interaction', fontsize=25)
    # save_dir = os.path.join(fig_dir, 'cca-analysis')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(os.path.join(fig_dir, f"nterCN-{trg_cell.replace(' ', '-')}.png"))
    return g1,g2

def cca_analysis(islet_info, test_trg_cells:list, n_cluster=6, 
                 fig_dir=None, all_cell_DF=None, all_freqs=None):
    if all_cell_DF is None:
        patient_col = 'Donor'
        neigh_col = 'community_label'
        group_patients = {'ND': ['ABI2259', 'ACHM315', 'AFES372', 'AFFN281', 'AFG1440', 'AGBA390'],
                        'T2D': ['ABHQ115', 'ABIC495', 'ABIQ254', 'ACIA085', 'ADBI307', 'ADIX484', 'ADLE098', 'AEDN413', 'AEJR177', 'AFCM451']}
        good_patients = group_patients['ND'] + group_patients['T2D']
        
        #work out neighborhood counts
        all_feats_list = []
        for islet in islet_info:
            all_feats_list.append(islet['features'])
        
        all_cell_DF = pd.concat(all_feats_list).reset_index(drop=True)
    if all_freqs is None:
        all_freqs = all_cell_DF.groupby(['Donor', 'community_label']).apply(lambda x: x['celltype'].value_counts()).unstack().reset_index().fillna(0)
    
    nbd_counts = all_cell_DF.groupby([patient_col, neigh_col]).size().unstack().loc[good_patients]
    pat_counts = all_cell_DF.groupby([patient_col]).size().loc[good_patients]
    nbs = [i for i in range(n_cluster)]
    
    ## test starts
    if fig_dir is not None:
        save_dir = os.path.join(fig_dir, 'cca-analysis')
    for trg_cell in test_trg_cells:
        chks = [trg_cell]
        x = all_freqs.reset_index().iloc[:,1:]
        df_list = []
        for nb in nbs:
            arr = x.loc[x[neigh_col]==nb,:]
            nb_patients = list(set(arr[patient_col].values))
            arr = arr.set_index(patient_col).loc[nb_patients,chks].values
            arr /= nbd_counts.loc[nb_patients][nb].values[:,None]
            df_list.append(pd.DataFrame(arr, index = nb_patients, columns = [chk+'_'+str(nb) for chk in chks]))
            
        X_density = pd.concat(df_list, axis = 1)
        # X_density.fillna(value=0, inplace=True) # added

        X_density = np.log(1e-3+X_density.loc[:, X_density.apply(np.std,axis = 0)>0])
        X_density.fillna(value=0, inplace=True) # added   
        np.random.seed(42)
        try:
            stats = do_cca(X_density, 
                        group_patients, 
                        n_perms=1000, chks=chks, nbs=nbs, fun = 'pearson')
            draw_stats(stats, trg_cell, nbs, cutoff = 0.2, fig_dir=save_dir)
        except:
            pass
    if save_dir is not None:
        print(f'Fig(CCA analysis) saved to {save_dir}! 🥝')
