#!/bin/bash

cd /mlodata1/sfan/CFIDF
pip install -r requirements.txt
sudo apt install graphviz

TRG_CSV_PATH='put the path to original cell-id csv here'
DATA_DIR='put the path to the directory for saving preprocessed data'
FIG_DIR='put the path to the directory for saving generated figures'

N_CLUSTER=6
RESOLUTION=0.5

python run.py --trg_csv_path $TRG_CSV_PATH --data_dir $DATA_DIR --fig_dir $FIG_DIR --n_cluster $N_CLUSTER --resolution $RESOLUTION --enrichment_analysis --diff_analysis --voronoi_analysis --cca_analysis