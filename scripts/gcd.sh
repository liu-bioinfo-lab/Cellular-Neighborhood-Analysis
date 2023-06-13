#!/bin/bash

cd /mlodata1/sfan/CFIDF
pip install -r requirements.txt

TRG_CSV_PATH='put the path to original cell-id csv here'
DATA_DIR='put the path to the directory for saving preprocessed data'
FIG_DIR='put the path to the directory for saving generated figures'

python gcd.py --trg_csv_path $TRG_CSV_PATH --data_dir $DATA_DIR --fig_dir $FIG_DIR
