#!/bin/bash

# Load environment variables
cd ../../src && source .env 

python3 main.py --run inference \
                --edges_as_classes 0 \
                --checkpoint_model_path -1 \
                --dataset webnlg \
                --checkpoint_model_id -1 \
                --default_root_dir ${STORAGE_DRIVE}/data/core/grapher/output \
                --inference_input_text "Danielle Harris had a main role in Super Capers, a 98 minute long movie."

# Problems with interpreting 0 as False in main.py
# --detect_anomaly 0 \
