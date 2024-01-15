<div align="center">    
 
# Knowledge Graph Generation From Text

[![Paper](https://img.shields.io/badge/Paper-ArXiv.2211.10511-blue)](https://arxiv.org/abs/2211.10511)
[![Conference](https://img.shields.io/badge/EMNLP-2022-orange)](https://2022.emnlp.org/)
 
<!--  
Conference   
-->   
</div>

## Description   
Grapher is an end-to-end multi-stage Knowledge Graph (KG) construction system, that separates the overall generation process  into  two  stages.
<p align="center">
  <img src="imgs/overview_grapher.png" width="50%">
</p>
The  graph  nodes  are generated first using pretrained language model, such as T5.The input text is transformed into a sequence of text entities. The features corresponding to each entity (node) is extracted and then sent to the edge generation module.
<p align="center">
  <img src="imgs/node_gen.png" width="50%">
</p>
Edge construction, using generation (e.g.,GRU) or a classifier head. Blue circles represent the features corresponding to the actual graph edges (solid lines) and the white circles are the features that are decoded into ⟨NO_EDGE⟩ (dashed line).
<p align="center">
  <img src="imgs/edges_gen.png" width="50%">
</p>

## Environment
To run this code, please install PyTorch and Pytorch Lightning (we tested the code on Pytorch 1.13 and Pytorch Lightning 1.8.1)
  

## Setup   
Install dependencies   
```bash
# clone project   
git clone git@github.com:IBM/Grapher.git

# navigate to the directory
cd Grapher

# clone an external repository for reading the data
git submodule add https://gitlab.com/webnlg/corpus-reader.git corpusreader
git submodule add https://gitlab.com/webnlg/corpus-reader.git corpusreader

# clone another external repositories for scoring the results
git submodule add https://github.com/WebNLG/WebNLG-Text-to-triples.git WebNLG_Text_to_triples
git submodule add https://github.com/WebNLG/WebNLG-Text-to-triples.git WebNLG_Text_to_triples
 ```   
## Data

WebNLG 3.0 dataset   
 ```bash
# download the dataset   
git clone https://gitlab.com/shimorina/webnlg-dataset.git
```

## How to train
There are two scripts to run two versions of the algorithm
```bash
# naviagate to scripts directory
cd scripts

# run Grapher with the edge generation head
./train_gen.sh

# run Grapher with the classifier edge head
./train_class.sh
```

## How to test
```bash
# run the test on experiment "webnlg_version_1" using latest checkpoint last.ckpt
python3 main.py --run test --version 1 --default_root_dir output --data_path webnlg-dataset/release_v3.0/en

# run the test on experiment "webnlg_version_1" using checkpoint at iteration 5000
python3 main.py --run test --version 1 --default_root_dir output --data_path webnlg-dataset/release_v3.0/en --checkpoint_model_id 5000
```

## How to run inference
```bash
# run inference on experiment "webnlg_version_1" using latest checkpoint last.ckpt
python main.py --run inference --version 1 --default_root_dir output --inference_input_text "Danielle Harris had a main role in Super Capers, a 98 minute long movie."
```

## Results
Results can be visualized in Tensorboard
```bash
tensorboard --logdir output
```

### Citation   
```
@inproceedings{grapher2022,
  title={Knowledge Graph Generation From Text},
  author={Igor Melnyk, Pierre Dognin, Payel Das},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (Findings of EMNLP)},
  year={2022}
}
```   

# Reproducibility study
The scripts ./train_gen and ./train_class.sh both produce an error during execution due to several non-existing arguments passed during the call of main.py (see below).
```main.py: error: unrecognized arguments: --max_epochs 100 --accelerator gpu --num_nodes 1 --num_sanity_val_steps 0 --fast_dev_run 0 --overfit_batches 0 --limit_train_batches 1.0 --limit_val_batches 1.0 --limit_test_batches 1.0 --accumulate_grad_batches 10 --detect_anomaly True --log_every_n_steps 100 --val_check_interval 1000```

Some additional libraries are required, such as `SentencePie`(See below).
```T5Tokenizer requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones that match your environment. Please note that you may need to restart your runtime after installation.```

Instead of just cloning the two separately needed git repositories corpusreader and WebNLG-Text-to-triples, I added them as submodules (See [Setup](#Setup) section).