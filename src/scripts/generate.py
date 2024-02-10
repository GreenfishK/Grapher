from datasets.webnlg.datamodule import GraphDataModule
from pytorch_lightning import loggers as pl_loggers
from argparse import ArgumentParser
from engines.grapher_lightning import LitGrapher
import pytorch_lightning as pl
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, EarlyStopping
from misc.utils import decode_graph
import logging
import torch
import nltk

# TODO: Build project structure according to this article:
# https://medium.com/@l.charteros/scalable-project-structure-for-machine-learning-projects-with-pytorch-and-pytorch-lightning-d5f1408d203e

def generate(args):

    # Create directories for validations, tests and checkpoints
    eval_dir = os.path.join(args.default_root_dir, args.dataset + '_version_' + args.version)
    checkpoint_dir = os.path.join(eval_dir, 'checkpoints')

    # Start from last checkpoint or a specific checkpoint. 
    if args.checkpoint_model_id < 0:
        checkpoint_model_path = os.path.join(checkpoint_dir, 'last.ckpt')
    else:
        checkpoint_model_path = os.path.join(checkpoint_dir, f"model-step={args.checkpoint_model_id}.ckpt")
    assert os.path.exists(checkpoint_model_path), 'Provided checkpoint does not exists, cannot do inference'

    # Specify the GPU device you want to use
    if torch.cuda.device_count() <= 1:
        device = torch.device(f"cuda:{os.environ['CUDA_VISIBLE_DEVICES']}") 

    logging.info(device)
    grapher = LitGrapher.load_from_checkpoint(checkpoint_path=checkpoint_model_path)
    grapher.to(device)
    grapher.eval()

    # 150 as max tokens was set based on what was found in the grapher.Grapher.sample function
    tokenizer = T5Tokenizer.from_pretrained(grapher.transformer_name, cache_dir=grapher.cache_dir,
                                            model_max_length=150, legacy=True)
    tokenizer.add_tokens('__no_node__')
    tokenizer.add_tokens('__no_edge__')
    tokenizer.add_tokens('__node_sep__')
    
    text_tok = tokenizer([args.inference_input_text],
                            add_special_tokens=True,
                            padding=True,
                            return_tensors='pt')

    text_input_ids, mask = text_tok['input_ids'], text_tok['attention_mask']

    # Set the device for input tensors
    text_input_ids = text_input_ids.to(device)
    mask = mask.to(device)

    seq_nodes, seq_edges = grapher.model.sample(text_input_ids, mask)
    dec_graph = decode_graph(tokenizer, grapher.edge_classes, seq_nodes, seq_edges, grapher.edges_as_classes,
                            grapher.node_sep_id, grapher.max_nodes, grapher.noedge_cl, grapher.noedge_id,
                            grapher.bos_token_id, grapher.eos_token_id)
    
    graph_str = ['-->'.join(tri) for tri in dec_graph[0]]
    
    print(f'Generated Graph: {graph_str}')
        
    
# --------------------------------------------------------------
# Start inference
# --------------------------------------------------------------
    
# Parsing arguments
parser = ArgumentParser(description='Arguments')

parser.add_argument("--dataset", type=str, default='webnlg')
parser.add_argument('--checkpoint_model_id', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument("--focal_loss_gamma", type=float, default=0.0)
parser.add_argument("--dropout_rate", type=float, default=0.5)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--eval_dump_only", type=int, default=0)

# pytorch lightning params
parser.add_argument("--default_root_dir", type=str, default="output")
parser.add_argument("--inference_input_text", type=str,
                    default='Danielle Harris had a main role in Super Capers, a 98 minute long movie.') 

args = parser.parse_args()
generate(args)

