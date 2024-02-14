from engines.grapher_lightning import LitGrapher
from misc.utils import decode_graph

from transformers import T5Tokenizer
import os
import logging
import torch

# TODO: Build project structure according to this article:
# https://medium.com/@l.charteros/scalable-project-structure-for-machine-learning-projects-with-pytorch-and-pytorch-lightning-d5f1408d203e

def generate(args, model_variant, device):

    # Create directories for validations, tests and checkpoints
    eval_dir = os.path.join(args.default_root_dir, args.dataset + '_model_variant=' + model_variant)
    checkpoint_dir = os.path.join(eval_dir, 'checkpoints')

    # Start from last checkpoint or a specific checkpoint. 
    if args.checkpoint_model_id < 0:
        checkpoint_model_path = os.path.join(checkpoint_dir, 'last.ckpt')
    else:
        checkpoint_model_path = os.path.join(checkpoint_dir, f"model-step={args.checkpoint_model_id}.ckpt")
    assert os.path.exists(checkpoint_model_path), 'Provided checkpoint does not exists, cannot do inference'
        
    grapher = LitGrapher.load_from_checkpoint(checkpoint_path=checkpoint_model_path)
    if device:
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
    if device:
        text_input_ids = text_input_ids.to(device)
        mask = mask.to(device)
    else:
        text_input_ids = text_input_ids.to('cuda')
        mask = mask.to('cuda')
    
    seq_nodes, seq_edges = grapher.model.sample(text_input_ids, mask)
    dec_graph = decode_graph(tokenizer, grapher.edge_classes, seq_nodes, seq_edges, grapher.edges_as_classes,
                            grapher.node_sep_id, grapher.max_nodes, grapher.noedge_cl, grapher.noedge_id,
                            grapher.bos_token_id, grapher.eos_token_id)
    
    graph_str = ['-->'.join(tri) for tri in dec_graph[0]]
    
    print(f'Generated Graph: {graph_str}')
        
