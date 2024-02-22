from engines.grapher_lightning import LitGrapher
from misc.utils import decode_graph, model_file_name

from transformers import T5Tokenizer
import os
import logging


def generate(args, device):
    # Load grapher
    # -------------------------------------------------------------------------------
    exec_dir = os.environ['EXEC_DIR']

    checkpoint_dir = os.path.join(exec_dir, 'checkpoints')
    if args.checkpoint_model_id == -1:
        logging.info(f"Generating using the last checkpoint model from: {exec_dir}")
        checkpoint_model_path = os.path.join(checkpoint_dir, 'last.ckpt')
    else:
        logging.info(f"Generating using the model at epoch {args.checkpoint_model_id} from {exec_dir}")
        checkpoint_model_path = os.path.join(exec_dir,
                                            'checkpoints',
                                            model_file_name(exec_dir, args.checkpoint_model_id))
    assert os.path.exists(checkpoint_model_path), 'Provided checkpoint does not exists, cannot do inference'
        
    grapher = LitGrapher.load_from_checkpoint(checkpoint_path=checkpoint_model_path)
    if device:
        grapher.to(device)
    grapher.eval()

    # Create input tokens
    # -------------------------------------------------------------------------------
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
        
