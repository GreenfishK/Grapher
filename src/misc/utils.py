from WebNLG_Text_to_triples import Evaluation_script_json
from misc.rdf import save_webnlg_rdf

import torch
import networkx as nx
import os
import json


failed_node = 'failed node'
failed_edge = 'failed edge'
nonode_str = '__no_node__'


def compute_loss(criterion, logits_nodes, logits_edges, target_nodes, target_edges, edges_as_classes, focal_loss_gamma):
    """
    This function computes the loss for the model during training. 
    It computes the cross-entropy loss for nodes and edges separately 
    and returns their sum as the total loss. In case focal_loss_gamma is not None, 
    it computes the focal loss for edges.
    Args:
        * criterion: Dictionary containing the loss functions for nodes and edges.
        * logits_nodes: Logits for nodes predicted by the model.
        * logits_edges: Logits for edges predicted by the model.
        * target_nodes: Ground truth labels for nodes.
        * target_edges: Ground truth labels for edges.
        * edges_as_classes: Boolean indicating whether edges are treated as classes or not.
        * focal_loss_gamma: Parameter for the focal loss function.
    
    """
    # --------- Node Loss ---   ------
    # shift forward 1 step to create labels
    # The shift ensures that the lengths of the predicted logits 
    # and the target labels remain consistent
    labels = torch.cat([target_nodes[:, 1:], torch.zeros_like(target_nodes[:, -2:-1])], 1)
    loss_nodes = criterion['ce'](logits_nodes.transpose(1,2), labels).mean()

    # --------- Edge Loss ---------
    if edges_as_classes:
        target_edges = target_edges.permute(2, 0, 1)
        logits_edges = logits_edges.permute(2, 3, 0, 1)
        if focal_loss_gamma:
            loss_edges = criterion['focal'](logits_edges, target_edges).mean()
        else:
            loss_edges = criterion['ce'](logits_edges, target_edges).mean()

    else:  # full
        target_edges = target_edges.permute(2, 0, 1, 3)
        logits_edges = logits_edges.permute(2, 4, 0, 1, 3)
        loss_edges = criterion['ce'](logits_edges, target_edges).mean()

    loss = loss_nodes + loss_edges

    return loss


def compute_scores(hyp, ref, iteration, eval_dir, split, rank):
    """
    Compute evaluation scores (prec, rec, F1) for generated triples against reference triples.
    Args:
        * hyp: List of generated triples (hypotheses).
        * ref: List of reference triples.
        * iteration: Iteration or step number of the training process.
        * eval_dir: Directory path where evaluation files will be stored.
        * split: Name of the data split (e.g., 'valid' or 'test').
        * rank: Rank of the process (used for naming files).

    Convert the hypotheses and references into RDF/xml format files and saves them as xml files.
    Compute evaluation scores based on these XML files and save them as JSON files.
    Load the JSON files and extract evaluation scores from them.

    Returns:
        The evaluation scores as dict.

    """
    refs = [[' | '.join(i) for i in t] for t in ref]
    hyps = [[' | '.join(i) for i in t] for t in hyp]
    categories = [' '] * len(refs)

    ref_fname, hyp_fname = save_webnlg_rdf(hyps, refs, categories, os.path.join(eval_dir, split), f'{iteration}_{rank}')
    scores_fname = os.path.join(eval_dir, split, f'scores_{iteration}_{rank}.json')
    Evaluation_script_json.main(ref_fname, hyp_fname, scores_fname)

    scores = json.load(open(scores_fname))
    scores = {'Precision': scores['Total_scores']['Exact']['Precision'],
              'Recall': scores['Total_scores']['Exact']['Recall'],
              'F1': scores['Total_scores']['Exact']['F1']}

    return scores


def decode_text(tokenizer, text_input_ids, bos_token_id, eos_token_id):
    """
    This function decodes a batch of text sequences represented as token IDs into a list of strings.
    Args:
        * tokenizer: Tokenizer object for decoding.
        * text_input_ids: Batch of input token IDs.
        * bos_token_id: Token ID for the beginning-of-sequence token.
        * eos_token_id: Token ID for the end-of-sequence token.
    It iterates over each text sequence in the batch, finds the beginning-of-sequence and end-of-sequence tokens, 
    and decodes the text between them using the tokenizer.
    It returns a list of decoded text strings.
    """
    text_decoded = []

    for text in text_input_ids:
        bos_mask = (text == bos_token_id).nonzero(as_tuple=False)
        eos_mask = (text == eos_token_id).nonzero(as_tuple=False)
        text_dec = tokenizer._decode(text[bos_mask[0] + 1:eos_mask[0]])
        text_decoded.append(text_dec)

    return text_decoded


def decode_graph(tokenizer, edge_classes, bnodes, bedges, edges_as_classes, node_sep_id,
                 max_nodes, noedge_cl, noedge_id, bos_token_id, eos_token_id):
    """
    This function decodes a batch of node and edge tokens into a list of triples.
    It constructs a directed graph using NetworkX library based on the input node and edge sequences.
    It decodes the nodes and edges into strings and forms triples.
    Args:
        * tokenizer: Tokenizer object for decoding.
        * edge_classes: List of edge classes.
        * bnodes: Batch of node sequences represented as token IDs.
        * bedges: Batch of edge sequences represented as token IDs.
        * edges_as_classes: Boolean indicating whether edges are treated as classes or not.
        * node_sep_id: Token ID for separating nodes.
        * max_nodes: Maximum number of nodes in a graph.
        * noedge_cl: Class representing no edge.
        * noedge_id: Token ID representing no edge.
        * bos_token_id: Token ID for the beginning-of-sequence token.
        * eos_token_id: Token ID for the end-of-sequence token.
    Returns:
        * A list of decoded triples for each graph in the batch.

    """
    if edges_as_classes:
        bedges = bedges.permute(2, 0, 1)
    else:
        bedges = bedges.permute(2, 0, 1, 3)

    # bnodes: batch_size X num_nodes X seq_len_node
    # bedges: batch_size X num_nodes X num_nodes X seq_len_edge [FULL]
    # bedges: batch_size X num_nodes X num_nodes [CLASSES]

    triples_decoded = []

    for b_ind, (nodes, edges) in enumerate(zip(bnodes, bedges)):

        G = nx.DiGraph()

        nodes_decoded = []
        all_nodes = tokenizer._decode(nodes).split(tokenizer._decode(node_sep_id))

        for n in all_nodes:
            s = n.replace('<pad>', '').replace('</s>', '').strip()
            # empty or white space
            if not s or not s.strip():
                s = failed_node
            nodes_decoded.append(s)

        nodes_decoded = nodes_decoded[:max_nodes]
        nodes_decoded += (max_nodes - len(nodes_decoded)) * [failed_node]

        if edges_as_classes:
            noedge_mask = ~(bedges == noedge_cl)
            for i in range(max_nodes):
                for j in range(max_nodes):
                    if i == j: continue
                    if noedge_mask[b_ind][i, j] > 0:
                        edge = edges[i, j].detach()

                        if edge == noedge_cl:
                            s = failed_edge
                        else:
                            s = edge_classes[edge]

                        if nodes_decoded[i] != failed_node and nodes_decoded[j] != failed_node and s != failed_edge and \
                           nonode_str not in nodes_decoded[i] and nonode_str not in nodes_decoded[j]:
                            G.add_edge(nodes_decoded[i], nodes_decoded[j], edge=s)
        else:  # full
            noedge_mask = 1 - torch.sum(bedges == noedge_id, -1)
            for i in range(max_nodes):
                for j in range(max_nodes):
                    if i == j: continue
                    if noedge_mask[b_ind][i, j] > 0:
                        edge = edges[i, j]

                        s = _decode(edge, bos_token_id, eos_token_id, tokenizer, failed_edge)

                        # empty or white space
                        if not s or not s.strip():
                            s = failed_edge

                        if failed_node not in nodes_decoded[i] and failed_node not in nodes_decoded[j] and s != failed_edge and nonode_str not in nodes_decoded[i] and nonode_str not in nodes_decoded[j]:
                            G.add_edge(nodes_decoded[i], nodes_decoded[j], edge=s)

        # make sure there are at least 2 nodes and 1 edge
        if nx.is_empty(G):
            node1 = nodes_decoded[0] if len(nodes_decoded)>0 else failed_node
            node2 = nodes_decoded[1] if len(nodes_decoded)>1 else failed_node
            G.add_edge(node1, node2, edge=failed_edge)

        tri = []
        for ind, (u, v, d) in enumerate(G.edges(data=True)):

            # decode up to 8 paths, discard others (because eval fails with too many paths)
            if ind >= 8:
                break

            tri.append([u, d['edge'], v])

        triples_decoded.append(tri)

    return triples_decoded


# Helper function for the decode_graph function.
def _decode(cand, bos_token_id, eos_token_id, tokenizer, failed=failed_node):
    """
    This function decodes a sequence of tokens into a string.
    Args:
        * cand: Tensor representing the sequence of tokens.
        * bos_token_id: Token ID for the beginning-of-sequence token.
        * eos_token_id: Token ID for the end-of-sequence token.
        * tokenizer: Tokenizer object for decoding.
        * failed: String to return if decoding fails.
    It first finds the indices of the beginning-of-sequence and end-of-sequence tokens in the tensor.
    Then, it decodes the sequence between these tokens using the tokenizer.
    If decoding fails (no beginning-of-sequence token found), it returns the failed string.
    """
    bos_mask = (cand == bos_token_id).nonzero(as_tuple=False)
    if len(bos_mask) > 0:
        eos_mask = (cand == eos_token_id).nonzero(as_tuple=False)
        if len(eos_mask) > 0:
            s = tokenizer._decode(cand[bos_mask[0] + 1:eos_mask[0]])
        else:
            s = failed
    else:
        s = failed

    return s


