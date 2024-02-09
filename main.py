from data.dataset import GraphDataModule
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

def main(args):

    # Create directories for validations, tests and checkpoints
    eval_dir = os.path.join(args.default_root_dir, args.dataset + '_version_' + args.version)
    checkpoint_dir = os.path.join(eval_dir, 'checkpoints')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(eval_dir, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(eval_dir, 'test'), exist_ok=True)

    # Download punkt tokenizer
    punkt_dir = f"{args.default_root_dir}/../lib/punkt"
    os.makedirs(punkt_dir, exist_ok=True)
    nltk.download('punkt', download_dir=punkt_dir)
    nltk.data.path.append(punkt_dir)

    # Logger for TensorBoard
    TB = pl_loggers.TensorBoardLogger(save_dir=args.default_root_dir, name='', version=args.dataset + '_version_' + args.version, default_hp_metric=False)

    # Start from last checkpoint or a specific checkpoint. 
    if args.checkpoint_model_id < 0:
        checkpoint_model_path = os.path.join(checkpoint_dir, 'last.ckpt')
    else:
        checkpoint_model_path = os.path.join(checkpoint_dir, f"model-step={args.checkpoint_model_id}.ckpt")
    
    # Specify the GPU device you want to use
    if torch.cuda.device_count() <= 1:
        device = torch.device(f"cuda:{os.environ['CUDA_VISIBLE_DEVICES']}") 

    if args.run == 'train':
        # -------------------- Data module ---------------------
        dm = GraphDataModule(cache_dir=args.cache_dir,
                             data_path=args.data_path,
                             dataset=args.dataset,
                             tokenizer_class=T5Tokenizer,
                             tokenizer_name=args.pretrained_model,                           
                             batch_size=args.batch_size,
                             num_data_workers=args.num_data_workers,
                             max_nodes=args.max_nodes,
                             max_edges=args.max_edges,
                             edges_as_classes=args.edges_as_classes)

        # Download data and create train, dev and test splits
        dm.prepare_data()

        # Load training data (train.text, train.graph) into GraphDataset
        dm.setup(stage='fit')

        # Load validation data (dev.text, dev.graph) into GraphDataset
        dm.setup(stage='validate')

        # -------------------- Model ---------------------
        grapher = LitGrapher(eval_dir=eval_dir,
                             cache_dir=args.cache_dir,
                             transformer_class=T5ForConditionalGeneration,
                             transformer_name=args.pretrained_model,
                             tokenizer=dm.tokenizer,
                             dropout_rate=args.dropout_rate,
                             focal_loss_gamma=args.focal_loss_gamma,
                             lr=args.lr,
                             num_layers=args.num_layers,
                             # Grapher-specific parameters
                             max_nodes=args.max_nodes,
                             max_edges=args.max_edges,
                             edges_as_classes=args.edges_as_classes,
                             default_seq_len_edge=args.default_seq_len_edge,
                             # Tokenizer params
                             vocab_size=len(dm.tokenizer.get_vocab()),
                             bos_token_id=dm.tokenizer.pad_token_id,
                             eos_token_id=dm.tokenizer.eos_token_id,
                             nonode_id=dm.tokenizer.convert_tokens_to_ids('__no_node__'),
                             noedge_id=dm.tokenizer.convert_tokens_to_ids('__no_edge__'),
                             node_sep_id=dm.tokenizer.convert_tokens_to_ids('__node_sep__'),
                             # Edge classification
                             noedge_cl=len(dm.dataset_train.edge_classes) - 1,
                             edge_classes=dm.dataset_train.edge_classes,
                             num_classes=len(dm.dataset_train.edge_classes))
        
        # Wrap the model with DataParallel
        if torch.cuda.device_count() <= 1:
            grapher.to(device)

        # disable randomness, dropout, etc...
        grapher.eval()

        # -------------------- Trainer ---------------------
        # Create plan to save the model periodically   
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='model-{epoch:02d}-{train_loss:.2f}-{F1:.2f}',
            every_n_epochs=args.every_n_epochs, # Saves a checkpoint every n epochs 
            save_on_train_epoch_end=False, # Checkpointing runs at the end of validation
            save_last=True, # saves a last.ckpt whenever a checkpoint file gets saved
            save_top_k=3, # Save top 3
            mode="min",
            monitor="train_loss",
        )

        # If three consecutive validation checks yield no improvement, the trainer stops.
        # Monitor validation loss to prevent overfitting
        early_stopping_callback = EarlyStopping(
            monitor="F1",
            mode="max",
            patience=3  
        )

        # Validation checks are done every check_val_every_n_epoch epoch.
        trainer = pl.Trainer(default_root_dir=args.default_root_dir,
                            accelerator=args.accelerator, 
                            max_epochs=args.max_epochs,
                            num_sanity_val_steps=args.num_sanity_val_steps,
                            fast_dev_run=args.fast_dev_run,
                            overfit_batches=args.overfit_batches,
                            limit_train_batches=args.limit_train_batches,
                            limit_val_batches=args.limit_val_batches,
                            limit_test_batches=args.limit_test_batches,
                            accumulate_grad_batches=args.accumulate_grad_batches,
                            detect_anomaly=args.detect_anomaly,
                            log_every_n_steps=args.log_every_n_steps,
                            check_val_every_n_epoch=args.check_val_every_n_epoch,
                            # val_check_interval=0.10, # Just for debugging
                            logger=TB,
                            callbacks=[checkpoint_callback, early_stopping_callback, RichProgressBar(10)],
                            num_nodes=args.num_nodes)

        trainer.fit(model=grapher, datamodule=dm,
                    ckpt_path=checkpoint_model_path if os.path.exists(checkpoint_model_path) else None)
        
    elif args.run == 'test':

        assert os.path.exists(checkpoint_model_path), 'Provided checkpoint does not exists, cannot run the test'

        grapher = LitGrapher.load_from_checkpoint(checkpoint_path=checkpoint_model_path)
        grapher.to(device)
        grapher.eval()

        dm = GraphDataModule(tokenizer_class=T5Tokenizer,
                             tokenizer_name=grapher.transformer_name,
                             cache_dir=grapher.cache_dir,
                             data_path=args.data_path,
                             dataset=args.dataset,
                             batch_size=args.batch_size,
                             num_data_workers=args.num_data_workers,
                             max_nodes=grapher.max_nodes,
                             max_edges=grapher.max_edges,
                             edges_as_classes=grapher.edges_as_classes)

        dm.setup(stage='test')
        trainer = pl.Trainer(default_root_dir=args.default_root_dir,
                            accelerator=args.accelerator, 
                            max_epochs=args.max_epochs,
                            num_sanity_val_steps=args.num_sanity_val_steps,
                            fast_dev_run=args.fast_dev_run,
                            overfit_batches=args.overfit_batches,
                            limit_train_batches=args.limit_train_batches,
                            limit_val_batches=args.limit_val_batches,
                            limit_test_batches=args.limit_test_batches,
                            accumulate_grad_batches=args.accumulate_grad_batches,
                            detect_anomaly=args.detect_anomaly,
                            log_every_n_steps=args.log_every_n_steps,
                            check_val_every_n_epoch=args.check_val_every_n_epoch, 
                            logger=TB,
                            num_nodes=args.num_nodes)
        trainer.test(grapher, datamodule=dm)

    else: # single inference

        assert os.path.exists(checkpoint_model_path), 'Provided checkpoint does not exists, cannot do inference'

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
        
    
if __name__ == "__main__":
    
    # Parsing arguments
    parser = ArgumentParser(description='Arguments')

    parser.add_argument("--dataset", type=str, default='webnlg')
    parser.add_argument("--run", type=str, default='train')
    parser.add_argument('--pretrained_model', type=str, default='t5-large')
    parser.add_argument('--version', type=str, default='0')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--num_data_workers', type=int, default=3)
    parser.add_argument('--every_n_epochs', type=int, default=-1)
    parser.add_argument('--checkpoint_model_id', type=int, default=-1)
    parser.add_argument('--max_nodes', type=int, default=8)
    parser.add_argument('--max_edges', type=int, default=7)
    parser.add_argument('--default_seq_len_node', type=int, default=20)
    parser.add_argument('--default_seq_len_edge', type=int, default=20)
    parser.add_argument('--edges_as_classes', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument("--focal_loss_gamma", type=float, default=0.0)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--eval_dump_only", type=int, default=0)
    
    # pytorch lightning params
    parser.add_argument("--default_root_dir", type=str, default="output")
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)
    parser.add_argument("--fast_dev_run", type=int, default=0)
    parser.add_argument("--overfit_batches", type=int, default=0)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--limit_test_batches", type=float, default=1.0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=10)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--inference_input_text", type=str,
                        default='Danielle Harris had a main role in Super Capers, a 98 minute long movie.') 
    
    args = parser.parse_args()
    main(args)

