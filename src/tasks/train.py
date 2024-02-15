from data.webnlg.datamodule import GraphDataModule
from engines.grapher_lightning import LitGrapher
from misc.utils import last_model_path

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, EarlyStopping
from datetime import datetime
import torch
import logging

def train(args, model_variant, device):

    # Create directories for validations, tests and checkpoints
    training_start_tmstmp = str(datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    eval_dir = os.path.join(args.default_root_dir, args.dataset + '_model_variant=' + model_variant)
    checkpoint_dir = os.path.join(eval_dir, training_start_tmstmp, 'checkpoints')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(eval_dir, training_start_tmstmp, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(eval_dir, training_start_tmstmp, 'test'), exist_ok=True)

    # Logger for TensorBoard
    TB = pl_loggers.TensorBoardLogger(save_dir=args.default_root_dir,
                                      name='',
                                      version=args.dataset + '_model_variant=' + model_variant, 
                                      default_hp_metric=False)

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

    # -------------------- Engine incl. Model ---------------------
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
    if device:
        grapher.to(device)

    # disable randomness, dropout, etc...
    grapher.eval()

    # -------------------- Trainer ---------------------

    # Trade off precision for performance
    lst_tensor_devices = ['NVIDIA A40', 'NVIDIA A100']
    cnt_devices = torch.cuda.device_count()
    for i in range(cnt_devices):
        logging.info(torch.cuda.get_device_name(i))
        if torch.cuda.get_device_name(i) in lst_tensor_devices:
           torch.set_float32_matmul_precision('high')
           break

    # Create plan to save the model periodically.
    # {train_loss_epoch} is the arithmetic mean of {train_loss} from each step.
    # {train_loss_epoch} gets logged for the previous epoch .  
    # Do not use {train_epoch}! This is just the loss from the last step of the epoch.
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='model-{epoch:02d}-{train_loss_epoch:.2f}-{F1:.2f}',
        every_n_epochs=args.every_n_epochs, # Saves a checkpoint every n epochs 
        save_on_train_epoch_end=False, # Checkpointing runs at the end of validation
        save_last=True, # saves a last.ckpt whenever a checkpoint file gets saved
        save_top_k=-1, # Save all models
    )

    # If three consecutive validation checks yield no improvement, the trainer stops.
    # Monitor validation loss to prevent overfitting
    early_stopping_callback = EarlyStopping(
        monitor="train_loss_epoch",
        min_delta=0.005,
        mode="min",
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
    
    # Start from last checkpoint or a specific checkpoint. 
    last_model_path_str = last_model_path(eval_dir)
    if args.checkpoint_model_id < 0:
        checkpoint_model_path = os.path.join(last_model_path_str, 'checkpoints', 'last.ckpt')
    else:
        checkpoint_model_path = os.path.join(last_model_path_str, 'checkpoints', f"model-epoch={args.epoch}-train_loss={args.train_loss}-F1={args.F1}.ckpt")

    trainer.fit(model=grapher, 
                datamodule=dm,
                ckpt_path=checkpoint_model_path if os.path.exists(checkpoint_model_path) else None)    
        
