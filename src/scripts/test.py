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

def test(args):

    # Logger for TensorBoard
    TB = pl_loggers.TensorBoardLogger(save_dir=args.default_root_dir, name='', version=args.dataset + '_version_' + args.version, default_hp_metric=False)

    # Start from last checkpoint or a specific checkpoint. 
    eval_dir = os.path.join(args.default_root_dir, args.dataset + '_version_' + args.version)
    checkpoint_dir = os.path.join(eval_dir, 'checkpoints')
    if args.checkpoint_model_id < 0:
        checkpoint_model_path = os.path.join(checkpoint_dir, 'last.ckpt')
    else:
        checkpoint_model_path = os.path.join(checkpoint_dir, f"model-step={args.checkpoint_model_id}.ckpt")
    
    # Specify the GPU device you want to use
    if torch.cuda.device_count() <= 1:
        device = torch.device(f"cuda:{os.environ['CUDA_VISIBLE_DEVICES']}") 

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


# --------------------------------------------------------------
# Start testing
# --------------------------------------------------------------
    
# Parsing arguments
parser = ArgumentParser(description='Arguments')

parser.add_argument("--dataset", type=str, default='webnlg')
parser.add_argument('--version', type=str, default='0')
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--num_data_workers', type=int, default=3)
parser.add_argument('--checkpoint_model_id', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=10)

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

args = parser.parse_args()
test(args)

