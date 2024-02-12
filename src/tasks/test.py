from data.webnlg.datamodule import GraphDataModule
from engines.grapher_lightning import LitGrapher

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from transformers import T5Tokenizer
import os
import torch

# TODO: Build project structure according to this article:
# https://medium.com/@l.charteros/scalable-project-structure-for-machine-learning-projects-with-pytorch-and-pytorch-lightning-d5f1408d203e

def test(args, model_variant):

    # Logger for TensorBoard
    TB = pl_loggers.TensorBoardLogger(save_dir=args.default_root_dir, name='', version=args.dataset + '_model_variant=' + model_variant, default_hp_metric=False)

    # Start from last checkpoint or a specific checkpoint. 
    eval_dir = os.path.join(args.default_root_dir, args.dataset + '_model_variant=' + model_variant)
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
