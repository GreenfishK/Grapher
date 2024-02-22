from data.webnlg.datamodule import GraphDataModule
from engines.grapher_lightning import LitGrapher
from misc.utils import model_file_name
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from transformers import T5Tokenizer
import os
import logging


# TODO: Build project structure according to this article:
# https://medium.com/@l.charteros/scalable-project-structure-for-machine-learning-projects-with-pytorch-and-pytorch-lightning-d5f1408d203e

def test(args,  device):
    # Load grapher
    # -------------------------------------------------------------------------------
    exec_dir = os.environ['EXEC_DIR']
    
    checkpoint_dir = os.path.join(exec_dir, 'checkpoints')
    if args.checkpoint_model_id == -1:
        logging.info(f"Resuming test from location: {exec_dir}")
        checkpoint_model_path = os.path.join(checkpoint_dir, 'last.ckpt')
    else:
        logging.info(f"Resuming test from location: {exec_dir} and model at epoch {args.checkpoint_model_id}")
        checkpoint_model_path = os.path.join(exec_dir,
                                            'checkpoints',
                                            model_file_name(exec_dir, args.checkpoint_model_id))

    assert os.path.exists(checkpoint_model_path), 'Provided checkpoint does not exists, cannot run the test'

    grapher = LitGrapher.load_from_checkpoint(checkpoint_path=checkpoint_model_path)
    if device:
        grapher.to(device)
    grapher.eval()
    

    # Load data module
    # -------------------------------------------------------------------------------
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


    # Load trainer
    # -------------------------------------------------------------------------------
    # Logger for TensorBoard
    TB = pl_loggers.TensorBoardLogger(save_dir=os.environ['EVAL_DIR'],
                                      name='',
                                      version=exec_dir.split('/')[-1], 
                                      default_hp_metric=False)
    
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
