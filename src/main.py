from argparse import ArgumentParser
from tasks import train, test, generate
import torch
import os
import nltk

# Parsing arguments
parser = ArgumentParser(description='Arguments')

parser.add_argument("--run", type=str, default='train')
parser.add_argument("--default_root_dir", type=str, default="output")
parser.add_argument('--pretrained_model', type=str, default='t5-large')
parser.add_argument('--cache_dir', type=str, default='cache')
parser.add_argument('--num_data_workers', type=int, default=3)
parser.add_argument('--every_n_epochs', type=int, default=-1)
parser.add_argument('--max_nodes', type=int, default=8)
parser.add_argument('--max_edges', type=int, default=7)
parser.add_argument('--default_seq_len_node', type=int, default=20)
parser.add_argument('--default_seq_len_edge', type=int, default=20)
parser.add_argument('--edges_as_classes', type=int, default=0)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument("--focal_loss_gamma", type=float, default=0.0)
parser.add_argument("--dropout_rate", type=float, default=0.5)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--eval_dump_only", type=int, default=0)
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
parser.add_argument("--dataset", type=str, default='webnlg')
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--checkpoint_model_id', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--prec_perf_tradeoff', type=int, default=0)
parser.add_argument("--inference_input_text", type=str,
                    default='Danielle Harris had a main role in Super Capers, a 98 minute long movie.') 

args = parser.parse_args()


# Run train, test or generate task
model_variant = "gen" if args.edges_as_classes == 0 else "class"
# For single GPU, specify device
if torch.cuda.device_count() <= 1:
    device = torch.device(f"cuda:{os.environ['CUDA_VISIBLE_DEVICES']}") 
else:
    device = None

if args.run in ['train', 'test']:
    # Download punkt tokenizer
    punkt_dir = f"{args.default_root_dir}/../lib/punkt"
    os.makedirs(punkt_dir, exist_ok=True)
    nltk.download('punkt', download_dir=punkt_dir)
    nltk.data.path.append(punkt_dir)

if args.run == "train":
    train.train(args, device)
elif args.run == "test":
    test.test(args, model_variant, device)
elif args.run == "inference":
    generate.generate(args, model_variant, device)
else:
    raise ValueError("The train argument must be one of the three values: train, test, inference.")