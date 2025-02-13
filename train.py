from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
import wandb  # Import wandb for logging

from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics

def train(args):
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Adjust default dtype.
    # Note: Apple MPS does not fully support double precision. In that case, we warn and use float32.
    if int(args.double_precision):
        # If double precision is requested but we will use MPS, warn and force float32.
        if not (int(args.cuda) >= 0) and torch.backends.mps.is_available():
            logging.warning("Double precision requested but Apple MPS does not fully support float64. Using float32 instead.")
            torch.set_default_dtype(torch.float32)
        else:
            torch.set_default_dtype(torch.float64)
    
    # Device selection:
    # Priority: CUDA if available and requested, then MPS if available, else CPU.
    if int(args.cuda) >= 0 and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda:" + str(args.cuda))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    args.device = device  # save device in args for later use

    # Configure patience
    args.patience = args.epochs if not args.patience else int(args.patience)
    
    # Set up logging.
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using device: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    # Initialize wandb
    wandb.init(project="MLP_reproduction_with_Bayesian_sweeping",
           config=vars(args),
           name=args.run_time)

    # Load data
    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape

    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel
        else:
            Model = RECModel
            # No validation for reconstruction task
            args.eval_freq = args.epochs + 1

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Instantiate model and optimizer
    model = Model(args)
    logging.info(str(model))
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")

    # If using CUDA, set environment variable (not needed for MPS)
    if device.type == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

    # Move model and data to the chosen device.
    model = model.to(device)
    for key, value in data.items():
        if torch.is_tensor(value):
            data[key] = value.to(device)

    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()

        # Log training metrics
        if (epoch + 1) % args.log_freq == 0:
            log_msg = " ".join(['Epoch: {:04d}'.format(epoch + 1),
                                'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                format_metrics(train_metrics, 'train'),
                                'time: {:.4f}s'.format(time.time() - t)])
            logging.info(log_msg)

            log_data = {
                "epoch": epoch + 1,
                "lr": lr_scheduler.get_last_lr()[0],
                "train_loss": train_metrics["loss"].item(),
                "train_time": time.time() - t
            }
            if "acc" in train_metrics:
                log_data["train_acc"] = train_metrics["acc"]
            if "f1" in train_metrics:
                log_data["train_f1"] = train_metrics["f1"]
            if 'roc' in train_metrics:
                log_data['roc'] = train_metrics['roc']
            if 'ap' in train_metrics:
                log_data['ap'] = train_metrics['ap']

            wandb.log(log_data)

        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, 'val')
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
            log_data = {
                "epoch": epoch + 1,
                "val_loss": val_metrics["loss"].item()
            }
            if "acc" in val_metrics:
                log_data["val_acc"] = val_metrics["acc"]
            if "f1" in val_metrics:
                log_data["val_f1"] = val_metrics["f1"]
            if 'roc' in val_metrics:
                log_data['roc'] = val_metrics['roc']    
            if 'ap' in val_metrics:
                log_data['ap'] = val_metrics['ap']
            wandb.log(log_data)

            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logging.info("Early stopping")
                    break

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    log_data = {}
    if "acc" in best_val_metrics:
        log_data["best_val_acc"] = best_val_metrics["acc"]
        log_data["test_acc"] = best_test_metrics["acc"]
    if "f1" in best_val_metrics:
        log_data["best_val_f1"] = best_val_metrics["f1"]
        log_data["test_f1"] = best_test_metrics["f1"]
    if 'roc' in best_val_metrics:
        log_data['val_roc'] = best_val_metrics['roc']
        log_data['test_roc'] = best_test_metrics['roc']
    if 'ap' in best_val_metrics:
        log_data['val_ap'] = best_val_metrics['ap']
        log_data['test_ap'] = best_test_metrics['ap']
    wandb.log(log_data)

    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f"Saved model in {save_dir}")

    wandb.finish()

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)