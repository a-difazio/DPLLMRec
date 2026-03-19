import os
import time
import wandb
import argparse
import random
import json
from torch.utils.data import DataLoader

from model import SASRec
from utils import *


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=200, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='mps', type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true', default=False)
    parser.add_argument('--eval_interval', default=10, type=int)
    parser.add_argument('--patience', default=5, type=int)
    return parser.parse_args()


def setup_dirs(args):
    base_dir = 'experiments'
    output_path = os.path.join(base_dir, f'{args.dataset}_{args.run_name}')
    checkpoint_dir = os.path.join(output_path, 'checkpoints')
    splits_dir = os.path.join(output_path, 'splits')

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    return output_path, checkpoint_dir

def save_config(args, output_path):
    path = os.path.join(output_path, 'config.json')

    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=4)

def init_wandb(args):

    run = wandb.init(
        entity="angela-politecnico-di-bari",
        project="SASRec",
        name=f"{args.dataset}_{args.run_name}",
        config=vars(args)
    )

    return run

def setup_logger(output_path):
    log_path = os.path.join(output_path, 'metrics.jsonl')
    logger = open(log_path, 'w')
    print(f"Initialized metrics logger file: {log_path}")

    return logger

def reset_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    elif torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024**2
    else:
        return 0

if __name__ == '__main__':

    # arguments
    args = parser()

    # setup experiment environment
    output_path, checkpoint_dir = setup_dirs(args)
    save_config(args, output_path)
    set_seed(args.seed)
    run = init_wandb(args)
    logger = setup_logger(output_path)

    # load the dataset and splits it
    dataset = data_partition(os.path.join('data', f'{args.dataset}.tsv'))
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    # computes the average sequence length
    total_interactions = 0.0
    for u in user_train:
        total_interactions += len(user_train[u])
    average_sequence_length = total_interactions / len(user_train)
    print(f'average sequence length: {average_sequence_length:2f}')

    # Dataloader
    train_dataloader = DataLoader(SASRecDataset(user_train, usernum, args.maxlen),
                                  batch_size=args.batch_size,
                                  num_workers=3,
                                  shuffle=True)
    val_dataloader = DataLoader(SASRecDataset(user_valid, usernum, args.maxlen),
                                batch_size=args.batch_size,
                                num_workers=3,
                                shuffle=False,
                                pin_memory=True)
    test_dataloader = DataLoader(SASRecDataset(user_test, usernum, args.maxlen),
                                 batch_size=args.batch_size,
                                 num_workers=3,
                                 shuffle=False,
                                 pin_memory=True)

    num_batch = len(train_dataloader)
    print(f'number of batches: {num_batch}')

    # model initialization
    model = SASRec(usernum, itemnum, args).to(args.device)  # no ReLU activation in original SASRec implementation?

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    # loss and optimizer
    ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # checkpoint
    epoch_start_idx = 1
    if args.checkpoint is not None:
        try:
            checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=args.device)

            model.load_state_dict(checkpoint["model_state_dict"])
            adam_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            epoch_start_idx = checkpoint["epoch"] + 1
        except Exception as e:
            print(f'Failed to load checkpoint. Error: {e}')

    # initialization of metrics and time
    counter = 0
    patience = args.patience
    best_val = float('inf')
    T = 0.0
    t0 = time.time()
    reset_memory_stats()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        current_epoch_loss = 0.0

        for step, batch in enumerate(train_dataloader):

            udis, seqs, targets = batch
            udis = udis.to(args.device)
            seqs = seqs.to(args.device)
            targets = targets.to(args.device)

            adam_optimizer.zero_grad()

            logits = model(udis, seqs)

            batch_size, maxlen, num_items = logits.shape
            logits = logits.view(-1, num_items)  # (batch_size*maxlen, num_items)
            targets = targets.view(-1)  # (batch_size*maxlen,)

            loss = ce_criterion(logits, targets)

            # l2 regularization
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.sum(param ** 2)

            loss.backward()
            adam_optimizer.step()


            current_epoch_loss += loss.item()

        avg_epoch_loss = current_epoch_loss / num_batch
        print(f"Epoch {epoch} | loss: {avg_epoch_loss:.4f} | time: {time.time() - t0:.1f}s")


        if epoch % args.eval_interval  == 0:
            t1 = time.time() - t0
            T += t1

            print('Evaluating')
            t_valid = evaluate(model, val_dataloader, args)
            t_test = evaluate(model, test_dataloader, args)

            metrics = {
                "epoch": epoch,
                "elapsed_s": T,
                "avg_epoch_loss": avg_epoch_loss,
                "valid_loss": t_valid[0],
                "valid_ppl": t_valid[1],
                "valid_top1": t_valid[2],
                "valid_top10": t_valid[3],
                "test_loss": t_test[0],
                "test_ppl": t_test[1],
                "test_top1": t_test[2],
                "test_top10": t_test[3],
            }

            print(
                f"Eval epoch: {metrics['epoch']}, time: {metrics['elapsed_s']:.1f}s, "
                f"train_loss: {metrics['avg_epoch_loss']:.4f}, "
                f"valid(loss: {metrics['valid_loss']:.4f}, ppl: {metrics['valid_ppl']:.4f}, "
                f"top1: {metrics['valid_top1']:.4f}, top10: {metrics['valid_top10']:.4f}), "
                f"test(loss: {metrics['test_loss']:.4f}, ppl: {metrics['test_ppl']:.4f}, "
                f"top1: {metrics['test_top1']:.4f}, top10: {metrics['test_top10']:.4f})"
            )

            fname = f'checkpoint_last.pth'
            checkpoint_path = os.path.join(checkpoint_dir, fname)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": adam_optimizer.state_dict(),
                "args": vars(args)
            }, checkpoint_path)

            if t_valid[0] < best_val:
                counter = 0
                best_val = min(t_valid[0], best_val)
                fname = f'checkpoint_best.pth'
                checkpoint_path = os.path.join(checkpoint_dir, fname)
                torch.save(model.state_dict(), checkpoint_path)
                print(f'Saved best checkpoint at epoch {epoch} (val_loss={t_valid[0]:.4f})')
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch}.')
                    break

            run.save(checkpoint_path)
            run.log({
                "train_loss": metrics['avg_epoch_loss'],
                "valid_loss": metrics['valid_loss'],
                "valid_top1": metrics['valid_top1'],
                "valid_top10": metrics['valid_top10'],
                "test_loss": metrics['test_loss'],
                "test_top1": metrics['test_top1'],
                "test_top10": metrics['test_top10'],
            }, step=epoch)

            json.dump(metrics, logger)
            logger.write('\n')
            logger.flush()

            t0 = time.time()

    mem = get_memory()
    print(f"[STATS] Batch size: {args.batch_size}")
    print(f"[STATS] Max GPU memory: {mem:.2f} MB")

    logger.close()
    run.finish()
    print("Done")
