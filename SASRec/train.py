import os
import time
import wandb
import argparse
import random
import numpy as np
import torch

from model import SASRec
from utils import *


def setup_experiment(args):
    # argument file
    output_path = os.path.join(args.dataset + '_' + args.train_dir)
    if not os.path.isdir(output_path):
        os.makedirs(args.dataset + '_' + args.train_dir)

    with open(os.path.join(output_path, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    # log file
    log_file_path = os.path.join(output_path, 'log.txt')
    log = open(log_file_path, 'w')
    log.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
    print(f"Initialized log file: {log_file_path}")

    # wandb init
    run = wandb.init(
        entity="angela-politecnico-di-bari",
        project="SASRec",
        config={
            "learning_rate": args.lr,
            "architecture": "SASRec_Transformer",
            "dataset": args.dataset,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "maxlen": args.maxlen,
            "hidden_units": args.hidden_units,
            "num_blocks": args.num_blocks,
            "num_heads": args.num_heads,
            "dropout_rate": args.dropout_rate,
            "l2_emb": args.l2_emb,
            "device": args.device,
            "norm_first": args.norm_first,
            "eval_interval": args.eval_interval,
        },
    )

    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {args.seed}")

    return output_path, run, log


if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--train_dir', required=True)
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
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true', default=False)
    parser.add_argument('--eval_interval', default=20, type=int)
    args = parser.parse_args()

    # setup experiment environment
    output_path, run, log = setup_experiment(args)
    
    # load the dataset and splits it
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    # num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    # computes the number of batches
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    print(f'number of batches: {num_batch}')

    # computes the average sequence length
    total_interactions = 0.0
    for u in user_train:
        total_interactions += len(user_train[u])
    average_sequence_length = total_interactions / len(user_train)
    print('average sequence length: %.2f' % average_sequence_length)

    # Multi-process sampler
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    # model initialization
    model = SASRec(usernum, itemnum, args).to(args.device)  # no ReLU activation in original SASRec implementation?
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    # enable model training
    model.train()

    # checkpoint
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
            print(f"Model loaded from {args.state_dict_path}, resuming training from epoch {epoch_start_idx}")
        except Exception as e:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print(f'Failed loading state_dicts from {args.state_dict_path}. Error: {e}')
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb
            pdb.set_trace()

    # loss and optimizer
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # initialization of metrics and time
    best_val_ndcg, best_val_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        current_epoch_loss = 0.0

        for step in range(num_batch):   # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            # anche questo training loop è molto specifico ovviamente per il negative sampling
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            # torch.norm(param) returns the square root of the sum of squared weights (‖w‖₂), 
            # should be torch.norm(param)**2 or the way below which is faster.
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.sum(param ** 2)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step,
                                                             loss.item()))  # expected 0.4~0.6 after init few epochs
            current_epoch_loss += loss.item()

        avg_epoch_loss = current_epoch_loss / num_batch
        print(f"Epoch {epoch} finished. Average loss: {avg_epoch_loss:.4f}")

        if epoch % args.eval_interval == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            # questa cosa non ha molto senso, di solito si ottimizza solo per una metrica di validation
            # if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr:
            if t_valid[0] > best_val_ndcg:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                # best_val_hr = max(t_valid[1], best_val_hr)
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                model_path = os.path.join(output_path, fname)
                torch.save(model.state_dict(), model_path)
                run.save(model_path)

            run.log({
                "epoch": epoch,
                "avg_epoch_loss": avg_epoch_loss,
                "HR@10_test": t_test[1],
                "NDCG@10_test": t_test[0],
                "HR@10_valid": t_valid[1],
                "NDCG@10_valid": t_valid[0],
            })
            log.write(str(epoch) + ' ' + str(t_valid) + ' ' + str(t_test) + '\n')
            log.flush()
            t0 = time.time()
            model.train()

        # TODO: aggiungere early stopping e rimuovere questo salvataggio
        if epoch == args.num_epochs:
            fname = 'SASRec.final.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            model_path = os.path.join(output_path, fname)
            torch.save(model.state_dict(), model_path)
            run.save(model_path)
            print(f"Final model saved to {model_path} and uploaded to W&B.")
    
    log.close()
    sampler.close()
    run.finish()
    print("Done")
