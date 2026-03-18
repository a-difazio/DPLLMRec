import sys
import copy
import random
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict


class SASRecDataset(Dataset):
    def __init__(self, user_sequences, usernum, maxlen):
        self.user_sequences = user_sequences
        self.maxlen = maxlen
        self.users = [u for u in range(1, usernum+1) if len(user_sequences[u]) > 1]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        uid = self.users[index]
        history = self.user_sequences[uid]

        input_items = history[:-1]
        target_items = history[1:]

        input_items = input_items[-self.maxlen:]
        target_items = target_items[-self.maxlen:]

        # passare direttamente a tensori se riesco
        # seq = torch.zeros(self.maxlen, dtype=torch.long)
        seq = np.zeros([self.maxlen], dtype=np.int64)
        target = np.zeros([self.maxlen], dtype=np.int64)

        # seq[-len(input_items):] = torch.tensor(input_items, dtype=torch.long)
        # target[-len(target_items):] = torch.tensor(target_items, dtype=torch.long)

        seq[-len(input_items):] = input_items
        target[-len(target_items):] = target_items

        return uid, seq, target

def evaluate(model, dataloader, args):
    model.eval()

    ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    total_loss = 0.0
    total_predictions = 0
    correct1 = 0.0
    correctk = 0.0

    for batch in dataloader:
        uids, seqs, targets = batch
        uids = uids.to(args.device)
        seqs = seqs.to(args.device)
        targets = targets.to(args.device)

        with torch.no_grad():
            logits = model(uids, seqs)
            batch_size, maxlen, num_items = logits.shape
            logits = logits.view(-1, num_items)  # (batch_size*maxlen, num_items)
            targets = targets.view(-1)  # (batch_size*maxlen,)

            mask = targets != 0
            valid_logits = logits[mask]
            valid_targets = targets[mask]

            pred1 = valid_logits.argmax(-1)
            correct1 += (pred1 == valid_targets).sum().item()

            topk_idx = valid_logits.topk(10, dim=1).indices
            correctk += (topk_idx == valid_targets.unsqueeze(1)).any(dim=1).sum().item()

            loss = ce_criterion(logits, targets)

        total_loss += loss.item()
        total_predictions += (targets != 0).sum().item()

    avg_loss = total_loss / total_predictions
    perplexity = np.exp(avg_loss)
    top1 = correct1 / total_predictions
    topk = correctk / total_predictions

    return avg_loss, perplexity, top1, topk

# divido caricamento e splitting, che mi sembra la cosa migliore effettivamente
def load_interactions(fname):

    user_sequences = defaultdict(list)
    usernum = 0
    itemnum = 0

    with open(fname, 'r') as f:
        for line in f:
            u, i = line.rstrip().split('\t')
            u = int(u)
            i = int(i)

            user_sequences[u].append(i)

            usernum = max(u, usernum)
            itemnum = max(i, itemnum)

    return user_sequences, usernum, itemnum


def temporal_split(user_sequences, val_ratio=0.1, test_ratio=0.1, min_interaction=5):

    user_train = {}
    user_valid = {}
    user_test = {}

    for user, seq in user_sequences.items():

        n = len(seq)

        if n < min_interaction:
            user_train[user] = seq
            user_valid[user] = []
            user_test[user] = []
            continue

        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        train_end = n - (n_val + n_test)
        val_end = n - n_val

        user_train[user] = seq[:train_end]
        user_valid[user] = seq[train_end:val_end]
        user_test[user] = seq[val_end:]

    return user_train, user_valid, user_test


def data_partition(fname, val_ratio=0.1, test_ratio=0.1):

    user_sequences, usernum, itemnum = load_interactions(fname)
    user_train, user_valid, user_test = temporal_split(user_sequences, val_ratio, test_ratio)

    return [user_train, user_valid, user_test, usernum, itemnum]

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}.")
